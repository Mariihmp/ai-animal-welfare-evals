"""Shared NVIDIA NIM client: key rotation, 429 backoff, on-disk JSONL cache, thread pool.

Keys come from the repo-root .env: NVIDIA_API_KEY, NVIDIA_API_KEY_2, NVIDIA_API_KEY_3, ...
Never print or log keys.
"""
import hashlib
import itertools
import json
import os
import pathlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_URL = "https://integrate.api.nvidia.com/v1"
load_dotenv(REPO / ".env")


def api_keys():
    keys = []
    for name in ["NVIDIA_API_KEY", "NVIDIA_KEY"] + [f"NVIDIA_API_KEY_{i}" for i in range(2, 10)]:
        v = os.environ.get(name, "").strip()
        if v and not v.startswith("nvapi-your"):
            keys.append(v)
    return keys


def have_key():
    return bool(api_keys())


class NIM:
    def __init__(self, cache_path, max_retries=8):
        keys = api_keys()
        if not keys:
            raise RuntimeError("No NVIDIA_API_KEY in .env")
        self.clients = [OpenAI(base_url=BASE_URL, api_key=k, timeout=600, max_retries=0) for k in keys]
        self._cycle = itertools.cycle(range(len(self.clients)))
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.n429 = 0
        self.cooldown = [0.0] * len(self.clients)  # per-key pause after a 403/429 burst block
        self.cache_path = pathlib.Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
                except json.JSONDecodeError:
                    pass

    def _pick(self):
        """Next key not in cooldown (round robin); if all are cooling down, the one that frees up first."""
        with self._lock:
            now = time.time()
            for _ in range(len(self.clients)):
                i = next(self._cycle)
                if self.cooldown[i] <= now:
                    return i, 0.0
            i = min(range(len(self.clients)), key=lambda j: self.cooldown[j])
            return i, self.cooldown[i] - now

    @staticmethod
    def make_key(model, messages, params):
        blob = json.dumps(dict(m=model, msg=messages, p=params), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def chat(self, model, messages, tag=None, **params):
        """Return a cached record {key, model, content, logprobs, error, ...}."""
        key = self.make_key(model, messages, params)
        if key in self.cache and self._usable(self.cache[key]):
            return self.cache[key]
        err, rec = None, None
        for attempt in range(self.max_retries):
            ki, wait = self._pick()
            if wait > 0:
                time.sleep(wait + random.random() * 5)
            client = self.clients[ki]
            try:
                if "logprobs" in params:  # reward models: short, non-streamed
                    resp = client.chat.completions.create(model=model, messages=messages, **params)
                    ch = resp.choices[0]
                    lp = None
                    if getattr(ch, "logprobs", None) and getattr(ch.logprobs, "content", None):
                        lp = [dict(token=t.token, logprob=t.logprob) for t in ch.logprobs.content[:8]]
                    content, finish, n_reason = ch.message.content, ch.finish_reason, 0
                else:  # stream so long generations don't hit the ~90 s idle cut-off seen on this network
                    lp = None
                    content, finish, n_reason = self._stream(client, model, messages, params)
                rec = dict(key=key, tag=tag, model=model, messages=messages, params=params,
                           content=content, logprobs=lp, finish_reason=finish, reasoning_chars=n_reason,
                           error=None, t=time.time())
                if not self._usable(rec):
                    err = f"empty content (finish={finish}, reasoning_chars={n_reason})"
                    rec = None
                    continue
                break
            except Exception as e:  # rate limits, timeouts, 5xx
                err = f"{type(e).__name__}: {str(e)[:300]}"
                status = getattr(e, "status_code", None)
                print(f"    ! key{ki + 1} {model.split('/')[-1]} attempt {attempt}: {status} {type(e).__name__} {str(e)[:80]!r}",
                      flush=True)
                if status in (403, 429):  # burst block / rate limit: pause this key, retry on another
                    self.n429 += 1
                    with self._lock:
                        self.cooldown[ki] = max(self.cooldown[ki], time.time() + 90)
                    continue
                if status in (400, 401, 404, 422):
                    break  # not retryable
                time.sleep(min(60, (2 ** attempt) + random.random() * 2))
        if rec is None:
            rec = dict(key=key, tag=tag, model=model, messages=messages, params=params,
                       content=None, logprobs=None, error=err, t=time.time())
        with self._lock:
            self.cache[key] = rec
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    @staticmethod
    def _usable(rec):
        if rec.get("error"):
            return False
        return not (not (rec.get("content") or "").strip() and rec.get("finish_reason") == "length")

    def _stream(self, client, model, messages, params):
        parts, reasoning, finish = [], 0, None
        stream = client.chat.completions.create(model=model, messages=messages, stream=True, **params)
        for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            if getattr(d, "content", None):
                parts.append(d.content)
            r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if r:
                reasoning += len(r)
            if ch.finish_reason:
                finish = ch.finish_reason
        return "".join(parts), finish, reasoning

    def map(self, jobs, workers=8, progress_every=25):
        """jobs: list of dicts with keys model, messages, tag, params. Returns list of records in order."""
        out = [None] * len(jobs)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.chat, j["model"], j["messages"], j.get("tag"), **j.get("params", {})): i
                    for i, j in enumerate(jobs)}
            for n, fut in enumerate(as_completed(futs), 1):
                out[futs[fut]] = fut.result()
                if n % progress_every == 0:
                    errs = sum(1 for r in out if r and r.get("error"))
                    print(f"  {n}/{len(jobs)} done, {errs} errors, {self.n429} rate-limit hits, "
                          f"{time.time() - t0:.0f}s", flush=True)
        return out


def list_models():
    keys = api_keys()
    c = OpenAI(base_url=BASE_URL, api_key=keys[0], timeout=60)
    return sorted(m.id for m in c.models.list().data)
