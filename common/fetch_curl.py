"""Robust model fetcher for a flaky connection.

One fresh `curl -C -` process per attempt (survives DNS drops that wedge in-process HTTP clients),
byte-range resume, sha256 verification against Hugging Face LFS metadata.
Files land in $HF_LOCAL_DIR/<org>/<name>/<file> (default E:/hf_cache/direct).
Usage: python fetch_curl.py [--parallel N] <repo_id> [...]   -> prints DOWNLOADED <repo> / FAILED <repo>
--parallel downloads N files of a repo at once (Hugging Face throttles each connection, so shards go faster).
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DEST = pathlib.Path(os.environ.get("HF_LOCAL_DIR", "E:/hf_cache/direct"))
KEEP = re.compile(r"\.(json|safetensors|model|txt|jinja|tiktoken)$|^pytorch_model\.bin$")
SKIP = re.compile(r"^(training_args\.bin|README|\.gitattributes|LICENSE)")


def log(*a):
    print(*a, flush=True)


def curl(args, timeout):
    try:
        return subprocess.run(["curl", "-sS", "-L", "--connect-timeout", "30", *args],
                              capture_output=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 28


def manifest(repo):
    tmp = DEST / repo / ".api.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for i in range(30):
        if curl(["-m", "60", "-o", str(tmp), f"https://huggingface.co/api/models/{repo}?blobs=true"], 90) == 0:
            try:
                d = json.loads(tmp.read_text(encoding="utf-8"))
                return [(s["rfilename"], (s.get("lfs") or {}).get("sha256"), s.get("size"))
                        for s in d["siblings"]
                        if "/" not in s["rfilename"] and KEEP.search(s["rfilename"]) and not SKIP.search(s["rfilename"])]
            except Exception as e:
                log("  manifest parse error", e)
        time.sleep(min(60, 5 * (i + 1)))
    raise RuntimeError("could not fetch manifest")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def good(p, sha, size):
    if not p.exists():
        return False
    if size is not None and p.stat().st_size != size:
        return False
    return sha is None or sha256(p) == sha


def fetch_file(repo, name, sha, size):
    p = DEST / repo / name
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    for i in range(80):
        if good(p, sha, size):
            return True
        if p.exists() and size is not None and p.stat().st_size >= size:  # full size but bad hash: corrupt
            log(f"  corrupt {name}, restarting")
            p.unlink()
        before = p.stat().st_size if p.exists() else 0
        rc = curl(["-C", "-", "-m", "1800", "-Y", "20000", "-y", "45", "-o", str(p), url], 1900)
        after = p.stat().st_size if p.exists() else 0
        if rc != 0:
            log(f"  {name}: curl rc={rc}, {after / 1e6:.0f}/{(size or 0) / 1e6:.0f} MB")
            if after == before:
                time.sleep(min(60, 5 * (i + 1)))
    return good(p, sha, size)


def fetch(repo, parallel=1):
    files = manifest(repo)
    with ThreadPoolExecutor(parallel) as ex:
        ok = list(ex.map(lambda f: fetch_file(repo, *f), files))
    bad = [f[0] for f, k in zip(files, ok) if not k]
    if bad:
        raise RuntimeError(f"gave up on {bad}")


if __name__ == "__main__":
    args = sys.argv[1:]
    par = 1
    if args and args[0] == "--parallel":
        par, args = int(args[1]), args[2:]
    for repo in args:
        t0 = time.time()
        try:
            fetch(repo, par)
            log(f"DOWNLOADED {repo} in {time.time() - t0:.0f}s")
        except Exception as e:
            log(f"FAILED {repo}: {e}")
