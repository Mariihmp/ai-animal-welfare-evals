"""Download the primary sources behind WelfareFacts into references/ (the files every answer key is verified against).

- EU law: the original acts as HTML from the Publications Office CELLAR service (the machine-access route to EUR-Lex;
  eur-lex.europa.eu itself answers scripted requests with a bot challenge). Reuse is permitted with acknowledgement
  (Commission Decision 2011/833/EU).
  The .txt files are the HTML with tags stripped; 01_build_prompts.py normalises whitespace before matching quotes.
- UK legislation (context for AnimalXSTest's legal bases, not used in scoring): legislation.gov.uk, Open Government
  Licence v3.0.
- FAOSTAT Production (QCL) bulk download, CC BY 4.0. Only the World "Producing Animals/Slaughtered" rows for the 7 species
  used are kept, in faostat_world_slaughtered.csv. The 34 MB zip and 545 MB CSV stay out of git. FAOSTAT revises past
  years, so the committed extract is the October 2026 snapshot the facts were checked against.

Usage: python 00_fetch_references.py [--out DIR] [--zip EXISTING_ZIP] [--skip-faostat]
"""
import argparse
import html
import pathlib
import re
import subprocess
import tempfile
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLAR = "http://publications.europa.eu/resource/celex/{}"  # content negotiation: HTML, English
EU = {  # file stem -> CELEX number
    "dir_1999_74_laying_hens": "31999L0074",
    "dir_2008_120_pigs": "32008L0120",
    "dir_2007_43_broilers": "32007L0043",
    "dir_2008_119_calves": "32008L0119",
    "reg_2005_1_transport": "32005R0001",
    "reg_2009_1099_slaughter": "32009R1099",
}
UK = {  # file stem -> legislation.gov.uk path
    "uk_ecollars_wales_2010": "wsi/2010/943/made",
    "uk_glue_traps_act_2022": "ukpga/2022/26/enacted",
    "uk_sentience_act_2022": "ukpga/2022/22/enacted",
}
UK_URL = "https://www.legislation.gov.uk/{}/data.xht?view=snippet&wrap=true"
FAO_URL = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
FAO_ITEMS = [867, 977, 1017, 1035, 1058, 1069, 1080]  # cattle, sheep, goat, pig, chickens, ducks, turkeys
FAO_COLS = ["Area Code", "Area", "Item Code", "Item", "Element", "Year", "Unit", "Value"]


def curl(url, dest, headers=()):
    h = [x for k in headers for x in ("-H", k)]
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", *h, "-o", str(dest), url], check=True)
    if pathlib.Path(dest).stat().st_size == 0:
        raise RuntimeError(f"empty response from {url}")


def html_to_text(s):
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(s)).strip()


def fetch_pages(out):
    for stem, celex in EU.items():
        for mime in ("application/xhtml+xml", "text/html"):  # acts from ~2005 on are stored as XHTML, older ones as HTML
            try:
                curl(CELLAR.format(celex), out / f"{stem}.html", (f"Accept: {mime}", "Accept-Language: eng"))
                break
            except (subprocess.CalledProcessError, RuntimeError):
                continue
        else:
            raise RuntimeError(f"no HTML or XHTML manifestation for CELEX {celex}")
    for stem, path in UK.items():
        curl(UK_URL.format(path), out / f"{stem}.html")
    for p in sorted(out.glob("*.html")):
        p.with_suffix(".txt").write_text(html_to_text(p.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        print("fetched", p.name)


def faostat_extract(zip_path, out):
    z = zipfile.ZipFile(zip_path)
    name = next(n for n in z.namelist() if n.endswith("All_Data_(Normalized).csv"))
    keep = []
    with z.open(name) as f:
        for chunk in pd.read_csv(f, usecols=FAO_COLS, chunksize=500_000, encoding="latin-1", low_memory=False):
            c = chunk[(chunk["Area Code"] == 5000) & (chunk["Element"] == "Producing Animals/Slaughtered")
                      & chunk["Item Code"].isin(FAO_ITEMS)]
            keep.append(c)
    d = pd.concat(keep).sort_values(["Item Code", "Year"])
    d.to_csv(out / "faostat_world_slaughtered.csv", index=False)
    print(f"faostat_world_slaughtered.csv: {len(d)} rows, years {d.Year.min()}-{d.Year.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "references"))
    ap.add_argument("--zip", help="use an already-downloaded FAOSTAT zip instead of downloading it")
    ap.add_argument("--skip-faostat", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch_pages(out)
    if not a.skip_faostat:
        if a.zip:
            faostat_extract(a.zip, out)
        else:
            with tempfile.TemporaryDirectory() as td:
                zp = pathlib.Path(td) / "qcl.zip"
                curl(FAO_URL, zp)
                faostat_extract(zp, out)


if __name__ == "__main__":
    main()
