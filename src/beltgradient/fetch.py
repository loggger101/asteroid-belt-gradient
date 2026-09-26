"""Download and verify the two inputs.

* The AsteroidCatalog release ``CATALOG_RELEASE`` (``data-2026-09-26``: 1,567,657 bodies, data
  contract 1.4.1). A release is one frozen build that is never rebuilt under its tag; the sha256
  below is the one its ``manifest.json`` lists for ``asteroid_catalog.parquet``.
* ``ast.nesvorny.families_V2_0``: Nesvorný HCM families and proper elements, PDS Small Bodies Node.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from . import __version__
from .config import CATALOG_RELEASE, Paths

CATALOG_URL = f"https://github.com/loggger101/AsteroidCatalog/releases/download/{CATALOG_RELEASE}/asteroid_catalog.parquet"
CATALOG_SHA256 = "1b923a41b8eda45c7823f17d87e2c88473c2f1c1931c0c9417235fa73c59a33d"
NESVORNY_URL = "https://sbnarchive.psi.edu/pds4/non_mission/ast.nesvorny.families_V2_0.zip"
NESVORNY_SHA256 = "4adf5a341eaea3f1fb209ccb4c875188f224a65b5f260d1d99e10be28f1b3145"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    if dest.exists() and sha256(dest) == expected:
        print(f"ok        {dest.name}")
        return
    print(f"download  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # the PDS archive refuses Python's default User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": f"beltgradient/{__version__}"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise RuntimeError(f"{dest.name}: sha256 {got} != expected {expected}")
    tmp.replace(dest)


def fetch(paths: Paths | None = None, keep_zip: bool = False) -> None:
    paths = paths or Paths.default()
    paths.data.mkdir(parents=True, exist_ok=True)
    download(CATALOG_URL, paths.catalog, CATALOG_SHA256)
    if (paths.nesvorny / "data" / "proper_catalog24.tab").exists():
        print(f"ok        {paths.nesvorny.name}/")
        return
    z = paths.data / "nesvorny_families_v2.zip"
    download(NESVORNY_URL, z, NESVORNY_SHA256)
    with zipfile.ZipFile(z) as zf:
        top = {n.split("/")[0] for n in zf.namelist()}
        zf.extractall(paths.data if top == {paths.nesvorny.name} else paths.nesvorny)
    if not keep_zip:
        z.unlink()
    print(f"extracted {paths.nesvorny}")
