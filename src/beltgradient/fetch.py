"""Download and verify the two inputs.

* ``catalog_snapshot.parquet``: the columns of the AsteroidCatalog build of 2026-08-11 (pipeline
  1.1.0) used here, frozen because JPL adds bodies daily and the build cannot be refetched
  identically. Attached to this repository's ``data-v1`` release.
* ``ast.nesvorny.families_V2_0``: Nesvorný HCM families and proper elements, PDS Small Bodies Node.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from . import __version__
from .config import Paths

SNAPSHOT_URL = "https://github.com/loggger101/asteroid-belt-gradient/releases/download/data-v1/catalog_snapshot.parquet"
SNAPSHOT_SHA256 = "f5ac7e68f6db758dba8e786092d7f7967d83b14b93d97473665f739c4e36d498"
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
    download(SNAPSHOT_URL, paths.snapshot, SNAPSHOT_SHA256)
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
