#!/usr/bin/env python3
"""
A1 — Acquire & pin OptimusKG.

The optimuskg client CANNOT pin a dataset version: every call resolves
`latestVersion` from the Dataverse API. `set_doi()` selects a *dataset*, not a
release. So "pinning" here means: resolve the version, record it plus per-file
checksums in a manifest, and fail loudly on any later drift.

Usage:
    python scripts/a1_pin_okg.py                 # download the default file set
    python scripts/a1_pin_okg.py --list          # list remote files, download nothing
    python scripts/a1_pin_okg.py --all           # download every file in the release
    python scripts/a1_pin_okg.py --verify        # re-verify manifest against cache + remote
    python scripts/a1_pin_okg.py --files a b c   # download specific relative paths
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

import optimuskg

# --------------------------------------------------------------------------
# Config. Repo-relative so the script is runnable from the repo root only.
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "okg_cache"
MANIFEST = REPO / "data" / "manifest" / "okg_manifest.json"

DOI = os.environ.get("OPTIMUSKG_DOI", "doi:10.7910/DVN/IYNGEV")
SERVER = os.environ.get("OPTIMUSKG_SERVER", "https://dataverse.harvard.edu")

# Flat tables: `properties` is a JSON *string*. Wanted for the Neo4j bulk
# import in A3, where you only need id/label/from/to/relation anyway.
FLAT_LCC = [
    "largest_connected_component_nodes.parquet",
    "largest_connected_component_edges.parquet",
]
FLAT_FULL = ["nodes.parquet", "edges.parquet"]

# Stratified per-type tables: `properties` is expanded to a native Polars
# Struct. Much cheaper for A2/A4/A6 than JSON-parsing 21M rows.
STRATIFIED = [
    "nodes/drug.parquet",       # carries inchi_key, canonical_smiles, is_approved
    "nodes/disease.parquet",    # carries umls_cui, concept_ids, xrefs
    "nodes/gene.parquet",       # carries HGNC symbol
    "edges/drug_disease.parquet",   # INDICATION / CONTRAINDICATION / OFF_LABEL_USE
    "edges/drug_gene.parquet",      # mechanism edges
    "edges/disease_gene.parquet",
    "edges/pathway_gene.parquet",
]

DEFAULT_FILES = FLAT_LCC + STRATIFIED


# --------------------------------------------------------------------------
# Dataverse metadata (raw API — the client drops checksums and sizes)
# --------------------------------------------------------------------------
def fetch_release_metadata() -> dict:
    """Resolve the latest published version of DOI and its file table."""
    url = f"{SERVER}/api/datasets/:persistentId/"
    resp = requests.get(url, params={"persistentId": DOI}, timeout=60)
    resp.raise_for_status()
    latest = resp.json()["data"]["latestVersion"]

    major, minor = latest.get("versionNumber"), latest.get("versionMinorNumber")
    version = f"{major}.{minor}" if major is not None and minor is not None else str(
        latest.get("versionState", "DRAFT")
    ).lower()

    files = {}
    for raw in latest.get("files", []):
        df = raw["dataFile"]
        directory = (raw.get("directoryLabel") or "").strip("/")
        rel = f"{directory}/{df['filename']}" if directory else df["filename"]
        checksum = df.get("checksum") or {}
        files[rel] = {
            "dataverse_file_id": int(df["id"]),
            "filesize_bytes": df.get("filesize"),
            "remote_checksum_type": checksum.get("type") or ("MD5" if df.get("md5") else None),
            "remote_checksum_value": checksum.get("value") or df.get("md5"),
        }

    return {
        "doi": DOI,
        "server": SERVER,
        "dataset_version": version,
        "version_state": latest.get("versionState"),
        "release_time": latest.get("releaseTime"),
        "files": files,
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list remote files and exit")
    ap.add_argument("--all", action="store_true", help="download every file in the release")
    ap.add_argument("--files", nargs="+", help="specific relative paths to download")
    ap.add_argument("--verify", action="store_true", help="re-verify an existing manifest")
    ap.add_argument("--allow-checksum-dispute", action="store_true",
                    help="proceed when a deterministic download disagrees with the "
                         "checksum Dataverse publishes; the dispute is recorded in "
                         "the manifest")
    args = ap.parse_args()

    optimuskg.set_cache_dir(CACHE_DIR)
    optimuskg.set_doi(DOI)
    optimuskg.set_server(SERVER)

    print(f"DOI    : {DOI}")
    print(f"Server : {SERVER}")
    print(f"Cache  : {CACHE_DIR}")

    meta = fetch_release_metadata()
    print(f"Version: {meta['dataset_version']}  (state={meta['version_state']}, "
          f"released={meta['release_time']})")

    if meta["version_state"] != "RELEASED":
        print("\n!! latestVersion is not RELEASED. A draft is mutable and unciteable.\n"
              "!! Do not build a holdout on it.", file=sys.stderr)
        return 2

    if args.list:
        print(f"\n{len(meta['files'])} files in release {meta['dataset_version']}:")
        for rel, info in sorted(meta["files"].items()):
            print(f"  {human(info['filesize_bytes']):>9}  {rel}")
        total = sum(f["filesize_bytes"] or 0 for f in meta["files"].values())
        print(f"  {'-'*9}\n  {human(total):>9}  TOTAL")
        return 0

    # --- drift check against a previous manifest ---------------------------
    if MANIFEST.exists():
        prev = json.loads(MANIFEST.read_text())
        if prev["dataset_version"] != meta["dataset_version"]:
            print(
                f"\n!! DRIFT: manifest pins version {prev['dataset_version']}, "
                f"Dataverse now serves {meta['dataset_version']}.\n"
                f"!! The client always follows latestVersion, so a fresh download "
                f"would silently give you different data.\n"
                f"!! Delete data/manifest/okg_manifest.json only if you intend to "
                f"re-baseline the whole of Phase A.",
                file=sys.stderr,
            )
            return 3
        print(f"Manifest agrees with remote (version {meta['dataset_version']}).")

    wanted = list(meta["files"]) if args.all else (args.files or DEFAULT_FILES)

    missing = [w for w in wanted if w not in meta["files"]]
    if missing:
        print(f"\n!! Not in this release: {missing}", file=sys.stderr)
        print(f"!! Run --list to see the real paths.", file=sys.stderr)
        return 4

    # --- disk headroom check ----------------------------------------------
    need = sum(meta["files"][w]["filesize_bytes"] or 0 for w in wanted)
    free = shutil.disk_usage(CACHE_DIR.parent if CACHE_DIR.parent.exists() else REPO).free
    print(f"\nDownload set: {len(wanted)} files, {human(need)} "
          f"({human(free)} free on this volume)")
    if free < need * 1.3:
        print("!! Not enough headroom (want ~1.3x the download size). "
              "Grow the EBS volume first.", file=sys.stderr)
        return 5

    # --- download + verify -------------------------------------------------
    records = {}
    for rel in wanted:
        info = meta["files"][rel]
        print(f"\n-> {rel}  ({human(info['filesize_bytes'])})")
        if args.verify:
            # don't re-download; just locate what get_file would return
            path = CACHE_DIR / DOI.replace(":", "_").replace("/", "_").replace(".", "_") \
                   / meta["dataset_version"] / rel
            if not path.exists():
                print(f"   MISSING from cache", file=sys.stderr)
                return 6
        else:
            path = optimuskg.get_file(rel)

        size = path.stat().st_size
        if info["filesize_bytes"] and size != info["filesize_bytes"]:
            print(f"   !! SHORT: local {size:,} B vs remote {info['filesize_bytes']:,} B "
                  f"— truncated download, retrying once", file=sys.stderr)
            path = optimuskg.get_file(rel, force=True)
            size = path.stat().st_size
            if info["filesize_bytes"] and size != info["filesize_bytes"]:
                print(f"   !! still short after retry — give up", file=sys.stderr)
                return 7

        # get_file() does no integrity check of its own, so do it here.
        # A size match with a hash mismatch is NOT truncation. Either the bytes
        # were mangled in flight (a retry gives a different hash) or Dataverse's
        # recorded checksum is wrong (a retry gives the same hash). Those are
        # different problems; find out which before failing.
        dispute = None
        algo = (info["remote_checksum_type"] or "").upper()
        expected = (info["remote_checksum_value"] or "").lower()
        hasher = {"MD5": md5_of, "SHA-256": sha256_of}.get(algo)

        if not expected:
            print(f"   ?? no remote checksum published — sha256 recorded, unverified")
        elif hasher is None:
            print(f"   ?? remote checksum is {algo}, which this script can't compute "
                  f"— sha256 recorded, unverified")
            dispute = {"reason": "unsupported_algorithm", "algorithm": algo}
        else:
            local = hasher(path)
            if local != expected:
                print(f"   !! {algo} mismatch (size is correct — not truncation)\n"
                      f"      remote : {expected}\n"
                      f"      local  : {local}\n"
                      f"      re-downloading to test determinism…", file=sys.stderr)
                path = optimuskg.get_file(rel, force=True)
                again = hasher(path)
                if again == expected:
                    print(f"      retry matches — first download was corrupt in flight")
                    local = again
                elif again == local:
                    print(f"      !! retry is byte-identical ({again})\n"
                          f"      !! download is deterministic; the REMOTE checksum is "
                          f"the outlier.\n"
                          f"      !! Verify the parquet reads, then re-run with "
                          f"--allow-checksum-dispute to record this and continue.",
                          file=sys.stderr)
                    if not args.allow_checksum_dispute:
                        return 8
                    dispute = {"reason": "remote_checksum_disagrees",
                               "algorithm": algo, "remote": expected, "local": again}
                else:
                    print(f"      !! two downloads disagree with each other "
                          f"({local} vs {again}) — the transport is unreliable, "
                          f"not the file. Retry on a fresh connection.", file=sys.stderr)
                    return 8
            if dispute is None:
                print(f"   {algo.lower()} ok ({local[:12]}…)")

        digest = sha256_of(path)
        print(f"   sha256 {digest[:12]}…")
        records[rel] = {
            **info,
            "local_path": str(path.relative_to(REPO)),
            "sha256": digest,
            **({"checksum_dispute": dispute} if dispute else {}),
        }

    # --- write manifest ----------------------------------------------------
    manifest = {
        "step": "A1",
        "doi": DOI,
        "server": SERVER,
        "dataset_version": meta["dataset_version"],
        "version_state": meta["version_state"],
        "release_time": meta["release_time"],
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "client": {
            "package": "optimuskg",
            "dist_version": _dist_version(),
            "dunder_version": optimuskg.__version__,  # stale in 1.0.0 — do not cite this
        },
        "pinning_note": (
            "The optimuskg client always resolves Dataverse latestVersion; set_doi() "
            "selects a dataset, not a release. Reproducibility rests on dataset_version "
            "plus the sha256 values below, not on the DOI alone."
        ),
        "files": records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    disputed = [r for r, v in records.items() if "checksum_dispute" in v]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWrote {MANIFEST.relative_to(REPO)}")
    print(f"Pinned: {DOI} @ version {meta['dataset_version']} "
          f"({len(records)} files, {len(records) - len(disputed)} checksum-verified)")
    if disputed:
        print(f"\n!! {len(disputed)} file(s) carry a recorded checksum dispute:")
        for r in disputed:
            print(f"     {r}")
        print("!! Your sha256 values still pin these bytes exactly. Say so in the\n"
              "!! Phase A writeup rather than leaving it implicit.")
    return 0


def _dist_version() -> str:
    from importlib.metadata import version
    try:
        return version("optimuskg")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
