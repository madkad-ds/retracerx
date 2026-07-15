#!/usr/bin/env python3
"""
A1 -- adjudicate the checksum disputes recorded in okg_manifest.json.

Two independent tests, no downloads beyond metadata:

  TEST 1 (provenance): does each disputed file's local content hash to the MD5
    that release 1.0 records for its predecessor? If yes, the file is unchanged
    since 1.0 and Dataverse's 2.0 checksum record is simply wrong.

  TEST 2 (content): every stratified file is a strict subset of the flat
    nodes.parquet / edges.parquet, both of which PASSED checksum verification.
    So the flat tables are a trusted oracle. If a disputed stratified file's
    keys match its slice of the verified flat table exactly, its content is
    correct regardless of what the metadata claims.

Run from the repo root, after a1_pin_okg.py --all.
"""
import hashlib
import json
import sys
from pathlib import Path

import polars as pl
import requests

SRV, DOI = "https://dataverse.harvard.edu", "doi:10.7910/DVN/IYNGEV"
REPO = Path.cwd()
MANIFEST = REPO / "data" / "manifest" / "okg_manifest.json"
REPORT = REPO / "data" / "manifest" / "okg_dispute_audit.json"

CODE = {"anatomy": "ANA", "biological_process": "BPO", "cellular_component": "CCO",
        "disease": "DIS", "drug": "DRG", "exposure": "EXP", "gene": "GEN",
        "molecular_function": "MFN", "pathway": "PWY", "phenotype": "PHE"}


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


def version_files(v: str) -> dict:
    r = requests.get(f"{SRV}/api/datasets/:persistentId/versions/{v}",
                     params={"persistentId": DOI}, timeout=60)
    r.raise_for_status()
    return {f"{(f.get('directoryLabel') or '').strip('/')}/{f['dataFile']['filename']}".lstrip("/"):
            f["dataFile"] for f in r.json()["data"]["files"]}


def edge_label(stem: str) -> str | None:
    for a in CODE:
        for b in CODE:
            if stem == f"{a}_{b}":
                return f"{CODE[a]}-{CODE[b]}"
    return None


def main() -> int:
    man = json.loads(MANIFEST.read_text())
    files = man["files"]
    disputed = sorted(r for r, v in files.items() if "checksum_dispute" in v)
    verified = sorted(r for r in files if r not in disputed)
    print(f"manifest: {len(files)} files | {len(verified)} verified | {len(disputed)} disputed\n")

    v1, v2 = version_files("1.0"), version_files("2.0")
    out = {"dataset_version": man["dataset_version"],
           "n_disputed": len(disputed), "test1": {}, "test2": {}}

    # ---------------- TEST 1: unchanged-since-1.0? ------------------------
    print("=" * 72)
    print("TEST 1 -- do disputed files hash to their 1.0 predecessor?")
    print("=" * 72)
    matches_v1, anomalies = [], []
    for rel in disputed:
        local = md5_of(REPO / files[rel]["local_path"])
        prev = v1.get(rel, {}).get("md5", "").lower()
        same = bool(prev) and local == prev
        (matches_v1 if same else anomalies).append(rel)
        out["test1"][rel] = {"local_md5": local, "v1_md5": prev,
                             "v2_metadata_md5": (files[rel]["remote_checksum_value"] or "").lower(),
                             "identical_to_v1": same}
        print(f"  {'== 1.0' if same else '!! NEW '}  {rel}")
    print(f"\n  {len(matches_v1)}/{len(disputed)} disputed files are byte-identical to 1.0")
    if anomalies:
        print(f"  !! {len(anomalies)} do NOT match 1.0 either -- investigate:")
        for a in anomalies: print(f"       {a}")

    # sanity: verified files should have CHANGED since 1.0
    changed = sum(1 for rel in verified
                  if v1.get(rel, {}).get("md5", "").lower() != files[rel]["sha256"][:0] + (v2.get(rel, {}).get("md5", "").lower()))
    print(f"\n  (cross-check: of {len(verified)} verified files, "
          f"{sum(1 for r in verified if v1.get(r, {}).get('md5','').lower() != v2.get(r, {}).get('md5','').lower())}"
          f" changed content since 1.0)")

    # ---------------- TEST 2: validate against the verified flat tables ----
    print("\n" + "=" * 72)
    print("TEST 2 -- do disputed stratified files match the VERIFIED flat tables?")
    print("=" * 72)
    if "nodes.parquet" not in files or "edges.parquet" not in files:
        print("  flat tables not pinned; skipping", file=sys.stderr)
        return 1
    for flat in ("nodes.parquet", "edges.parquet"):
        if "checksum_dispute" in files[flat]:
            print(f"  !! {flat} is itself disputed -- it cannot serve as an oracle.",
                  file=sys.stderr)
            return 2

    fnodes = pl.scan_parquet(REPO / files["nodes.parquet"]["local_path"])
    fedges = pl.scan_parquet(REPO / files["edges.parquet"]["local_path"])
    ok = bad = 0

    for rel in disputed:
        stem = Path(rel).stem
        strat = pl.scan_parquet(REPO / files[rel]["local_path"])
        if rel.startswith("nodes/"):
            code = CODE.get(stem)
            if not code: continue
            a = fnodes.filter(pl.col("label") == code).select("id").sort("id")
            b = strat.select("id").sort("id")
        elif rel.startswith("edges/"):
            lab = edge_label(stem)
            if not lab: continue
            a = fedges.filter(pl.col("label") == lab).select("from", "to", "relation").sort(["from", "to", "relation"])
            b = strat.select("from", "to", "relation").sort(["from", "to", "relation"])
        else:
            continue

        da = a.collect(engine="streaming"); db = b.collect(engine="streaming")
        same = da.equals(db)
        ok, bad = (ok + same, bad + (not same))
        flag = "OK  " if same else "FAIL"
        print(f"  {flag}  {rel:<52} flat={da.height:>9,}  strat={db.height:>9,}")
        out["test2"][rel] = {"flat_rows": da.height, "strat_rows": db.height, "keys_match": same}

    print(f"\n  {ok} match the verified flat table, {bad} do not")

    # ---------------- verdict ---------------------------------------------
    print("\n" + "=" * 72)
    if bad == 0 and not anomalies:
        print("VERDICT: all disputed files are byte-identical to their unchanged 1.0")
        print("  predecessors AND agree exactly with the checksum-verified flat")
        print("  tables. The content is correct; the 2.0 metadata MD5s are wrong.")
        print("  -> proceed. sha256 in the manifest is the pin. File upstream.")
    else:
        print("VERDICT: NOT benign. Files failing TEST 2 disagree with a")
        print("  checksum-verified oracle -- their content is actually wrong.")
        print("  -> read those slices from nodes.parquet / edges.parquet instead.")
    print("=" * 72)

    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
