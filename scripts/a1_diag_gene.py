#!/usr/bin/env python3
"""
Did the GEN nodes change between OptimusKG 1.0 and 2.0?

Compares the GEN subset of the LCC nodes table across both releases. Both files
are MD5-verified against their own release metadata before anything is compared,
and both store `properties` as a JSON string, so this is a like-for-like diff.

  genes identical  -> nodes/gene.parquet being byte-identical to its 1.0
                      predecessor is expected. The 2.0 metadata MD5 is wrong.
                      Pin it with --allow-checksum-dispute.
  genes differ     -> nodes/gene.parquet is serving stale 1.0 content.
                      Drop it from the pin set; read GEN from nodes.parquet.

Run from the repo root.
"""
import hashlib
import sys
from pathlib import Path

import polars as pl
import requests

SRV = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/IYNGEV"
REL = "largest_connected_component_nodes.parquet"
SCRATCH = Path.cwd() / "data" / "okg_cache" / "_diag_v1"


def version_files(version: str) -> dict:
    r = requests.get(f"{SRV}/api/datasets/:persistentId/versions/{version}",
                     params={"persistentId": DOI}, timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json()["data"]["files"]:
        d = f["dataFile"]
        key = f"{(f.get('directoryLabel') or '').strip('/')}/{d['filename']}".lstrip("/")
        out[key] = d
    return out


def fetch_verified(meta: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != meta["filesize"]:
        print(f"   downloading id={meta['id']} ({meta['filesize']/1e6:.1f} MB)…")
        h = hashlib.md5()
        with requests.get(f"{SRV}/api/access/datafile/{meta['id']}",
                          stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for c in r.iter_content(1 << 22):
                    fh.write(c); h.update(c)
            tmp.replace(dest)
        got = h.hexdigest()
    else:
        h = hashlib.md5()
        with dest.open("rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""): h.update(c)
        got = h.hexdigest()
    ok = got == meta["md5"].lower()
    print(f"   md5 {'OK' if ok else 'MISMATCH'}: {got}")
    if not ok:
        print(f"   !! expected {meta['md5']} — this file is untrustworthy too.",
              file=sys.stderr)
        sys.exit(9)
    return dest


def gen_frame(path: Path) -> pl.DataFrame:
    return (pl.scan_parquet(path)
              .filter(pl.col("label") == "GEN")
              .select("id", "properties")
              .sort("id")
              .collect(engine="streaming"))


def main() -> int:
    v1, v2 = version_files("1.0"), version_files("2.0")
    print(f"1.0 {REL}: id={v1[REL]['id']}")
    print(f"2.0 {REL}: id={v2[REL]['id']}")
    if v1[REL]["md5"] == v2[REL]["md5"]:
        print("\nThe whole LCC nodes table is unchanged between releases.")

    print("\n[1.0]"); p1 = fetch_verified(v1[REL], SCRATCH / "lcc_nodes_v1.parquet")
    print("[2.0]"); p2 = fetch_verified(
        v2[REL], Path.cwd() / "data" / "okg_cache"
        / DOI.replace(":", "_").replace("/", "_").replace(".", "_") / "2.0" / REL)

    a, b = gen_frame(p1), gen_frame(p2)
    print(f"\nGEN rows: 1.0={a.height:,}  2.0={b.height:,}")

    ids_same = a["id"].to_list() == b["id"].to_list()
    print(f"GEN id sets identical      : {ids_same}")
    if not ids_same:
        only1 = set(a["id"]) - set(b["id"]); only2 = set(b["id"]) - set(a["id"])
        print(f"  only in 1.0: {len(only1)}   only in 2.0: {len(only2)}")

    props_same = a["properties"].to_list() == b["properties"].to_list()
    print(f"GEN properties identical   : {props_same}")
    if ids_same and not props_same:
        diff = sum(1 for x, y in zip(a["properties"], b["properties"]) if x != y)
        print(f"  {diff:,} of {a.height:,} gene rows changed content in 2.0")

    print("\n" + "=" * 68)
    if ids_same and props_same:
        print("VERDICT: GEN nodes are byte-for-byte unchanged from 1.0 to 2.0.")
        print("  nodes/gene.parquet matching its predecessor is CORRECT.")
        print("  The 2.0 metadata MD5 (e19d8d38…) is the error.")
        print("  -> python scripts/a1_pin_okg.py --all --allow-checksum-dispute")
    else:
        print("VERDICT: GEN nodes CHANGED in 2.0, but nodes/gene.parquet serves")
        print("  1.0 bytes. The file is STALE and the metadata MD5 was right.")
        print("  -> drop nodes/gene.parquet from the pin set")
        print("  -> read GEN rows from nodes.parquet (JSON properties) instead")
        print("  -> report the storage fault to the OptimusKG maintainers")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
