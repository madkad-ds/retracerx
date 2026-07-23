#!/usr/bin/env python3
"""
a2_diag_dis_keys.py — M3 says umls_cui covers 2.95% of DIS nodes. Before A6 redesigns
around that, ask whether 36,345 is even the right denominator.

A6 joins repoDB to diseases. repoDB only reaches diseases that carry drug edges, so
the population that matters is the DIS nodes touching DRG-DIS (and DRG-PHE) — not
every disease in the graph. Coverage on that subset is the number A6 lives or dies by.
Everything else here is about finding a second bridge if the first one is thin.

Read-only. Writes nothing. Run from the repo root after A2:
    python scripts/a2_diag_dis_keys.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
NODES = REPO / "data" / "canon" / "canon_nodes.parquet"
EDGES = REPO / "data" / "canon" / "canon_edges"

GOLD = {"INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"}


def pct(n: int, d: int) -> str:
    return f"{n:,}/{d:,} = {n / d:.1%}" if d else f"{n:,}/0"


def main() -> None:
    for p in (NODES, EDGES):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} missing — run A2 first (or set RETRACERX_ROOT)")

    dis = pl.scan_parquet(NODES).filter(pl.col("label") == "DIS")
    total = dis.select(pl.len()).collect().item()
    with_cui = dis.filter(pl.col("umls_cui").is_not_null()).select(pl.len()).collect().item()
    print(f"DIS nodes: {total:,} | umls_cui present: {pct(with_cui, total)}\n")

    # ---- 1. id namespaces. A1 said MONDO is only ~32% of DIS; confirm from canon.
    print("id_prefix distribution on DIS (the merge shows up here):")
    print(dis.group_by("id_prefix").agg(pl.len().alias("nodes"))
          .sort("nodes", descending=True).collect())

    # ---- 2. THE denominator that matters: diseases repoDB could actually reach.
    dd = EDGES / "DRG-DIS.parquet"
    if not dd.exists():
        raise SystemExit(f"FATAL: {dd} missing")
    drg_dis = pl.scan_parquet(dd)

    # DRG-DIS is undirected and stored once, so a disease can sit on either end.
    touched = (pl.concat([drg_dis.select(pl.col("from").alias("id")),
                          drg_dis.select(pl.col("to").alias("id"))])
               .unique()
               .join(dis.select("id", "umls_cui", "name", "properties_json"),
                     on="id", how="inner"))

    gold = (drg_dis.filter(pl.col("relation").is_in(GOLD)))
    gold_touched = (pl.concat([gold.select(pl.col("from").alias("id")),
                               gold.select(pl.col("to").alias("id"))])
                    .unique()
                    .join(dis.select("id", "umls_cui"), on="id", how="inner"))

    for name, lf in (("any DRG-DIS edge", touched), ("gold-set relations only", gold_touched)):
        n = lf.select(pl.len()).collect().item()
        k = lf.filter(pl.col("umls_cui").is_not_null()).select(pl.len()).collect().item()
        print(f"\nDIS nodes touched by {name}: {n:,}")
        print(f"  umls_cui present: {pct(k, n)}   <- the A6 number for this population")

    # ---- 3. xrefs: the only unexplored bridge (23.9% of DIS). What's in it?
    print("\nxrefs — is there a second route to a CUI?")
    xr = (touched.filter(pl.col("properties_json").str.contains(r'"xrefs"\s*:\s*\[')) 
          .select("properties_json").head(400).collect())
    vocabs: dict[str, int] = {}
    sample: list[str] = []
    for row in xr["properties_json"]:
        try:
            vals = json.loads(row).get("xrefs") or []
        except Exception:
            continue
        for v in vals:
            if not isinstance(v, str):
                continue
            pre = v.split(":", 1)[0] if ":" in v else v.split("_", 1)[0]
            vocabs[pre] = vocabs.get(pre, 0) + 1
            if len(sample) < 12:
                sample.append(v)
    if vocabs:
        print("  vocabularies seen in xrefs (sample of up to 400 drug-touched DIS nodes):")
        for k, v in sorted(vocabs.items(), key=lambda x: -x[1])[:15]:
            print(f"    {k:<14} {v:,}")
        print(f"  examples: {sample}")
        umls_like = [k for k in vocabs if "UMLS" in k.upper() or "CUI" in k.upper()]
        print(f"  -> UMLS-ish vocabularies in xrefs: {umls_like or 'NONE'}")
    else:
        print("  no populated xrefs among the sampled drug-touched DIS nodes")

    # ---- 4. Does xrefs rescue nodes that umls_cui misses, or the same ones?
    has_x = touched.filter(
        pl.col("properties_json").str.contains(r'"xrefs"\s*:\s*\[\s*"')
    )
    n_touched = touched.select(pl.len()).collect().item()
    n_x = has_x.select(pl.len()).collect().item()
    n_x_no_cui = has_x.filter(pl.col("umls_cui").is_null()).select(pl.len()).collect().item()
    print(f"\ndrug-touched DIS with non-empty xrefs: {pct(n_x, n_touched)}")
    print(f"  of those, WITHOUT a umls_cui: {n_x_no_cui:,}  "
          f"<- what an xrefs route would add on top")

    print("\nread this as: if the gold-set coverage above is high, A6 stays a join and")
    print("M3's 2.95% was the wrong denominator. If it is also ~3%, A6 needs an external")
    print("mapping (MONDO/EFO -> UMLS via OxO or MRCONSO) and that is a design change,")
    print("not a column change. Either way A2 stands: it measured, it did not guess.")


if __name__ == "__main__":
    main()
