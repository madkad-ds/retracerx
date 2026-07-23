#!/usr/bin/env python3
"""
a2_diag_undirected.py — adjudicate G5: A2 measures 13 undirected labels, A1's census
says 12. Nobody edits EXPECT until we know which number is right.

The question this answers: do `edges.parquet` (what A2 read, and the file A1 froze as
the source of truth) and the stratified `edges/*.parquet` files agree on the
`undirected` column, label by label?

This matters because A1's dispute adjudication cannot see it. okg_dispute_audit.json
test2 compares flat_rows/strat_rows and keys_match — row counts and endpoint keys. A
column-level disagreement on `undirected` passes that test untouched.

Read-only. Writes nothing. Run from the repo root:
    python scripts/a2_diag_undirected.py
    python scripts/a2_diag_undirected.py --json      # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
MAN = REPO / "data" / "manifest"


def load(name: str):
    p = MAN / name
    if not p.exists():
        sys.exit(f"FATAL: missing {p} (run from the repo root, or set RETRACERX_ROOT)")
    return json.loads(p.read_text())


def resolve(meta: dict, name: str) -> Path | None:
    lp = meta.get("local_path")
    if lp:
        cand = REPO / lp
        if cand.exists():
            return cand
    print(f"  WARN: {name} listed in the manifest but not on disk — skipped")
    return None


def profile(path: Path) -> pl.DataFrame:
    """Per label: is `undirected` all-true, any-true, and how many rows."""
    return (
        pl.scan_parquet(path)
        .group_by("label")
        .agg(
            pl.col("undirected").all().alias("all_true"),
            pl.col("undirected").any().alias("any_true"),
            pl.col("undirected").null_count().alias("nulls"),
            pl.len().alias("rows"),
        )
        .collect()
    )


def find_undirected_in_census(obj, path: str = "") -> list[tuple[str, object]]:
    """A1's census schema isn't fixed; hunt for anything mentioning `undirected`."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else k
            if "undirected" in k.lower():
                hits.append((here, v))
            hits += find_undirected_in_census(v, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            hits += find_undirected_in_census(v, f"{path}[{i}]")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    files = load("okg_manifest.json").get("files")
    if not isinstance(files, dict):
        sys.exit("FATAL: okg_manifest.json .files is not a dict keyed by relative path")

    if "edges.parquet" not in files:
        sys.exit("FATAL: edges.parquet not in the manifest")
    flat_path = resolve(files["edges.parquet"], "edges.parquet")
    if flat_path is None:
        sys.exit("FATAL: edges.parquet missing on disk")

    if not args.json:
        print(f"flat file : {flat_path}")
    flat = profile(flat_path).sort("label")

    strat_rows = []
    for name, meta in sorted(files.items()):
        if not name.startswith("edges/"):
            continue
        p = resolve(meta, name)
        if p is None:
            continue
        for r in profile(p).iter_rows(named=True):
            strat_rows.append({**r, "file": name})
    strat = pl.DataFrame(strat_rows) if strat_rows else pl.DataFrame(
        schema={"label": pl.Utf8, "all_true": pl.Boolean, "any_true": pl.Boolean,
                "nulls": pl.UInt32, "rows": pl.UInt32, "file": pl.Utf8}
    )

    flat_true = set(flat.filter(pl.col("all_true"))["label"].to_list())
    strat_true = set(strat.filter(pl.col("all_true"))["label"].to_list()) if len(strat) else set()

    # A label whose `undirected` is neither all-true nor all-false is a different and
    # worse problem than a miscount: the column would be meaningless for that label.
    mixed_flat = flat.filter(pl.col("any_true") & ~pl.col("all_true"))["label"].to_list()
    mixed_strat = (strat.filter(pl.col("any_true") & ~pl.col("all_true"))["label"].to_list()
                   if len(strat) else [])

    # Join the two views to name the labels that actually disagree.
    # A label the stratified set doesn't cover is NOT a disagreement — it's a coverage
    # gap, and conflating the two would manufacture a defect out of a missing file.
    disagree, uncovered = [], []
    if len(strat):
        j = flat.join(
            strat.select("label", "all_true", "rows", "file"),
            on="label", how="full", suffix="_strat", coalesce=True,
        )
        for r in j.iter_rows(named=True):
            a, b = r.get("all_true"), r.get("all_true_strat")
            row = {
                "label": r["label"],
                "flat_undirected": a,
                "strat_undirected": b,
                "flat_rows": r.get("rows"),
                "strat_rows": r.get("rows_strat"),
                "strat_file": r.get("file"),
            }
            if a is None or b is None:
                uncovered.append(row)
            elif a != b:
                disagree.append(row)

    census_hits = find_undirected_in_census(load("okg_census.json"))

    result = {
        "flat_undirected_count": len(flat_true),
        "strat_undirected_count": len(strat_true) if len(strat) else None,
        "flat_only": sorted(flat_true - strat_true) if len(strat) else None,
        "strat_only": sorted(strat_true - flat_true) if len(strat) else None,
        "disagreements": disagree,
        "uncovered_by_stratified": uncovered,
        "mixed_within_label_flat": mixed_flat,
        "mixed_within_label_strat": mixed_strat,
        "nulls_in_flat": flat.filter(pl.col("nulls") > 0).to_dicts(),
        "census_undirected_records": [{"path": p, "value": v} for p, v in census_hits],
        "flat_undirected_labels": sorted(flat_true),
    }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"\nflat edges.parquet : {len(flat_true)} undirected labels")
    if len(strat):
        print(f"stratified edges/* : {len(strat_true)} undirected labels")
        print(f"  undirected in flat only      : {sorted(flat_true - strat_true) or 'none'}")
        print(f"  undirected in stratified only: {sorted(strat_true - flat_true) or 'none'}")
    else:
        print("stratified edges/* : none found on disk — cannot cross-check")

    if disagree:
        print("\n*** FLAT vs STRATIFIED DISAGREE — this is a 2.0 data defect, not a miscount.")
        print("    test2's keys_match cannot see it; it compares endpoints, not columns.")
        print(pl.DataFrame(disagree))
    elif len(strat):
        print("\nflat and stratified agree on `undirected` for every label they share.")

    if uncovered:
        print(f"\n{len(uncovered)} label(s) present in only one representation "
              "(coverage gap, not a defect):")
        for r in uncovered[:15]:
            side = "flat only" if r["strat_undirected"] is None else "stratified only"
            print(f"  {r['label']:<12} {side}")

    if mixed_flat or mixed_strat:
        print(f"\n*** MIXED WITHIN LABEL (worse than a miscount): flat={mixed_flat} "
              f"strat={mixed_strat}")

    print("\ncensus records mentioning `undirected`:")
    if census_hits:
        for p, v in census_hits:
            s = json.dumps(v, default=str)
            print(f"  .{p} = {s[:300]}{'…' if len(s) > 300 else ''}")
    else:
        print("  none — the census never recorded an undirected count, so EXPECT's 12 "
              "came from A1 prose, not from a measurement.")

    print(f"\nthe 13 undirected labels A2 measured:\n  {sorted(flat_true)}")
    print("\nverdict:")
    if disagree:
        print("  the two representations differ -> errata item; the flat pair is the")
        print("  frozen source and the undisputed one, so A2's 13 stands. Record the")
        print("  defect, correct EXPECT['undirected_labels'] to 13, cite this run.")
    elif not census_hits:
        print("  no measurement to contradict A2 -> A1's 12 was prose. Correct")
        print("  EXPECT['undirected_labels'] to 13 and note it in HOWTO_ERRATA.md.")
    else:
        print("  compare A2's 13 against the census record printed above before editing")
        print("  anything. If the census counted 12 from the same bytes, one of the two")
        print("  counts has a bug — find it, do not paper over it.")


if __name__ == "__main__":
    main()
