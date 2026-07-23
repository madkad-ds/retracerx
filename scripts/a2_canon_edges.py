#!/usr/bin/env python3
"""
a2_canon_edges.py — build data/canon/canon_edges/<LABEL>.parquet (27 files,
21,813,816 rows total) from the routing frozen in a2_source_policy.json.

Output schema (per edge):

    from             str        verbatim node id — NOT `subject`
    to               str        verbatim node id — NOT `object`
    label            str        e.g. DRG-DIS  (27 of these)
    relation         str        e.g. INDICATION (36 distinct, across the 27 labels)
    undirected       bool       read it; never infer direction from Biolink semantics
    endpoint_key     str        unordered when undirected:true, ordered otherwise.
                                See --- why endpoint_key --- below.
    sources_direct   list[str]  properties.sources.direct  — provenance, plural
    sources_indirect list[str]  properties.sources.indirect — plural
    properties_json  str        whole blob verbatim, nothing dropped
    src_file         str        which of the 41 files this row came from
    src_checksum     str        verified | disputed_adjudicated_benign | ...

--- why endpoint_key ---
A7 removes held-out pairs by anti-join and A8 anti-joins the same set out of the
KGE triples. Both key on (from, to). But DRG-DIS is `undirected: true`, so
whether a given indication is stored as (drug, disease) or (disease, drug) is a
storage detail of the upstream build — and a directed anti-join silently misses
every edge stored the other way round. A missed removal is leakage, which is
pitfall #2 and invalidates results *silently*. endpoint_key makes the removal
orientation-safe without mutating the graph: A2 records, A7 decides.
a2_verify_canon.py measures whether both orientations are actually stored.

Written one file per label so the run is restartable and never materializes
21.8M rows; `pl.scan_parquet("data/canon/canon_edges/*.parquet")` reads it back
as a single table (the label lives in a column, not in the path).

Usage:
    python scripts/a2_canon_edges.py
    python scripts/a2_canon_edges.py --only DRG-DIS DRG-GEN
    python scripts/a2_canon_edges.py --dry-run
"""
from __future__ import annotations

import argparse

import polars as pl

from a2_common import (
    A2_POLICY, CANON_EDGES_DIR, EXPECT,
    coerce, die, load_json, resolve_path, revive, schema_of, struct_fields, warn,
)

SOURCES_TARGET = pl.List(pl.Utf8)
# A1 established this shape on the real snapshot: properties.sources.{direct,indirect},
# both LISTS. It is a *fact about 2.0*, not something to infer per label. A sampled
# inference degenerates to Null/List(Null) whenever a label's first N rows happen to
# carry empty lists, and a Null field is a decode error the moment a real string
# arrives ("error deserializing value String(\"CGI\") as null"). Decode with the fixed
# shape; the policy inventory decides *presence*, never dtype.
SOURCES_STRUCT = pl.Struct({"direct": SOURCES_TARGET, "indirect": SOURCES_TARGET})


def _degenerate(dt: pl.DataType | None) -> bool:
    """A sample-inferred dtype that carries no type information (all-null/empty rows)."""
    return dt is None or dt == pl.Null or dt == pl.List(pl.Null)


def sources_expr(entry: dict, path, repr_: str) -> tuple[pl.Expr, pl.Expr, bool]:
    """(direct, indirect, present). Provenance is properties.sources.{direct,indirect}
    and both are LISTS — there is no `provided_by` column anywhere in 2.0."""
    null = pl.lit(None, dtype=SOURCES_TARGET)

    if repr_ == "struct":
        live = struct_fields(schema_of(path)["properties"])
        if "sources" not in live:
            return null.alias("sources_direct"), null.alias("sources_indirect"), False
        sub = struct_fields(live["sources"])
        base = pl.col("properties").struct.field("sources")
        out = []
        for side in ("direct", "indirect"):
            if side in sub:
                out.append(coerce(base.struct.field(side), sub[side], SOURCES_TARGET)
                           .alias(f"sources_{side}"))
            else:
                out.append(null.alias(f"sources_{side}"))
        return out[0], out[1], True

    sources_dt = revive(entry["property_fields"].get("sources"))
    if sources_dt is None or not isinstance(sources_dt, pl.Struct):
        return null.alias("sources_direct"), null.alias("sources_indirect"), False

    # Presence check only. A degenerate side is fine — decoding it as List(String)
    # yields the empty/null lists that were actually there. A side inferred as a
    # genuine non-list scalar means 2.0 disagrees with A1, and that is not ours to
    # paper over.
    sub = struct_fields(sources_dt)
    for side, dt in sub.items():
        if not _degenerate(dt) and not isinstance(dt, pl.List):
            die(f"properties.sources.{side} inferred as {dt} — A1 established both sides "
                f"are lists. The snapshot disagrees with the pin; stop and re-adjudicate.")

    base = (pl.col("properties")
            .str.json_decode(pl.Struct({"sources": SOURCES_STRUCT}))
            .struct.field("sources"))
    return (base.struct.field("direct").alias("sources_direct"),
            base.struct.field("indirect").alias("sources_indirect"),
            True)


def build_label(label: str, entry: dict) -> pl.LazyFrame:
    path = resolve_path(entry["file"])
    if path is None:
        die(f"{entry['file']} vanished since a2_source_policy.py ran")
    schema = schema_of(path)
    for col in ("from", "to", "label", "relation", "undirected"):
        if col not in schema:
            die(f"{entry['file']} has no `{col}` column (found {list(schema)}). "
                f"The 2.0 edge schema is from/to/label/relation/undirected/properties — "
                f"if this file disagrees, the pin moved.")

    lf = pl.scan_parquet(path)
    if not entry["stratified"]:
        lf = lf.filter(pl.col("label") == label)

    repr_ = entry["properties_repr"]
    direct, indirect, has_prov = sources_expr(entry, path, repr_)
    if not has_prov:
        warn(f"{label}: no properties.sources — provenance columns will be null. "
             f"Pitfall #5: it is expensive to recover later.")
    props_json = (pl.col("properties").struct.json_encode() if repr_ == "struct"
                  else pl.col("properties").cast(pl.Utf8))

    f, t, u = pl.col("from"), pl.col("to"), pl.col("undirected")
    endpoint_key = (
        pl.when(u & (f > t))
        .then(pl.concat_str([t, f], separator="\u241f"))
        .otherwise(pl.concat_str([f, t], separator="\u241f"))
        .alias("endpoint_key")
    )

    return lf.select(
        f.cast(pl.Utf8), t.cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8), pl.col("relation").cast(pl.Utf8),
        u.cast(pl.Boolean),
        endpoint_key,
        direct, indirect,
        props_json.alias("properties_json"),
        pl.lit(entry["file"]).alias("src_file"),
        pl.lit(entry["checksum"]).alias("src_checksum"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="edge labels to (re)build")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    policy = load_json(A2_POLICY)
    if policy.get("graph_variant") != "full":
        die(f"policy graph_variant={policy.get('graph_variant')!r}; A1.5 froze 'full'.")

    labels = sorted(policy["edges"])
    if args.only:
        unknown = set(args.only) - set(labels)
        if unknown:
            die(f"unknown edge labels {sorted(unknown)}; routed labels are {labels}")
        labels = sorted(args.only)

    CANON_EDGES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for label in labels:
        entry = policy["edges"][label]
        out = CANON_EDGES_DIR / f"{label}.parquet"
        print(f"  {label:<12} <- {entry['file']:<40} "
              f"[{entry['properties_repr']}, {entry['checksum']}]", flush=True)
        if args.dry_run:
            continue
        build_label(label, entry).sink_parquet(out, compression="zstd")
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        total += n
        print(f"      -> {out.name} {n:,} rows", flush=True)

    if args.dry_run:
        print("\n--dry-run: plan only, nothing written")
        return

    print(f"\nwrote {len(labels)} label files, {total:,} rows")
    if not args.only and total != EXPECT["edges_total"]:
        die(f"{total:,} edges written, census says {EXPECT['edges_total']:,}. "
            f"Do not proceed — reconcile against okg_census.json first.")
    print("next: python scripts/a2_verify_canon.py")


if __name__ == "__main__":
    main()
