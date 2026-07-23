#!/usr/bin/env python3
"""
A3 — Emit Neo4j bulk-importer CSVs from the A2 canonical Parquet.

Jargon, defined once:
  bulk importer   Neo4j's OFFLINE loader (`neo4j-admin database import`). It writes store
                  files directly against a STOPPED database instead of executing queries,
                  which is why it is orders of magnitude faster than LOAD CSV / MERGE.
  typed header    The importer must be told which CSV column is the primary key (`:ID`),
                  the node label (`:LABEL`), the relationship endpoints (`:START_ID` /
                  `:END_ID`) and the relationship type (`:TYPE`).
  anti-join       Keep only the left-table rows with NO match on the right. Here: keep the
                  edges that are NOT in A7's removed set.
  endpoint_key    A2's orientation-insensitive edge key. The 13 undirected labels are each
                  stored exactly ONCE, in whichever orientation the source had, so a
                  (from, to) match misses an edge recorded the other way round — silently.
                  endpoint_key sorts the two endpoints when the label is undirected, so both
                  orientations produce the same string.
  composite TYPE  LABEL__RELATION, e.g. DRG_DIS__INDICATION. Frozen in a3_decisions.json.

Reads
  data/manifest/a1_decisions.json     graph_variant (never assumed)
  data/manifest/a3_decisions.json     the frozen A3 choices
  data/manifest/okg_census.json       node/edge totals + the A7 leakage surface
  data/manifest/a2_integrity.json     edges_by_label_relation -> the 57 composite types
  data/canon/canon_nodes.parquet
  data/canon/canon_edges/*.parquet

Writes (into the mounted import dir, default ./neo4j/import)
  nodes.csv                 emitted ONCE, identical for both variants
  edges_full.csv            all edges
  edges_filtered.csv        leakage-filtered (only with --removed)
  a3_meta_<variant>.cypher  the variant marker, values baked in at emit time
  data/manifest/a3_manifest.json

NOTHING in this file is a hardcoded count. Every expectation is read from a manifest.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from a3_common import (composite_type, fatal, info, load_json,
                       normalize_pair_counts, parse_composite, pick, sha256_file,
                       warn, write_manifest)

SAFE_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAMPLE_ROWS = 10_000          # used only to MEASURE row width for the disk guard


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------
def read_expectations(manifest_dir: Path) -> dict:
    a1 = load_json(manifest_dir / "a1_decisions.json", "a1_decisions.json")
    a3d = load_json(manifest_dir / "a3_decisions.json", "a3_decisions.json")
    census = load_json(manifest_dir / "okg_census.json", "okg_census.json")
    integ = load_json(manifest_dir / "a2_integrity.json", "a2_integrity.json")

    graph_variant = pick(a1, ["graph_variant", "graph_variant.value", "decisions.graph_variant"],
                         "graph_variant", "a1_decisions.json")
    if isinstance(graph_variant, dict):
        graph_variant = graph_variant.get("value", graph_variant)

    dataset_version = pick(census, ["dataset_version", "meta.dataset_version", "pin.dataset_version"],
                           "dataset_version", "okg_census.json")

    n_nodes = int(pick(census, ["nodes_total", "counts.nodes", "totals.nodes", "nodes.total",
                                "graph.nodes", "n_nodes"],
                       "total node count", "okg_census.json"))
    n_edges = int(pick(census, ["edges_total", "counts.edges", "totals.edges", "edges.total",
                                "graph.edges", "n_edges"],
                       "total edge count", "okg_census.json"))

    pairs_raw = pick(integ, ["edges_by_label_relation", "measurements.edges_by_label_relation",
                             "M.edges_by_label_relation"],
                     "edges_by_label_relation", "a2_integrity.json")
    pair_counts = normalize_pair_counts(pairs_raw, "a2_integrity.edges_by_label_relation")

    leak_raw = pick(census, ["leakage_surface", "a7_leakage_surface", "leakage",
                             "leakage_surface.pairs"],
                    "leakage surface", "okg_census.json")
    leak_counts = normalize_pair_counts(leak_raw, "okg_census.leakage_surface")

    # The leakage surface must be a subset of the measured pair census, or the two
    # manifests disagree about the graph and A3 must not paper over it.
    stray = sorted(set(leak_counts) - set(pair_counts))
    if stray:
        fatal("leakage surface contains (label, relation) pairs absent from a2_integrity",
              f"stray pairs: {stray}")

    return dict(graph_variant=graph_variant, dataset_version=dataset_version,
                n_nodes=n_nodes, n_edges=n_edges,
                pair_counts=pair_counts, leak_counts=leak_counts, a3_decisions=a3d)


# ---------------------------------------------------------------------------
# composite TYPE
# ---------------------------------------------------------------------------
def build_type_table(pair_counts: dict[tuple[str, str], int]) -> pl.DataFrame:
    """
    Build the 57-row (label, relation) -> composite TYPE table and PROVE it is safe.
    Every assertion here is a FATAL: no fallback, no repair, no sanitising regex.
    """
    labels = {lab for lab, _ in pair_counts}
    relations = {rel for _, rel in pair_counts}

    bad_lab = sorted(l for l in labels if "_" in l)
    if bad_lab:
        fatal("edge label contains '_', which breaks the LABEL__RELATION round-trip",
              f"labels: {bad_lab}", "Re-freeze the naming scheme in a3_decisions.json.")
    bad_rel = sorted(r for r in relations if "__" in r)
    if bad_rel:
        fatal("relation contains '__', which breaks the LABEL__RELATION round-trip",
              f"relations: {bad_rel}")

    rows = []
    for (lab, rel), cnt in sorted(pair_counts.items()):
        t = composite_type(lab, rel)
        if not SAFE_TYPE.match(t):
            fatal(f"composite type {t!r} would require backticks in Cypher")
        back_lab, back_rel = parse_composite(t)
        if (back_lab, back_rel) != (lab, rel):
            fatal(f"composite type {t!r} does not round-trip: got {(back_lab, back_rel)}")
        rows.append({"label": lab, "relation": rel, "type": t, "census_count": cnt})

    tbl = pl.DataFrame(rows)
    if tbl["type"].n_unique() != tbl.height:
        dupes = tbl.group_by("type").len().filter(pl.col("len") > 1)["type"].to_list()
        fatal("composite type collision", f"colliding names: {dupes}")

    info(f"composite TYPE table built: {tbl.height} distinct types "
         f"from {len(labels)} labels x {len(relations)} relations (not their product)")
    return tbl


# ---------------------------------------------------------------------------
# disk guard
# ---------------------------------------------------------------------------
def estimate_and_guard(lf: pl.LazyFrame, n_rows: int, out_dir: Path, factor: float, what: str) -> int:
    """
    MEASURE the row width by sinking a small sample, then require headroom.
    Measured, not guessed — the pattern A1's pin script uses for downloads.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tf:
        lf.head(SAMPLE_ROWS).sink_csv(tf.name)
        sample_bytes = Path(tf.name).stat().st_size
    per_row = sample_bytes / max(SAMPLE_ROWS, 1)
    est = int(per_row * n_rows)
    free = shutil.disk_usage(out_dir).free
    need = int(est * factor)
    info(f"{what}: ~{per_row:.0f} B/row x {n_rows:,} rows = ~{est/1e9:.2f} GB estimated; "
         f"free {free/1e9:.2f} GB; required {need/1e9:.2f} GB (x{factor})")
    if free < need:
        fatal(f"insufficient disk headroom for {what}",
              f"free {free/1e9:.2f} GB < required {need/1e9:.2f} GB",
              "Both CSV variants plus the Neo4j store live on this volume. "
              "Lower --headroom-factor only if you know where the store lives.")
    return est


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def emit_nodes(nodes_path: Path, out: Path, expected_n: int, out_dir: Path, factor: float) -> dict:
    nodes = pl.scan_parquet(nodes_path)
    have = set(nodes.collect_schema().names())
    missing = {"id", "label", "name"} - have
    if missing:
        fatal(f"canon_nodes.parquet missing required columns: {sorted(missing)}",
              f"columns present: {sorted(have)}")

    n = nodes.select(pl.len()).collect().item()
    if n != expected_n:
        fatal(f"node count {n:,} != census {expected_n:,}",
              "Either the canonical table or the census is stale. Do not proceed.")

    # Declared transformation (a3_decisions.declared_transformations): CR/LF/TAB inside
    # `name` are flattened to a space in nodes.csv ONLY, so the importer does not need
    # --multiline-fields, which would slow the 21.8M-edge import for a node-only problem.
    # canon_nodes.parquet is untouched; the affected count is recorded in the manifest.
    dirty = (nodes.filter(pl.col("name").str.contains(r"[\r\n\t]").fill_null(False))
                  .select(pl.len()).collect().item())
    if dirty:
        warn(f"{dirty:,} node names contain CR/LF/TAB; flattening to spaces in nodes.csv only "
             f"(canonical Parquet untouched). Recorded in a3_manifest.json.")

    optional = [c for c in ("id_prefix", "inchi_key", "umls_cui", "symbol") if c in have]
    for c in ("id_prefix", "inchi_key", "umls_cui", "symbol"):
        if c not in have:
            warn(f"column {c!r} absent from canon_nodes.parquet; omitted from nodes.csv")

    sel = [
        pl.col("id").alias("id:ID"),
        (pl.col("label") + pl.lit(";Node")).alias(":LABEL"),   # shared Node label -> ONE constraint
        pl.col("label").alias("label"),
        pl.col("name").str.replace_all(r"[\r\n\t]+", " ").alias("name"),
    ] + [pl.col(c) for c in optional]

    lf = nodes.select(sel)
    estimate_and_guard(lf, n, out_dir, factor, "nodes.csv")
    lf.sink_csv(out)
    cols = ["id:ID", ":LABEL", "label", "name"] + optional
    info(f"nodes.csv: {n:,} nodes, columns {cols} -> {out}")
    return {"rows": n, "names_whitespace_flattened": int(dirty), "columns": cols}


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------
def load_removed(removed: Path, leak_counts: dict[tuple[str, str], int]) -> pl.DataFrame:
    rem = pl.read_csv(removed)          # ~79,949 rows; small enough to materialise
    cols = set(rem.columns)

    if not {"endpoint_key", "relation"}.issubset(cols):
        fatal(f"{removed} lacks endpoint_key and/or relation",
              f"columns present: {sorted(cols)}",
              "A (from, to) fallback UNDER-REMOVES on the 13 undirected labels with no visible "
              "symptom — leakage edges survive into the eval graph. That is the exact failure "
              "endpoint_key exists to prevent, so there is deliberately no fallback path here.",
              "Fix A7 to emit endpoint_key (a3_decisions.json#a7_contract), then re-run.")
    if "label" not in cols:
        fatal(f"{removed} lacks `label`; the (label, relation) PAIR is the unit of the leakage surface")

    # Every removed pair must sit inside the census leakage surface. Anything else means
    # A7 is removing edges A3 has no warrant for.
    rem_pairs = {tuple(r) for r in rem.select(["label", "relation"]).unique().rows()}
    outside = sorted(rem_pairs - set(leak_counts))
    if outside:
        fatal("removed set contains (label, relation) pairs outside the census leakage surface",
              f"unexpected pairs: {outside}",
              "Either the census is stale or A7 is over-removing. Adjudicate before importing.")

    # Coverage report, per pair. DRG-DIS is ENTIRELY leakage (errata E3: 57,601 + 11,718 +
    # 1,061 = 70,380 = every row in the label), so partial DRG-DIS coverage is fatal.
    per_pair = {(r[0], r[1]): r[2] for r in rem.group_by(["label", "relation"]).len().rows()}
    info("leakage coverage (removed / census):")
    shortfalls = []
    for pair, census_n in sorted(leak_counts.items(), key=lambda kv: -kv[1]):
        got = per_pair.get(pair, 0)
        flag = "" if got == census_n else "   <-- partial"
        info(f"   {pair[0]:<9} {pair[1]:<18} {got:>7,} / {census_n:>7,}{flag}")
        if got != census_n:
            shortfalls.append((pair, got, census_n))

    drg_dis_short = [s for s in shortfalls if s[0][0] == "DRG-DIS"]
    if drg_dis_short:
        fatal("DRG-DIS is entirely leakage (errata E3) but the removed set does not cover it",
              *[f"{p[0]}/{p[1]}: {g:,} of {c:,}" for p, g, c in drg_dis_short],
              "The filtered graph must contain ZERO drug-disease edges.")
    if shortfalls:
        warn("partial coverage on: " + ", ".join(f"{p[0]}/{p[1]}" for p, _, _ in shortfalls))
        warn("Legitimate ONLY if it is A7's explicit CONTRAINDICATION policy (remove for "
             "held-out pairs only). State it in the writeup.")
    return rem


def emit_edges(edges_glob: str, out: Path, type_tbl: pl.DataFrame, expected_n: int,
               removed_df: pl.DataFrame | None, out_dir: Path, factor: float) -> dict:
    edges = pl.scan_parquet(edges_glob)
    have = set(edges.collect_schema().names())
    missing = {"from", "to", "label", "relation", "undirected", "endpoint_key"} - have
    if missing:
        fatal(f"canon_edges missing required columns: {sorted(missing)}",
              f"columns present: {sorted(have)}")

    total = edges.select(pl.len()).collect().item()
    if total != expected_n:
        fatal(f"edge count {total:,} != census {expected_n:,}")

    nulls = edges.select(pl.col("relation").is_null().sum()).collect().item()
    if nulls:
        fatal(f"{nulls:,} edges have a NULL relation",
              "There is deliberately no coalesce-to-label fallback: that would silently mint "
              "a type absent from the census.")

    filtered = edges
    removed_rows = matched_keys = 0
    if removed_df is not None:
        removed_rows = removed_df.height
        keys = removed_df.select(["endpoint_key", "relation"]).unique()
        n_keys = keys.height
        # Do the removal keys actually reach the graph? A key matching nothing means A7's
        # set was built against a different graph than canon_edges/.
        matched_keys = (keys.lazy()
                        .join(edges.select(["endpoint_key", "relation"]).unique(),
                              on=["endpoint_key", "relation"], how="semi")
                        .select(pl.len()).collect().item())
        if matched_keys != n_keys:
            fatal(f"{n_keys - matched_keys:,} of {n_keys:,} removal keys match no edge",
                  "A7's removed set disagrees with canon_edges/.",
                  "Check dataset_version and graph_variant in both manifests.")
        filtered = edges.join(keys.lazy(), on=["endpoint_key", "relation"], how="anti")

    kept = filtered.select(pl.len()).collect().item() if removed_df is not None else total
    if removed_df is not None:
        info(f"anti-join on endpoint_key+relation: {total:,} -> {kept:,} ({total - kept:,} removed)")

    # Composite TYPE comes from joining the frozen 57-row table, not from string-building
    # in-line, so the emitter and the verifier cannot drift apart.
    out_lf = (filtered
              .join(type_tbl.lazy().select(["label", "relation", "type"]),
                    on=["label", "relation"], how="left")
              .select(
                  pl.col("from").alias(":START_ID"),
                  pl.col("to").alias(":END_ID"),
                  pl.col("type").alias(":TYPE"),
                  pl.col("label"),
                  pl.col("relation"),
                  pl.col("undirected").alias("undirected:boolean"),
                  pl.col("endpoint_key"),
              ))

    unmatched = out_lf.select(pl.col(":TYPE").is_null().sum()).collect().item()
    if unmatched:
        fatal(f"{unmatched:,} edges carry a (label, relation) pair absent from the census",
              "a2_integrity.json and canon_edges/ disagree. Do not import.")

    types_present = out_lf.select(pl.col(":TYPE").n_unique()).collect().item()
    estimate_and_guard(out_lf, kept, out_dir, factor, out.name)
    out_lf.sink_csv(out)
    info(f"{out.name}: {kept:,} edges, {types_present} distinct composite types -> {out}")
    return {"rows": kept, "total_before_filter": total, "removed_rows": removed_rows,
            "matched_removal_keys": matched_keys, "distinct_types": types_present}


def measure_newly_edgeless_drugs(edges_glob: str, nodes_path: Path,
                                 removed_df: pl.DataFrame) -> dict:
    """
    M-A3-1: drugs that become edgeless ONCE the leakage surface is removed.

    This is a THIRD population, distinct from A1's 982 edgeless drugs (edgeless in the raw
    graph) and A4's 984 unscoreable Launched compounds (no OKG node at all). With DRG-DIS
    gone entirely, any drug whose only edges were indications drops out of every path query.
    Cheap here, expensive to notice at A7.
    """
    edges = pl.scan_parquet(edges_glob)
    drg = (pl.scan_parquet(nodes_path).filter(pl.col("label") == "DRG")
           .select(pl.col("id")).collect())

    keys = removed_df.select(["endpoint_key", "relation"]).unique().lazy()
    filt = edges.join(keys, on=["endpoint_key", "relation"], how="anti")

    def touched(lf):
        return (lf.select(pl.col("from").alias("id")).unique()
                .join(lf.select(pl.col("to").alias("id")).unique(), on="id", how="full",
                      coalesce=True).select("id"))

    before = drg.lazy().join(touched(edges), on="id", how="semi").select(pl.len()).collect().item()
    after = drg.lazy().join(touched(filt), on="id", how="semi").select(pl.len()).collect().item()
    out = {"drg_nodes": drg.height, "with_edges_full": before, "with_edges_filtered": after,
           "newly_edgeless": before - after}
    info(f"M-A3-1 newly edgeless drugs: {before:,} -> {after:,} "
         f"({out['newly_edgeless']:,} lose their last edge to leakage removal)")
    info("      NOTE: distinct from A1's 982 edgeless drugs and A4's 984 unscoreable compounds.")
    return out


# ---------------------------------------------------------------------------
# variant marker
# ---------------------------------------------------------------------------
def write_meta_cypher(path: Path, variant: str, dataset_version: str, edges_sha: str,
                      removed_count: int, edge_rows: int, node_rows: int,
                      distinct_types: int) -> None:
    # A singleton marker node. Neo4j Community hosts ONE database and nothing inside it says
    # which variant is loaded; without this you can evaluate against the full graph and never
    # know. _Meta deliberately does NOT carry the :Node label, so it sits outside the
    # uniqueness constraint and outside every path query.
    path.write_text(f"""// A3 variant marker — generated by a3_emit_csvs.py; values baked in at emit time.
MERGE (m:_Meta {{singleton: true}})
SET m.graph_variant      = '{variant}',
    m.dataset_version    = '{dataset_version}',
    m.edges_csv_sha256   = '{edges_sha}',
    m.removed_edge_count = {removed_count},
    m.expected_edges     = {edge_rows},
    m.expected_nodes     = {node_rows},
    m.expected_types     = {distinct_types},
    m.imported_at        = datetime();
""")
    info(f"variant marker cypher: {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Emit Neo4j bulk-importer CSVs for A3.")
    ap.add_argument("--nodes", default="data/canon/canon_nodes.parquet")
    ap.add_argument("--edges-glob", default="data/canon/canon_edges/*.parquet")
    ap.add_argument("--manifest-dir", default="data/manifest")
    ap.add_argument("--import-dir", default="neo4j/import")
    ap.add_argument("--removed", default=None,
                    help="A7 removed_edge_ids.csv -> emits edges_filtered.csv. "
                         "Omit for the full exploration graph.")
    ap.add_argument("--skip-nodes", action="store_true",
                    help="nodes.csv is identical for both variants; skip if already emitted.")
    ap.add_argument("--headroom-factor", type=float, default=3.0)
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    out_dir = Path(args.import_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = read_expectations(mdir)
    variant_suffix = "filtered" if args.removed else "full"
    info(f"graph_variant={exp['graph_variant']} (from a1_decisions.json) | "
         f"dataset_version={exp['dataset_version']} | emitting '{variant_suffix}'")

    type_tbl = build_type_table(exp["pair_counts"])
    type_tbl.write_parquet(mdir / "a3_type_table.parquet")

    node_stats: dict = {"skipped": True}
    nodes_csv = out_dir / "nodes.csv"
    if args.skip_nodes:
        if not nodes_csv.exists():
            fatal("--skip-nodes given but nodes.csv does not exist")
        info("nodes.csv reused (identical across variants under re-import)")
    else:
        node_stats = emit_nodes(Path(args.nodes), nodes_csv, exp["n_nodes"], out_dir,
                                args.headroom_factor)

    removed_df = load_removed(Path(args.removed), exp["leak_counts"]) if args.removed else None
    edges_csv = out_dir / f"edges_{variant_suffix}.csv"
    edge_stats = emit_edges(args.edges_glob, edges_csv, type_tbl, exp["n_edges"],
                            removed_df, out_dir, args.headroom_factor)

    measurement = None
    if removed_df is not None:
        measurement = measure_newly_edgeless_drugs(args.edges_glob, Path(args.nodes), removed_df)

    nodes_sha = sha256_file(nodes_csv)
    edges_sha = sha256_file(edges_csv)
    node_rows = node_stats.get("rows") or exp["n_nodes"]

    write_meta_cypher(out_dir / f"a3_meta_{variant_suffix}.cypher", variant_suffix,
                      str(exp["dataset_version"]), edges_sha, edge_stats["removed_rows"],
                      edge_stats["rows"], node_rows, edge_stats["distinct_types"])

    mpath = mdir / "a3_manifest.json"
    manifest = load_json(mpath, "a3_manifest.json") if mpath.exists() else {}
    manifest.update({
        "step": "A3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": exp["dataset_version"],
        "graph_variant": exp["graph_variant"],
        "relationship_type_scheme": "composite LABEL__RELATION",
        "distinct_composite_types_in_census": type_tbl.height,
        "nodes_csv": {"path": str(nodes_csv), "sha256": nodes_sha, **node_stats},
    })
    manifest.setdefault("variants", {})[variant_suffix] = {
        "edges_csv": str(edges_csv),
        "edges_csv_sha256": edges_sha,
        "removed_edge_ids_csv": str(args.removed) if args.removed else None,
        "removed_edge_ids_sha256": sha256_file(args.removed) if args.removed else None,
        "expected_nodes": node_rows,
        "expected_edges": edge_stats["rows"],
        "expected_distinct_types": edge_stats["distinct_types"],
        "edges_before_filter": edge_stats["total_before_filter"],
        "removed_rows": edge_stats["removed_rows"],
        "m_a3_1_newly_edgeless_drugs": measurement,
    }
    write_manifest(mpath, manifest)

    info(f"done. Next: bash scripts/a3_import.sh --variant {variant_suffix}")


if __name__ == "__main__":
    main()
