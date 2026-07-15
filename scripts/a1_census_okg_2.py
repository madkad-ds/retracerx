#!/usr/bin/env python3
"""
A1 - Verify the pinned OptimusKG snapshot and emit a census.

Reads paths from data/manifest/okg_manifest.json (written by a1_pin_okg.py).

The published figures at optimuskg.ai describe release 1.0. The client resolves
latestVersion, which is 2.0 as of 2026-05-06. Nothing on that docs site is a
safe assertion target. This script measures; it does not assume.

Usage:
    python scripts/a1_census_okg.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest" / "okg_manifest.json"
CENSUS = REPO / "data" / "manifest" / "okg_census.json"

# Reference only - the v1.0 figures from optimuskg.ai/docs. Reported as a delta
# so you can see how far 2.0 has moved. Never asserted.
REFERENCE_V1 = {"version": "1.0", "nodes": 190_531, "edges": 21_813_816, "relations": 27}

LEAKY_RELATIONS = ["INDICATION", "OFF_LABEL_USE", "CONTRAINDICATION"]


def scan(rel: str, manifest: dict) -> pl.LazyFrame:
    return pl.scan_parquet(REPO / manifest["files"][rel]["local_path"])


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def census_pair(nodes: pl.LazyFrame, edges: pl.LazyFrame, tag: str) -> dict:
    """Count one (nodes, edges) variant."""
    n_nodes = collect(nodes.select(pl.len())).item()
    n_edges = collect(edges.select(pl.len())).item()
    print(f"\n[{tag}] nodes={n_nodes:,}  edges={n_edges:,}")
    return {"n_nodes": n_nodes, "n_edges": n_edges}


def main() -> int:
    if not MANIFEST.exists():
        print("No manifest. Run scripts/a1_pin_okg.py first.", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    have = set(manifest["files"])
    version = manifest["dataset_version"]

    out: dict = {
        "doi": manifest["doi"],
        "dataset_version": version,
        "reference_release": REFERENCE_V1,
    }
    print(f"doi     : {manifest['doi']}")
    print(f"version : {version}   (docs describe {REFERENCE_V1['version']})")

    has_full = "nodes.parquet" in have
    has_lcc = "largest_connected_component_nodes.parquet" in have
    if not (has_full or has_lcc):
        print("No flat node/edge tables in the manifest.", file=sys.stderr)
        return 1

    # ---- variant counts, and the true LCC delta if we have both -----------
    variants = {}
    if has_full:
        variants["full"] = census_pair(
            scan("nodes.parquet", manifest), scan("edges.parquet", manifest), "full"
        )
    if has_lcc:
        variants["lcc"] = census_pair(
            scan("largest_connected_component_nodes.parquet", manifest),
            scan("largest_connected_component_edges.parquet", manifest),
            "lcc",
        )
    out["variants"] = variants

    if has_full and has_lcc:
        dn = variants["full"]["n_nodes"] - variants["lcc"]["n_nodes"]
        de = variants["full"]["n_edges"] - variants["lcc"]["n_edges"]
        pct_n = 100 * dn / variants["full"]["n_nodes"]
        pct_e = 100 * de / variants["full"]["n_edges"]
        print(f"\nLCC drops {dn:,} nodes ({pct_n:.3f}%) and {de:,} edges ({pct_e:.3f}%).")
        print("  -> A1.5: if this is ~0, pick either and record it. If a gold drug or")
        print("     disease sits in the dropped set, it silently leaves your eval set.")
        out["lcc_delta"] = {"nodes_dropped": dn, "edges_dropped": de,
                           "pct_nodes": pct_n, "pct_edges": pct_e}

    if has_full:
        d_nodes = variants["full"]["n_nodes"] - REFERENCE_V1["nodes"]
        d_edges = variants["full"]["n_edges"] - REFERENCE_V1["edges"]
        print(f"\nvs published {REFERENCE_V1['version']}: "
              f"{d_nodes:+,} nodes, {d_edges:+,} edges")
        out["drift_vs_reference"] = {"nodes": d_nodes, "edges": d_edges}

    # ---- schema + censuses, from the richest variant we have -------------
    primary = "nodes.parquet" if has_full else "largest_connected_component_nodes.parquet"
    primary_e = "edges.parquet" if has_full else "largest_connected_component_edges.parquet"
    nodes, edges = scan(primary, manifest), scan(primary_e, manifest)
    out["census_variant"] = "full" if has_full else "lcc"

    out["node_columns"] = nodes.collect_schema().names()
    out["edge_columns"] = edges.collect_schema().names()
    print(f"\nnode columns : {out['node_columns']}")
    print(f"edge columns : {out['edge_columns']}")

    node_census = collect(nodes.group_by("label").len().sort("len", descending=True))
    print("\nnode types:")
    print(node_census)
    out["nodes_by_label"] = node_census.to_dicts()

    edge_census = collect(
        edges.group_by(["label", "relation"]).len().sort("len", descending=True)
    )
    n_labels = int(edge_census["label"].n_unique())
    n_rels = int(edge_census["relation"].n_unique())
    print(f"\nedge label x relation ({edge_census.height} combos, "
          f"{n_labels} labels, {n_rels} distinct relations; "
          f"docs claim {REFERENCE_V1['relations']} edge types for 1.0):")
    with pl.Config(tbl_rows=120):
        print(edge_census)
    out["edges_by_label_relation"] = edge_census.to_dicts()
    out["n_edge_labels"], out["n_relations"] = n_labels, n_rels

    out["directionality"] = collect(
        edges.group_by(["label", "undirected"]).len().sort("label")
    ).to_dicts()

    # ---- the A7 leakage surface ------------------------------------------
    leakage = collect(
        edges.filter(pl.col("relation").is_in(LEAKY_RELATIONS))
        .group_by(["label", "relation"]).len().sort(["label", "relation"])
    )
    print("\nA7 leakage surface:")
    print(leakage)
    out["leakage_surface"] = leakage.to_dicts()
    out["leakage_total_edges"] = int(leakage["len"].sum()) if leakage.height else 0

    # ---- ID shapes A6 will join on ---------------------------------------
    samples = collect(
        nodes.filter(pl.col("label").is_in(["DRG", "DIS", "GEN"]))
        .group_by("label").agg(pl.col("id").first().alias("example_id"))
    )
    print("\nID formats:")
    print(samples)
    out["id_examples"] = samples.to_dicts()

    CENSUS.parent.mkdir(parents=True, exist_ok=True)
    CENSUS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {CENSUS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
