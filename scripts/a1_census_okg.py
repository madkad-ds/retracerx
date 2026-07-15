#!/usr/bin/env python3
"""
A1 — Verify the pinned OptimusKG snapshot and emit a census.

Reads paths from data/manifest/okg_manifest.json (written by a1_pin_okg.py).
Everything is a lazy scan: 21.8M edges carry a JSON `properties` string, and an
eager read on a 32 GiB box is a coin flip. `optimuskg.load_graph()` is eager —
this script deliberately does not use it.

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

# Published figures for the full graph (optimuskg.ai/docs). The docs and the
# PyPI blurb disagree on relation-type count (27 vs 26) — trust the census.
EXPECTED_FULL_NODES = 190_531
EXPECTED_FULL_EDGES = 21_813_816


def scan(rel: str, manifest: dict) -> pl.LazyFrame:
    path = REPO / manifest["files"][rel]["local_path"]
    return pl.scan_parquet(path)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # older polars
        return lf.collect(streaming=True)


def main() -> int:
    if not MANIFEST.exists():
        print("No manifest. Run scripts/a1_pin_okg.py first.", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    have = set(manifest["files"])
    out: dict = {
        "doi": manifest["doi"],
        "dataset_version": manifest["dataset_version"],
        "graph_variant": None,
    }

    # Which flat pair did we pin?
    if "largest_connected_component_nodes.parquet" in have:
        nodes_f = "largest_connected_component_nodes.parquet"
        edges_f = "largest_connected_component_edges.parquet"
        out["graph_variant"] = "lcc"
    elif "nodes.parquet" in have:
        nodes_f, edges_f = "nodes.parquet", "edges.parquet"
        out["graph_variant"] = "full"
    else:
        print("No flat node/edge tables in the manifest.", file=sys.stderr)
        return 1

    nodes = scan(nodes_f, manifest)
    edges = scan(edges_f, manifest)

    print(f"variant       : {out['graph_variant']}")
    print(f"node columns  : {nodes.collect_schema().names()}")
    print(f"edge columns  : {edges.collect_schema().names()}")
    out["node_columns"] = nodes.collect_schema().names()
    out["edge_columns"] = edges.collect_schema().names()

    # --- counts -----------------------------------------------------------
    n_nodes = collect(nodes.select(pl.len())).item()
    n_edges = collect(edges.select(pl.len())).item()
    out["n_nodes"], out["n_edges"] = n_nodes, n_edges
    print(f"\nnodes: {n_nodes:,}")
    print(f"edges: {n_edges:,}")

    if out["graph_variant"] == "lcc":
        print(f"  LCC drops {EXPECTED_FULL_NODES - n_nodes:,} nodes and "
              f"{EXPECTED_FULL_EDGES - n_edges:,} edges vs the published full graph "
              f"({EXPECTED_FULL_NODES:,} / {EXPECTED_FULL_EDGES:,}).")
        out["lcc_nodes_dropped"] = EXPECTED_FULL_NODES - n_nodes
        out["lcc_edges_dropped"] = EXPECTED_FULL_EDGES - n_edges
    else:
        if n_nodes != EXPECTED_FULL_NODES or n_edges != EXPECTED_FULL_EDGES:
            print(f"  !! counts differ from published "
                  f"({EXPECTED_FULL_NODES:,}/{EXPECTED_FULL_EDGES:,}) — "
                  f"the release moved. Record this.", file=sys.stderr)

    # --- node label census ------------------------------------------------
    node_census = collect(
        nodes.group_by("label").len().sort("len", descending=True)
    )
    print("\nnode types:")
    print(node_census)
    out["nodes_by_label"] = node_census.to_dicts()

    # --- edge label x relation census -------------------------------------
    edge_census = collect(
        edges.group_by(["label", "relation"]).len().sort("len", descending=True)
    )
    print(f"\nedge label x relation ({edge_census.height} combinations, "
          f"{edge_census['label'].n_unique()} labels, "
          f"{edge_census['relation'].n_unique()} distinct relations):")
    with pl.Config(tbl_rows=100):
        print(edge_census)
    out["edges_by_label_relation"] = edge_census.to_dicts()
    out["n_edge_labels"] = int(edge_census["label"].n_unique())
    out["n_relations"] = int(edge_census["relation"].n_unique())

    # --- directionality ---------------------------------------------------
    directionality = collect(
        edges.group_by(["label", "undirected"]).len().sort("label")
    )
    out["directionality"] = directionality.to_dicts()

    # --- the A7 leakage surface -------------------------------------------
    # These are the label/relation pairs that encode "this drug treats this
    # thing". Every one of them has to come out before evaluation, not just
    # DRG-DIS/INDICATION.
    leakage = collect(
        edges.filter(
            pl.col("relation").is_in(["INDICATION", "OFF_LABEL_USE", "CONTRAINDICATION"])
        )
        .group_by(["label", "relation"])
        .len()
        .sort(["label", "relation"])
    )
    print("\nA7 leakage surface (drug->indication-ish edges):")
    print(leakage)
    out["leakage_surface"] = leakage.to_dicts()
    out["leakage_total_edges"] = int(leakage["len"].sum())

    # --- ID shape spot-check ----------------------------------------------
    # Confirms the join keys A6 depends on before you write any crosswalk code.
    samples = collect(
        nodes.filter(pl.col("label").is_in(["DRG", "DIS", "GEN"]))
        .group_by("label")
        .agg(pl.col("id").first().alias("example_id"))
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
