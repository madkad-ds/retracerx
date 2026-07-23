#!/usr/bin/env python3
"""
a2_verify_canon.py — prove the canonical tables are the pinned graph, and
measure the three things A3/A6/A7/A8 are about to assume.

Gates (any failure => exit 1, do not proceed to A3):
    G1  node count == census (190,531)
    G2  node `id` is globally unique      -> A3's uniqueness constraint is real
    G3  edge count == census (21,813,816)
    G4  27 edge labels / 36 distinct relations   (the docs' 26 is wrong)
    G5  `undirected` is constant per label, and exactly 12 labels are undirected
    G6  no dangling edge endpoints (graph_variant=full => every endpoint resolves)
    G7  the A7 leakage surface is intact: 7 label/relation pairs, 79,949 edges
    G8  spot-check counts from the A1 census survived canonicalization

Measured and reported (not gated — these are decisions downstream, not errors):
    M1  orientation: are undirected edges stored once, or in both directions?
        -> decides whether A8 emits both orientations, and confirms A7 must
           anti-join on endpoint_key rather than (from, to).
    M2  provenance coverage per label (pitfall #5)
    M3  promoted key coverage: inchi_key on DRG, umls_cui on DIS, symbol on GEN
        -> the premise that "A6 is a join, not a project". a1_keycoverage.py is
           the real A6 gate; this is the early warning.
    M4  self-loops and duplicate (from,to,label,relation) tuples
    M5  disease join keys on the population A6 joins (drug-touched DIS), native
        umls_cui vs the xrefs route, plus the other vocabularies xrefs exposes

Writes data/manifest/a2_canon_manifest.json (dataset_version + graph_variant +
sha256 of every output — the row A7's MANIFEST.json inherits) and
data/manifest/a2_integrity.json.

Usage:
    python scripts/a2_verify_canon.py
"""
from __future__ import annotations

import re
from collections import Counter

import polars as pl

from a2_common import (
    A2_CANON_MANIFEST, A2_DECISIONS, A2_INTEGRITY, A2_POLICY, CANON_EDGES_DIR,
    CANON_NODES, EXPECT, NODE_PROMOTE, census, census_expect, load_json, now_iso,
    sha256_file, sha256_tree, warn, write_json,
)

# A1 census facts that canonicalization must not perturb.
LEAKAGE_SURFACE = {
    ("DRG-DIS", "INDICATION"): 57_601,
    ("DRG-DIS", "CONTRAINDICATION"): 11_718,
    ("DRG-DIS", "OFF_LABEL_USE"): 1_061,
    ("DRG-PHE", "CONTRAINDICATION"): 8_279,
    ("DRG-PHE", "INDICATION"): 1_027,
    ("DRG-PHE", "OFF_LABEL_USE"): 201,
    ("DRG-BPO", "INDICATION"): 62,
}
SPOT_CHECKS = {
    ("DIS-GEN", "ASSOCIATED_WITH"): 9_734_774,
    ("GEN-GEN", "INTERACTS_WITH"): 327_924,
    ("PWY-GEN", "INTERACTS_WITH"): 46_977,
}
KEY_OWNER = {"inchi_key": "DRG", "umls_cui": "DIS", "symbol": "GEN"}

gates: list[dict] = []


EDGE_LABEL_RE = re.compile(r"^[A-Z]{3}-[A-Z]{3}$")


def census_directionality() -> dict[str, bool] | None:
    """label -> undirected, read from okg_census.json .directionality.

    A1 *measured* this, one entry per edge label. EXPECT's scalar count came from A1's
    prose handover instead, and prose is not a measurement: the handover said 12, the
    census array says 13. Prefer the array — and compare label by label, because two
    counts can agree while disagreeing about which labels they name.
    """
    rows = census().get("directionality")
    if not isinstance(rows, list) or not rows:
        return None
    out: dict[str, bool] = {}
    for r in rows:
        if not isinstance(r, dict) or "undirected" not in r:
            return None
        # The label field's name isn't pinned by A1; find the value that looks like one.
        lab = next((v for v in r.values()
                    if isinstance(v, str) and EDGE_LABEL_RE.match(v)), None)
        if lab is None:
            return None
        out[lab] = bool(r["undirected"])
    return out


def gate(name: str, ok: bool, detail: str) -> None:
    gates.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    if not CANON_NODES.exists() or not any(CANON_EDGES_DIR.glob("*.parquet")):
        raise SystemExit("FATAL: canonical tables missing — run a2_canon_nodes.py "
                         "and a2_canon_edges.py first.")
    census_expect(census())
    policy = load_json(A2_POLICY)

    nodes = pl.scan_parquet(CANON_NODES)
    edges = pl.scan_parquet(str(CANON_EDGES_DIR / "*.parquet"))

    # ---------------- G1 / G2 nodes
    n_nodes = nodes.select(pl.len()).collect().item()
    gate("G1 node count", n_nodes == EXPECT["nodes_total"],
         f"{n_nodes:,} (census {EXPECT['nodes_total']:,})")

    n_unique = nodes.select(pl.col("id").n_unique()).collect().item()
    gate("G2 node id unique", n_unique == n_nodes,
         f"{n_unique:,} distinct ids of {n_nodes:,} rows"
         + ("" if n_unique == n_nodes else " — A3's uniqueness constraint would fail"))

    nodes_by_label = (nodes.group_by("label").len().sort("len", descending=True)
                      .collect().to_dicts())
    gate("G1b node labels", len(nodes_by_label) == EXPECT["node_labels"],
         f"{len(nodes_by_label)} labels (census {EXPECT['node_labels']})")

    # ---------------- G3 / G4 edges
    n_edges = edges.select(pl.len()).collect().item()
    gate("G3 edge count", n_edges == EXPECT["edges_total"],
         f"{n_edges:,} (census {EXPECT['edges_total']:,})")

    lr = (edges.group_by(["label", "relation"]).len()
          .sort("len", descending=True).collect())
    n_labels = lr["label"].n_unique()
    n_relations = lr["relation"].n_unique()
    gate("G4 labels/relations",
         n_labels == EXPECT["edge_labels"] and n_relations == EXPECT["relations"],
         f"{n_labels} labels / {n_relations} distinct relations "
         f"(census {EXPECT['edge_labels']}/{EXPECT['relations']}; PyPI's 26 is wrong)")

    # ---------------- G5 undirected
    dirs = (edges.group_by("label")
            .agg(pl.col("undirected").n_unique().alias("n_vals"),
                 pl.col("undirected").any().alias("any_true"),
                 pl.len().alias("edges"))
            .sort("label").collect())
    mixed = dirs.filter(pl.col("n_vals") > 1)["label"].to_list()
    measured = set(dirs.filter(pl.col("any_true"))["label"].to_list())
    cen_dir = census_directionality()

    if cen_dir is None:
        # No measurement to check against; fall back to the scalar, and say so.
        ok = not mixed and len(measured) == EXPECT["undirected_labels"]
        detail = (f"{len(measured)} undirected labels (EXPECT {EXPECT['undirected_labels']}"
                  f", census has no .directionality array to check against)")
    else:
        expected = {lb for lb, u in cen_dir.items() if u}
        extra, missing = sorted(measured - expected), sorted(expected - measured)
        ok = not mixed and not extra and not missing
        detail = f"{len(measured)} undirected labels (census .directionality: {len(expected)})"
        if extra:
            detail += f"; undirected here but not in census: {extra}"
        if missing:
            detail += f"; undirected in census but not here: {missing}"
        if len(expected) != EXPECT["undirected_labels"]:
            warn(f"EXPECT['undirected_labels']={EXPECT['undirected_labels']} disagrees with "
                 f"the census's own .directionality array ({len(expected)}). The array is a "
                 f"measurement; the constant came from prose. Trusting the array.")
    gate("G5 undirected per label", ok,
         detail + (f"; MIXED within {mixed}" if mixed else "; constant within every label"))

    # ---------------- G6 dangling endpoints
    endpoints = pl.concat([
        edges.select(pl.col("from").alias("id")),
        edges.select(pl.col("to").alias("id")),
    ]).unique()
    dangling = (endpoints.join(nodes.select("id"), on="id", how="anti")
                .select(pl.len()).collect().item())
    gate("G6 dangling endpoints", dangling == 0,
         f"{dangling:,} edge endpoints with no node row "
         f"(graph_variant={policy['graph_variant']})")

    # ---------------- G7 leakage surface
    counts = {(r["label"], r["relation"]): r["len"] for r in lr.to_dicts()}
    surface_rows, surface_total, surface_bad = [], 0, []
    for k, expected in LEAKAGE_SURFACE.items():
        got = counts.get(k, 0)
        surface_total += got
        surface_rows.append({"label": k[0], "relation": k[1], "edges": got,
                             "census": expected, "match": got == expected})
        if got != expected:
            surface_bad.append(f"{k[0]}/{k[1]} {got:,}!={expected:,}")
    gate("G7 leakage surface", not surface_bad and surface_total == 79_949,
         f"{len(LEAKAGE_SURFACE)} pairs, {surface_total:,} edges (census 79,949)"
         + (f"; MISMATCH {surface_bad}" if surface_bad else "")
         + f"; DRG-DIS/INDICATION alone would leave "
           f"{surface_total - counts.get(('DRG-DIS','INDICATION'), 0):,} behind")

    # ---------------- G8 spot checks
    bad = [f"{k[0]}/{k[1]} {counts.get(k, 0):,}!={v:,}"
           for k, v in SPOT_CHECKS.items() if counts.get(k, 0) != v]
    gate("G8 census spot checks", not bad,
         "all match" if not bad else f"MISMATCH {bad}")

    # ---------------- M1 orientation
    und_labels = dirs.filter(pl.col("any_true"))["label"].to_list()
    orient = (edges.filter(pl.col("label").is_in(und_labels))
              .group_by("label")
              .agg(pl.len().alias("edges"),
                   pl.col("endpoint_key").n_unique().alias("distinct_pairs"))
              .sort("label").collect()
              .with_columns((pl.col("edges") / pl.col("distinct_pairs")).round(3)
                            .alias("rows_per_pair"))
              .to_dicts())
    both = [r["label"] for r in orient if r["rows_per_pair"] > 1.5]
    once = [r["label"] for r in orient if r["rows_per_pair"] <= 1.5]
    print(f"\n[M1] undirected storage: stored once -> {once}\n"
          f"                        both orientations -> {both or 'none'}\n"
          f"     => A7 must anti-join on endpoint_key (a (from,to) anti-join misses "
          f"any pair stored the other way round);\n"
          f"     => A8 must decide explicitly whether to emit both orientations "
          f"(PyKEEN treats triples as directed).")

    # ---------------- M2 provenance
    prov = (edges.group_by("label")
            .agg(pl.len().alias("edges"),
                 (pl.col("sources_direct").list.len().fill_null(0) > 0)
                 .mean().alias("direct_cov"),
                 (pl.col("sources_indirect").list.len().fill_null(0) > 0)
                 .mean().alias("indirect_cov"))
            .sort("edges", descending=True).collect())
    print("\n[M2] provenance coverage (properties.sources.*, lists — not provided_by):")
    print(prov.head(10))
    no_prov = prov.filter(pl.col("direct_cov") == 0)["label"].to_list()
    if no_prov:
        print(f"     labels with ZERO direct provenance: {no_prov}")

    # ---------------- M3 promoted keys
    keycov = []
    for field, owner in KEY_OWNER.items():
        if field not in nodes.collect_schema().names():
            keycov.append({"field": field, "owner": owner, "present": False})
            continue
        r = (nodes.filter(pl.col("label") == owner)
             .select(pl.len().alias("n"),
                     pl.col(field).is_not_null().sum().alias("have"))
             .collect().row(0))
        keycov.append({"field": field, "owner": owner, "present": True,
                       "nodes": r[0], "with_key": r[1],
                       "coverage": round(r[1] / r[0], 4) if r[0] else None})
    print("\n[M3] promoted key coverage (the 'A6 is a join, not a project' premise):")
    for k in keycov:
        print("     " + str(k))
    print("     -> a1_keycoverage.py is the A6 gate; run it against canon_nodes.parquet.")

    # ---------------- M4 hygiene
    self_loops = edges.filter(pl.col("from") == pl.col("to")).select(pl.len()).collect().item()
    dupes = (edges.group_by(["from", "to", "label", "relation"]).len()
             .filter(pl.col("len") > 1)
             .select(pl.len().alias("tuples"), (pl.col("len") - 1).sum().alias("extra"))
             .collect().row(0))
    print(f"\n[M4] self-loops: {self_loops:,} | duplicate (from,to,label,relation) tuples: "
          f"{dupes[0]:,} ({dupes[1]:,} extra rows). Not removed here — A2 canonicalizes, "
          f"it does not edit the graph.")

    # ---------------- M5 disease join keys, on the population A6 actually joins
    # M3's denominator is every DIS node. A6 only ever needs a key for diseases that
    # carry drug edges, so the honest number is coverage on THAT subset. Both are
    # reported; the drug-touched one is the one A6 should be designed against.
    dis = nodes.filter(pl.col("label") == "DIS")
    m5: dict = {"schema_has_umls_cui_xrefs": "umls_cui_xrefs" in nodes.collect_schema().names()}
    if not m5["schema_has_umls_cui_xrefs"]:
        warn("canon_nodes.parquet predates umls_cui_xrefs — re-run a2_canon_nodes.py "
             "to measure the xrefs route.")
    else:
        dd = CANON_EDGES_DIR / "DRG-DIS.parquet"
        touched = None
        if dd.exists():
            e = pl.scan_parquet(dd)
            touched = (pl.concat([e.select(pl.col("from").alias("id")),
                                  e.select(pl.col("to").alias("id"))]).unique())

        def cov(lf: pl.LazyFrame, note: str) -> dict:
            has_native = pl.col("umls_cui").is_not_null()
            has_xref = pl.col("umls_cui_xrefs").list.len() > 0
            r = lf.select(
                pl.len().alias("nodes"),
                has_native.sum().alias("umls_cui"),
                has_xref.sum().alias("via_xrefs"),
                (has_native | has_xref).sum().alias("either"),
                (has_xref & ~has_native).sum().alias("xrefs_only"),
                (pl.col("umls_cui_xrefs").list.len() > 1).sum().alias("ambiguous_multi_cui"),
            ).collect().to_dicts()[0]
            r["population"] = note
            for k in ("umls_cui", "via_xrefs", "either", "xrefs_only"):
                r[f"{k}_pct"] = round(r[k] / r["nodes"], 4) if r["nodes"] else None
            return r

        m5["all_dis"] = cov(dis, "every DIS node")
        if touched is not None:
            m5["drug_touched_dis"] = cov(
                dis.join(touched, on="id", how="semi"), "DIS nodes touching a DRG-DIS edge")

        print("\n[M5] disease join keys (M3's denominator is every DIS node; A6 only "
              "needs the drug-touched ones):")
        for key in ("all_dis", "drug_touched_dis"):
            if key not in m5:
                continue
            r = m5[key]
            print(f"     {r['population']}: {r['nodes']:,} nodes")
            print(f"       umls_cui         {r['umls_cui']:,} ({r['umls_cui_pct']:.1%})")
            print(f"       via xrefs        {r['via_xrefs']:,} ({r['via_xrefs_pct']:.1%})")
            print(f"       either           {r['either']:,} ({r['either_pct']:.1%})  <- A6's real ceiling")
            print(f"       xrefs adds       {r['xrefs_only']:,} nodes the native key misses")
            if r["ambiguous_multi_cui"]:
                print(f"       AMBIGUOUS        {r['ambiguous_multi_cui']:,} nodes carry >1 UMLS "
                      f"xref — A6 must pick a rule, A2 kept them all")

        # Counted in Python on purpose: DIS is ~36k rows, and every Polars explode form
        # trips the 1.x->2.0 `empty_as_null` deprecation. Not worth a warning filter.
        vlists = (dis.filter(pl.col("xref_vocabs").list.len() > 0)
                  .select("xref_vocabs").collect())["xref_vocabs"].to_list()
        counter: Counter = Counter(v for lst in vlists for v in (lst or []) if v)
        vocab_top = [{"vocab": k, "dis_nodes": n} for k, n in counter.most_common(15)]
        m5["xref_vocabs_top"] = vocab_top
        print("     other bridges visible in xrefs (top vocabularies across DIS):")
        print(f"       {[r['vocab'] for r in vocab_top]}")

    # ---------------- manifests
    write_json(A2_INTEGRITY, {
        "generated_at": now_iso(),
        "dataset_version": EXPECT["dataset_version"],
        "graph_variant": policy["graph_variant"],
        "gates": gates,
        "nodes_by_label": nodes_by_label,
        "edges_by_label_relation": lr.to_dicts(),
        "leakage_surface": surface_rows,
        "leakage_surface_total": surface_total,
        "undirected_labels": und_labels,
        "orientation": orient,
        "provenance_coverage": prov.to_dicts(),
        "promoted_key_coverage": keycov,
        "disease_join_keys": m5,
        "self_loops": self_loops,
        "duplicate_tuples": dupes[0],
        "duplicate_extra_rows": dupes[1],
        "dangling_endpoints": dangling,
    })

    write_json(A2_CANON_MANIFEST, {
        "generated_at": now_iso(),
        "dataset_version": EXPECT["dataset_version"],
        "graph_variant": policy["graph_variant"],
        "stratified_policy": policy["stratified_policy"],
        "inputs": {
            "a2_source_policy.json": sha256_file(A2_POLICY),
            "a2_decisions.json": sha256_file(A2_DECISIONS),
        },
        "outputs": {
            "data/canon/canon_nodes.parquet": {
                "sha256": sha256_file(CANON_NODES),
                "rows": n_nodes,
                "bytes": CANON_NODES.stat().st_size,
            },
            "data/canon/canon_edges/": {
                "sha256_tree": sha256_tree(CANON_EDGES_DIR),
                "rows": n_edges,
                "files": sorted(p.name for p in CANON_EDGES_DIR.glob("*.parquet")),
                "bytes": sum(p.stat().st_size for p in CANON_EDGES_DIR.glob("*.parquet")),
            },
        },
        "source_files_used": sorted({e["file"] for k in ("nodes", "edges")
                                     for e in policy[k].values()}),
        "source_files_not_checksum_verified": sorted(
            {e["file"] for k in ("nodes", "edges") for e in policy[k].values()
             if e["checksum"] != "verified"}),
        "all_gates_passed": all(g["pass"] for g in gates),
    })

    failed = [g["gate"] for g in gates if not g["pass"]]
    if failed:
        raise SystemExit(f"\nFAILED GATES: {failed} — do not proceed to A3.")
    print("\nA2 gates all pass. canon_nodes.parquet + canon_edges/ are the pinned graph.")


if __name__ == "__main__":
    main()
