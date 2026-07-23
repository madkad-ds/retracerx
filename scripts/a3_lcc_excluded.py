#!/usr/bin/env python3
"""
A3.5 — Derive the LCC-excluded node set.

WHY THIS EXISTS
  The how-to (A4) says 982 drugs have "no edges at all" and tells you to exclude them by
  anti-joining edges.parquet on from/to. Measured against the imported full graph, that
  predicate selects NOTHING: zero nodes of any label have degree 0. The census already
  contained the disproof — lcc_delta drops 1,167 nodes AND 1,086 edges, and removing truly
  isolated nodes would drop zero edges.

  The COUNT is right; the DEFINITION is wrong. The 1,167 are nodes outside the largest
  connected component, sitting in small components with edges among themselves. So the
  exclusion A1.5 deliberately deferred to A6/A7 cannot be expressed as a degree filter,
  and any script that tries will silently exclude no one.

  Jargon:
    connected component   a maximal set of nodes where every node is reachable from every
                          other by some path.
    LCC                   largest connected component — the biggest such set.
    anti-join             keep only left-table rows with NO match on the right.
    drift gate            a check that a value measured today still equals the value pinned
                          on the first run; it fails loudly instead of quietly updating.

METHOD
  Primary: anti-join canon_nodes against the release's OWN lcc node table. That uses the
  upstream definition rather than a reconstruction of it. Recomputing is available behind
  an explicit --compute-lcc flag and is RECORDED as the method used — it is never a silent
  substitute for the shipped table.

OUTPUTS
  data/manifest/a3_lcc_excluded.parquet   id, label, degree, component_id, component_size
  data/manifest/a3_lcc_excluded.json      counts, per-label breakdown, component histogram,
                                          method, sha256 — and the pins for drift detection

CONSUMERS
  A4  candidate whitelist: anti-join Broad Launched compounds against this set.
  A7  exclusion #3 at inference: unreachable-from-the-main-graph drugs.
  Both must key on THIS artifact, not on a degree filter.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from a3_common import fatal, info, load_json, pick, sha256_file, warn, write_manifest


# ---------------------------------------------------------------------------
# locate the release's own LCC node table
# ---------------------------------------------------------------------------
def discover_lcc_nodes(manifest_dir: Path, cache_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            fatal(f"--lcc-nodes {p} does not exist")
        info(f"LCC node table (explicit): {p}")
        return p

    okg = load_json(manifest_dir / "okg_manifest.json", "okg_manifest.json")
    files = pick(okg, ["files"], "the pinned file inventory", "okg_manifest.json")
    if isinstance(files, dict):
        relpaths = list(files.keys())
    elif isinstance(files, list):
        relpaths = [f.get("name") or f.get("path") or "" for f in files]
    else:
        fatal(f"okg_manifest.json#files is {type(files).__name__}, expected dict or list")

    cand = [r for r in relpaths
            if r.lower().endswith(".parquet") and "lcc" in r.lower() and "node" in r.lower()]
    if not cand:
        fatal("no LCC node table found in the pinned file inventory",
              f"parquet files present: {sorted(r for r in relpaths if r.endswith('.parquet'))}",
              "Pass --lcc-nodes <path> if it is named differently, or --compute-lcc to derive "
              "components locally (recorded as a different method).")
    if len(cand) > 1:
        exact = [c for c in cand if Path(c).name in ("nodes.parquet", "node.parquet")]
        if len(exact) != 1:
            fatal("ambiguous LCC node table", f"candidates: {sorted(cand)}",
                  "Disambiguate with --lcc-nodes <path>.")
        cand = exact

    rel = cand[0]
    for base in (cache_dir, cache_dir / "okg", Path(".")):
        p = base / rel
        if p.exists():
            info(f"LCC node table: {p}  (from okg_manifest.json#files)")
            return p
    hits = list(cache_dir.rglob(Path(rel).name))
    lcc_hits = [h for h in hits if "lcc" in str(h).lower()]
    if len(lcc_hits) == 1:
        info(f"LCC node table: {lcc_hits[0]}  (resolved by search under {cache_dir})")
        return lcc_hits[0]
    fatal(f"LCC node table '{rel}' is in the manifest but not on disk under {cache_dir}",
          f"search hits: {[str(h) for h in hits]}",
          "Re-run scripts/a1_pin_okg.py to fetch it, or pass --lcc-nodes.")


def compute_lcc_ids(nodes: pl.LazyFrame, edges_glob: str) -> pl.DataFrame:
    """Opt-in fallback: derive the LCC locally. Recorded as a DIFFERENT method."""
    try:
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as e:
        fatal(f"--compute-lcc needs numpy + scipy ({e})",
              "pip install scipy --break-system-packages",
              "Or drop the flag and use the release's own LCC table.")

    ids = nodes.select("id").collect()["id"]
    index = {v: i for i, v in enumerate(ids.to_list())}
    n = len(index)
    e = (pl.scan_parquet(edges_glob).select(["from", "to"]).collect())
    src = np.fromiter((index[x] for x in e["from"].to_list()), dtype=np.int64, count=e.height)
    dst = np.fromiter((index[x] for x in e["to"].to_list()), dtype=np.int64, count=e.height)
    g = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels)
    biggest = int(np.argmax(sizes))
    info(f"computed {ncomp:,} components; LCC has {int(sizes[biggest]):,} of {n:,} nodes")
    return pl.DataFrame({"id": ids.filter(pl.Series(labels == biggest))})


# ---------------------------------------------------------------------------
# components within the excluded subgraph (tiny: ~1,167 nodes / ~1,086 edges)
# ---------------------------------------------------------------------------
def components(excluded_ids: list[str], edge_pairs: list[tuple[str, str]]) -> dict[str, int]:
    parent = {i: i for i in excluded_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {i: find(i) for i in excluded_ids}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Derive the LCC-excluded node set (A3.5).")
    ap.add_argument("--nodes", default="data/canon/canon_nodes.parquet")
    ap.add_argument("--edges-glob", default="data/canon/canon_edges/*.parquet")
    ap.add_argument("--manifest-dir", default="data/manifest")
    ap.add_argument("--cache-dir", default=os.environ.get("OPTIMUSKG_CACHE_DIR", "data/okg_cache"))
    ap.add_argument("--lcc-nodes", default=None, help="explicit path to the release's LCC node table")
    ap.add_argument("--compute-lcc", action="store_true",
                    help="derive components locally instead of using the shipped table")
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    a1 = load_json(mdir / "a1_decisions.json", "a1_decisions.json")
    census = load_json(mdir / "okg_census.json", "okg_census.json")

    gv = pick(a1, ["graph_variant", "graph_variant.value"], "graph_variant", "a1_decisions.json")
    if isinstance(gv, dict):
        gv = gv.get("value", gv)
    if gv != "full":
        fatal(f"graph_variant is '{gv}', not 'full'",
              "On the lcc variant the excluded nodes are already absent; this artifact is moot.")

    exp_nodes = int(pick(census, ["lcc_delta.nodes_dropped"], "lcc_delta.nodes_dropped",
                         "okg_census.json"))
    exp_edges = int(pick(census, ["lcc_delta.edges_dropped"], "lcc_delta.edges_dropped",
                         "okg_census.json"))

    nodes = pl.scan_parquet(args.nodes)

    # --- the excluded set ------------------------------------------------------------
    if args.compute_lcc:
        method = "computed_scipy"
        warn("deriving the LCC locally (--compute-lcc). This is a RECONSTRUCTION of the "
             "release's definition, not the release's own table. Recorded as such.")
        lcc = compute_lcc_ids(nodes, args.edges_glob).lazy()
    else:
        method = "upstream_lcc_table"
        lcc = pl.scan_parquet(discover_lcc_nodes(mdir, Path(args.cache_dir), args.lcc_nodes)) \
                .select("id")

    excluded = nodes.join(lcc, on="id", how="anti").select(["id", "label"]).collect()
    info(f"excluded nodes: {excluded.height:,} (method={method})")

    if excluded.height != exp_nodes:
        fatal(f"excluded {excluded.height:,} nodes but census lcc_delta says {exp_nodes:,}",
              "The LCC table and the canonical nodes describe different graphs.")

    # --- the edges they carry: the whole point ---------------------------------------
    ids = excluded["id"].to_list()
    idset = set(ids)
    inc = (pl.scan_parquet(args.edges_glob)
           .filter(pl.col("from").is_in(ids) | pl.col("to").is_in(ids))
           .select(["from", "to"]).collect())

    crossing = [(a, b) for a, b in zip(inc["from"].to_list(), inc["to"].to_list())
                if (a in idset) != (b in idset)]
    if crossing:
        fatal(f"{len(crossing):,} edges cross the LCC boundary",
              f"examples: {crossing[:5]}",
              "An excluded node with an edge INTO the LCC would be in the LCC by definition. "
              "The LCC table and canon_edges disagree.")
    if inc.height != exp_edges:
        fatal(f"excluded nodes carry {inc.height:,} edges but census says {exp_edges:,}")

    info(f"excluded nodes carry {inc.height:,} edges, all internal — this is why a degree "
         f"filter selects nothing")

    # --- components, degrees ----------------------------------------------------------
    pairs = list(zip(inc["from"].to_list(), inc["to"].to_list()))
    comp_of = components(ids, pairs)
    deg = (pl.concat([inc.select(pl.col("from").alias("id")), inc.select(pl.col("to").alias("id"))])
           .group_by("id").len().rename({"len": "degree"}))

    out = (excluded
           .join(deg, on="id", how="left")
           .with_columns(pl.col("degree").fill_null(0),
                         pl.Series("component_id", [comp_of[i] for i in ids]))
           )
    csize = out.group_by("component_id").len().rename({"len": "component_size"})
    out = out.join(csize, on="component_id", how="left").sort(["component_size", "label", "id"])

    isolated = out.filter(pl.col("degree") == 0).height
    if isolated:
        fatal(f"{isolated:,} excluded nodes have degree 0",
              "The imported graph measured ZERO isolated nodes; this contradicts it.")

    by_label = {r[0]: r[1] for r in out.group_by("label").len().sort("len", descending=True).rows()}
    hist = {str(r[0]): r[1] for r in
            csize.group_by("component_size").len().sort("component_size").rows()}
    info(f"per-label: {by_label}")
    info(f"components: {csize.height:,}; size histogram (size: count) {hist}")
    info(f"degree range: {out['degree'].min()}–{out['degree'].max()}")

    # --- write, then drift-gate against the pin --------------------------------------
    ppath = mdir / "a3_lcc_excluded.parquet"
    out.write_parquet(ppath)

    jpath = mdir / "a3_lcc_excluded.json"
    prev = load_json(jpath, "a3_lcc_excluded.json") if jpath.exists() else None
    payload = {
        "step": "A3.5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset_version": census.get("dataset_version"),
        "graph_variant": gv,
        "excluded_nodes": out.height,
        "excluded_edges": inc.height,
        "by_label": by_label,
        "n_components": csize.height,
        "component_size_histogram": hist,
        "degree_min": int(out["degree"].min()),
        "degree_max": int(out["degree"].max()),
        "parquet": str(ppath),
        "parquet_sha256": sha256_file(ppath),
        "warrant": ("The how-to's 'no edges at all' is false: 0 nodes have degree 0. "
                    "These nodes sit outside the LCC with edges among themselves. "
                    "A4/A7 must anti-join THIS artifact, not filter on degree."),
    }

    if prev is not None:
        for k in ("excluded_nodes", "excluded_edges", "by_label", "n_components"):
            if prev.get(k) is not None and prev[k] != payload[k]:
                fatal(f"drift in {k}: pinned {prev[k]}, measured {payload[k]}",
                      "The excluded set changed. Adjudicate before A4/A7 consume it.")
        if prev.get("method") != method:
            warn(f"method changed: pinned '{prev.get('method')}', now '{method}'. "
                 f"Counts agree, but the definitions are not identical by construction.")
        info("drift gate: measured set matches the pin")
    else:
        info("no previous pin — this run establishes it")

    write_manifest(jpath, payload)
    info("done. A4 whitelist and A7 exclusion #3 anti-join a3_lcc_excluded.parquet on `id`.")


if __name__ == "__main__":
    main()
