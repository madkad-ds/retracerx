#!/usr/bin/env python3
"""
a2_source_policy.py — decide, once and on evidence, which file each node/edge
label is canonicalized FROM, and record why.

Why this step exists
--------------------
A1 established two facts that collide in A2:

  * `properties` is a JSON string in the flat tables but a native Struct in the
    stratified per-type files. Struct is the difference between JSON-parsing
    21.8M rows and reading a typed column.
  * 27 of 41 files publish an MD5 that doesn't match their bytes. They were
    adjudicated benign — but "adjudicated benign" is a judgement, and
    "checksum-verified" is a fact. They are not the same warrant.

So the fast representation and the verified bytes are not always the same file.
That choice must be made per label, on the record, before any bytes are read —
not improvised inside the canonicalizer.

What it does
------------
  1. Restricts to the tables A1.5 froze (a1_decisions.json .source_tables),
     which is also what keeps `full` from silently mixing with `lcc` files.
  2. Reads every candidate's real schema and its real `label` values — the
     filename is only a hypothesis, never the evidence.
  3. Joins that against okg_dispute_audit.json to get each file's checksum tier.
  4. Ranks candidates per label and writes the routing to a2_source_policy.json.
  5. Measures the per-label `properties` field inventory for the CHOSEN file
     only, so a2_canon_* does typed extraction instead of schema guessing.
  6. Freezes the A2 policy choices in a2_decisions.json (A1.5 pattern).

Usage
-----
    source scripts/okg_env.sh
    python scripts/a2_source_policy.py                       # verified bytes win
    python scripts/a2_source_policy.py --stratified-policy adjudicated_ok
    python scripts/a2_source_policy.py --infer-rows 5000
"""
from __future__ import annotations

import argparse

import polars as pl

from a2_common import (
    A2_DECISIONS, A2_POLICY, EXPECT, NODE_PROMOTE, STEM_TO_LABEL,
    a1_source_tables, census, census_expect, die, dispute_status, graph_variant,
    kind_of, manifest_files, now_iso, properties_repr, resolve_path,
    schema_of, sha256_file, struct_fields, warn, write_json,
)

CHECKSUM_TIER = {
    "verified": 2,
    "disputed_adjudicated_benign": 1,
    "disputed_unadjudicated": 0,
    "unknown": 0,
}


def profile_file(name: str, status: dict[str, str]) -> dict | None:
    path = resolve_path(name)
    if path is None:
        warn(f"{name}: not found on disk (or ambiguous basename) — skipped")
        return None
    if path.suffix != ".parquet":
        return None
    schema = schema_of(path)
    kind = kind_of(schema)
    if kind is None:
        return None  # not a node/edge table (e.g. a README or an index)
    labels = (
        pl.scan_parquet(path).select(pl.col("label").unique()).collect()
        ["label"].drop_nulls().sort().to_list()
    )
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    return {
        "name": name,
        "path": str(path),
        "kind": kind,
        "labels": labels,
        "stratified": len(labels) == 1,
        "rows": rows,
        "properties_repr": properties_repr(schema),
        "columns": {c: str(d) for c, d in schema.items()},
        "checksum": status.get(name, "unknown"),
        "filename_hypothesis": STEM_TO_LABEL.get(path.stem),
    }


def rank(cand: dict, prefer_stratified_adjudicated: bool) -> tuple:
    tier = CHECKSUM_TIER[cand["checksum"]]
    strat = 1 if cand["stratified"] else 0
    fast = 1 if cand["properties_repr"] == "struct" else 0
    if prefer_stratified_adjudicated:
        # speed first, but only among files with a warrant (tier >= 1)
        return (1 if tier >= 1 else 0, fast, strat, tier, cand["rows"])
    # default: verified bytes beat a fast representation
    return (tier, strat, fast, cand["rows"])


def infer_property_fields(cand: dict, label: str, n: int) -> dict[str, str]:
    """Field inventory of `properties` for one label, measured from the chosen file."""
    path = cand["path"]
    if cand["properties_repr"] == "struct":
        dt = schema_of(resolve_path(cand["name"]))["properties"]
        return {k: str(v) for k, v in struct_fields(dt).items()}
    if cand["properties_repr"] != "json":
        return {}
    sample = (
        pl.scan_parquet(path).filter(pl.col("label") == label)
        .select("properties").drop_nulls().head(n).collect()
    )
    if sample.is_empty():
        warn(f"{label}: no non-null properties in the first {n} rows of {cand['name']}")
        return {}
    # Series-level decode: the expression API requires an explicit dtype, which is
    # exactly what we don't have yet. Inference happens once, on a sample, and the
    # result is written to a2_source_policy.json so the canon step never infers.
    decoded = sample["properties"].str.json_decode(infer_schema_length=n)
    return {k: str(v) for k, v in struct_fields(decoded.dtype).items()}


def infer_rows_for(kind: str, args) -> int:
    """How many rows to sample when inventorying `properties`.

    Nodes: ALL of them. nodes.parquet is 190,531 rows total — sampling it is a false
    economy that buys milliseconds and costs correctness. A 2,000-row window over DIS
    (36,345 rows) landed entirely in a block with null xrefs, so the shape came back
    unprovable and the UMLS bridge for ~400 drug-touched diseases went unmeasured. The
    2,000 default was reasoned about 21.8M-row EDGE files and wrongly applied to nodes.

    Edges: keep the window. 21.8M rows is a real cost, and nothing in the edge path
    depends on inference any more — properties.sources decodes at a fixed dtype.
    """
    if kind == "nodes":
        return args.node_infer_rows
    return args.infer_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratified-policy", choices=["verified_only", "adjudicated_ok"],
                    default="verified_only",
                    help="verified_only (default): a disputed file is used only if no "
                         "checksum-verified file carries that label. adjudicated_ok: allow "
                         "files the A1 audit adjudicated benign to win on speed.")
    ap.add_argument("--infer-rows", type=int, default=2000,
                    help="rows sampled per EDGE label to inventory JSON properties")
    ap.add_argument("--node-infer-rows", type=int, default=250_000,
                    help="rows sampled per NODE label. Defaults above the 190,531-row "
                         "node total, i.e. no sampling: a sparse field (xrefs on DIS) "
                         "is invisible in a small window and A2 then refuses to promote it.")
    args = ap.parse_args()

    variant = graph_variant()
    cen = census()
    census_expect(cen)
    status = dispute_status()
    allow = a1_source_tables()

    names = [f.get("name") or f.get("filename") for f in manifest_files()]
    names = [n for n in names if n]
    if allow:
        # Exact relative paths only. Basename matching would readmit lcc/<name>.parquet
        # alongside its full-graph twin — silently mixing the two variants, which is
        # the one thing A1.5 froze a decision to prevent.
        kept = [n for n in names if n in set(allow)]
        if not kept:
            base = {n.split("/")[-1] for n in allow}
            collide = [n for n in names if n.split("/")[-1] in base]
            die("a1_decisions.source_tables matched no manifest path exactly.\n"
                f"  source_tables[:3] = {allow[:3]}\n"
                f"  manifest names[:3] = {names[:3]}\n"
                f"  basename-only matches would be: {collide[:6]}\n"
                "Fix the two by hand. Do NOT fall back to basenames: `lcc/x.parquet` and "
                "`x.parquet` share a basename, and mixing variants corrupts every "
                "downstream count.")
        dropped = sorted(set(names) - set(kept))
        print(f"restricted to {len(kept)}/{len(names)} manifest files via "
              f"a1_decisions.source_tables (dropped: {dropped or 'none'})")
        names = kept

    cands = [p for p in (profile_file(n, status) for n in names) if p]
    if not cands:
        die("no node/edge parquet tables found in the pinned snapshot")

    by_label: dict[tuple[str, str], list[dict]] = {}
    for c in cands:
        for lb in c["labels"]:
            by_label.setdefault((c["kind"], lb), []).append(c)

    warnings: list[str] = []
    for c in cands:
        if c["stratified"] and c["filename_hypothesis"] and c["filename_hypothesis"] not in c["labels"]:
            warnings.append(
                f"{c['name']}: filename implies {c['filename_hypothesis']} but the file's own "
                f"label column says {c['labels']} — trusting the column."
            )

    prefer = args.stratified_policy == "adjudicated_ok"
    routing: dict[str, dict[str, dict]] = {"nodes": {}, "edges": {}}
    for (kind, label), options in sorted(by_label.items()):
        best = max(options, key=lambda c: rank(c, prefer))
        entry = {
            "file": best["name"],
            "path": best["path"],
            "stratified": best["stratified"],
            "properties_repr": best["properties_repr"],
            "checksum": best["checksum"],
            "rows_in_file": best["rows"],
            "columns": best["columns"],
            "chosen_because": (
                f"checksum={best['checksum']}, "
                f"{'stratified' if best['stratified'] else 'flat'}, "
                f"properties={best['properties_repr']}, policy={args.stratified_policy}"
            ),
            "alternatives": [
                {"file": o["name"], "checksum": o["checksum"],
                 "properties_repr": o["properties_repr"], "stratified": o["stratified"]}
                for o in options if o["name"] != best["name"]
            ],
            "property_fields": infer_property_fields(best, label, infer_rows_for(kind, args)),
        }
        if best["checksum"] != "verified":
            verified_alt = [o["name"] for o in options if o["checksum"] == "verified"]
            why = (f"policy={args.stratified_policy} preferred it over verified {verified_alt}"
                   if verified_alt else "NO checksum-verified source carries this label")
            warnings.append(f"{kind}/{label}: canonicalized from {best['name']} "
                            f"({best['checksum']}) — {why}. Warrant is the A1 audit "
                            f"(okg_dispute_audit.json), not the published MD5.")
        if kind == "nodes":
            entry["promotable"] = {
                f: NODE_PROMOTE[f] for f in NODE_PROMOTE if f in entry["property_fields"]
            }
        else:
            src = entry["property_fields"].get("sources")
            entry["provenance"] = {
                "present": src is not None,
                "sources_dtype": src,
            }
            if src is None:
                warnings.append(f"edges/{label}: no `sources` field in properties — "
                                f"provenance will be null (pitfall #5).")
        routing[kind][label] = entry

    n_nodes = len(routing["nodes"])
    n_edges = len(routing["edges"])
    if n_nodes != EXPECT["node_labels"]:
        die(f"routed {n_nodes} node labels, census says {EXPECT['node_labels']}. "
            f"Found: {sorted(routing['nodes'])}")
    if n_edges != EXPECT["edge_labels"]:
        die(f"routed {n_edges} edge labels, census says {EXPECT['edge_labels']} "
            f"(labels, not relations — the docs' 26 is wrong). Found: {sorted(routing['edges'])}")

    policy = {
        "generated_at": now_iso(),
        "dataset_version": EXPECT["dataset_version"],
        "graph_variant": variant,
        "stratified_policy": args.stratified_policy,
        # On the record: an inventory is only as good as the window it was measured in.
        "infer_rows_edges": args.infer_rows,
        "infer_rows_nodes": args.node_infer_rows,
        "nodes": routing["nodes"],
        "edges": routing["edges"],
        "unused_files": sorted(
            c["name"] for c in cands
            if c["name"] not in {e["file"] for k in routing for e in routing[k].values()}
        ),
        "warnings": warnings,
    }
    write_json(A2_POLICY, policy)

    write_json(A2_DECISIONS, {
        "generated_at": now_iso(),
        "dataset_version": EXPECT["dataset_version"],
        "graph_variant": variant,
        "frozen": True,
        "decisions": {
            "primary_key": {
                "value": "nodes.id verbatim; edges keyed (from, to, label, relation)",
                "rationale": "The how-to's A2 snippet aliases id -> id_curie, but A3's Neo4j "
                             "constraint and A7's removed_edge_ids.csv both key on `id` / "
                             "(from,to,label,relation). Renaming here makes the A3 constraint a "
                             "silent no-op. The key is never rewritten; normalization is emitted "
                             "as derived columns (id_prefix, id_local) alongside it.",
            },
            "properties": {
                "value": "retain verbatim as properties_json + typed promotions",
                "rationale": "Struct schemas differ per label, so a single canonical table cannot "
                             "hold native structs. Promote the fields A3/A4/A6 need to typed "
                             "columns; keep the rest as JSON so nothing is lost.",
            },
            "provenance": {
                "value": "sources_direct / sources_indirect as List(str) on every edge",
                "rationale": "There is no provided_by column. Provenance is "
                             "properties.sources.{direct,indirect} and both are LISTS — "
                             "'which datasets contributed or referenced this edge', plural. "
                             "Pitfall #5: expensive to recover later.",
            },
            "undirected": {
                "value": "stored orientation preserved; no reorientation, no dedupe; "
                         "edge_key added",
                "rationale": "DRG-DIS is undirected:true. A7 anti-joins on (from,to) and A8 emits "
                             "directed triples; both break if an undirected edge is stored in the "
                             "opposite orientation to the gold pair. edge_key = unordered "
                             "endpoint key makes A7's removal orientation-safe without mutating "
                             "the graph here. Whether A8 emits both orientations stays an A8 "
                             "decision, informed by a2_integrity.json.",
            },
            "source_policy": {
                "value": args.stratified_policy,
                "rationale": "verified_only: 'adjudicated benign' is a judgement, "
                             "'checksum-verified' is a fact. Where both exist, the fact wins; "
                             "the cost is JSON-decoding instead of a typed read.",
            },
        },
        "inputs": {
            "a2_source_policy.json.sha256": sha256_file(A2_POLICY),
        },
    })

    print(f"\nrouted {n_nodes} node labels / {n_edges} edge labels")
    disputed = [f"{k}/{lb}" for k in ("nodes", "edges") for lb, e in routing[k].items()
                if e["checksum"] != "verified"]
    print(f"labels sourced from non-verified files: {len(disputed)}"
          + (f" -> {disputed}" if disputed else ""))
    for w in warnings:
        print("  WARN " + w)


if __name__ == "__main__":
    main()
