#!/usr/bin/env python3
"""
a6_dis_headroom.py — 717 of repoDB's 1,462 CUIs reach an OKG disease. Before building
a MeSH/OMIM/MedGen -> UMLS crosswalk to chase the other 745, find out whether those
745 diseases are IN the graph at all. A crosswalk to a disease OKG does not contain
buys nothing, and the crosswalk is a licensed ~1GB download plus a new pinned source.

THE CONSTRAINT: repoDB's only disease key is ind_id (a UMLS CUI). There is no MeSH,
OMIM or MedDRA column. So you cannot "join by MeSH" — every route must TERMINATE at a
CUI. A crosswalk expands OKG's CUI set; the join itself never changes.

This measures three things:
  1. CEILING  — of the 745 unmatched CUIs, how many have an ind_name that matches an
                OKG DIS name/synonym? That bounds what ANY crosswalk could recover.
                Name matching is NOT a join (too unreliable to ship) — it is a probe.
  2. PAYOFF   — for drug-touched DIS nodes with no CUI, which xref vocabularies do they
                actually carry? That says WHICH crosswalk to buy, if any.
  3. AMBIGUITY— CUIs hitting >1 OKG node, and OKG nodes hitting >1 CUI. A cascade needs
                a stated rule for both; "first match wins" is a decision, not a default.

Read-only. Writes data/manifest/a6_dis_headroom.json.

    python scripts/a6_dis_headroom.py --repodb data/sources/repodb_full.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
NODES = REPO / "data" / "canon" / "canon_nodes.parquet"
EDGES = REPO / "data" / "canon" / "canon_edges"
OUT = REPO / "data" / "manifest" / "a6_dis_headroom.json"

NORM = re.compile(r"[^a-z0-9]+")


def norm(s: str | None) -> str | None:
    """Lowercase, strip punctuation. Crude on purpose — this bounds a ceiling, it does
    not produce a mapping. 'Alzheimer's disease' and 'alzheimer disease' should collide;
    anything subtler than that is exactly why name matching must not become the join."""
    if not s:
        return None
    v = NORM.sub(" ", s.lower()).strip()
    return v or None


def pct(n: int, d: int) -> str:
    return f"{n:,}/{d:,} = {n/d:5.1%}" if d else f"{n:,}/0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repodb", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    if not NODES.exists():
        raise SystemExit(f"FATAL: {NODES} missing — run A2 first (or set RETRACERX_ROOT)")

    rdb = pl.read_csv(args.repodb, infer_schema_length=10_000, truncate_ragged_lines=True)
    if "ind_id" not in rdb.columns:
        raise SystemExit("FATAL: no ind_id column — repoDB's export format changed.")

    ind = (rdb.select(pl.col("ind_id").str.strip_chars(),
                      pl.col("ind_name").str.strip_chars())
           .unique("ind_id").drop_nulls("ind_id"))
    n_ind = ind.height

    dis = pl.scan_parquet(NODES).filter(pl.col("label") == "DIS")
    dd = EDGES / "DRG-DIS.parquet"
    touched = None
    if dd.exists():
        e = pl.scan_parquet(dd)
        touched = pl.concat([e.select(pl.col("from").alias("id")),
                             e.select(pl.col("to").alias("id"))]).unique()

    # ---- every CUI OKG can currently produce (native + UMLS xrefs)
    have = dis.select("id", "name", "umls_cui", "umls_cui_xrefs", "properties_json").collect()
    okg_cuis: set[str] = set(c for c in have["umls_cui"].to_list() if c)
    for lst in have["umls_cui_xrefs"].to_list():
        okg_cuis.update(c for c in (lst or []) if c)

    matched = ind.filter(pl.col("ind_id").is_in(list(okg_cuis)))
    unmatched = ind.filter(~pl.col("ind_id").is_in(list(okg_cuis)))
    print(f"repoDB unique CUIs        : {n_ind:,}")
    print(f"  reachable today         : {pct(matched.height, n_ind)}")
    print(f"  UNMATCHED               : {pct(unmatched.height, n_ind)}   <- the target")

    # ---- 1. CEILING: do the unmatched diseases exist in OKG under any name?
    names: dict[str, list[str]] = {}
    for row in have.iter_rows(named=True):
        for nm in (row["name"],):
            k = norm(nm)
            if k:
                names.setdefault(k, []).append(row["id"])
    # synonyms live in the blob; include them — they cost one pass and they are exactly
    # what a crosswalk would end up matching anyway
    syn_hits = 0
    for row in have.iter_rows(named=True):
        try:
            d = json.loads(row["properties_json"])
        except Exception:
            continue
        for field in ("exact_synonyms", "related_synonyms", "broad_synonyms", "narrow_synonyms"):
            for s in (d.get(field) or []):
                k = norm(s if isinstance(s, str) else None)
                if k:
                    names.setdefault(k, []).append(row["id"])
                    syn_hits += 1

    hit, miss = [], []
    for r in unmatched.iter_rows(named=True):
        k = norm(r["ind_name"])
        (hit if (k and k in names) else miss).append(r)

    print(f"\n[1] CEILING — unmatched CUIs whose ind_name matches an OKG DIS name/synonym:")
    print(f"    {pct(len(hit), unmatched.height)} of the unmatched"
          f"   ({len(names):,} normalised names/synonyms indexed, {syn_hits:,} from synonyms)")
    print(f"    -> a crosswalk could plausibly recover AT MOST ~{len(hit):,} more CUIs,")
    print(f"       taking coverage from {pct(matched.height, n_ind)} to at best "
          f"{pct(matched.height + len(hit), n_ind)}")
    print(f"    -> {len(miss):,} unmatched CUIs have NO name-alike in OKG at all: those")
    print(f"       diseases are absent from the graph and NO crosswalk reaches them.")
    for r in hit[:args.examples]:
        print(f"       recoverable  {r['ind_id']}  {r['ind_name'][:52]}")
    for r in miss[:4]:
        print(f"       absent       {r['ind_id']}  {r['ind_name'][:52]}")

    # ---- 2. PAYOFF: which crosswalk would actually pay?
    pop = dis if touched is None else dis.join(touched, on="id", how="semi")
    # umls_cui_xrefs is NULL (not []) when the node has no xrefs field at all, so
    # `list.len() == 0` is null and the filter drops exactly the nodes we are trying to
    # count. fill_null(0) — keyless is keyless whether the field is empty or absent.
    nocui_df = (pop.filter(pl.col("umls_cui").is_null() &
                           (pl.col("umls_cui_xrefs").list.len().fill_null(0) == 0))
                .select("id_prefix", "xref_vocabs").collect())
    counter: Counter = Counter()
    for lst in nocui_df["xref_vocabs"].to_list():
        counter.update(v for v in (lst or []) if v)
    print(f"\n[2] PAYOFF — drug-touched DIS nodes with NO CUI by any current route: "
          f"{nocui_df.height:,}")

    # Which ONTOLOGY are the keyless nodes in? This decides whether a MONDO/DOID ->
    # UMLS crosswalk could pay at all. A MONDO crosswalk cannot help an EFO or OBA node.
    # (OBA is not a disease ontology; those nodes should never have reached DIS.)
    prefix = Counter(p or "<none>" for p in nocui_df["id_prefix"].to_list())
    print(f"    id namespace of those keyless nodes (a MONDO crosswalk only reaches MONDO):")
    for p, n in prefix.most_common(10):
        print(f"      {p:<20} {n:,}")
    print(f"    vocabularies they DO carry (this is which crosswalk to buy):")
    for v, n in counter.most_common(12):
        print(f"      {v:<20} {n:,} nodes")
    if not counter:
        print("      none — these nodes carry no xrefs at all, so no crosswalk reaches them")
    print(f"    CAP: any crosswalk is bounded by repoDB's {unmatched.height:,} unmatched CUIs,")
    print(f"         NOT by these {nocui_df.height:,} nodes. Enriching a node repoDB never")
    print(f"         mentions adds nothing. And note the xrefs route already gave 406 nodes")
    print(f"         a CUI and produced ZERO extra repoDB matches — ontology-lineage CUIs")
    print(f"         and repoDB's DrugCentral/AACT-lineage CUIs barely intersect.")

    # ---- 3. AMBIGUITY: a cascade needs a rule, in both directions
    cui2node: dict[str, list[str]] = {}
    for row in have.iter_rows(named=True):
        cs = set()
        if row["umls_cui"]:
            cs.add(row["umls_cui"])
        cs.update(c for c in (row["umls_cui_xrefs"] or []) if c)
        for c in cs:
            cui2node.setdefault(c, []).append(row["id"])
    multi_node = {c: v for c, v in cui2node.items() if len(v) > 1}
    n_multi_cui = sum(1 for row in have.iter_rows(named=True)
                      if len({*( [row["umls_cui"]] if row["umls_cui"] else [] ),
                              *(row["umls_cui_xrefs"] or [])}) > 1)
    print(f"\n[3] AMBIGUITY — a cascade must state a rule for both of these:")
    print(f"    CUIs mapping to >1 OKG DIS node : {len(multi_node):,}"
          f"   (join duplicates gold pairs; corrupts holdout counts)")
    print(f"    OKG DIS nodes with >1 CUI       : {n_multi_cui:,}")
    for c, v in list(multi_node.items())[:3]:
        print(f"      {c} -> {v[:3]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "repodb_cuis": n_ind, "reachable_today": matched.height,
        "unmatched": unmatched.height,
        "ceiling_name_alike": len(hit), "absent_from_graph": len(miss),
        "best_case_coverage": round((matched.height + len(hit)) / n_ind, 4),
        "nocui_drug_touched": nocui_df.height,
        "nocui_id_prefixes": dict(prefix.most_common()),
        "vocabs_on_nocui_nodes": dict(counter.most_common()),
        "cuis_to_many_nodes": len(multi_node), "nodes_with_many_cuis": n_multi_cui,
        "note": "repoDB's only disease key is ind_id (UMLS CUI). MeSH/OMIM/MedDRA cannot "
                "be joined directly — a crosswalk expands OKG's CUI set, then the join "
                "runs on CUI as before. Name matching here bounds a ceiling; it is not "
                "a mapping and must not become one.",
    }, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
