#!/usr/bin/env python3
"""
A3 — verification gate. Exits non-zero on any failure.

  gate    a check that exits non-zero, so a broken import cannot be mistaken for a good
          one. A1/A2 both end in a gate; A3 previously only PRINTED counts for a human
          to eyeball, which is the same class of hazard as a silent no-op.
  probe   an active test rather than a passive read — here, attempting a duplicate insert
          inside a transaction that is always rolled back, to prove the uniqueness
          constraint actually enforces rather than merely existing.

Every expectation is read from data/manifest/a3_manifest.json (written by the emitter),
never hardcoded. Usage:

  python scripts/a3_verify.py --variant full
  python scripts/a3_verify.py --variant filtered
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, Neo4jError

from a3_common import composite_type, info, load_json, normalize_pair_counts, pick, warn

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          file=sys.stdout if ok else sys.stderr)
    return ok


def one(session, cypher: str, **params):
    rec = session.run(cypher, **params).single()
    return None if rec is None else rec[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["full", "filtered"])
    ap.add_argument("--manifest-dir", default="data/manifest")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="retracerx")
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    man = load_json(mdir / "a3_manifest.json", "a3_manifest.json")
    census = load_json(mdir / "okg_census.json", "okg_census.json")

    if args.variant not in man.get("variants", {}):
        print(f"[A3][FATAL] variant '{args.variant}' absent from a3_manifest.json; "
              f"present: {sorted(man.get('variants', {}))}", file=sys.stderr)
        return 1
    v = man["variants"][args.variant]

    exp_nodes = int(v["expected_nodes"])
    exp_edges = int(v["expected_edges"])
    exp_types = int(v["expected_distinct_types"])
    census_types = int(man["distinct_composite_types_in_census"])

    leak_counts = normalize_pair_counts(
        pick(census, ["leakage_surface", "a7_leakage_surface", "leakage", "leakage_surface.pairs"],
             "leakage surface", "okg_census.json"),
        "okg_census.leakage_surface")
    leak_types = sorted(composite_type(l, r) for l, r in leak_counts)

    drv = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with drv.session() as s:
            # --- G-A3-1/2: counts ------------------------------------------------
            # _Meta is excluded: it is a marker, not graph data.
            n = one(s, "MATCH (n) WHERE NOT n:_Meta RETURN count(n)")
            check("G-A3-1 node count", n == exp_nodes, f"{n:,} vs expected {exp_nodes:,}")

            e = one(s, "MATCH ()-[r]->() RETURN count(r)")
            check("G-A3-2 relationship count", e == exp_edges, f"{e:,} vs expected {exp_edges:,}")

            # --- G-A3-3: composite types -----------------------------------------
            t = one(s, "MATCH ()-[r]->() RETURN count(DISTINCT type(r))")
            check("G-A3-3 distinct composite types", t == exp_types,
                  f"{t} vs expected {exp_types} (census has {census_types})")

            bad = one(s, "CALL db.relationshipTypes() YIELD relationshipType AS t "
                         "WITH t WHERE NOT t =~ '^[A-Za-z_][A-Za-z0-9_]*$' RETURN count(t)")
            check("G-A3-4 no backtick-forcing type names", bad == 0, f"{bad} offending names")

            # --- G-A3-5: the constraint is REAL, not a silent no-op ---------------
            cons = s.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
                         "RETURN name, labelsOrTypes, properties").data()
            targeted = any("Node" in (c.get("labelsOrTypes") or [])
                           and "id" in (c.get("properties") or []) for c in cons)
            check("G-A3-5a constraint targets (:Node).id", targeted, f"{len(cons)} constraint(s) found")

            probe_id = one(s, "MATCH (n:Node) RETURN n.id LIMIT 1")
            enforced = False
            if probe_id is not None:
                tx = s.begin_transaction()
                try:
                    tx.run("CREATE (n:Node {id:$i})", i=probe_id).consume()
                except (ClientError, Neo4jError):
                    enforced = True       # rejected == the constraint bites
                finally:
                    tx.rollback()          # always; the probe never persists
            check("G-A3-5b duplicate-insert probe rejected", enforced,
                  "constraint enforces" if enforced else
                  "duplicate ACCEPTED — the constraint is a no-op")

            # --- G-A3-6/7: leakage absence (filtered only) ------------------------
            if args.variant == "filtered":
                nonzero = []
                for tname in leak_types:
                    c = one(s, f"MATCH ()-[r:{tname}]-() RETURN count(r)")
                    if c:
                        nonzero.append(f"{tname}={c:,}")
                check("G-A3-6 all 7 leakage types empty", not nonzero,
                      "; ".join(nonzero) if nonzero else f"{len(leak_types)} types, all zero")

                drg_dis = one(s, "CALL db.relationshipTypes() YIELD relationshipType AS t "
                                 "WITH t WHERE t STARTS WITH 'DRG_DIS__' RETURN count(t)")
                check("G-A3-7 no DRG-DIS types survive", drg_dis == 0,
                      f"{drg_dis} DRG_DIS__* type(s) present (errata E3: the label is "
                      f"entirely leakage)")
            else:
                info("G-A3-6/7 skipped (full variant retains the leakage surface by design)")

            # --- G-A3-8: undirected traversal ------------------------------------
            # 13 labels are undirected and stored ONE-WAY only. A directed pattern from the
            # wrong end silently returns nothing. This proves -[r]- reaches what -[r]-> misses.
            row = s.run(
                "MATCH (a)-[r]->(b) WHERE r.undirected = true "
                "WITH a, b, r LIMIT 1 "
                "OPTIONAL MATCH (b)-[r2]->(a) WHERE r2.undirected = true "
                "MATCH (b)-[r3]-(a) WHERE r3.undirected = true "
                "RETURN count(r2) AS reverse_directed, count(r3) AS undirected_pattern"
            ).single()
            if row is None:
                check("G-A3-8 undirected traversal probe", False, "no undirected edge found")
            else:
                ok = row["reverse_directed"] == 0 and row["undirected_pattern"] > 0
                check("G-A3-8 undirected edges need -[r]-", ok,
                      f"reverse -[r]-> matched {row['reverse_directed']}, "
                      f"-[r]- matched {row['undirected_pattern']}")

            # --- G-A3-9: variant marker agrees -----------------------------------
            m = s.run("MATCH (m:_Meta {singleton:true}) RETURN m.graph_variant AS variant, "
                      "m.expected_edges AS edges, m.expected_nodes AS nodes, "
                      "m.edges_csv_sha256 AS sha, m.dataset_version AS dsv").single()
            if m is None:
                check("G-A3-9 variant marker present", False,
                      "no (:_Meta) node — which variant is loaded is unknowable")
            else:
                agree = (m["variant"] == args.variant and m["edges"] == e
                         and m["nodes"] == n and m["sha"] == v["edges_csv_sha256"])
                check("G-A3-9 marker agrees with measurement", agree,
                      f"marker: variant={m['variant']} edges={m['edges']:,} "
                      f"nodes={m['nodes']:,} dsv={m['dsv']}")

            # --- M-A3-1: measurement, not a gate ---------------------------------
            deg = one(s, "MATCH (d:DRG) WHERE NOT (d)-[]-() RETURN count(d)")
            info(f"M-A3-1 drugs with zero edges in this variant: {deg:,}")
            info("      Distinct from A1's 982 edgeless drugs and A4's 984 unscoreable "
                 "compounds — coincident integers, different populations.")
    finally:
        drv.close()

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n[A3] {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
    if failed:
        print("[A3][FATAL] failing checks: " + ", ".join(r[0] for r in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
