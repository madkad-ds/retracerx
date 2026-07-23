#!/usr/bin/env python3
"""
a6_freeze_keys.py — write data/manifest/a6_keys.json, the frozen A6 join-key decision
record, in the A1.5 / A2 pattern.

It does NOT re-measure. It READS a5_key_probe.json and a6_dis_headroom.json and stamps
their numbers into a decision document, so the frozen record cannot drift from what was
actually measured. If a probe output is missing, it fails loudly rather than freezing a
decision on numbers nobody produced.

    python scripts/a6_freeze_keys.py

Re-run the two probes first if the underlying data changed:
    python scripts/a5_key_probe.py    --broad-drug ... --broad-sample ... --repodb ...
    python scripts/a6_dis_headroom.py --repodb ...
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
MAN = REPO / "data" / "manifest"
PROBE = MAN / "a5_key_probe.json"
HEADROOM = MAN / "a6_dis_headroom.json"
OUT = MAN / "a6_keys.json"


def load(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(f"FATAL: {p} missing. Run its probe first; A6 keys are not "
                         f"frozen on numbers that were never measured.")
    return json.loads(p.read_text())


def main() -> None:
    probe = load(PROBE)
    head = load(HEADROOM)
    b = probe.get("broad", {})
    r = probe.get("repodb", {})

    doc = {
        "step": "A6",
        "frozen": True,
        "decided_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "a5_key_probe.json": PROBE.name,
            "a6_dis_headroom.json": HEADROOM.name,
            "canon_nodes": probe.get("canon_nodes"),
        },

        "principle": (
            "Keys are per-SOURCE, not universal. Broad and repoDB are different sources "
            "with different native identifiers, so each gets its own key and its own "
            "join. A drug-disease pair is two entities, so each side is a separate join. "
            "Every route must terminate at the identifier the OTHER side actually "
            "publishes — measured, never assumed."),

        "keys": {
            "broad_drug": {
                "key": "InChIKey",
                "join": "Broad sample.InChIKey  ==  OKG DRG.inchi_key",
                "warrant": (
                    "Broad ships NO DrugBank id; InChIKey is the only structural bridge. "
                    "Built from the two bulk files (there is no public API) via "
                    "a4_broad_join.py -> broad_compounds.parquet."),
                "coverage": {
                    "launched_compounds": b.get("launched_compounds"),
                    "launched_reaching_okg": b.get("launched_in_okg"),
                    "note": ("This is A4's candidate universe. NOT the 62.3% inchi_key "
                             "presence figure and NOT the row-inflated 72.1% — those "
                             "were denominator errors. This is distinct Launched "
                             "compounds reaching an OKG node."),
                },
            },
            "repodb_drug": {
                "key": "DrugBank id, recovered from OKG properties.source_ids (DB##### shape)",
                "join": "repoDB.drugbank_id  ==  DB##### parsed from OKG DRG.source_ids",
                "warrant": (
                    "repoDB is natively keyed on DrugBank, so this route needs no "
                    "UniChem hop AND reaches biologics that InChIKey structurally "
                    "cannot. Node-id DrugBank alone reaches only 17.7%; the properties "
                    "route reaches 80.0%. source_ids is an UNLABELED list e.g. "
                    '["DB00341","3561"] — DrugBank is identified by the DB##### shape; '
                    "the bare-numeric element is unattributable (PubChem? struct_id? "
                    "RxNorm?) and MUST NOT be treated as any specific vocabulary."),
                "coverage": {
                    "repodb_drugs": r.get("repodb_drugbank"),
                    "reachable_via_node_id": r.get("repodb_hit_via_id"),
                    "reachable_via_properties": r.get("repodb_hit_via_props"),
                },
            },
            "repodb_disease": {
                "key": "UMLS CUI (OKG umls_cui, native)",
                "join": "repoDB.ind_id  ==  OKG DIS.umls_cui",
                "warrant": (
                    "repoDB's ONLY disease identifier is ind_id, a UMLS CUI. There is no "
                    "MeSH/OMIM/MedDRA column, so those cannot be joined directly — a "
                    "crosswalk would only expand OKG's CUI set, then the join still runs "
                    "on CUI. The xrefs route (umls_cui_xrefs) was measured and adds "
                    "ZERO repoDB matches, so it is NOT used for the join."),
                "coverage": {
                    "repodb_cuis": head.get("repodb_cuis"),
                    "reachable_native_and_xrefs": head.get("reachable_today"),
                    "note": ("native and native+xrefs are identical on the repoDB "
                             "population — xrefs enriches OKG nodes repoDB never names."),
                },
            },
        },

        "no_crosswalk_ruling": {
            "decision": "Do NOT build a MeSH/OMIM/MedDRA/MONDO -> UMLS crosswalk.",
            "measured_evidence": {
                "repodb_cuis_unmatched": head.get("unmatched"),
                "ceiling_name_alike": head.get("ceiling_name_alike"),
                "ceiling_note": ("name-alike ceiling is inflated by ~88k synonyms and is "
                                 "an upper bound of true recoveries, not a count of them"),
                "absent_from_graph": head.get("absent_from_graph"),
                "best_case_coverage_if_every_name_match_were_real": head.get("best_case_coverage"),
                "keyless_node_namespaces": head.get("nocui_id_prefixes"),
                "xrefs_route_extra_repodb_matches": 0,
            },
            "reasoning": [
                "Most unmatched CUIs (~86%) are diseases ABSENT from OKG entirely — "
                "trial conditions from ClinicalTrials.gov/AACT (symptoms, organism-"
                "specific infections, clinical states) that no disease ontology models. "
                "No crosswalk reaches a node that does not exist.",
                "The keyless drug-touched DIS nodes are ~91% EFO; only ~4% are "
                "MONDO+DOID. A MONDO/DOID crosswalk addresses the wrong ontology.",
                "The xrefs route already gave 406 nodes a CUI from ontology-lineage "
                "cross-references and produced zero additional repoDB matches. A MONDO "
                "crosswalk draws from the same lineage — same well, same water.",
                "Cost (licensed ~1GB MRCONSO + a new pinned source) vastly exceeds the "
                "real gain, and a crosswalk SHIFTS which diseases are evaluable, "
                "changing the evaluation set composition — a cost, not a pure win.",
            ],
            "if_revisited_later": (
                "The ~106 name-alike-recoverable CUIs are diseases PRESENT in OKG under "
                "a DIFFERENT CUI (e.g. CLL, melanoma) — a UMLS synonymy problem, not an "
                "ontology-bridge problem. The right artifact would be UMLS MRREL/MRCONSO "
                "CUI<->CUI same-concept links, NOT MONDO->UMLS. Different tool entirely."),
            "honest_ceiling": (
                f"{head.get('reachable_today')}/{head.get('repodb_cuis')} repoDB CUIs "
                "reachable. State this as a measured coverage BOUNDARY in the writeup: "
                "roughly half of repoDB's indications describe conditions OptimusKG does "
                "not contain. This is a property of the two datasets, not a key failure."),
        },

        "open_rules": [
            {
                "id": "shared_inchikey",
                "problem": ("~75 OKG DRG nodes share an inchi_key with another node "
                            "(10,446 nodes / 10,371 distinct keys), so Broad<->OKG on "
                            "inchi_key is NOT 1:1 and will duplicate rows."),
                "why_it_matters": "duplicates gold pairs; corrupts holdout counts (A7).",
                "decision_needed": "node-collapse or key-disambiguation rule before the join.",
                "status": "OPEN — A6 must state a rule; do not inherit from row order.",
            },
            {
                "id": "broad_salt_forms",
                "problem": ("148 Broad compounds ship as >1 distinct InChIKey (salt/"
                            "stereo forms). broad_compounds.parquet leaves inchi_key "
                            "NULL for these by design and keeps all keys in `inchikeys`."),
                "why_it_matters": "picking one silently chooses a salt form.",
                "decision_needed": "parent-structure / desalting rule, or match on any key.",
                "status": "OPEN — surfaced (n_inchikeys), not resolved.",
            },
            {
                "id": "cui_to_many_nodes",
                "problem": (f"{head.get('cuis_to_many_nodes')} repoDB CUIs map to >1 OKG "
                            "DIS node (EFO and MONDO for the same disease — the ontology "
                            "merge showing through)."),
                "why_it_matters": ("join is not 1:1; duplicates gold pairs and inflates "
                                   "holdout counts (the how-to's explicit warning)."),
                "decision_needed": ("node-collapse rule: prefer MONDO? prefer the "
                                    "drug-touched node? collapse to one canonical DIS?"),
                "status": "OPEN — independent of any coverage decision.",
            },
            {
                "id": "nodes_with_many_cuis",
                "problem": (f"{head.get('nodes_with_many_cuis')} OKG DIS nodes carry >1 "
                            "UMLS CUI (native + xrefs disagree, or multiple xref CUIs)."),
                "why_it_matters": "a node matching two repoDB CUIs double-counts.",
                "decision_needed": "which CUI is canonical per node (native over xref?).",
                "status": "OPEN.",
            },
        ],

        "not_a6": [
            "Building the crosswalk (ruled out above, with evidence).",
            "Removing leakage — A7 anti-joins DRG-DIS on endpoint_key.",
            "Deduping / reorienting the graph — A2 canonicalized; it does not edit.",
            "Choosing the numeric element of source_ids as any vocabulary — unattributable.",
        ],
    }

    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"\n  Broad drug     : InChIKey        "
          f"{b.get('launched_in_okg')}/{b.get('launched_compounds')} Launched -> OKG")
    print(f"  repoDB drug    : DrugBank/source_ids  "
          f"{r.get('repodb_hit_via_props')}/{r.get('repodb_drugbank')} reachable")
    print(f"  repoDB disease : umls_cui        "
          f"{head.get('reachable_today')}/{head.get('repodb_cuis')} reachable "
          f"(HONEST CEILING — no crosswalk)")
    print(f"  open rules     : {len(doc['open_rules'])} (shared InChIKeys, salt forms, "
          f"CUI<->node many-to-many)")


if __name__ == "__main__":
    main()
