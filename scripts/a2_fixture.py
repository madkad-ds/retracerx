"""Build a miniature but structurally faithful OptimusKG 2.0 snapshot + A1 artifacts."""
import json, hashlib, random, sys
from pathlib import Path
import polars as pl

ROOT = Path(sys.argv[1]).resolve()
SNAP = ROOT / "data" / "okg_cache"
MAN = ROOT / "data" / "manifest"
(SNAP / "nodes").mkdir(parents=True, exist_ok=True)
(SNAP / "edges").mkdir(parents=True, exist_ok=True)
(SNAP / "lcc").mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)
random.seed(7)

LABELS = ["GEN", "DIS", "DRG", "BPO", "PHE", "ANA", "MFN", "CCO", "PWY", "EXP"]
STEM = {"GEN": "gene", "DIS": "disease", "DRG": "drug", "BPO": "biological_process",
        "PHE": "phenotype", "ANA": "anatomy", "MFN": "molecular_function",
        "CCO": "cellular_component", "PWY": "pathway", "EXP": "exposure"}

# ---- nodes -----------------------------------------------------------------
nodes = []
def add(label, i, props):
    if label == "GEN":
        nid = f"ENSG{i:011d}"            # no separator at all
    elif label == "DIS":
        nid = [f"MONDO:{i:07d}", f"DOID_{i:07d}", f"meddra:{i}"][i % 3]
    elif label == "DRG":
        nid = [f"CHEMBL:{i}", f"DrugBank:DB{i:05d}"][i % 2]
    else:
        nid = f"{label}:{i:05d}"
    nodes.append({"id": nid, "label": label, "properties": props})

for i in range(6):
    # GEN's xrefs are a DIFFERENT SHAPE from DIS's — List(Struct{id,source}), not
    # List(String). This is real OptimusKG 2.0 behaviour and it broke a build that
    # assumed one shape dataset-wide. The fixture carries both forms deliberately.
    add("GEN", i, {"name": f"gene{i}", "symbol": f"SYM{i}",
                   "xrefs": None if i % 3 == 0 else
                   [{"id": f"{51600 + i}", "source": "HGNC"},
                    {"id": f"C{i:07d}", "source": "UMLS"}]})
for i in range(6):
    # REGRESSION (real snapshot): the LEADING DIS rows carry null xrefs — the OBA /
    # attribute block sorts first — so a 2,000-row sample window over 36,345 DIS rows
    # saw nothing but nulls and the shape came back unprovable. Node inference must
    # read the whole label. First 3 DIS nodes here have no xrefs at all.
    # Real DIS xrefs mix CURIEs with URLs. Splitting a URL on ":" reports the scheme
    # ("https") as a vocabulary — the artifact this case exists to catch.
    _xr = {
        0: None,
        1: ["MEDGEN:267607", "NCIT:C35772", "UMLS:C1510471",
            "http://purl.obolibrary.org/obo/DOID_0050890"],
        2: [],
        3: ["UMLS:C0221023", "UMLS:C9999999", "MESH:D001361",
            "https://omim.org/entry/123456"],
    }
    add("DIS", i, {"name": f"dis{i}", "umls_cui": f"C{i:07d}" if i % 4 else None,
                   "xrefs": None if i < 3 else _xr[i % 4]})
for i in range(6):
    add("DRG", i, {"name": f"drug{i}", "inchi_key": f"KEY{i:022d}" if i % 5 else None,
                   "canonical_smiles": "CCO"})
for lb in LABELS:
    if lb in ("GEN", "DIS", "DRG"):
        continue
    for i in range(3):
        add(lb, i, {"name": f"{lb.lower()}{i}"})

ids_by_label = {}
for n in nodes:
    ids_by_label.setdefault(n["label"], []).append(n["id"])

# stratified node files: properties as a native Struct (per label schema)
for lb in LABELS:
    rows = [n for n in nodes if n["label"] == lb]
    pl.DataFrame({
        "id": [r["id"] for r in rows],
        "label": [lb] * len(rows),
        "properties": [r["properties"] for r in rows],
    }).write_parquet(SNAP / "nodes" / f"{STEM[lb]}.parquet")

# flat node table: properties as a JSON string
pl.DataFrame({
    "id": [n["id"] for n in nodes],
    "label": [n["label"] for n in nodes],
    "properties": [json.dumps(n["properties"]) for n in nodes],
}).write_parquet(SNAP / "nodes.parquet")

# ---- edges -----------------------------------------------------------------
PAIRS = [("DRG", "DIS"), ("DRG", "PHE"), ("DRG", "BPO"), ("DRG", "GEN"), ("DIS", "GEN"),
         ("GEN", "GEN"), ("PWY", "GEN"), ("ANA", "GEN"), ("MFN", "GEN"), ("CCO", "GEN"),
         ("BPO", "GEN"), ("EXP", "GEN"), ("DIS", "PHE"), ("DIS", "DIS"), ("DRG", "DRG"),
         ("PHE", "PHE"), ("BPO", "BPO"), ("ANA", "ANA"), ("MFN", "MFN"), ("CCO", "CCO"),
         ("PWY", "PWY"), ("EXP", "DIS"), ("EXP", "BPO"), ("EXP", "ANA"), ("EXP", "MFN"),
         ("EXP", "CCO"), ("EXP", "PHE")]
assert len(PAIRS) == 27
UNDIRECTED = {f"{a}-{b}" for a, b in PAIRS[:12]}   # exactly 12 undirected labels

LEAK = {("DRG-DIS", "INDICATION"): 5, ("DRG-DIS", "CONTRAINDICATION"): 4,
        ("DRG-DIS", "OFF_LABEL_USE"): 3, ("DRG-PHE", "CONTRAINDICATION"): 2,
        ("DRG-PHE", "INDICATION"): 2, ("DRG-PHE", "OFF_LABEL_USE"): 1,
        ("DRG-BPO", "INDICATION"): 1}
SPOT = {("DIS-GEN", "ASSOCIATED_WITH"): 9, ("GEN-GEN", "INTERACTS_WITH"): 7,
        ("PWY-GEN", "INTERACTS_WITH"): 4}

def mk(a, b, rel, k, label):
    out = []
    for i in range(k):
        f = ids_by_label[a][i % len(ids_by_label[a])]
        t = ids_by_label[b][(i + 1) % len(ids_by_label[b])]
        if label in UNDIRECTED and i % 2:      # store some the "wrong" way round
            f, t = t, f
        # REGRESSION (real snapshot, DIS-GEN): the leading rows of a label can carry
        # empty sources lists, so a sampled infer_schema_length returns List(Null) --
        # and feeding that back in as a *decode* dtype explodes on the first real
        # string further down the file. Empty early, populated late, same label.
        early = label == "DIS-GEN" and i < max(1, k - 1)
        src = ({"direct": [], "indirect": []} if early else
               {"direct": ["DrugCentral"], "indirect": ["DrugBank", "CGI"]})
        out.append({"from": f, "to": t, "label": label, "relation": rel,
                    "undirected": label in UNDIRECTED,
                    "properties": {"sources": src, "structure_id": i}})
    return out

edges = []
for (label, rel), k in {**LEAK, **SPOT}.items():
    a, b = label.split("-")
    edges += mk(a, b, rel, k, label)
covered = {k[0] for k in {**LEAK, **SPOT}}
for a, b in PAIRS:
    label = f"{a}-{b}"
    if label in covered:
        continue
    rels = ["TARGET", "ENZYME"] if label == "DRG-GEN" else ["ASSOCIATED_WITH"]
    for rel in rels:
        edges += mk(a, b, rel, 2, label)

nrel = len({e["relation"] for e in edges})
print(f"fixture: {len(nodes)} nodes / {len(edges)} edges / "
      f"{len({e['label'] for e in edges})} labels / {nrel} relations")

def edge_frame(rows, as_json):
    return pl.DataFrame({
        "from": [r["from"] for r in rows], "to": [r["to"] for r in rows],
        "label": [r["label"] for r in rows], "relation": [r["relation"] for r in rows],
        "undirected": [r["undirected"] for r in rows],
        "properties": [json.dumps(r["properties"]) if as_json else r["properties"]
                       for r in rows],
    })

edge_frame(edges, True).write_parquet(SNAP / "edges.parquet")
STRAT_EDGE = {"DRG-DIS": "drug_disease", "DRG-GEN": "drug_gene", "DIS-GEN": "disease_gene",
              "DRG-PHE": "drug_phenotype", "DRG-BPO": "drug_biological_process"}
for label, stem in STRAT_EDGE.items():
    edge_frame([e for e in edges if e["label"] == label], False).write_parquet(
        SNAP / "edges" / f"{stem}.parquet")

# an LCC variant file that must NOT be picked up (A1.5 froze `full`)
edge_frame(edges[:3], False).write_parquet(SNAP / "lcc" / "drug_disease.parquet")

# ---- A1 artifacts ----------------------------------------------------------
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

full_files = (["nodes.parquet", "edges.parquet"]
              + [f"nodes/{STEM[l]}.parquet" for l in LABELS]
              + [f"edges/{s}.parquet" for s in STRAT_EDGE.values()])
all_files = full_files + ["lcc/drug_disease.parquet"]

# Shapes below mirror the real A1 artifacts exactly: files/test1/test2 are dicts
# keyed by relative path, a dispute is the *presence* of a checksum_dispute block,
# and the audit records evidence (identical_to_v1 / keys_match) rather than a verdict.
disputed = {"nodes/gene.parquet", "edges/drug_gene.parquet"}

json.dump({
    "step": "A1", "doi": "doi:10.7910/DVN/IYNGEV", "dataset_version": "2.0",
    "version_state": "RELEASED", "release_time": "2026-05-06T00:00:00Z",
    "files": {
        n: {
            "dataverse_file_id": 1000 + i,
            "filesize_bytes": (SNAP / n).stat().st_size,
            "remote_checksum_type": "MD5",
            "remote_checksum_value": "deadbeef" if n in disputed else "cafef00d",
            "local_path": f"data/okg_cache/{n}",
            "sha256": sha(SNAP / n),
            **({"checksum_dispute": {"reason": "remote_checksum_disagrees",
                                     "algorithm": "MD5", "remote": "deadbeef",
                                     "local": "62bbd145216c7061b210fc8702317f53"}}
               if n in disputed else {}),
        }
        for i, n in enumerate(all_files)
    },
}, (MAN / "okg_manifest.json").open("w"), indent=2)

json.dump({
    "dataset_version": "2.0",
    "n_disputed": len(disputed),
    "test1": {n: {"local_md5": "62bb", "v1_md5": "62bb",
                  "v2_metadata_md5": "deadbeef", "identical_to_v1": True}
              for n in sorted(disputed)},
    "test2": {n: {"flat_rows": 1, "strat_rows": 1, "keys_match": True}
              for n in sorted(disputed)},
}, (MAN / "okg_dispute_audit.json").open("w"), indent=2)

# A1 froze A2's scope to the two flat tables, exactly as the live repo does.
json.dump({"step": "A1", "doi": "doi:10.7910/DVN/IYNGEV", "dataset_version": "2.0",
           "graph_variant": "full", "source_tables": ["nodes.parquet", "edges.parquet"],
           "rationale": "LCC costs 982 DRG (5.86%)", "frozen": True},
          (MAN / "a1_decisions.json").open("w"), indent=2)

json.dump({"dataset_version": "2.0",
           "nodes": {"total": len(nodes)},
           "edges": {"total": len(edges)},
           # The real census records directionality per label as an ARRAY — the
           # measurement G5 checks against. A1's prose handover miscounted it, so the
           # fixture must carry the array or it can't exercise that path at all.
           "directionality": [{"label": lb, "undirected": lb in UNDIRECTED}
                              for lb in sorted({e["label"] for e in edges})],
           "drift_vs_reference": {"nodes": 0, "edges": 0}},
          (MAN / "okg_census.json").open("w"), indent=2)

print(json.dumps({"nodes_total": len(nodes), "edges_total": len(edges),
                  "relations": nrel,
                  "leak_total": sum(LEAK.values()),
                  "leak": {f"{k[0]}|{k[1]}": v for k, v in LEAK.items()},
                  "spot": {f"{k[0]}|{k[1]}": v for k, v in SPOT.items()}}))
