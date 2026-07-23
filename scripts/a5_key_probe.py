#!/usr/bin/env python3
"""
a5_key_probe.py — find every candidate join key, and measure each one SEPARATELY
against Broad and repoDB. Does not join. Does not decide. Measures.

Why this exists: `inchi_key` covers 62.3% of DRG and `umls_cui` covers 2.95% of DIS —
both numbers looked alarming until the denominator was corrected (M5: 43.5% -> 60.0%
on the population that matters). The same trap applies here. Before A6 designs around
a key, measure every candidate on the population that actually needs one.

VERIFIED (2026-07, from source, not from the papers):
  repoDB full.csv columns:  drug_name, drugbank_id, ind_name, ind_id, NCT, status,
                            phase, DetailedStatus
    * ind_id IS the UMLS CUI.
    * drugbank_id is the ONLY drug id — there is NO DrugCentral struct_id in the
      export, despite repoDB being built from DrugCentral.
    * status has exactly 4 levels: Approved / Suspended / Terminated / Withdrawn.
      `status == "Terminated"` alone silently drops Suspended + Withdrawn negatives.
    * rows are (drug, indication, TRIAL) — NOT unique pairs. Dedupe before counting.
    * sem_type exists upstream but is NOT exported.
  Broad drug file (VERIFIED 8/18/2025, 7,540 rows): pert_iname, clinical_phase, moa,
                            target, disease_area, indication
  Broad sample file (VERIFIED 8/18/2025, 22,612 rows): broad_id, pert_iname,
                            qc_incompatible, purity, vendor, catalog_no, vendor_name,
                            expected_mass, smiles, InChIKey, pubchem_cid,
                            deprecated_broad_id
    * '!' header lines are TSV rows padded with tabs to the column count — strip()
      anything parsed out of them or values compare unequal while printing identically.
    * !File_date is the version. The FILENAME IS NOT: the link labelled 8/19/2025 serves
      repo-sample-annotation-20240610.txt whose header says 8/18/2025. Broad refreshes
      content behind stable URLs. (The DOI pins nothing (A1); the filename pins nothing.)
    * 22,612 samples vs 7,540 drugs -> the sample file is PHYSICAL-SAMPLE level. One
      compound can have several samples AND several distinct InChIKeys (salt forms).
    * Broad ships NO DrugBank id. It DOES ship pubchem_cid, so InChIKey is not
      necessarily the only bridge — measured below.

Read-only. Writes data/manifest/a5_key_probe.json. Run from the repo root after A2:

    python scripts/a5_key_probe.py \
        --broad-drug   data/sources/broad_drug_annotation.txt \
        --broad-sample data/sources/broad_samples.txt \
        --repodb       data/sources/repodb_full.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
NODES = REPO / "data" / "canon" / "canon_nodes.parquet"
EDGES = REPO / "data" / "canon" / "canon_edges"
OUT = REPO / "data" / "manifest" / "a5_key_probe.json"

# Identifier shapes, so a candidate field is classified by what it CONTAINS rather
# than by what it is named. A field called `accession_numbers` may hold DrugBank ids;
# a field called `code` may hold a URL. Names lie; values don't.
SHAPES = {
    "drugbank": re.compile(r"^DB\d{5}$"),
    "chembl": re.compile(r"^CHEMBL\d+$"),
    "inchikey": re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"),
    "umls_cui": re.compile(r"^C\d{7}$"),
    "unii": re.compile(r"^[0-9A-Z]{10}$"),
    "cas": re.compile(r"^\d{2,7}-\d{2}-\d$"),
    # NOT discriminative — a bare integer is a PubChem CID, a DrugCentral struct_id, a
    # cd_id, a year, or a molecular weight. Reported as "numeric", never as a vocabulary.
    # (Calling this "pubchem" produced `struct_id pubchem×754`, which is simply false.)
    "numeric": re.compile(r"^\d{2,9}$"),
}
# Shapes that actually identify a vocabulary. `numeric` is a shape, not an identity.
DISCRIMINATIVE = {"drugbank", "chembl", "inchikey", "umls_cui", "unii", "cas"}


def classify(val: str) -> str | None:
    for name, rx in SHAPES.items():
        if rx.match(val):
            return name
    return None


def pct(n: int, d: int) -> str:
    return f"{n:,}/{d:,} = {n/d:6.1%}" if d else f"{n:,}/0"


def head(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


# ------------------------------------------------------------------ OKG side


def okg_candidates(label: str, sample_n: int = 4000) -> dict:
    """Every properties field, classified by the id shapes its VALUES match."""
    lf = pl.scan_parquet(NODES).filter(pl.col("label") == label)
    total = lf.select(pl.len()).collect().item()
    blobs = lf.select("properties_json").head(sample_n).collect()["properties_json"]

    hits: dict[str, dict[str, int]] = {}
    examples: dict[str, str] = {}
    seen = 0
    for b in blobs:
        try:
            d = json.loads(b)
        except Exception:
            continue
        seen += 1
        for k, v in d.items():
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, dict):          # e.g. GEN xrefs {"id","source"}
                    x = x.get("id")
                if not isinstance(x, str):
                    continue
                shape = classify(x.strip())
                if shape:
                    hits.setdefault(k, {}).setdefault(shape, 0)
                    hits[k][shape] += 1
                    if shape in DISCRIMINATIVE and k not in examples:
                        # keep the RAW field value, so a multi-vocabulary container
                        # (source_ids) is legible rather than just tallied
                        examples[k] = json.dumps(v)[:160]
    return {"label": label, "nodes": total, "sampled": seen,
            "fields": hits, "examples": examples}


def report_candidates(c: dict) -> None:
    print(f"\n{c['label']}: {c['nodes']:,} nodes ({c['sampled']:,} blobs sampled)")
    if not c["fields"]:
        print("  no fields carry values matching a known identifier shape")
        return
    def score(kv):
        return -sum(v for k, v in kv[1].items() if k in DISCRIMINATIVE)
    for field, shapes in sorted(c["fields"].items(), key=score):
        real = {k: v for k, v in shapes.items() if k in DISCRIMINATIVE}
        noise = {k: v for k, v in shapes.items() if k not in DISCRIMINATIVE}
        s = ", ".join(f"{k}×{v:,}" for k, v in sorted(real.items(), key=lambda x: -x[1]))
        n = ", ".join(f"{k}×{v:,}" for k, v in sorted(noise.items(), key=lambda x: -x[1]))
        line = f"  {field:<34} {s}"
        if n:
            line += f"   [{n} — shape only, no vocabulary implied]" if not s else f"   [{n}]"
        print(line)
        ex = c["examples"].get(field)
        if ex and real:
            print(f"  {'':<34} e.g. {ex}")


def okg_drug_keys() -> pl.LazyFrame:
    """DRG with every plausible key surfaced, including ones dug out of properties."""
    d = pl.scan_parquet(NODES).filter(pl.col("label") == "DRG")
    return d.with_columns(
        # DrugBank ids wherever they hide: as the node id itself, or anywhere in the blob.
        pl.when(pl.col("id_prefix").str.to_uppercase() == "DRUGBANK")
          .then(pl.col("id_local"))
          .otherwise(None).alias("drugbank_from_id"),
        pl.col("properties_json").str.extract_all(r"DB\d{5}").alias("drugbank_from_props"),
    )


# ------------------------------------------------------------------ external side


def read_broad(path: Path) -> pl.DataFrame:
    """Broad ships '!'-prefixed header/comment lines. Tab separated."""
    df = pl.read_csv(path, separator="\t", comment_prefix="!",
                     infer_schema_length=10_000, truncate_ragged_lines=True)
    print(f"  {path.name}: {df.height:,} rows × {df.width} cols")
    print(f"  columns: {df.columns}")
    return df


def read_repodb(path: Path) -> pl.DataFrame | None:
    df = pl.read_csv(path, infer_schema_length=10_000, truncate_ragged_lines=True)
    print(f"  {path.name}: {df.height:,} rows × {df.width} cols")
    print(f"  columns: {df.columns}")
    missing = {"drugbank_id", "ind_id", "status"} - set(df.columns)
    if missing:
        print(f"  *** the export format CHANGED: expected columns absent: {sorted(missing)}.\n"
              f"      Verified schema (from repoDB's own assemble.R + app.R):\n"
              f"        drug_name, drugbank_id, ind_name, ind_id, NCT, status, phase, DetailedStatus\n"
              f"      Re-read the export before trusting anything downstream. Not guessing.")
        return None
    return df


# ------------------------------------------------------------------ checks


def check_broad(broad_drug: pl.DataFrame, broad_sample: pl.DataFrame, res: dict) -> None:
    head("BROAD  <->  OptimusKG   (InChIKey is the only bridge — Broad ships no DrugBank id)")

    ik_col = next((c for c in broad_sample.columns if c.lower() in
                   ("inchikey", "inchi_key")), None)
    if ik_col is None:
        print("  *** no InChIKey column in the sample file — cannot bridge to OKG at all")
        return

    b = (broad_sample.lazy()
         .select(pl.col(ik_col).str.strip_chars().alias("inchi_key"),
                 pl.col("pert_iname").str.strip_chars().alias("pert_iname"))
         .filter(pl.col("inchi_key").is_not_null() & (pl.col("inchi_key") != "")))
    b_keys = b.select("inchi_key").unique()
    n_b = b_keys.select(pl.len()).collect().item()

    okg = (pl.scan_parquet(NODES).filter(pl.col("label") == "DRG")
           .select("id", "inchi_key")
           .filter(pl.col("inchi_key").is_not_null()))
    n_okg_rows = okg.select(pl.len()).collect().item()
    okg_keys = okg.select("inchi_key").unique()
    n_okg = okg_keys.select(pl.len()).collect().item()
    both = okg_keys.join(b_keys, on="inchi_key", how="semi").select(pl.len()).collect().item()

    print(f"  Broad samples with an InChIKey : {n_b:,} unique keys")
    print(f"  OKG DRG with an inchi_key      : {n_okg_rows:,} nodes / {n_okg:,} unique keys"
          + ("  <- keys are SHARED by multiple nodes" if n_okg_rows != n_okg else ""))
    print(f"  intersection (unique keys)     : {both:,}")
    print(f"    -> of OKG's keys, matched by Broad : {pct(both, n_okg)}")
    print(f"    -> of Broad's keys, found in OKG   : {pct(both, n_b)}")
    print("  NOTE: the second number is the one A4 cares about — the candidate universe")
    print("        is Broad's, and a Broad drug absent from OKG is unscoreable.")

    # ---- the sample file is PHYSICAL-SAMPLE level (~3 rows per compound: different
    # vendors, catalog numbers, batches). The how-to does samples.unique("pert_iname"),
    # which picks ONE row per drug arbitrarily. If two samples of one compound are
    # different SALT FORMS they carry different InChIKeys, and unique() then silently
    # chooses a salt — the how-to's own salt/stereo warning, one step earlier than it
    # expects it.
    per = (broad_sample.lazy()
           .filter(pl.col(ik_col).is_not_null() & (pl.col(ik_col) != ""))
           .group_by("pert_iname")
           .agg(pl.col(ik_col).n_unique().alias("distinct_keys"),
                pl.len().alias("samples"))
           .collect())
    multi = per.filter(pl.col("distinct_keys") > 1)
    print(f"\n  sample rows {broad_sample.height:,} vs distinct pert_iname {per.height:,}"
          f"  ({broad_sample.height / max(per.height,1):.1f} samples per compound)")
    print(f"  pert_iname with >1 DISTINCT InChIKey : {pct(multi.height, per.height)}")
    if multi.height:
        print(f"    -> .unique('pert_iname') would silently pick one salt/stereo form for")
        print(f"       {multi.height:,} compounds. Decide the rule; don't inherit it from row order.")
        ex = multi.sort("distinct_keys", descending=True).head(3)
        for r in ex.iter_rows(named=True):
            print(f"       e.g. {r['pert_iname']}: {r['distinct_keys']} keys across {r['samples']} samples")
    res.setdefault("broad_salt", {}).update(
        {"sample_rows": broad_sample.height, "distinct_pert_iname": per.height,
         "pert_iname_multi_key": multi.height})

    # ---- PubChem: a SECOND bridge, independent of structure registration. Broad ships
    # pubchem_cid; whether OKG carries PubChem ids is the open half.
    pc_col = next((c for c in broad_sample.columns if "pubchem" in c.lower()), None)
    if pc_col:
        b_pc = (broad_sample.lazy()
                .select(pl.col(pc_col).cast(pl.Utf8).str.strip_chars().alias("pubchem"))
                .filter(pl.col("pubchem").is_not_null() & (pl.col("pubchem") != "")
                        & (pl.col("pubchem") != "null"))
                .unique())
        n_bpc = b_pc.select(pl.len()).collect().item()
        # OKG has NO field named pubchem*. If PubChem ids exist they are inside a
        # multi-vocabulary container (source_ids), keyed by vocabulary name. Look for
        # that shape; do not guess a top-level field name.
        okg_pc = (pl.scan_parquet(NODES).filter(pl.col("label") == "DRG")
                  .select(pl.col("properties_json")
                          .str.extract(r'"(?i:pubchem[a-z_]*)"\s*:\s*"?(\d+)"?', 1)
                          .alias("pubchem"))
                  .filter(pl.col("pubchem").is_not_null()).unique())
        n_opc = okg_pc.select(pl.len()).collect().item()
        hit = okg_pc.join(b_pc, on="pubchem", how="semi").select(pl.len()).collect().item()
        print(f"\n  PubChem route (Broad ships {pc_col}; InChIKey is NOT the only bridge):")
        print(f"    Broad unique pubchem ids     : {n_bpc:,}")
        print(f"    OKG DRG with a pubchem id    : {n_opc:,}  (0 => no PubChem field in properties)")
        print(f"    intersection                 : {hit:,}")
        res.setdefault("broad_pubchem", {}).update(
            {"broad": n_bpc, "okg": n_opc, "intersection": hit})

    # The whitelist population specifically.
    # The sample file is SAMPLE-level (3.1 rows per compound), so joining drugs onto it
    # multiplies drug rows by the sample count. Aggregate keys per compound FIRST, then
    # join 1:1. (Getting this wrong reports 9,009 "Launched drugs" from a file whose
    # Launched count is 2,718 — percentages of sample rows masquerading as drugs.)
    phase = next((c for c in broad_drug.columns if "phase" in c.lower()), None)
    if phase:
        vals = broad_drug[phase].value_counts().sort("count", descending=True)
        print(f"\n  {phase} values (the whitelist filter):")
        for r in vals.head(8).iter_rows(named=True):
            print(f"    {str(r[phase])[:40]:<42} {r['count']:,}")

        n_rows, n_drugs = broad_drug.height, broad_drug["pert_iname"].n_unique()
        if n_rows != n_drugs:
            print(f"  NOTE: drug file has {n_rows:,} rows but {n_drugs:,} distinct pert_iname")

        sample_keys = (broad_sample.lazy()
                       .filter(pl.col(ik_col).is_not_null() & (pl.col(ik_col) != ""))
                       .group_by("pert_iname")
                       .agg(pl.col(ik_col).str.strip_chars().unique().alias("keys")))
        launched = (broad_drug.lazy().filter(pl.col(phase) == "Launched")
                    .select("pert_iname").unique()
                    .join(sample_keys, on="pert_iname", how="left")
                    .collect())

        okg_set = set(okg.select("inchi_key").unique().collect()["inchi_key"].to_list())
        n_l = launched.height
        rows = launched.to_dicts()
        n_lk = sum(1 for r in rows if r["keys"])
        # ANY of a compound's keys reaching OKG counts — that is the salt-form-tolerant
        # reading, and it is deliberate: picking one key per compound would undercount.
        in_okg = sum(1 for r in rows if r["keys"] and any(k in okg_set for k in r["keys"]))

        print(f"\n  Launched compounds (distinct pert_iname) : {n_l:,}")
        print(f"    with at least one InChIKey             : {pct(n_lk, n_l)}")
        print(f"    reaching an OKG DRG node               : {pct(in_okg, n_l)}   <- A4's real candidate universe")
        print(f"    unreachable, therefore unscoreable     : {n_l - in_okg:,}")
        res["broad"] = {"launched_compounds": n_l, "launched_with_key": n_lk,
                        "launched_in_okg": in_okg, "broad_keys": n_b,
                        "okg_keyed": n_okg, "intersection": both}
    else:
        res["broad"] = {"broad_keys": n_b, "okg_keyed": n_okg, "intersection": both}


def check_repodb(repodb: pl.DataFrame, res: dict) -> None:
    head("repoDB  <->  OptimusKG   (drugbank_id and ind_id/UMLS CUI — checked separately)")

    if "status" in repodb.columns:
        print("  status levels (Approved/Suspended/Terminated/Withdrawn — NOT just Terminated):")
        for r in repodb["status"].value_counts().sort("count", descending=True).iter_rows(named=True):
            print(f"    {str(r['status'])[:30]:<32} {r['count']:,} rows")
        if "NCT" in repodb.columns:
            pairs = repodb.select("drugbank_id", "ind_id").unique().height
            print(f"\n  rows {repodb.height:,} vs UNIQUE (drug,indication) pairs {pairs:,}"
                  f"  <- rows are per-TRIAL; the paper's 6,677 is a pair count")

    # ---- drug side
    r_db = (repodb.lazy().select(pl.col("drugbank_id").str.strip_chars())
            .filter(pl.col("drugbank_id").is_not_null()).unique())
    n_rdb = r_db.select(pl.len()).collect().item()

    okg = okg_drug_keys()
    n_drg = okg.select(pl.len()).collect().item()
    via_id = okg.filter(pl.col("drugbank_from_id").is_not_null())
    n_via_id = via_id.select(pl.len()).collect().item()
    via_props = okg.filter(pl.col("drugbank_from_props").list.len() > 0)
    n_via_props = via_props.select(pl.len()).collect().item()

    print(f"\n  repoDB unique drugbank_id      : {n_rdb:,}")
    print(f"  OKG DRG nodes                  : {n_drg:,}")
    print(f"    DrugBank as the node id      : {pct(n_via_id, n_drg)}   (A1 said ~1,227)")
    print(f"    DB##### found in properties  : {pct(n_via_props, n_drg)}   <- THE unmeasured one")

    hit_id = (via_id.select(pl.col("drugbank_from_id").alias("drugbank_id"))
              .join(r_db, on="drugbank_id", how="semi").select(pl.len()).collect().item())
    dbp = (via_props.filter(pl.col("drugbank_from_props").list.len() > 0)
           .select("drugbank_from_props").collect())["drugbank_from_props"].to_list()
    flat_db = sorted({x for lst in dbp for x in (lst or []) if x})
    exploded = pl.LazyFrame({"drugbank_id": flat_db}, schema={"drugbank_id": pl.Utf8})
    hit_props = exploded.join(r_db, on="drugbank_id", how="semi").select(pl.len()).collect().item()
    print(f"\n  repoDB drugs reachable via OKG node-id DrugBank : {pct(hit_id, n_rdb)}")
    print(f"  repoDB drugs reachable via properties DrugBank  : {pct(hit_props, n_rdb)}")
    print("  (if the second is much larger, the DrugBank route beats InChIKey->UniChem")
    print("   AND reaches the biologics inchi_key structurally cannot)")

    # ---- disease side, on both denominators (the M5 lesson)
    r_cui = (repodb.lazy().select(pl.col("ind_id").str.strip_chars())
             .filter(pl.col("ind_id").is_not_null()).unique())
    n_rcui = r_cui.select(pl.len()).collect().item()

    dis = pl.scan_parquet(NODES).filter(pl.col("label") == "DIS")
    dd = EDGES / "DRG-DIS.parquet"
    touched = None
    if dd.exists():
        e = pl.scan_parquet(dd)
        touched = pl.concat([e.select(pl.col("from").alias("id")),
                             e.select(pl.col("to").alias("id"))]).unique()

    print(f"\n  repoDB unique ind_id (UMLS CUI): {n_rcui:,}")
    for name, pop in (("every DIS node", dis),
                      ("drug-touched DIS", dis.join(touched, on="id", how="semi")
                       if touched is not None else None)):
        if pop is None:
            continue
        n = pop.select(pl.len()).collect().item()
        native = (pop.select(pl.col("umls_cui").alias("ind_id"))
                  .filter(pl.col("ind_id").is_not_null()).unique())
        xref = None
        if "umls_cui_xrefs" in pop.collect_schema().names():
            # Counted in Python: DIS is ~36k rows and every Polars explode form trips
            # the 1.x->2.0 `empty_as_null` deprecation. Not worth a warning filter.
            lists = (pop.filter(pl.col("umls_cui_xrefs").list.len() > 0)
                     .select("umls_cui_xrefs").collect())["umls_cui_xrefs"].to_list()
            flat = sorted({c for lst in lists for c in (lst or []) if c})
            xref = pl.LazyFrame({"ind_id": flat}, schema={"ind_id": pl.Utf8})
        h_native = native.join(r_cui, on="ind_id", how="semi").select(pl.len()).collect().item()
        h_either = h_native
        if xref is not None:
            h_either = (pl.concat([native, xref]).unique()
                        .join(r_cui, on="ind_id", how="semi").select(pl.len()).collect().item())
        print(f"    {name} ({n:,}): repoDB CUIs matched — native {pct(h_native, n_rcui)}"
              f" | +xrefs {pct(h_either, n_rcui)}")

    res["repodb"] = {"repodb_drugbank": n_rdb, "repodb_cui": n_rcui,
                     "okg_drg": n_drg, "drugbank_as_id": n_via_id,
                     "drugbank_in_props": n_via_props,
                     "repodb_hit_via_id": hit_id, "repodb_hit_via_props": hit_props}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad-drug", type=Path)
    ap.add_argument("--broad-sample", type=Path)
    ap.add_argument("--repodb", type=Path)
    ap.add_argument("--sample-rows", type=int, default=4000,
                    help="property blobs sampled per label for the shape inventory")
    args = ap.parse_args()

    if not NODES.exists():
        raise SystemExit(f"FATAL: {NODES} missing — run A2 first (or set RETRACERX_ROOT)")

    res: dict = {"canon_nodes": str(NODES)}

    head("PART 1 — every OptimusKG field whose VALUES look like an identifier")
    print("Fields are classified by what they contain, not what they are named.")
    for label in ("DRG", "DIS"):
        c = okg_candidates(label, args.sample_rows)
        report_candidates(c)
        res.setdefault("okg_candidates", {})[label] = c["fields"]

    if args.broad_drug and args.broad_sample:
        head("PART 2 — Broad files as they actually are")
        if args.broad_drug.exists() and args.broad_sample.exists():
            bd, bs = read_broad(args.broad_drug), read_broad(args.broad_sample)
            check_broad(bd, bs, res)
        else:
            print(f"  missing: {[str(p) for p in (args.broad_drug, args.broad_sample) if not p.exists()]}")

    if args.repodb:
        head("PART 3 — repoDB as it actually is")
        if args.repodb.exists():
            rdb = read_repodb(args.repodb)
            if rdb is not None:
                check_repodb(rdb, res)
        else:
            print(f"  missing: {args.repodb}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}")
    print("\nNo joins were made and no key was chosen. That is A6's call, on these numbers.")


if __name__ == "__main__":
    main()
