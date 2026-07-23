"""
a2_common.py — shared helpers for ReTraceRx Phase A2 (canonicalize OptimusKG).

Design rules inherited from A1 and enforced here:
  * Never guess a schema. Measure it, record it, and fail loudly on disagreement.
  * The DOI pins nothing. `dataset_version` + per-file sha256 are the pin.
  * Every A2 output carries the source file and its checksum status.

All paths are repo-relative. Override the repo root with RETRACERX_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import polars as pl

# ---------------------------------------------------------------- paths

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
MANIFEST_DIR = REPO / "data" / "manifest"
CANON_DIR = REPO / "data" / "canon"
LOG_DIR = REPO / "logs"

OKG_MANIFEST = MANIFEST_DIR / "okg_manifest.json"
OKG_CENSUS = MANIFEST_DIR / "okg_census.json"
OKG_DISPUTES = MANIFEST_DIR / "okg_dispute_audit.json"
A1_DECISIONS = MANIFEST_DIR / "a1_decisions.json"

A2_POLICY = MANIFEST_DIR / "a2_source_policy.json"
A2_DECISIONS = MANIFEST_DIR / "a2_decisions.json"
A2_CANON_MANIFEST = MANIFEST_DIR / "a2_canon_manifest.json"
A2_INTEGRITY = MANIFEST_DIR / "a2_integrity.json"

CANON_NODES = CANON_DIR / "canon_nodes.parquet"
CANON_EDGES_DIR = CANON_DIR / "canon_edges"  # one parquet per edge label

# ---------------------------------------------------------------- expectations
# Source: data/manifest/okg_census.json (A1.6), as reported in the A1 handover.
# These are cross-checked against the census at runtime; a disagreement is a
# hard error, because it means the pinned snapshot moved.
EXPECT = {
    "dataset_version": "2.0",
    "nodes_total": 190_531,
    "edges_total": 21_813_816,
    "node_labels": 10,
    "edge_labels": 27,
    "relations": 36,
    # 13, not the 12 in A1's prose handover. okg_census.json .directionality is the
    # measurement (13 entries true); the handover was a hand-count. G5 reads the array
    # and only falls back to this scalar if the array is absent.
    "undirected_labels": 13,
}

NODE_LABELS = ["GEN", "DIS", "DRG", "BPO", "PHE", "ANA", "MFN", "CCO", "PWY", "EXP"]

# filename stem -> node label. Used only to *propose* a mapping; every proposal
# is verified by reading the file's own `label` column before it is used.
STEM_TO_LABEL = {
    "gene": "GEN",
    "disease": "DIS",
    "drug": "DRG",
    "biological_process": "BPO",
    "phenotype": "PHE",
    "anatomy": "ANA",
    "molecular_function": "MFN",
    "cellular_component": "CCO",
    "pathway": "PWY",
    "exposure": "EXP",
}

# Node `properties` fields A2 promotes to typed columns, and who needs them.
NODE_PROMOTE = {
    "name": "A3 (Neo4j import needs it as a column)",
    "description": "A3 / demo surface",
    "inchi_key": "A6 drug join key",
    "umls_cui": "A6 disease join key",
    "symbol": "A4 Broad HGNC->Ensembl bridge",
}

# ---------------------------------------------------------------- io helpers


def die(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            die(f"missing {path} — A2 depends on A1 artifacts. Run A1 first.")
        return None
    with path.open() as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    print(f"wrote {path}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    """Order-stable digest of a directory of parquet parts."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.parquet")):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def deep_find(obj: Any, aliases: Iterable[str], _path: str = "") -> tuple[Any, str] | tuple[None, None]:
    """Find the first dict key matching any alias. Returns (value, json_path)."""
    aliases = list(aliases)
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{_path}.{k}" if _path else k
            if k in aliases:
                return v, here
        for k, v in obj.items():
            here = f"{_path}.{k}" if _path else k
            found, p = deep_find(v, aliases, here)
            if found is not None:
                return found, p
    return None, None


# ---------------------------------------------------------------- A1 artifacts


def snapshot_root() -> Path:
    """Where a1_pin_okg_2.py downloaded the 41 files."""
    raw = os.environ.get("OPTIMUSKG_CACHE_DIR") or str(REPO / "data" / "okg_cache")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (REPO / root).resolve()
    if not root.exists():
        die(f"snapshot root {root} does not exist — `source scripts/okg_env.sh` first.")
    return root


@lru_cache(maxsize=1)
def manifest_files() -> tuple[dict, ...]:
    """A1 writes files as {relpath: {local_path, sha256, checksum_dispute?}}.

    Normalised to: name (relpath), local_path, sha256, disputed (bool).
    `checksum_dispute` present == the published MD5 disagrees with our bytes.
    """
    man = load_json(OKG_MANIFEST)
    files = man.get("files")
    if not isinstance(files, dict) or not files:
        die(f"{OKG_MANIFEST}: expected .files to be a dict keyed by relative path, got "
            f"{type(files).__name__} (top-level keys: {list(man)}). A1's format changed — "
            "fix this loader by hand rather than guessing.")
    if man.get("dataset_version") not in (None, EXPECT["dataset_version"]):
        die(f"manifest dataset_version={man['dataset_version']!r} but A2 expects "
            f"{EXPECT['dataset_version']!r}. The pin moved — stop and re-adjudicate.")
    out = []
    for name, meta in files.items():
        if not isinstance(meta, dict):
            die(f"{OKG_MANIFEST}: .files[{name!r}] is {type(meta).__name__}, expected dict")
        if not meta.get("sha256"):
            die(f"{OKG_MANIFEST}: .files[{name!r}] has no sha256. The sha256 is the pin "
                "(the DOI is not) — A2 will not canonicalize unpinnable bytes.")
        out.append({
            "name": name,
            "local_path": meta.get("local_path"),
            "sha256": meta["sha256"],
            "disputed": "checksum_dispute" in meta,
        })
    n_disp = sum(f["disputed"] for f in out)
    print(f"okg_manifest: {len(out)} files, {n_disp} with a published-MD5 dispute, "
          f"dataset_version={man.get('dataset_version')!r}")
    return tuple(out)


def resolve_path(name: str) -> Path | None:
    """A1's manifest carries an explicit repo-relative local_path — trust it first."""
    for f in manifest_files():
        if f["name"] == name and f["local_path"]:
            cand = REPO / f["local_path"]
            if cand.exists():
                return cand
            die(f"manifest lists {name} at {f['local_path']} but that file is missing. "
                "The snapshot and the manifest disagree — re-run A1's fetch.")
    cand = snapshot_root() / name
    return cand if cand.exists() else None


@lru_cache(maxsize=1)
def dispute_status() -> dict[str, str]:
    """relpath -> 'verified' | 'disputed_adjudicated_benign' | 'disputed_unadjudicated'.

    A1 records no verdict field; it records *evidence*, and A2 re-derives the verdict:
      manifest .checksum_dispute absent            -> verified (bytes match published MD5)
      test1[f].identical_to_v1 is True             -> the bytes are byte-identical to the
                                                      v1 release, so the v2 *metadata* MD5
                                                      is what's stale, not the file
      test2[f].keys_match is True (when present)   -> flat and stratified copies agree
    Both available tests must pass, or the dispute stands unadjudicated.
    """
    audit = load_json(OKG_DISPUTES, required=False)
    if audit is None:
        die(f"{OKG_DISPUTES} not found. 27 of 41 files carry an MD5 dispute; without the "
            "A1 audit there is no warrant for reading any of them. Re-run A1.")
    t1 = audit.get("test1")
    t2 = audit.get("test2")
    if not isinstance(t1, dict):
        die(f"{OKG_DISPUTES}: expected .test1 to be a dict keyed by relative path, got "
            f"{type(t1).__name__} (keys: {list(audit)}). Fix this loader by hand — "
            "silently treating disputed bytes as UNKNOWN is how a bad file gets read.")
    if not isinstance(t2, dict):
        t2 = {}
        warn(f"{OKG_DISPUTES}: no .test2 (flat-vs-stratified) — adjudication rests on test1 alone.")

    status: dict[str, str] = {}
    for f in manifest_files():
        name = f["name"]
        if not f["disputed"]:
            status[name] = "verified"
            continue
        e1, e2 = t1.get(name), t2.get(name)
        ok1 = isinstance(e1, dict) and e1.get("identical_to_v1") is True
        ok2 = True if e2 is None else (isinstance(e2, dict) and e2.get("keys_match") is True)
        status[name] = "disputed_adjudicated_benign" if (ok1 and ok2) else "disputed_unadjudicated"

    tally = {k: sum(v == k for v in status.values()) for k in
             ("verified", "disputed_adjudicated_benign", "disputed_unadjudicated")}
    print(f"dispute audit: {tally['verified']} verified / "
          f"{tally['disputed_adjudicated_benign']} disputed-but-benign / "
          f"{tally['disputed_unadjudicated']} unadjudicated")
    if audit.get("n_disputed") not in (None, tally["disputed_adjudicated_benign"]
                                       + tally["disputed_unadjudicated"]):
        warn(f"audit says n_disputed={audit['n_disputed']} but the manifest carries "
             f"{tally['disputed_adjudicated_benign'] + tally['disputed_unadjudicated']} "
             "checksum_dispute blocks — the two A1 artefacts disagree.")
    return status


def graph_variant() -> str:
    dec = load_json(A1_DECISIONS)
    variant = dec.get("graph_variant")
    if variant is None:
        variant, _ = deep_find(dec, ["graph_variant"])
    if variant != "full":
        die(f"a1_decisions.json graph_variant={variant!r}. A2's counts and gates are written "
            f"against 'full' (frozen at A1.5). Re-run A1.5 or pass --allow-variant.")
    if dec.get("frozen") is False:
        die("a1_decisions.json is not frozen — freeze A1.5 before canonicalizing.")
    return variant


def a1_source_tables() -> list[str] | None:
    """A1.5 froze which tables Phase A reads. Authoritative when present."""
    dec = load_json(A1_DECISIONS)
    tables, path = deep_find(dec, ["source_tables"])
    if tables is None:
        warn("a1_decisions.json has no source_tables — A2 will consider every parquet in the snapshot.")
        return None
    if isinstance(tables, dict):
        flat: list[str] = []
        for v in tables.values():
            flat.extend(v if isinstance(v, list) else [v])
        tables = flat
    if not isinstance(tables, list):
        die(f"a1_decisions.json .{path} has unexpected type {type(tables).__name__}")
    print(f"a1_decisions.source_tables: {len(tables)} entries — A2 restricted to these.")
    return [str(t) for t in tables]


def census() -> dict:
    return load_json(OKG_CENSUS)


def census_expect(cen: dict) -> None:
    """Cross-check the hardcoded EXPECT block against the census. Loud on drift."""
    checks = [
        (["nodes_total", "n_nodes", "node_count", "nodes"], "nodes_total"),
        (["edges_total", "n_edges", "edge_count", "edges"], "edges_total"),
    ]
    for aliases, key in checks:
        val, path = deep_find(cen, aliases)
        if isinstance(val, dict):
            val = val.get("total")
        if isinstance(val, int):
            if val != EXPECT[key]:
                die(f"census .{path} = {val:,} but A2 expects {EXPECT[key]:,}. "
                    f"The snapshot or the census changed — stop.")
            print(f"census cross-check OK: {key} = {val:,} (.{path})")
        else:
            warn(f"could not cross-check {key} against the census; using EXPECT={EXPECT[key]:,}")


# ---------------------------------------------------------------- schema helpers


def lazy(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def schema_of(path: Path) -> dict[str, pl.DataType]:
    return dict(pl.scan_parquet(path).collect_schema())


def struct_fields(dtype: pl.DataType) -> dict[str, pl.DataType]:
    return {f.name: f.dtype for f in dtype.fields} if isinstance(dtype, pl.Struct) else {}


def kind_of(schema: dict[str, pl.DataType]) -> str | None:
    cols = set(schema)
    if {"from", "to", "label", "relation"} <= cols:
        return "edges"
    if {"id", "label"} <= cols:
        return "nodes"
    return None


def properties_repr(schema: dict[str, pl.DataType]) -> str:
    dt = schema.get("properties")
    if dt is None:
        return "absent"
    if isinstance(dt, pl.Struct):
        return "struct"
    if dt == pl.Utf8:
        return "json"
    return f"other:{dt}"


# Polars dtype reprs are recorded as strings in a2_source_policy.json and revived
# here through a closed namespace — no builtins, no arbitrary evaluation.
_DTYPE_NS = {
    "String": pl.Utf8, "Utf8": pl.Utf8, "Int64": pl.Int64, "Int32": pl.Int32,
    "Float64": pl.Float64, "Float32": pl.Float32, "Boolean": pl.Boolean,
    "List": pl.List, "Struct": pl.Struct, "Null": pl.Null, "UInt32": pl.UInt32,
    "UInt64": pl.UInt64, "Date": pl.Date, "Categorical": pl.Categorical,
}

# Separator forms measured in A1: MONDO:0005148 (colon), DOID_0050890
# (underscore), meddra (lower-case). Ensembl IDs have no separator at all.
ID_RE = r"^([A-Za-z][A-Za-z0-9.]*)[:_](.+)$"


def revive(dtype_str: str | None) -> pl.DataType | None:
    """'List(String)' -> pl.List(pl.Utf8). None when the repr isn't supported."""
    if not dtype_str:
        return None
    try:
        return eval(dtype_str, {"__builtins__": {}}, _DTYPE_NS)  # noqa: S307 — closed namespace
    except Exception:
        return None


def coerce(expr: pl.Expr, src: pl.DataType, target: pl.DataType) -> pl.Expr:
    """Align a promoted field to one dtype across labels, without inventing values."""
    if isinstance(target, pl.List) and not isinstance(src, pl.List):
        return pl.when(expr.is_null()).then(None).otherwise(pl.concat_list(expr.cast(pl.Utf8)))
    if isinstance(target, pl.List):
        return expr.cast(pl.List(pl.Utf8))
    return expr.cast(pl.Utf8)
