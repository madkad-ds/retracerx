#!/usr/bin/env python3
"""
a2_canon_nodes.py — build data/canon/canon_nodes.parquet from the routing frozen
in a2_source_policy.json.

Output schema (one row per node, 190,531 rows expected):

    id              str          verbatim graph key. NEVER rewritten.
    label           str          GEN/DIS/DRG/BPO/PHE/ANA/MFN/CCO/PWY/EXP
    id_prefix       str|null     normalized: MONDO, DOID, MEDDRA (upper-cased)
    id_local        str          the local part; == id when there is no separator
    id_norm         str          prefix:local — an ANNOTATION for A6, not a key
    name            str|null     lifted out of properties (A3 needs it as a column)
    description     str|null     lifted out of properties
    inchi_key       str|null     DRG only — A6's drug join key, already on the node
    umls_cui        *|null       DIS only — A6's disease join key, already on the node
    umls_cui_xrefs  list[str]|-  UMLS CUIs recovered from properties.xrefs. ALL of them.
    xref_vocabs     list[str]|-  which vocabularies this node cross-references
    symbol          str|null     GEN only — A4's Broad HGNC bridge
    properties_json str          the whole properties blob, verbatim, nothing dropped
    src_file        str          which of the 41 files this row came from
    src_checksum    str          verified | disputed_adjudicated_benign | ...

Why the promotions: A6 is "a join, not a project" only because inchi_key and
umls_cui are already on the nodes. Lifting them here means A6 reads a typed
column instead of re-parsing JSON, and a1_keycoverage.py can gate on this file.

Why umls_cui_xrefs: the native umls_cui covers 43.5% of drug-touched DIS nodes.
properties.xrefs carries UMLS:C....... entries for a further ~400 of them. Both are
already in the blob; A2's job is to surface them, not to choose between them. They
stay in SEPARATE columns so A6 can weigh a curated key against a cross-reference
rather than inheriting a merge A2 made silently.

Usage:
    python scripts/a2_canon_nodes.py
    python scripts/a2_canon_nodes.py --dry-run     # print the plan, write nothing
"""
from __future__ import annotations

import argparse

import polars as pl

from a2_common import (
    A2_POLICY, CANON_DIR, CANON_NODES, EXPECT, ID_RE, NODE_PROMOTE,
    coerce, die, load_json, resolve_path, revive, schema_of, struct_fields, warn,
)

XREFS = "xrefs"
XREFS_STR_DT = pl.List(pl.Utf8)
UMLS_XREF_PREFIX = "UMLS"
# `xrefs` does NOT have one shape across labels. Measured on OptimusKG 2.0:
#   DIS -> List(String)                     e.g. "UMLS:C1510471", "MEDGEN:267607"
#   GEN -> List(Struct{id, source})         e.g. {"id":"51632","source":"HGNC"}
# So the shape is a per-label fact, not a dataset fact, and it has to be read from the
# inventory each time. Hardcoding either form onto every label is the same mistake as
# inferring a dtype from a sample — just in the opposite direction.
XREFS_OBJ_DT = pl.List(pl.Struct({"id": pl.Utf8, "source": pl.Utf8}))
_SOURCE_KEYS = ("source", "database", "db", "vocabulary", "vocab", "prefix")
_ID_KEYS = ("id", "identifier", "accession", "local_id", "value")
URL_RE = r"^https?://"
# last path segment of a PURL/identifiers.org URL, when it carries a CURIE-ish prefix:
#   .../obo/DOID_0050890      -> DOID
#   .../meddra:10053176       -> meddra
#   .../entry/123456          -> no match, falls back to "URL"
URL_VOCAB_RE = r"/([A-Za-z][A-Za-z0-9.]*)[_:][^/#]*$"


def xref_mode(dt: pl.DataType | None) -> tuple[str | None, pl.DataType | None, str]:
    """(mode, decode_dtype, why). mode is 'string' | 'object' | None.

    None means A2 could not prove the shape, so it promotes nothing and says so.
    Guessing here would either crash mid-stream or fabricate empty columns.
    """
    if dt is None:
        return None, None, "absent from this label's properties"
    if isinstance(dt, pl.List) and dt.inner == pl.Utf8:
        return "string", XREFS_STR_DT, "List(String), parsed as VOCAB:LOCAL"
    if isinstance(dt, pl.List) and isinstance(dt.inner, pl.Struct):
        names = {f.name.lower() for f in dt.inner.fields}
        src = next((k for k in _SOURCE_KEYS if k in names), None)
        idf = next((k for k in _ID_KEYS if k in names), None)
        if src and idf:
            return "object", pl.List(pl.Struct({idf: pl.Utf8, src: pl.Utf8})), \
                   f"List(Struct) with .{src}/.{idf}"
        return None, None, f"List(Struct) with unrecognised fields {sorted(names)}"
    if dt == pl.Null or (isinstance(dt, pl.List) and dt.inner == pl.Null):
        return None, None, ("sample saw only empty/null xrefs, so the element shape is "
                            "unproven — raise --infer-rows if this label matters")
    return None, None, f"unsupported dtype {dt}"


def xref_exprs(raw: pl.Expr, mode: str, dt: pl.DataType) -> tuple[pl.Expr, pl.Expr]:
    """(umls_cui_xrefs, xref_vocabs) for a proven shape."""
    if mode == "string":
        umls = (raw.list.eval(pl.element().filter(
                    pl.element().str.starts_with(f"{UMLS_XREF_PREFIX}:")))
                .list.eval(pl.element().str.strip_prefix(f"{UMLS_XREF_PREFIX}:")))
        # Some xrefs are URLs, not CURIEs ("http://purl.obolibrary.org/obo/DOID_0050890").
        # Splitting those on ":" names the URI scheme, which is how "https" turned up in
        # the vocabulary histogram — a scheme is not a vocabulary. Recover the real one
        # from the last path segment when it's a CURIE, else say plainly that it's a URL.
        #
        # Deliberately NOT done: harvesting CUIs out of URL paths. NCIT identifiers are
        # C-prefixed too (NCIT:C35772), so a /C\d+/ match would mint fake CUIs from NCIT
        # codes. umls_cui_xrefs stays on the exact UMLS: prefix.
        el = pl.element()
        vocabs = raw.list.eval(
            pl.when(el.str.contains(URL_RE))
            .then(pl.coalesce(el.str.extract(URL_VOCAB_RE, 1), pl.lit("URL")))
            .otherwise(el.str.split(":").list.first())
        )
    else:
        fields = list(dt.inner.fields)
        idf = next(f.name for f in fields if f.name.lower() in _ID_KEYS)
        src = next(f.name for f in fields if f.name.lower() in _SOURCE_KEYS)
        umls = raw.list.eval(
            pl.element().filter(
                pl.element().struct.field(src).str.to_uppercase() == UMLS_XREF_PREFIX
            ).struct.field(idf))
        vocabs = raw.list.eval(pl.element().struct.field(src))
    return (umls.alias("umls_cui_xrefs"),
            vocabs.list.unique().list.sort().alias("xref_vocabs"))


def target_dtypes(policy: dict) -> dict[str, pl.DataType]:
    """One dtype per promoted field, across all labels, so the concat aligns."""
    out: dict[str, pl.DataType] = {}
    for field in NODE_PROMOTE:
        seen = []
        for lb, entry in policy["nodes"].items():
            dt = revive(entry["property_fields"].get(field))
            if dt is not None:
                seen.append(dt)
        if not seen:
            continue
        out[field] = pl.List(pl.Utf8) if any(isinstance(d, pl.List) for d in seen) else pl.Utf8
    return out


def coerce(expr: pl.Expr, src: pl.DataType, target: pl.DataType) -> pl.Expr:
    if isinstance(target, pl.List) and not isinstance(src, pl.List):
        return pl.when(expr.is_null()).then(None).otherwise(pl.concat_list(expr.cast(pl.Utf8)))
    if isinstance(target, pl.List):
        return expr.cast(pl.List(pl.Utf8))
    return expr.cast(pl.Utf8)


def build_label(label: str, entry: dict, targets: dict[str, pl.DataType]) -> pl.LazyFrame:
    path = resolve_path(entry["file"])
    if path is None:
        die(f"{entry['file']} vanished since a2_source_policy.py ran")
    lf = pl.scan_parquet(path)
    if not entry["stratified"]:
        lf = lf.filter(pl.col("label") == label)

    repr_ = entry["properties_repr"]
    fields = {f: revive(t) for f, t in entry["property_fields"].items()}

    if repr_ == "struct":
        live = struct_fields(schema_of(path)["properties"])
        prop = pl.col("properties")
        get = lambda f: prop.struct.field(f)  # noqa: E731
        have = {f: dt for f, dt in live.items()}
        props_json = prop.struct.json_encode()
        xmode, xdt, xwhy = xref_mode(live.get(XREFS))
        xrefs_raw = prop.struct.field(XREFS) if xmode else None
    elif repr_ == "json":
        # Presence from the inventory, dtype from NODE_PROMOTE — never from the sample.
        # An all-null sample infers Null, and decoding a real string into a Null field
        # is a hard error (the same trap that bit properties.sources on the edges).
        wanted = {f: pl.Utf8 for f in fields if f in NODE_PROMOTE}
        have = dict(wanted)
        narrow_fields = dict(wanted)
        xmode, xdt, xwhy = xref_mode(fields.get(XREFS))
        if xmode:
            narrow_fields[XREFS] = xdt
        if narrow_fields:
            narrow = pl.Struct(narrow_fields)
            get = lambda f: pl.col("properties").str.json_decode(narrow).struct.field(f)  # noqa: E731
        else:
            get = lambda f: pl.lit(None, dtype=pl.Utf8)  # noqa: E731
        props_json = pl.col("properties")
        xrefs_raw = get(XREFS) if xmode else None
    else:
        die(f"{label}: properties repr {repr_!r} in {entry['file']} — unsupported. "
            f"Re-run a2_source_policy.py; do not guess.")

    if xmode:
        print(f"       {label} xrefs: {xwhy}")
    elif XREFS in fields:
        warn(f"{label}: xrefs present but not promoted — {xwhy}. "
             f"The blob is still intact in properties_json.")

    promoted = []
    for field, target in targets.items():
        if field in have:
            promoted.append(coerce(get(field), have[field], target).alias(field))
        else:
            promoted.append(pl.lit(None, dtype=target).alias(field))

    if xrefs_raw is None:
        umls_xrefs = pl.lit(None, dtype=XREFS_STR_DT).alias("umls_cui_xrefs")
        vocabs = pl.lit(None, dtype=XREFS_STR_DT).alias("xref_vocabs")
    else:
        # ALL of them, not the first. A node with two CUIs is an ambiguity A6 has to
        # resolve; silently taking [0] would hide the decision inside A2. M5 counts them.
        # The other bridges (MEDGEN, MESH, SCTID, OMIM, MedDRA…) land in xref_vocabs:
        # A2 inventories them, choosing one is A6's call.
        umls_xrefs, vocabs = xref_exprs(xrefs_raw, xmode, xdt)

    return lf.select(
        pl.col("id").cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8),
        pl.col("id").str.extract(ID_RE, 1).str.to_uppercase().alias("id_prefix"),
        pl.coalesce(pl.col("id").str.extract(ID_RE, 2), pl.col("id")).alias("id_local"),
        pl.coalesce(
            pl.concat_str([
                pl.col("id").str.extract(ID_RE, 1).str.to_uppercase(),
                pl.col("id").str.extract(ID_RE, 2),
            ], separator=":"),
            pl.col("id"),
        ).alias("id_norm"),
        *promoted,
        umls_xrefs,
        vocabs,
        props_json.alias("properties_json"),
        pl.lit(entry["file"]).alias("src_file"),
        pl.lit(entry["checksum"]).alias("src_checksum"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    policy = load_json(A2_POLICY)
    if policy.get("graph_variant") != "full":
        die(f"policy graph_variant={policy.get('graph_variant')!r}; A1.5 froze 'full'.")
    targets = target_dtypes(policy)
    print("promoted node fields:", {k: str(v) for k, v in targets.items()} or "NONE")
    for field, why in NODE_PROMOTE.items():
        if field not in targets:
            warn(f"`{field}` not found in any node label's properties ({why}) — "
                 f"downstream must fall back.")

    frames = []
    for label in sorted(policy["nodes"]):
        entry = policy["nodes"][label]
        print(f"  {label:<4} <- {entry['file']:<40} "
              f"[{entry['properties_repr']}, {entry['checksum']}]")
        frames.append(build_label(label, entry, targets))

    if args.dry_run:
        print("\n--dry-run: plan only, nothing written")
        return

    CANON_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(frames, how="vertical_relaxed").sink_parquet(CANON_NODES, compression="zstd")

    n = pl.scan_parquet(CANON_NODES).select(pl.len()).collect().item()
    print(f"\nwrote {CANON_NODES} — {n:,} rows")
    if n != EXPECT["nodes_total"]:
        die(f"{n:,} nodes written, census says {EXPECT['nodes_total']:,}. "
            f"Do not proceed to A3 — reconcile first.")


if __name__ == "__main__":
    main()
