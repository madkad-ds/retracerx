#!/usr/bin/env python3
"""
a3_common — shared helpers for the A3 scripts.

Design rule inherited from A2's failure log #1: NEVER guess the shape of a manifest
written by an earlier step. Every read goes through `pick()`, which tries a list of
candidate paths and FATALs with the actual available keys printed. There is no
"UNKNOWN" state and no silent default.

Jargon used below, defined once:
  * sha256      — a cryptographic checksum; two files with the same sha256 are byte-identical.
  * manifest    — a JSON record of what a step consumed and produced, so a later step can
                  verify it is reading the same bytes. Every A1/A2 step writes one; A7's
                  MANIFEST.json inherits their rows.
  * gate        — a check that exits non-zero on failure, so a broken step cannot be
                  mistaken for a passing one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# --- reuse A2's helpers where they exist; never silently reimplement -----------------
try:
    from a2_common import sha256_file as _a2_sha256  # type: ignore
except Exception:  # pragma: no cover - a2_common absent or renamed
    _a2_sha256 = None


def fatal(msg: str, *extra: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"[A3][FATAL] {msg}", file=sys.stderr)
    for e in extra:
        if e:                       # conditional details pass "" when not applicable
            print(f"           {e}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[A3][WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[A3] {msg}")


_A2_SHA_BROKEN = False


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """
    sha256 of a file. Delegates to a2_common so both steps agree on the implementation.

    a2_common.sha256_file takes a Path (it calls path.open). If its signature ever differs,
    fall back to the local implementation and SAY SO once — the output is byte-identical
    either way, so this is a declared substitution, not a silent repair.
    """
    global _A2_SHA_BROKEN
    path = Path(path)
    if _a2_sha256 is not None and not _A2_SHA_BROKEN:
        try:
            return _a2_sha256(path)
        except (AttributeError, TypeError) as e:
            _A2_SHA_BROKEN = True
            warn(f"a2_common.sha256_file rejected a Path ({e}); using the local "
                 f"implementation for the rest of this run. Output is identical.")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: str | Path, what: str) -> dict:
    p = Path(path)
    if not p.exists():
        fatal(f"{what} not found at {p}",
              "A3 reads its counts from the manifests. It does not carry hardcoded numbers.")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        fatal(f"{what} at {p} is not valid JSON: {e}")


def _walk(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def pick(obj: Any, candidates: Iterable[str], what: str, src: str) -> Any:
    """
    Return the first candidate dotted-path that resolves, or FATAL listing what IS there.

    This exists because A3 must not encode a guess about A1/A2's JSON layout.
    A2 failure #1 was exactly this: loaders guessed A1's shapes.
    """
    tried = list(candidates)
    for c in tried:
        try:
            return _walk(obj, c)
        except KeyError:
            continue
    top = sorted(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
    fatal(f"could not locate {what} in {src}",
          f"tried paths: {tried}",
          f"top-level keys present: {top}",
          "Fix the candidate list in a3_common/pick() against the real file. Do not hardcode the value.")


def normalize_pair_counts(raw: Any, src: str) -> dict[tuple[str, str], int]:
    """
    Accept the (label, relation) -> count census in any of the plausible shapes and
    return one canonical dict. FATALs on an unrecognised shape rather than guessing.

    Handles:
      [{"label":..,"relation":..,"count"/"edges"/"n":..}, ...]
      {"DRG-DIS": {"INDICATION": 57601, ...}, ...}
      {"DRG-DIS/INDICATION": 57601, ...}   (also accepts '|' and '::' separators)
    """
    out: dict[tuple[str, str], int] = {}

    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                fatal(f"{src}: list element is {type(row).__name__}, expected object")
            lab = row.get("label")
            rel = row.get("relation")
            cnt = row.get("count", row.get("edges", row.get("n", row.get("len"))))
            if lab is None or rel is None or cnt is None:
                fatal(f"{src}: list element missing label/relation/count", f"element keys: {sorted(row)}")
            out[(str(lab), str(rel))] = int(cnt)
        return out

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                for rel, cnt in v.items():
                    out[(str(k), str(rel))] = int(cnt)
            else:
                for sep in ("/", "|", "::"):
                    if sep in k:
                        lab, rel = k.split(sep, 1)
                        out[(lab.strip(), rel.strip())] = int(v)
                        break
                else:
                    fatal(f"{src}: key {k!r} is not a label{{/,|,::}}relation pair and its value is not a dict")
        return out

    fatal(f"{src}: pair census is {type(raw).__name__}, expected list or dict")


def write_manifest(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    info(f"manifest written: {p}")


# --- composite relationship TYPE ------------------------------------------------------
# One place, used by the emitter AND the verifier, so the two cannot drift.

COMPOSITE_SEP = "__"


def composite_type(label: str, relation: str) -> str:
    return f"{label.replace('-', '_')}{COMPOSITE_SEP}{relation}"


def parse_composite(type_name: str) -> tuple[str, str]:
    """Inverse of composite_type. Raises ValueError if the name is not reversible."""
    if type_name.count(COMPOSITE_SEP) != 1:
        raise ValueError(f"{type_name!r} does not contain exactly one {COMPOSITE_SEP!r}")
    lab_us, rel = type_name.split(COMPOSITE_SEP)
    return lab_us.replace("_", "-"), rel
