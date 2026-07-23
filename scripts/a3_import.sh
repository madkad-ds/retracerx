#!/usr/bin/env bash
# A3 — offline bulk import into Neo4j, then start, constrain, mark, and GATE.
#
#   variant   which edge set is loaded: `full` (all edges) or `filtered` (leakage removed).
#             Neo4j Community hosts ONE database, so only one variant exists at a time.
#             The frozen method is RE-IMPORT, not delete-in-place: the store is then a pure
#             function of sha256-pinned inputs (a3_decisions.json#leakage_filter_method).
#   offline   the importer runs against a STOPPED database, in a throwaway container sharing
#             the same volumes. It has its OWN memory budget (--max-off-heap-memory); the
#             server's heap and page-cache settings do NOTHING for this step — they matter
#             only for the Cypher queries afterwards.
#
# Usage:
#   bash scripts/a3_import.sh --variant full
#   bash scripts/a3_import.sh --variant filtered
#   OFFHEAP=6G bash scripts/a3_import.sh --variant full
set -euo pipefail

CONTAINER=${CONTAINER:-retracerx-neo4j}
NEO_USER=${NEO_USER:-neo4j}
NEO_PASS=${NEO_PASS:-retracerx}
OFFHEAP=${OFFHEAP:-4G}
IMPORT_HOST="$PWD/neo4j/import"
DATA_HOST="$PWD/neo4j/data"
LOG_DIR="$PWD/logs"

VARIANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="${2:-}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$VARIANT" in
  full|filtered) ;;
  *) echo "[A3][FATAL] --variant must be 'full' or 'filtered' (got '${VARIANT}')." >&2
     echo "            It is never inferred: importing the wrong one is invisible after the fact." >&2
     exit 1 ;;
esac

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/a3_import_${VARIANT}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "[A3] logging to $LOG"

EDGES_CSV="$IMPORT_HOST/edges_${VARIANT}.csv"
META_CYPHER="a3_meta_${VARIANT}.cypher"

for f in "$IMPORT_HOST/nodes.csv" "$EDGES_CSV" "$IMPORT_HOST/$META_CYPHER"; do
  test -f "$f" || { echo "[A3][FATAL] missing $f — run a3_emit_csvs.py first." >&2; exit 1; }
done
cp scripts/a3_constraints.cypher "$IMPORT_HOST/a3_constraints.cypher"

echo "[A3] variant=$VARIANT  edges=$(basename "$EDGES_CSV")"
echo "[A3] stopping DB for offline import..."
docker stop "$CONTAINER" >/dev/null

echo "[A3] running neo4j-admin database import full ..."
# NOTE: 'full' here is the IMPORT MODE (build the database from scratch). It is unrelated
# to the full-vs-lcc graph variant frozen in a1_decisions.json. Same word, different axis.
docker run --rm \
  -v "$DATA_HOST":/data \
  -v "$IMPORT_HOST":/var/lib/neo4j/import \
  neo4j:5-community \
  neo4j-admin database import full \
    --nodes=/var/lib/neo4j/import/nodes.csv \
    --relationships="/var/lib/neo4j/import/edges_${VARIANT}.csv" \
    --max-off-heap-memory="$OFFHEAP" \
    --overwrite-destination \
    neo4j
# a3_emit_csvs.py flattens CR/LF/TAB inside node names and never writes properties_json,
# so --multiline-fields=true should not be needed. If the importer ever reports a quoted
# newline, add it — but check the manifest's names_whitespace_flattened count first.

echo "[A3] starting DB..."
docker start "$CONTAINER" >/dev/null

echo "[A3] waiting for bolt (Neo4j's binary protocol, port 7687)..."
for i in $(seq 1 60); do
  if docker exec "$CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" "RETURN 1;" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  [[ $i -eq 60 ]] && { echo "[A3][FATAL] bolt not up after 120s" >&2; exit 1; }
done

echo "[A3] applying constraints/indexes..."
docker exec "$CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" \
  -f /var/lib/neo4j/import/a3_constraints.cypher

echo "[A3] waiting for indexes to come ONLINE..."
docker exec "$CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" \
  "CALL db.awaitIndexes(300);"

echo "[A3] writing variant marker..."
docker exec "$CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" \
  -f "/var/lib/neo4j/import/$META_CYPHER"

echo "[A3] running gate (a3_verify.py)..."
python scripts/a3_verify.py --variant "$VARIANT"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "[A3][FATAL] verification FAILED (exit $RC). The graph is loaded but NOT trusted." >&2
  echo "            Do not proceed to A7/A8 against it." >&2
  exit $RC
fi
echo "[A3] OK — variant '$VARIANT' imported, constrained, marked and verified."
