// A3 — constraints & indexes. Run AFTER import, with the database RUNNING.
//
//   constraint   a rule Neo4j enforces on write (here: no two :Node share an id).
//   index        a lookup structure that turns a full scan into a direct hit.
//
// WHY :Node and .id
//   A constraint on a label that does not exist, or on a property that does not exist,
//   is a SILENT NO-OP — it is created, it reports success, and it enforces nothing.
//   `id` is A2's verbatim key, measured globally unique across all nodes (gate G2).
//   The how-to's `id_curie` is not a column in canon_nodes.parquet.
//   `:Node` exists only because a3_emit_csvs.py emits ";Node" alongside each 3-letter
//   label (the importer splits :LABEL on ';'), so one constraint covers all ten types.
//   a3_verify.py proves the constraint is real with a duplicate-insert probe.

CREATE CONSTRAINT id_unique IF NOT EXISTS
FOR (n:Node) REQUIRE n.id IS UNIQUE;

// --- node lookup indexes ---------------------------------------------------
// `description` is NULL on every node (errata E4); `name` is the only display surface.
CREATE INDEX node_name      IF NOT EXISTS FOR (n:Node) ON (n.name);
CREATE INDEX node_id_prefix IF NOT EXISTS FOR (n:Node) ON (n.id_prefix);
CREATE INDEX drug_inchikey  IF NOT EXISTS FOR (n:DRG)  ON (n.inchi_key);
CREATE INDEX dis_umls       IF NOT EXISTS FOR (n:DIS)  ON (n.umls_cui);
CREATE INDEX gen_symbol     IF NOT EXISTS FOR (n:GEN)  ON (n.symbol);

// --- relationship indexes: deliberately NONE -------------------------------
// Neo4j 5 requires a relationship TYPE on a relationship property index:
//     FOR ()-[r]-() ON (r.prop)          <- INVALID, syntax error
//     FOR ()-[r:SOME_TYPE]-() ON (r.prop) <- valid
//
// Adding the type back is not the fix, because the composite TYPE already removed
// the problem the index was for:
//
//   * r.relation is CONSTANT within a type — every DRG_DIS__INDICATION edge has
//     relation='INDICATION' — so a per-type index on it indexes a single key.
//   * The cross-group question ("all INDICATION regardless of endpoint types") is
//     served by TYPE DISJUNCTION, which is backed by the relationship-type lookup
//     index Neo4j maintains automatically:
//         MATCH ()-[r:DRG_DIS__INDICATION|DRG_PHE__INDICATION|DRG_BPO__INDICATION]-()
//     That is an index-backed lookup, strictly better than a property scan.
//   * r.endpoint_key does vary within a type, but nothing queries it from Cypher —
//     A7's anti-join runs in Polars against canon_edges. 57 per-type indexes would
//     be cost with no consumer. Add one for a specific type if that ever changes.

// ---------------------------------------------------------------------------
// OPTION B — only if you did NOT emit the shared ";Node" label. Ten per-label
// constraints, used INSTEAD of id_unique above. Kept for the record; the frozen
// decision is the shared label (a3_decisions.json#node_label_scheme).
//
// CREATE CONSTRAINT gen_id IF NOT EXISTS FOR (n:GEN) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT dis_id IF NOT EXISTS FOR (n:DIS) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT drg_id IF NOT EXISTS FOR (n:DRG) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT bpo_id IF NOT EXISTS FOR (n:BPO) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT phe_id IF NOT EXISTS FOR (n:PHE) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT ana_id IF NOT EXISTS FOR (n:ANA) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT mfn_id IF NOT EXISTS FOR (n:MFN) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT cco_id IF NOT EXISTS FOR (n:CCO) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT pwy_id IF NOT EXISTS FOR (n:PWY) REQUIRE n.id IS UNIQUE;
// CREATE CONSTRAINT exp_id IF NOT EXISTS FOR (n:EXP) REQUIRE n.id IS UNIQUE;
