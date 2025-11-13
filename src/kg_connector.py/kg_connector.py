import os
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase  # removed RoutingControl
from marisa_trie import Trie  # optional

class KGConnector:
    """
    Neo4j connector helper with entity relationship retrieval.

    Notes:
    - Uses `elementId(...)` instead of deprecated `id(...)` in Cypher.
    - Driver is created lazily; calling methods after `close()` raises an error.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **driver_kwargs: Any,
    ):
        self.uri = uri or os.getenv("KG_URI")
        self.username = username or os.getenv("KG_USERNAME")
        self.password = password or os.getenv("KG_PASSWORD")
        self.database = database or os.getenv("KG_NAME", "neo4j")
        if not self.uri or not self.username or not self.password:
            raise EnvironmentError("KG_URI, KG_USERNAME and KG_PASSWORD must be set")

        self._driver = None  # create lazily
        self._driver_kwargs = driver_kwargs
        self._closed = False

    def _ensure_driver(self):
        if self._closed:
            raise RuntimeError("KGConnector driver has been closed; create a new KGConnector instance.")
        if self._driver is None:
            # create driver on first use
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), **self._driver_kwargs)

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None
        self._closed = True

    def __enter__(self) -> "KGConnector":
        # do not create driver until needed
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _use_execute_read(self, session, func, *args, **kwargs):
        # prefer execute_read (neo4j 4/5+), fallback to read_transaction
        if hasattr(session, "execute_read"):
            return session.execute_read(func, *args, **kwargs)
        return session.read_transaction(func, *args, **kwargs)

    def count_nodes(self) -> int:
        def _q(tx):
            res = tx.run("MATCH (n) RETURN count(n) AS total_nodes")
            row = res.single()
            return int(row["total_nodes"]) if row and row["total_nodes"] is not None else 0

        self._ensure_driver()
        with self._driver.session(database=self.database) as session:
            return self._use_execute_read(session, _q)

    def run_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        def _q(tx):
            res = tx.run(cypher, **params)
            return [record.data() for record in res]

        self._ensure_driver()
        with self._driver.session(database=self.database) as session:
            return self._use_execute_read(session, _q)

    def run_query_graph(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        params = params or {}
        nodes: Dict[Any, Dict[str, Any]] = {}
        rels: Dict[Any, Dict[str, Any]] = {}

        def _node_to_dict(n) -> Dict[str, Any]:
            nid = getattr(n, "element_id", None) if hasattr(n, "element_id") else None
            # If n doesn't expose element_id via driver object, use n.id replacement by caution:
            # but since we use elementId(...) in queries, the record will contain ids as strings already.
            try:
                props = dict(n)
                labels = list(getattr(n, "labels", []))
                # driver may not provide numeric id in records if we used elementId in cypher,
                # so handle both forms
                return {"id": nid or getattr(n, "id", None), "labels": labels, "properties": props}
            except Exception:
                return {"value": n}

        def _rel_to_dict(r) -> Dict[str, Any]:
            try:
                # records that include relation object may have attributes; best to return generic dict
                rid = getattr(r, "id", None)
                rtype = getattr(r, "type", getattr(r, "rel_type", None))
                props = dict(r)
                return {"id": rid, "type": rtype, "properties": props}
            except Exception:
                return {"value": r}

        def _q(tx):
            res = tx.run(cypher, **params)
            for record in res:
                for val in record.values():
                    if hasattr(val, "id") and hasattr(val, "items"):
                        nobj = _node_to_dict(val)
                        if nobj.get("id") is not None:
                            nodes[nobj["id"]] = nobj
                    elif hasattr(val, "type") or hasattr(val, "start_node"):
                        robj = _rel_to_dict(val)
                        if "id" in robj and robj["id"] is not None:
                            rels[robj["id"]] = robj
            return {"nodes": list(nodes.values()), "relationships": list(rels.values())}

        self._ensure_driver()
        with self._driver.session(database=self.database) as session:
            return self._use_execute_read(session, _q)

    def get_entity_connections(self, input_str: str) -> Dict[str, Any]:
        """
        Find entity by exact id or name and return its direct nodes and relationships.
        Use `elementId(...)` to avoid deprecated `id(...)`.

        Parameters:
        - input_str: str - entity id (as string) or exact name to search.

        Returns:
        - Dict with keys:
          - current_node: Dict with 'identity', 'labels', 'properties'
          - direct_node: List of Dicts with 'identity', 'labels', 'properties'
          - relations: List of Dicts with 'relation_name', 'start', 'end'

        Example:
        ```
        entity = kg.get_entity_connections("Huế")
        ```
        or better with:
        ```
        with KGConnector() as kg:
            entity_data = kg.get_entity_connections("Huế")
        ```
        -> ```{'current_node': {'identity': None, 'labels': ['Entity'], 'properties': {'name': 'Huế', 'id': 'Huế'}}, 'direct_node': [{'identity': 1307054, 'labels': ['Entity'], 'properties': {'name': 'Empire of Vietnam', 'id': 'Empire_of_Vietnam'}}, {'identity': 606675, 'labels': ['Entity'], 'properties': {'name': '"1993"', 'id': '"1993"'}}], 'relations': [{'relation_name': 'capital', 'start': '4:6de6b895-bb86-4aa5-9f9e-f625cd63cdad:1307054', 'end': '4:6de6b895-bb86-4aa5-9f9e-f625cd63cdad:966057'}, {'relation_name': 'year', 'start': '4:6de6b895-bb86-4aa5-9f9e-f625cd63cdad:966057', 'end': '4:6de6b895-bb86-4aa5-9f9e-f625cd63cdad:606675'}]}```
        """


        cypher = """
        MATCH (n:Entity)
        WHERE elementId(n) = $input OR n.name = $input
        MATCH (n)-[r]-(m:Entity)
        RETURN n AS current_node,
               collect(DISTINCT m) AS direct_node,
               collect({relation_name: type(r), start: elementId(startNode(r)), end: elementId(endNode(r))}) AS relations
        """

        def _q(tx):
            res = tx.run(cypher, input=input_str)
            row = res.single()
            if not row:
                return {}
            current_node = {
                "identity": row["current_node"] and (row["current_node"].get("elementId") if isinstance(row["current_node"], dict) else None)
                # if current_node is a node object, we may instead use properties mapped below
            }
            # But since we returned node object as `n`, we should extract properties directly:
            cur = row["current_node"]
            current_node = {
                "identity": row.get("current_node") and row.get("current_node").get("elementId", None) or None,
                "labels": list(cur.labels) if hasattr(cur, "labels") else [],
                "properties": dict(cur) if hasattr(cur, "items") else (cur or {})
            }

            direct_nodes = []
            for m in row["direct_node"]:
                direct_nodes.append({
                    "identity": (m.get("elementId") if isinstance(m, dict) else None) or getattr(m, "id", None),
                    "labels": list(m.labels) if hasattr(m, "labels") else [],
                    "properties": dict(m) if hasattr(m, "items") else (m or {})
                })

            relations = []
            for rel in row["relations"]:
                # relation entries built by cypher: relation_name, start, end (start/end are elementId strings)
                relations.append({
                    "relation_name": rel.get("relation_name"),
                    "start": rel.get("start"),
                    "end": rel.get("end")
                })

            return {
                "current_node": current_node,
                "direct_node": direct_nodes,
                "relations": relations
            }

        self._ensure_driver()
        with self._driver.session(database=self.database) as session:
            result = self._use_execute_read(session, _q)
            return result or {}

    def generate_trie(self, save_to: Optional[str] = None) -> Trie:
        def _q(tx):
            result = tx.run("MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name")
            return [record["name"] for record in result if record["name"]]

        self._ensure_driver()
        with self._driver.session(database=self.database) as session:
            entities = self._use_execute_read(session, _q)

        print(f"Total entities with names: {len(entities)}")
        trie = Trie(entities)
        if save_to:
            print(f"Saving trie to {save_to}")
            trie.save_to_file(save_to)
        return trie