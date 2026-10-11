"""Model-neutral exclusive-state pairing for the semantic gated-delta contract."""
from __future__ import annotations

from collections import Counter


def recurrent_pairs(graph):
    """Only overwrite state whose complete consumer set is one exact operator pair."""
    operations = graph["operations"]
    consumers = Counter(name for op in operations for name in op["inputs"])
    initial = {op["id"] for op in operations if op["operator"] == "state_initialize"
               and op["attributes"].get("state_kind") == "recurrent"}
    initial.update(row["id"] for row in graph["state_inputs"] if row["kind"] == "recurrent")
    retained = {row["id"] for row in graph["state_outputs"] if row["kind"] == "recurrent"}
    updates = {tuple(op["inputs"]): op for op in operations
               if op["operator"] == "gated_delta_state_update" and op["id"] in retained}
    pairs = {}
    for rule in operations:
        if rule["operator"] != "gated_delta_rule" or len(rule["inputs"]) != 6:
            continue
        state = rule["inputs"][-1]
        update = updates.get(tuple(rule["inputs"]))
        if (state not in initial or consumers[state] != 2 or update is None
                or update["attributes"] != rule["attributes"]
                or update.get("layer") != rule.get("layer")):
            continue
        pair = (rule, update)
        pairs[rule["id"]] = pairs[update["id"]] = pair
    return pairs


def recurrent_workspace_bytes(graphs):
    """One shared float32 outer-product, query/key and delta workspace."""
    return max((4 * shape[1] * (shape[2] * shape[3] + 2 * shape[2] + shape[3])
                for graph in graphs for row in graph["state_outputs"]
                if row["kind"] == "recurrent" for shape in (row["shape"],)), default=0)
