"""Conserved post-close stage-row counters; no material, IDs or prompt data."""
def material_accounting(audit):
    names = ("issued", "reserved", "claimed", "burned", "discarded")
    values = {name: audit.get("prepared_stage_rows_" + name) for name in names}
    if any(type(value) is not int or value < 0 for value in values.values()):
        return {"schema": "pllm.prepared_material_accounting.v1", "scope": "sealed stage rows",
                "counts": values, "conserved": None, "remaining_stage_rows": None}
    remaining = values["issued"] - sum(values[name] for name in ("claimed", "burned", "discarded"))
    return {"schema": "pllm.prepared_material_accounting.v1", "scope": "sealed stage rows after client close; claims burn on use, not successful output",
            "counts": values, "remaining_stage_rows": remaining,
            "conserved": remaining == 0 and values["reserved"] == values["claimed"] + values["burned"]}
