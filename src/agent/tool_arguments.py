"""Resolve explicit earlier-step result references for recorded sequences."""
from copy import deepcopy


def has_reference(value):
    if isinstance(value, dict):
        return "$ref" in value or any(has_reference(v) for v in value.values())
    return isinstance(value, list) and any(has_reference(v) for v in value)


def resolve_references(value, results):
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                raise ValueError("A result reference must contain only a string $ref")
            parts = value["$ref"].split(".")
            try:
                item = results[parts[0]]
                for key in parts[1:]:
                    item = item[int(key)] if isinstance(item, list) else item[key]
                return deepcopy(item)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("Unknown earlier result reference: " + value["$ref"]) from exc
        return {k: resolve_references(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, results) for v in value]
    return value
