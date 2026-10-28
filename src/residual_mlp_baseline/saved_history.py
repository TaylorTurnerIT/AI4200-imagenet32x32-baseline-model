"""Portable, dataset-specific experiment metadata; never load checkpoint weights."""

import json
from pathlib import Path


HISTORY_DIR = Path(__file__).with_name("histories")


def read_history(task):
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError("Unknown dataset")
    path = HISTORY_DIR / f"{task}.json"
    if not path.exists():
        return {"task": task, "entries": [], "legacy_rows": []}
    data = json.loads(path.read_text())
    if data["task"] != task or any(entry["config"]["task"] != task for entry in data["entries"]):
        raise ValueError("Archived history contains another dataset")
    return data


def merge_archived_runs(task, local_entries):
    """Use local versions when available and retain portable historical settings."""
    entries = {entry["id"]: entry for entry in read_history(task)["entries"]}
    for entry in local_entries:
        if entry["config"]["task"] != task:
            raise ValueError("Local history contains another dataset")
        entries[entry["id"]] = {**entries.get(entry["id"], {}), **entry}
    return sorted(entries.values(), key=lambda entry: entry["id"])
