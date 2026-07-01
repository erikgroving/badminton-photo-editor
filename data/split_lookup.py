"""
Builds a raw_path → split dict from mapping.json so that crop_gt.json and
color_gt.json records (which only store file paths, not splits) can be
partitioned into train/val/test without re-reading the raw folder structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.mapping import flat_entries, load_mapping


def build_split_lookup(mapping: dict | None = None) -> dict[str, str]:
    """Return {raw_path_str: split} for every entry with a raw file."""
    if mapping is None:
        mapping = load_mapping()
    return {e["raw"]: e["split"] for e in flat_entries(mapping) if e["raw"]}
