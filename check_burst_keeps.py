"""Check if Jay ever keeps multiple photos from the same 1-second burst."""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))
from data.mapping import flat_entries, load_mapping

from data.mapping import _read_cr3_ts, _ts_str_to_float

def get_timestamp_sec(path):
    """Return truncated-to-second string, or None."""
    ts = _read_cr3_ts(path)
    if ts is None:
        return None
    # Drop subsecond — we want the whole-second bucket
    return ts.split(".")[0]

mapping = load_mapping()
entries = [e for e in flat_entries(mapping) if e["raw"] and e["label"] is not None]
print(f"Total labeled entries with raw: {len(entries):,}")

# Group by (folder, timestamp-to-the-second)
burst_labels = defaultdict(list)
missing_ts = 0

for e in entries:
    folder = Path(e["raw"]).parent.name
    ts = get_timestamp_sec(e["raw"])
    if ts is None:
        missing_ts += 1
        continue
    burst_key = (folder, ts)
    burst_labels[burst_key].append(e["label"])

print(f"Missing timestamps: {missing_ts:,}")
print(f"Unique 1-second bursts: {len(burst_labels):,}")

keep_counts = Counter(sum(v) for v in burst_labels.values())
total = len(burst_labels)
print()
print("Keeps per 1-second burst:")
for n in sorted(keep_counts):
    pct = 100 * keep_counts[n] / total
    print(f"  {n} keep(s): {keep_counts[n]:,} bursts  ({pct:.1f}%)")

multi = {k: v for k, v in burst_labels.items() if sum(v) > 1}
print(f"\nBursts where Jay kept 2+: {len(multi):,}  ({100*len(multi)/total:.1f}%)")

if multi:
    print("\nSample multi-keep bursts:")
    for (folder, ts), labels in list(multi.items())[:5]:
        print(f"  {folder}  {ts}  -> {len(labels)} photos, {sum(labels)} kept")
