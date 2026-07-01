"""How often does Jay keep more than 1 photo from the same burst?"""
import json
from collections import Counter
from pathlib import Path

seq_file = Path("embeddings/burst_sequences.json")
with open(seq_file) as f:
    seqs = json.load(f)

print(f"Total bursts in burst_sequences.json: {len(seqs):,}")

# Each seq: {"split": "train", "photos": [{"raw": ..., "label": 0|1}, ...]}
burst_stats = []
for seq in seqs:
    photos = seq["photos"]
    n_total = len(photos)
    n_kept  = sum(1 for p in photos if p.get("label") == 1)
    burst_stats.append((n_total, n_kept))

total_bursts = len(burst_stats)
kept_dist = Counter(nk for _, nk in burst_stats)
size_dist = Counter(sz for sz, _ in burst_stats)

print(f"Total photos across all bursts: {sum(sz for sz, _ in burst_stats):,}")
print()

print("Burst SIZE (raws shot per burst):")
for sz in sorted(size_dist)[:25]:
    pct = size_dist[sz] / total_bursts * 100
    bar = "#" * int(pct / 2)
    print(f"  {sz:3d}: {size_dist[sz]:5,}  ({pct:5.1f}%)  {bar}")
if max(size_dist) >= 25:
    rest = sum(v for k, v in size_dist.items() if k >= 25)
    print(f"  25+: {rest:5,}  ({rest/total_bursts*100:.1f}%)")

print()
print("KEEPERS per burst (how many Jay selects per burst):")
for n in sorted(kept_dist)[:20]:
    pct = kept_dist[n] / total_bursts * 100
    bar = "#" * int(pct / 2)
    print(f"  {n:3d}: {kept_dist[n]:5,}  ({pct:5.1f}%)  {bar}")
if max(kept_dist) >= 20:
    rest = sum(v for k, v in kept_dist.items() if k >= 20)
    print(f"  20+: {rest:5,}  ({rest/total_bursts*100:.1f}%)")

print()
n_0  = kept_dist.get(0, 0)
n_1  = kept_dist.get(1, 0)
n_2p = sum(v for k, v in kept_dist.items() if k >= 2)
print(f"Bursts where Jay keeps 0 photos:   {n_0:5,}  ({n_0/total_bursts*100:.1f}%)")
print(f"Bursts where Jay keeps exactly 1:  {n_1:5,}  ({n_1/total_bursts*100:.1f}%)")
print(f"Bursts where Jay keeps 2 or more:  {n_2p:5,}  ({n_2p/total_bursts*100:.1f}%)")

print()
avg_size = sum(sz for sz, _ in burst_stats) / total_bursts
avg_kept = sum(nk for _, nk in burst_stats) / total_bursts
kept_given_any = (
    sum(nk for _, nk in burst_stats if nk > 0)
    / sum(1 for _, nk in burst_stats if nk > 0)
    if any(nk > 0 for _, nk in burst_stats) else 0
)
print(f"Avg burst size: {avg_size:.1f} raws")
print(f"Avg keepers per burst: {avg_kept:.2f}")
print(f"Avg keepers per burst (when keeping anything): {kept_given_any:.2f}")
