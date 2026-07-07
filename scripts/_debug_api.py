import json, os, requests, sys

HEADERS = {"Accept": "application/json", "User-Agent": "poker44-miner/1.0"}

# Debug: test releases endpoint
r = requests.get("https://api.poker44.net/api/v1/benchmark/releases?limit=5", headers=HEADERS, timeout=30)
d = r.json()
sys.stderr.write(f"DEBUG releases: status={r.status_code} keys={list(d.keys())} data_keys={list(d.get('data',{}).keys())}\n")
rels = d.get("data", {}).get("releases", [])
sys.stderr.write(f"DEBUG releases count: {len(rels)}\n")
for rel in rels[:3]:
    sys.stderr.write(f"  {rel.get('sourceDate')} chunks={rel.get('chunkCount')}\n")

# Test chunks endpoint
r2 = requests.get("https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-07&limit=24", headers=HEADERS, timeout=30)
d2 = r2.json()
cs = d2.get("data", {}).get("chunks", [])
sys.stderr.write(f"DEBUG chunks 2026-07-07: status={r2.status_code} count={len(cs)}\n")
if cs:
    sys.stderr.write(f"  first chunk keys: {list(cs[0].keys())}\n")

print(f"RELEASES={len(rels)} CHUNKS={len(cs)}")
