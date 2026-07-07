"""Fetch all Poker44 benchmark chunks for model training."""
import json, os, requests

HEADERS = {"Accept": "application/json", "User-Agent": "poker44-miner/1.0"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(REPO, "train_data")
os.makedirs(train_dir, exist_ok=True)

# Get all release dates
r = requests.get("https://api.poker44.net/api/v1/benchmark/releases?limit=50", headers=HEADERS, timeout=30)
rels = r.json().get("data", {}).get("releases", [])
dates = [rel["sourceDate"] for rel in rels if rel.get("chunkCount", 0) > 0]
print(f"Found {len(dates)} release dates: {dates[0]} ... {dates[-1]}")

all_chunks = []
for date in dates:
    cursor = None
    count = 0
    while True:
        url = f"https://api.poker44.net/api/v1/benchmark/chunks?sourceDate={date}&limit=24"
        if cursor:
            url += f"&cursor={cursor}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        cs = resp.json().get("data", {}).get("chunks", [])
        if not cs:
            break
        all_chunks.extend(cs)
        count += len(cs)
        cursor = resp.json().get("data", {}).get("nextCursor")
        if not cursor:
            break
    print(f"  {date}: {count} chunks")

bot = sum(1 for ch in all_chunks for g in (ch.get("groundTruth") or []) if g == 1)
human = sum(1 for ch in all_chunks for g in (ch.get("groundTruth") or []) if g == 0)
print(f"total: {len(all_chunks)} chunks, {bot}B + {human}H = {bot+human} examples")

# Save
batch, fn = [], 0
for ch in all_chunks:
    batch.append(ch)
    if len(batch) >= 24:
        with open(os.path.join(train_dir, f"chunks_{fn:04d}.json"), "w") as f:
            json.dump({"chunks": batch}, f)
        fn += 1
        batch = []
if batch:
    with open(os.path.join(train_dir, f"chunks_{fn:04d}.json"), "w") as f:
        json.dump({"chunks": batch}, f)
    fn += 1
print(f"saved {fn} files")
