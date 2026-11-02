#!/usr/bin/env python3
# get_streamflow_obs.py
# Fetches ALL JSON files from data.lv.is/cryoscope/
# Produces one CSV per catchment ID with all unique dates merged
# Output: /home/ubuntu/data/cryoscope/streamflow_obs/<ID>.csv

import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import csv
import os

# ── CONFIG ────────────────────────────────────────────────────
BASE_URL = "https://data.lv.is/cryoscope/"
OUT_DIR  = "/home/ubuntu/data/cryoscope/streamflow_obs"
# ─────────────────────────────────────────────────────────────

def get_directory_listing():
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".json")]

def parse_filename(fname):
    try:
        parts = unquote(fname).replace(".json", "").split("_")
        return {"id": int(parts[2])}
    except Exception:
        return None

def main():
    print("Fetching directory listing...")
    all_files = get_directory_listing()
    print(f"Found {len(all_files)} JSON files\n")

    # Collect all data per ID: {id: {timestamp: value}}
    # Dict keyed by timestamp ensures no duplicates
    data_by_id = {}

    for i, fname in enumerate(all_files, 1):
        info = parse_filename(fname)
        if not info:
            print(f"  [{i:>4}/{len(all_files)}] Skipping (could not parse): {fname}")
            continue

        cid = info["id"]
        url = BASE_URL + fname

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rows = data[0]["data"]

            if cid not in data_by_id:
                data_by_id[cid] = {}

            for ts, value in rows:
                data_by_id[cid][ts] = value   # newer file overwrites older for same date

            print(f"  [{i:>4}/{len(all_files)}] ID {cid:>4} — {unquote(fname)} ({len(rows)} rows)")

        except Exception as e:
            print(f"  [{i:>4}/{len(all_files)}] ERROR {unquote(fname)}: {e}")

    print(f"\nFound data for {len(data_by_id)} unique catchment IDs")
    print(f"Writing CSVs to {OUT_DIR}\n")

    # Write one CSV per ID
    os.makedirs(OUT_DIR, exist_ok=True)

    for cid in sorted(data_by_id.keys()):
        out_path = os.path.join(OUT_DIR, f"{cid}.csv")
        rows     = sorted(data_by_id[cid].items())   # sort by timestamp

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Value"])
            writer.writerows(rows)

        print(f"  ID {cid:>4} → {out_path}  ({len(rows)} rows, "
              f"{rows[0][0][:10]} to {rows[-1][0][:10]})")

    print(f"\nDone. {len(data_by_id)} CSV files written to {OUT_DIR}")

if __name__ == "__main__":
    main()