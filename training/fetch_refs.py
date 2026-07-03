"""Fetch official 30s preview clips of current chart tracks from Deezer's public
API — evaluation references only (never training data). Saved to training/refs/<genre>/.
"""
import sys
from pathlib import Path

import httpx

REFS = Path(__file__).parent / "refs"
# Deezer genre ids
GENRES = {"hip-hop": 116, "pop": 132, "rock": 152, "dance": 113, "rnb": 165, "latin": 197}
PER_GENRE = 8


def main():
    for name, gid in GENRES.items():
        out = REFS / name
        out.mkdir(parents=True, exist_ok=True)
        try:
            chart = httpx.get(f"https://api.deezer.com/chart/{gid}/tracks?limit=25", timeout=30).json()
        except Exception as e:
            print(f"{name}: chart fetch failed ({e})")
            continue
        got = 0
        for tr in chart.get("data", []):
            if got >= PER_GENRE:
                break
            url = tr.get("preview")
            if not url:
                continue
            dest = out / f"{tr['id']}.mp3"
            if not dest.exists():
                try:
                    dest.write_bytes(httpx.get(url, timeout=30, follow_redirects=True).content)
                except Exception:
                    continue
            got += 1
        print(f"{name}: {got} previews ({', '.join(t['artist']['name'] for t in chart.get('data', [])[:3])}...)")


if __name__ == "__main__":
    main()
