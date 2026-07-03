"""Download free, legally-trainable music from the Free Music Archive
(Creative Commons licensed, built for ML research).

  --source small   8,000 x 30s clips, ~7.4GB
  --source medium  25,000 x 30s clips, ~25GB (better pool for curation)

Usage: python get_free_data.py [--limit 600] [--genres "Hip-Hop,Electronic,Pop"] [--source medium]
Stages the `limit` MOST-LISTENED matching tracks into training/raw/ with genre
sidecars (play counts are the quality proxy — surfaces the well-produced stuff).
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
FMA = HERE / "fma"
RAW = HERE / "raw"
BASE = "https://os.unil.cloud.switch.ch/fma/"


def fetch(name):
    dest = FMA / name
    if not dest.exists():
        print(f"downloading {name} (this is big, be patient)...")
        FMA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(BASE + name, dest)
    marker = FMA / (name[:-4] + ".unzipped")
    if not marker.exists():
        print(f"unzipping {name}...")
        zipfile.ZipFile(dest).extractall(FMA)
        marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600, help="tracks to stage into raw/")
    ap.add_argument("--genres", default="", help="comma-separated genre_top filter, e.g. Hip-Hop,Electronic")
    ap.add_argument("--source", default="small", choices=["small", "medium"])
    args = ap.parse_args()
    wanted = {g.strip() for g in args.genres.split(",") if g.strip()}

    fetch("fma_metadata.zip")
    fetch(f"fma_{args.source}.zip")

    import pandas as pd

    tracks = pd.read_csv(FMA / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
    pool = tracks[[("track", "genre_top"), ("track", "listens")]].copy()
    pool.columns = ["genre", "listens"]
    if wanted:
        pool = pool[pool.genre.isin(wanted)]
    pool = pool.sort_values("listens", ascending=False)  # most-listened first

    audio_dirs = [d for d in (FMA / "fma_medium", FMA / "fma_small") if d.exists()]
    RAW.mkdir(exist_ok=True)
    staged = 0
    for tid, row in pool.iterrows():
        if staged >= args.limit:
            break
        name = f"{tid:06d}.mp3"
        for d in audio_dirs:
            src = d / name[:3] / name
            if src.exists():
                shutil.copy(src, RAW / name)
                (RAW / f"{tid:06d}.txt").write_text(str(row.genre), encoding="utf-8")
                staged += 1
                break
    print(f"{staged} CC-licensed tracks staged in {RAW} (top-listened first) — now run: python prep.py")


if __name__ == "__main__":
    main()
