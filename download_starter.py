import pandas as pd
import subprocess
from pathlib import Path

df = pd.read_csv("data/ACCIDENT/starter_25.csv")
output_dir = Path("data/ACCIDENT/real_videos")
output_dir.mkdir(parents=True, exist_ok=True)

for path in df["path"]:
    filename = Path(path).name
    print(f"\nDownloading: {filename}")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "picekl/accident",
        "-f", path,
        "-p", str(output_dir)
    ], check=True)

print("\nAll 25 videos downloaded.")
