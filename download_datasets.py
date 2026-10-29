from datasets import load_dataset
from pathlib import Path
import pandas as pd

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

datasets_to_download = {
    "recruitment_data": "katsufu/recruitment_data",
    "candidate_matching": "michaelozon/candidate-matching-synthetic",
}

for prefix, dataset_name in datasets_to_download.items():
    print(f"Downloading {dataset_name}...")

    ds = load_dataset(dataset_name)

    for split, data in ds.items():
        output_file = output_dir / f"{prefix}_{split}.csv"
        data.to_pandas().to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

print("Done!")
