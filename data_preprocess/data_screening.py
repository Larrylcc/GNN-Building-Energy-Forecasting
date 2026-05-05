from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DATA_DIR = WORKSPACE_ROOT / "preprocessed_data"
INPUT_PATH = PREPROCESSED_DATA_DIR / "preprocessed_train.csv"
OUTPUT_PATH = PREPROCESSED_DATA_DIR / "screened_preprocessed_train.csv"
CHUNK_SIZE = 500_000

EXCLUDED_BUILDING_IDS = {
    803,
    801,
    799,
    1088,
    993,
    794,
    881,
    904,
    921,
    927,
    954,
    955,
    983,
    1168,
}


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    meter_series = pd.to_numeric(chunk["meter"], errors="coerce")
    building_id_series = pd.to_numeric(chunk["building_id"], errors="coerce")

    filtered_chunk = chunk.loc[
        (meter_series == 0) & (~building_id_series.isin(EXCLUDED_BUILDING_IDS))
    ].copy()
    return filtered_chunk


def screen_preprocessed_train(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, str | int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0

    chunk_iterator = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

    for chunk_index, chunk in enumerate(
        tqdm(chunk_iterator, desc="Screening preprocessed train", unit="chunk")
    ):
        filtered_chunk = filter_chunk(chunk)

        total_rows += len(chunk)
        kept_rows += len(filtered_chunk)

        filtered_chunk.to_csv(
            output_path,
            mode="w" if chunk_index == 0 else "a",
            header=chunk_index == 0,
            index=False,
        )

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows": total_rows - kept_rows,
    }


def main() -> dict[str, str | int]:
    result = screen_preprocessed_train()
    print(result)
    return result


if __name__ == "__main__":
    main()
