from pathlib import Path

import pandas as pd

import process_dong_code as pdc


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

DEFAULT_INPUT = RAW_DIR / "capital_region_land_ratio.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "dong_land_ratio.csv"


REQUIRED_COLUMNS = {
    "행정동코드": "dong_code",
    "주거용지비율": "residential_land_ratio",
    "상업업무용지비율": "commercial_business_land_ratio",
    "공공시설용지비율": "public_facility_land_ratio",
}


def process_land_ratio(input_path: str | Path = DEFAULT_INPUT, output_path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"processing land ratio data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in land ratio CSV: {missing_columns}")

    df = df[list(REQUIRED_COLUMNS)].rename(columns=REQUIRED_COLUMNS)
    df = pdc.check_dong(df, "dong_code")

    ratio_cols = [
        "residential_land_ratio",
        "commercial_business_land_ratio",
        "public_facility_land_ratio",
    ]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_dropna = len(df)
    df = df.dropna(subset=ratio_cols).copy()
    if len(df) != before_dropna:
        print(f"warning: dropped {before_dropna - len(df)} rows with missing ratio values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"saved: {output_path}")
    print(f"rows: {len(df)}, duplicate dong_code: {df['dong_code'].duplicated().sum()}")
    return df


if __name__ == "__main__":
    process_land_ratio()
