from pathlib import Path

import pandas as pd


def default_od_dong_list_path() -> Path:
    return Path(__file__).resolve().parents[1] / "raw" / "OD_dong_list.xlsx"


def normalize_dong_code(series: pd.Series) -> pd.Series:
    code = pd.to_numeric(series, errors="coerce")
    return code.mask(code < 10_000_000, code * 10)


def check_dong(df: pd.DataFrame, dong_col_name: str, od_dong_list_path: str | Path | None = None) -> pd.DataFrame:
    if od_dong_list_path is None:
        od_dong_list_path = default_od_dong_list_path()

    od_dong_list = pd.read_excel(od_dong_list_path)
    valid_dongs = set(pd.to_numeric(od_dong_list["dong_code"], errors="coerce").dropna().astype(int))

    code_8digit = normalize_dong_code(df[dong_col_name])
    invalid_mask = code_8digit.isna() | ~code_8digit.astype("Int64").isin(valid_dongs)

    invalid_rows = df[invalid_mask]
    if not invalid_rows.empty:
        invalid_unique_codes = invalid_rows[dong_col_name].unique()
        print(f"warning: {len(invalid_rows)} rows have missing/non-OD dong codes and will be skipped.")
        print(f"invalid source codes: {invalid_unique_codes}")

    df_filtered = df[~invalid_mask].copy()
    df_filtered[dong_col_name] = code_8digit[~invalid_mask].astype(int)
    return df_filtered
