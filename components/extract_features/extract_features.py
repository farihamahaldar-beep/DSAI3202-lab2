import argparse
import os
import time
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

def parse_args():
    parser = argparse.ArgumentParser(description="Extract tsfresh features from turbofan time-series data")
    parser.add_argument("--input_train", type=str, required=True)
    parser.add_argument("--input_test", type=str, required=True)
    parser.add_argument("--input_labels", type=str, required=True)
    parser.add_argument("--output_train_features", type=str, required=True)
    parser.add_argument("--output_test_features", type=str, required=True)
    parser.add_argument("--output_labels", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=30)
    return parser.parse_args()


# Expected column names from Databricks notebook 05/06
EXPECTED_SENSOR_COLS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

ID_COLS = {"engine_id", "cycle", "entity_id", "sort_key",
           "rul", "target_rul"}


def detect_sensor_cols(df: pd.DataFrame) -> list:
    """
    Auto-detect sensor/feature columns from a DataFrame.
    Priority:
      1. Use EXPECTED_SENSOR_COLS if all present
      2. Use whichever expected cols are present (partial match)
      3. Fall back to all numeric columns excluding known ID columns
    """
    cols = set(df.columns)

    # Always print columns for debugging
    print(f"  DataFrame columns ({len(df.columns)}): {list(df.columns)}")

    # 1. All expected cols present
    if all(c in cols for c in EXPECTED_SENSOR_COLS):
        print(f"  Column detection: all {len(EXPECTED_SENSOR_COLS)} expected sensor columns found")
        return EXPECTED_SENSOR_COLS

    # 2. Partial match
    matched = [c for c in EXPECTED_SENSOR_COLS if c in cols]
    if len(matched) >= 5:
        missing = [c for c in EXPECTED_SENSOR_COLS if c not in cols]
        print(f"  Column detection: partial match — {len(matched)} cols found, missing: {missing}")
        return matched

    # 3. Fallback: all numeric non-ID columns
    numeric_cols = [
        c for c in df.select_dtypes(include=[float, int]).columns
        if c.lower() not in ID_COLS
    ]
    print(f"  Column detection: FALLBACK — using {len(numeric_cols)} numeric columns: {numeric_cols}")
    return numeric_cols


def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    dfs = [pd.read_parquet(os.path.join(folder_path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def detect_id_col(df: pd.DataFrame, candidates: list) -> str:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def build_rolling_windows(df: pd.DataFrame, window_size: int,
                          sensor_cols: list) -> pd.DataFrame:
    engine_col = detect_id_col(df, ["engine_id", "unit_nr", "unit", "id"])
    cycle_col  = detect_id_col(df, ["cycle", "time_in_cycles", "t"])

    records = []
    df = df.sort_values([engine_col, cycle_col]).reset_index(drop=True)

    for engine_id, grp in df.groupby(engine_col):
        grp = grp.reset_index(drop=True)
        cycles = grp[cycle_col].values
        n = len(cycles)

        for end_idx in range(window_size - 1, n):
            start_idx = end_idx - window_size + 1
            window = grp.iloc[start_idx: end_idx + 1].copy()
            end_cycle = cycles[end_idx]
            window["entity_id"] = f"{engine_id}_{end_cycle}"
            window["sort_key"] = range(window_size)
            records.append(window)

    return pd.concat(records, ignore_index=True)


def extract_tsfresh_features(df_windows: pd.DataFrame,
                             sensor_cols: list, n_jobs: int) -> pd.DataFrame:
    print(f"  Running tsfresh on {df_windows['entity_id'].nunique()} windows "
          f"with {len(sensor_cols)} sensors ...")
    t0 = time.time()

    extracted = extract_features(
        df_windows[["entity_id", "sort_key"] + sensor_cols],
        column_id="entity_id",
        column_sort="sort_key",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=n_jobs,
        show_warnings=False,
        disable_progressbar=False,
    )

    elapsed = time.time() - t0
    print(f"  tsfresh extraction done in {elapsed:.1f}s — shape: {extracted.shape}")
    impute(extracted)
    return extracted


def align_labels(labels_df: pd.DataFrame,
                 feature_index: pd.Index, window_size: int) -> pd.Series:
    engine_col = detect_id_col(labels_df, ["engine_id", "unit_nr", "unit", "id"])
    cycle_col  = detect_id_col(labels_df, ["cycle", "time_in_cycles", "t"])
    rul_col    = detect_id_col(labels_df, ["target_RUL", "RUL", "rul"])

    labels_df = labels_df.sort_values([engine_col, cycle_col])
    label_map = {}
    for _, row in labels_df.iterrows():
        key = f"{int(row[engine_col])}_{int(row[cycle_col])}"
        label_map[key] = row[rul_col]

    rul_values = []
    missing = 0
    for idx in feature_index:
        rul = label_map.get(str(idx))
        if rul is None:
            rul = np.nan
            missing += 1
        rul_values.append(rul)

    if missing > 0:
        print(f"  WARNING: {missing} feature rows could not be matched to RUL labels.")

    return pd.Series(rul_values, index=feature_index, name="target_RUL")


def main():
    args = parse_args()
    start_total = time.time()

    print("=" * 60)
    print("COMPONENT: extract_features")
    print("=" * 60)

    # --- Load ---
    print("\n[1/5] Loading input data ...")
    train_df  = load_parquet_folder(args.input_train)
    test_df   = load_parquet_folder(args.input_test)
    labels_df = load_parquet_folder(args.input_labels)

    print(f"\n  Train shape: {train_df.shape}")
    sensor_cols = detect_sensor_cols(train_df)

    print(f"\n  Test shape: {test_df.shape}")
    print(f"\n  Labels shape: {labels_df.shape}")
    print(f"  Label columns: {list(labels_df.columns)}")

    # --- Build rolling windows ---
    print(f"\n[2/5] Building rolling windows (window_size={args.window_size}) ...")
    t0 = time.time()
    train_windows = build_rolling_windows(train_df, args.window_size, sensor_cols)
    test_windows  = build_rolling_windows(test_df,  args.window_size, sensor_cols)
    print(f"  Train windows: {train_windows['entity_id'].nunique()} ({time.time()-t0:.1f}s)")

    # --- Extract features ---
    print("\n[3/5] Extracting tsfresh features (train) ...")
    train_features = extract_tsfresh_features(train_windows, sensor_cols, args.n_jobs)

    print("\n[4/5] Extracting tsfresh features (test) ...")
    test_features = extract_tsfresh_features(test_windows, sensor_cols, args.n_jobs)

    # --- Align labels ---
    print("\n[5/5] Aligning RUL labels ...")
    train_rul = align_labels(labels_df, train_features.index, args.window_size)

    valid_mask = train_rul.notna()
    train_features = train_features[valid_mask]
    train_rul = train_rul[valid_mask]
    print(f"  Final train feature rows: {len(train_features)}")
    print(f"  Final test  feature rows: {len(test_features)}")
    print(f"  Feature columns: {train_features.shape[1]}")

    # --- Save ---
    os.makedirs(args.output_train_features, exist_ok=True)
    os.makedirs(args.output_test_features,  exist_ok=True)
    os.makedirs(args.output_labels,         exist_ok=True)

    train_features.reset_index().rename(columns={"index": "entity_id"}).to_parquet(
        os.path.join(args.output_train_features, "data.parquet"), index=False
    )
    test_features.reset_index().rename(columns={"index": "entity_id"}).to_parquet(
        os.path.join(args.output_test_features, "data.parquet"), index=False
    )
    pd.DataFrame({"entity_id": train_rul.index, "target_RUL": train_rul.values}).to_parquet(
        os.path.join(args.output_labels, "data.parquet"), index=False
    )

    total_time = time.time() - start_total
    print(f"\n✅ extract_features complete in {total_time:.1f}s")
    print(f"   Train features → {args.output_train_features}")
    print(f"   Test features  → {args.output_test_features}")
    print(f"   RUL labels     → {args.output_labels}")


if __name__ == "__main__":
    main()