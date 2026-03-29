# ==============================================================================
# 00 — Schema & Folds (MultilabelStratifiedKFold)
# ==============================================================================

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = Path(".")
WORK_DIR = Path("prepared")

N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 100_000

# ============================================================
# HELPERS
# ============================================================
def parquet_info(path: Path):
    pf = pq.ParquetFile(path)
    schema_arrow = pf.schema_arrow
    cols = schema_arrow.names
    n_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    return pf, schema_arrow, cols, n_rows, n_row_groups


def detect_customer_id(columns):
    if "customer_id" in columns:
        return "customer_id"
    candidates = [c for c in columns if c.lower() == "customer_id" or c.lower().endswith("customer_id")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot determine customer_id. Candidates: {candidates}")


def split_main_columns(schema_arrow, customer_id_col: str):
    cat_cols, num_cols, other_cols = [], [], []
    for field in schema_arrow:
        name = field.name
        if name == customer_id_col:
            continue
        low = name.lower()
        if low.startswith("cat_feature_"):
            cat_cols.append(name)
        elif low.startswith("num_feature_"):
            num_cols.append(name)
        else:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type) or pa.types.is_dictionary(field.type):
                cat_cols.append(name)
            elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type):
                num_cols.append(name)
            else:
                other_cols.append(name)
    return cat_cols, num_cols, other_cols


def read_target_matrix_batched(path: Path, customer_id_col: str, target_cols, batch_size=100_000):
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    n_targets = len(target_cols)

    y = np.empty((n_rows, n_targets), dtype=np.uint8)
    customer_ids = np.empty(n_rows, dtype=np.int64)

    start = 0
    for i, batch in enumerate(
        pf.iter_batches(columns=[customer_id_col] + target_cols, batch_size=batch_size), start=1
    ):
        batch_df = batch.to_pandas(types_mapper=pd.ArrowDtype)
        bs = len(batch_df)
        customer_ids[start:start+bs] = batch_df[customer_id_col].astype("int64").to_numpy(copy=False)
        y[start:start+bs] = batch_df[target_cols].astype("uint8", copy=False).to_numpy(copy=True)
        start += bs
        print(f"  Read batch {i:02d}: rows={bs:,} | accumulated={start:,}/{n_rows:,}")
        del batch, batch_df
        gc.collect()

    return customer_ids, y


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    print("🚀 [00] Schema & Folds — MultilabelStratifiedKFold")

    PATHS = {
        "train_main": DATA_PATH / "train_main_features.parquet",
        "train_extra": DATA_PATH / "train_extra_features.parquet",
        "train_target": DATA_PATH / "train_target.parquet",
        "test_main": DATA_PATH / "test_main_features.parquet",
        "test_extra": DATA_PATH / "test_extra_features.parquet",
        "sample_submit": DATA_PATH / "sample_submit.parquet",
    }

    CONFIG_DIR = WORK_DIR / "artifacts" / "config"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in PATHS.items():
        print(f"  {name:>14}: {path} | exists={path.exists()}")

    # --- Read schemas ---
    print("\n📋 Reading parquet schemas...")
    _, train_main_schema, train_main_cols, n_train_main, _ = parquet_info(PATHS["train_main"])
    _, _, train_extra_cols, n_train_extra, _ = parquet_info(PATHS["train_extra"])
    _, _, train_target_cols_raw, n_train_target, _ = parquet_info(PATHS["train_target"])
    _, _, test_main_cols, n_test_main, _ = parquet_info(PATHS["test_main"])
    _, _, test_extra_cols, n_test_extra, _ = parquet_info(PATHS["test_extra"])

    print(f"  train_main  : {n_train_main:,}")
    print(f"  train_extra : {n_train_extra:,}")
    print(f"  train_target: {n_train_target:,}")
    print(f"  test_main   : {n_test_main:,}")
    print(f"  test_extra  : {n_test_extra:,}")

    assert n_train_main == n_train_extra == n_train_target
    assert n_test_main == n_test_extra

    # --- Detect columns ---
    customer_id_col = detect_customer_id(train_main_cols)
    target_cols = [c for c in train_target_cols_raw if c != customer_id_col]
    extra_cols = [c for c in train_extra_cols if c != customer_id_col]
    main_cat_cols, main_num_cols, main_other_cols = split_main_columns(train_main_schema, customer_id_col)

    print(f"\n  customer_id : {customer_id_col}")
    print(f"  targets     : {len(target_cols)}")
    print(f"  main_cat    : {len(main_cat_cols)}")
    print(f"  main_num    : {len(main_num_cols)}")
    print(f"  extra       : {len(extra_cols)}")

    # --- Schema checks ---
    test_main_non_id = [c for c in test_main_cols if c != customer_id_col]
    test_extra_non_id = [c for c in test_extra_cols if c != customer_id_col]
    assert set(main_cat_cols + main_num_cols + main_other_cols) == set(test_main_non_id)
    assert set(extra_cols) == set(test_extra_non_id)
    print("✅ Schema checks passed.")

    # --- Save configs ---
    schema_payload = {
        "customer_id_col": customer_id_col,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "n_train_rows": int(n_train_target),
        "n_test_rows": int(n_test_main),
        "n_targets": len(target_cols),
        "n_main_cat_cols": len(main_cat_cols),
        "n_main_num_cols": len(main_num_cols),
        "n_extra_cols": len(extra_cols),
    }

    save_json(schema_payload, CONFIG_DIR / "schema.json")
    save_json(target_cols, CONFIG_DIR / "target_cols.json")
    save_json(main_cat_cols, CONFIG_DIR / "main_cat_cols.json")
    save_json(main_num_cols, CONFIG_DIR / "main_num_cols.json")
    save_json(extra_cols, CONFIG_DIR / "extra_cols.json")
    print("💾 Saved config JSONs to", CONFIG_DIR)

    # --- Read target matrix ---
    print("\n📥 Reading target matrix in batches...")
    customer_ids, y = read_target_matrix_batched(
        path=PATHS["train_target"],
        customer_id_col=customer_id_col,
        target_cols=target_cols,
        batch_size=BATCH_SIZE,
    )
    print(f"  y shape: {y.shape} | mean positives per row: {float(y.sum(axis=1).mean()):.4f}")

    # --- 5-fold multilabel stratified split ---
    print(f"\n🔀 Building {N_SPLITS}-fold MultilabelStratifiedKFold...")
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_assign = np.empty(len(customer_ids), dtype=np.int8)

    for fold, (_, val_idx) in enumerate(mskf.split(np.zeros(len(customer_ids)), y)):
        fold_assign[val_idx] = fold
        print(f"  Fold {fold}: {len(val_idx):,} rows")

    folds_df = pd.DataFrame({customer_id_col: customer_ids, "fold": fold_assign.astype("int8")})
    folds_path = CONFIG_DIR / "folds.parquet"
    folds_df.to_parquet(folds_path, index=False)
    print(f"💾 Saved: {folds_path}")

    # --- Fold quality checks ---
    global_mean = y.mean(axis=0)
    fold_values = folds_df["fold"].to_numpy()
    for fold in range(N_SPLITS):
        mask = fold_values == fold
        fold_mean = y[mask].mean(axis=0)
        abs_diff = np.abs(fold_mean - global_mean)
        print(f"  Fold {fold}: max_target_rate_diff={abs_diff.max():.6f}, mean_diff={abs_diff.mean():.6f}")

    del customer_ids, y, fold_assign, folds_df
    gc.collect()

    print("\n🎉 Done! Config artifacts are ready in:", CONFIG_DIR)


if __name__ == "__main__":
    main()
