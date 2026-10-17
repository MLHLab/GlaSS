
# ============================================================
# STAGE 3: SEQUENTIAL REJECTION AND FINAL RETAINED DATASET
# ============================================================
# Input:
#   Glacier_Results/Stage2_SSL_GridSearch.pkl
#
# Output:
#   Glacier_Results/Stage3_Retained_FinalCV.pkl
#
# This stage:
#   1. Loads the completed Stage 2 checkpoint.
#   2. Recreates the five disjoint partitions using the SAME
#      random states used in the original workflow.
#   3. Applies sequential rejection in the order:
#         P5 -> P4 -> P3 -> P2 -> P1
#   4. For each partition, trains the stacking regressor on the
#      other four partitions, using refined versions of partitions
#      already processed.
#   5. Rejects samples whose absolute prediction error is above
#      the partition-specific 70th percentile.
#   6. Builds the final retained and rejected datasets.
#
# FINAL 5-FOLD CV is intentionally NOT performed here.
# It is performed in Stage 4.
# ============================================================

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Glacier_Results"

STAGE2_FILE = OUTPUT_DIR / "Stage2_SSL_GridSearch.pkl"
STAGE3_FILE = OUTPUT_DIR / "Stage3_Retained_FinalCV.pkl"

if not STAGE2_FILE.exists():
    raise FileNotFoundError(
        f"\nStage 2 checkpoint was not found:\n{STAGE2_FILE}\n\n"
        "Please run the corrected Stage 2 script first."
    )

print("=" * 75)
print("STAGE 3: SEQUENTIAL REJECTION AND FINAL RETAINED DATASET")
print("=" * 75)
print("Loading:", STAGE2_FILE)


# ============================================================
# 2. LOAD STAGE 2
# ============================================================

with open(STAGE2_FILE, "rb") as f:
    stage2 = pickle.load(f)

print("\nStage 2 checkpoint loaded successfully.")
print("Available objects:", len(stage2))


# ============================================================
# 3. REQUIRED OBJECTS FROM STAGE 2
# ============================================================

required_objects = [
    "X_embd_a",
    "y0",
    "X00",
    "y00",
]

missing_objects = [
    name for name in required_objects
    if name not in stage2
]

if missing_objects:
    raise KeyError(
        "\nThe Stage 2 checkpoint is missing:\n"
        + "\n".join(f"  - {x}" for x in missing_objects)
        + "\n\nPlease rerun the corrected Stage 2 script."
    )

X_embd_a = np.asarray(stage2["X_embd_a"], dtype=float)
y0 = np.asarray(stage2["y0"], dtype=float)

X00 = stage2["X00"].copy()
y00 = stage2["y00"].copy()

print("X_embd_a shape:", X_embd_a.shape)
print("y0 shape:", y0.shape)
print("X00 shape:", X00.shape)
print("y00 shape:", y00.shape)


# ============================================================
# 4. DETERMINE FINAL STACKING PARAMETERS
# ============================================================
# Prefer the actual GridSearch-selected parameters saved by
# Stage 2. If the Stage 2 checkpoint contains FINAL_* parameters,
# those are used. Otherwise BEST_* parameters are used.
#
# This makes Stage 3 compatible with the corrected Stage 2
# checkpoint and avoids hard-coding parameters that may differ
# from the completed GridSearch.
# ============================================================

if "FINAL_GBR_PARAMS" in stage2:
    FINAL_GBR_PARAMS = dict(stage2["FINAL_GBR_PARAMS"])
elif "BEST_GBR_PARAMS" in stage2:
    FINAL_GBR_PARAMS = dict(stage2["BEST_GBR_PARAMS"])
else:
    FINAL_GBR_PARAMS = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
    }

if "FINAL_XGB_PARAMS" in stage2:
    FINAL_XGB_PARAMS = dict(stage2["FINAL_XGB_PARAMS"])
elif "BEST_XGB_PARAMS" in stage2:
    FINAL_XGB_PARAMS = dict(stage2["BEST_XGB_PARAMS"])
else:
    FINAL_XGB_PARAMS = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

print("\nParameters used for rejection-stage stacking model:")
print("GBR:", FINAL_GBR_PARAMS)
print("XGBR:", FINAL_XGB_PARAMS)


# ============================================================
# 5. STACKING REGRESSOR
# ============================================================

def create_stacking_regressor():
    """
    Create the GBR + XGBR + LR stacking regressor.

    The GBR and XGBR parameters come from the completed
    Stage 2 GridSearch whenever available.
    """

    gbr = GradientBoostingRegressor(
        **FINAL_GBR_PARAMS,
        random_state=42
    )

    xgb = xgboost.XGBRegressor(
        **FINAL_XGB_PARAMS,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    lr = LinearRegression()

    return StackingRegressor(
        estimators=[
            ("gbr", gbr),
            ("xgb", xgb)
        ],
        final_estimator=lr,
        n_jobs=-1
    )


# ============================================================
# 6. HIGH-ERROR SAMPLE REJECTION FUNCTION
# ============================================================

def reject_high_error_samples(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    original_X,
    original_y,
    percentile=70
):
    """
    Reject samples whose absolute prediction error is above
    the specified percentile of the held-out partition's own
    error distribution.

    This implements the percentile-based refinement described
    in the manuscript.
    """

    # Predict held-out partition.
    y_pred_test = model.predict(X_test)

    y_test_array = np.asarray(y_test, dtype=float)
    y_pred_array = np.asarray(y_pred_test, dtype=float)

    # Absolute prediction error.
    absolute_error = np.abs(
        y_test_array - y_pred_array
    )

    # Partition-specific threshold.
    threshold = np.percentile(
        absolute_error,
        percentile
    )

    # IMPORTANT:
    # The manuscript says a sample is discarded if its error
    # EXCEEDS the 70th percentile. Therefore equality is retained.
    rejected_indices = np.where(
        absolute_error > threshold
    )[0]

    retained_mask = np.ones(
        len(y_test_array),
        dtype=bool
    )

    retained_mask[rejected_indices] = False

    # Model-space retained/rejected arrays.
    X_test_sr = np.asarray(X_test)[retained_mask]
    y_test_sr = y_test_array[retained_mask]

    X_test_r = np.asarray(X_test)[rejected_indices]
    y_test_r = y_test_array[rejected_indices]

    # Original untransformed data corresponding to this partition.
    original_X_reset = original_X.reset_index(drop=True)
    original_y_reset = original_y.reset_index(drop=True)

    original_X_retained = original_X_reset.iloc[
        np.where(retained_mask)[0]
    ].copy()

    original_y_retained = original_y_reset.iloc[
        np.where(retained_mask)[0]
    ].copy()

    original_X_rejected = original_X_reset.iloc[
        rejected_indices
    ].copy()

    original_y_rejected = original_y_reset.iloc[
        rejected_indices
    ].copy()

    sr_df = pd.DataFrame({
        "y_test": y_test_array,
        "y_pred_sr": y_pred_array,
        "absolute_error": absolute_error,
        "retained": retained_mask
    })

    return {
        "indices": rejected_indices.tolist(),
        "threshold": threshold,
        "sr_df": sr_df,

        "X_sr": X_test_sr,
        "y_sr": y_test_sr,

        "X_rejected": X_test_r,
        "y_rejected": y_test_r,

        "X_original_retained": original_X_retained,
        "y_original_retained": original_y_retained,

        "X_original_rejected": original_X_rejected,
        "y_original_rejected": original_y_rejected
    }


# ============================================================
# 7. RECREATE THE FIVE PARTITIONS
# ============================================================
# The same split sequence and random states are used so that
# the embedded arrays and original dataframe remain aligned.
#
# P1 = 20%
# Remaining 80% -> two 40% groups
# Each 40% group -> two 20% partitions
#
# Final partition order for rejection:
# P5 -> P4 -> P3 -> P2 -> P1
# ============================================================

print("\n" + "=" * 75)
print("RECREATING FIVE DATA PARTITIONS")
print("=" * 75)

# ---- Embedded/reconstructed feature space ----

X_train_80, X_P1, y_train_80, y_P1 = train_test_split(
    X_embd_a,
    y0,
    test_size=0.2,
    random_state=42
)

X_train_A, X_train_B, y_train_A, y_train_B = train_test_split(
    X_train_80,
    y_train_80,
    test_size=0.5,
    random_state=43
)

X_P2, X_P3, y_P2, y_P3 = train_test_split(
    X_train_A,
    y_train_A,
    test_size=0.5,
    random_state=44
)

X_P4, X_P5, y_P4, y_P5 = train_test_split(
    X_train_B,
    y_train_B,
    test_size=0.5,
    random_state=45
)


# ---- Corresponding original/reduced dataframe space ----

X00_train_80, X0_P1, y00_train_80, y0_P1 = train_test_split(
    X00,
    y00,
    test_size=0.2,
    random_state=42
)

X00_train_A, X00_train_B, y00_train_A, y00_train_B = train_test_split(
    X00_train_80,
    y00_train_80,
    test_size=0.5,
    random_state=43
)

X0_P2, X0_P3, y0_P2, y0_P3 = train_test_split(
    X00_train_A,
    y00_train_A,
    test_size=0.5,
    random_state=44
)

X0_P4, X0_P5, y0_P4, y0_P5 = train_test_split(
    X00_train_B,
    y00_train_B,
    test_size=0.5,
    random_state=45
)


print("\nInitial partition sizes:")
for name, Xp, yp in [
    ("P1", X_P1, y_P1),
    ("P2", X_P2, y_P2),
    ("P3", X_P3, y_P3),
    ("P4", X_P4, y_P4),
    ("P5", X_P5, y_P5),
]:
    print(
        f"{name}: X={Xp.shape}, y={yp.shape}"
    )


# ============================================================
# 8. FUNCTION TO PROCESS ONE PARTITION
# ============================================================

def process_partition(
    partition_name,
    X_train_parts,
    y_train_parts,
    X_test,
    y_test,
    original_X,
    original_y,
):
    """
    Train on the supplied four training components, predict the
    held-out partition, and perform 70th-percentile rejection.
    """

    print("\n" + "-" * 75)
    print(f"PROCESSING {partition_name}")
    print("-" * 75)

    X_train = np.concatenate(
        X_train_parts,
        axis=0
    )

    y_train = np.concatenate(
        y_train_parts,
        axis=0
    )

    print("Training shape:", X_train.shape)
    print("Held-out shape:", X_test.shape)

    model = create_stacking_regressor()

    model.fit(
        X_train,
        y_train
    )

    # Initial performance before rejection.
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\nBefore rejection:")
    print(
        f"Training R² = "
        f"{r2_score(y_train, y_pred_train):.6f}"
    )
    print(
        f"Testing R²  = "
        f"{r2_score(y_test, y_pred_test):.6f}"
    )
    print(
        f"Testing MAE = "
        f"{mean_absolute_error(y_test, y_pred_test):.6f}"
    )
    print(
        f"Testing RMSE = "
        f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.6f}"
    )

    result = reject_high_error_samples(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        original_X,
        original_y,
        percentile=70
    )

    print(
        f"\n70th-percentile error threshold = "
        f"{result['threshold']:.6f}"
    )

    print(
        "Retained samples:",
        len(result["y_sr"])
    )

    print(
        "Rejected samples:",
        len(result["y_rejected"])
    )

    # Performance after rejection.
    if len(result["y_sr"]) > 0:

        y_pred_retained = model.predict(
            result["X_sr"]
        )

        print("\nAfter rejection:")
        print(
            f"Retained R² = "
            f"{r2_score(result['y_sr'], y_pred_retained):.6f}"
        )
        print(
            f"Retained MAE = "
            f"{mean_absolute_error(result['y_sr'], y_pred_retained):.6f}"
        )
        print(
            f"Retained RMSE = "
            f"{np.sqrt(mean_squared_error(result['y_sr'], y_pred_retained)):.6f}"
        )

    return result


# ============================================================
# 9. SEQUENTIAL REJECTION
# ============================================================
# Exact sequence:
#
#   P5:
#       train = P1 + P2 + P3 + P4
#
#   P4:
#       train = P1 + P2 + retained(P5) 
#
#   P3:
#       train = P1 + P2 + retained(P4) + retained(P5)
#
#   P2:
#       train = P1 + retained(P3) + retained(P4) + retained(P5)
#
#   P1:
#       train = retained(P2) + retained(P3) +
#               retained(P4) + retained(P5)
# ============================================================

# ------------------------------------------------------------
# 9A. P5
# ------------------------------------------------------------

P5_result = process_partition(
    "P5",
    [X_P1, X_P2, X_P3, X_P4],
    [y_P1, y_P2, y_P3, y_P4],
    X_P5,
    y_P5,
    X0_P5,
    y0_P5
)

X_P5_sr = P5_result["X_sr"]
y_P5_sr = P5_result["y_sr"]

X_P5_r = P5_result["X_rejected"]
y_P5_r = P5_result["y_rejected"]

X0_P5_2 = P5_result["X_original_retained"]
y0_P5_2 = P5_result["y_original_retained"]

X0_P5_2r = P5_result["X_original_rejected"]
y0_P5_2r = P5_result["y_original_rejected"]


# ------------------------------------------------------------
# 9B. P4
# ------------------------------------------------------------

P4_result = process_partition(
    "P4",
    [X_P1, X_P2, X_P5_sr],
    [y_P1, y_P2, y_P5_sr],
    X_P4,
    y_P4,
    X0_P4,
    y0_P4
)

X_P4_sr = P4_result["X_sr"]
y_P4_sr = P4_result["y_sr"]

X_P4_r = P4_result["X_rejected"]
y_P4_r = P4_result["y_rejected"]

X0_P4_2 = P4_result["X_original_retained"]
y0_P4_2 = P4_result["y_original_retained"]

X0_P4_2r = P4_result["X_original_rejected"]
y0_P4_2r = P4_result["y_original_rejected"]


# ------------------------------------------------------------
# 9C. P3
# ------------------------------------------------------------

P3_result = process_partition(
    "P3",
    [X_P1, X_P2, X_P4_sr, X_P5_sr],
    [y_P1, y_P2, y_P4_sr, y_P5_sr],
    X_P3,
    y_P3,
    X0_P3,
    y0_P3
)

X_P3_sr = P3_result["X_sr"]
y_P3_sr = P3_result["y_sr"]

X_P3_r = P3_result["X_rejected"]
y_P3_r = P3_result["y_rejected"]

X0_P3_2 = P3_result["X_original_retained"]
y0_P3_2 = P3_result["y_original_retained"]

X0_P3_2r = P3_result["X_original_rejected"]
y0_P3_2r = P3_result["y_original_rejected"]


# ------------------------------------------------------------
# 9D. P2
# ------------------------------------------------------------

P2_result = process_partition(
    "P2",
    [X_P1, X_P3_sr, X_P4_sr, X_P5_sr],
    [y_P1, y_P3_sr, y_P4_sr, y_P5_sr],
    X_P2,
    y_P2,
    X0_P2,
    y0_P2
)

X_P2_sr = P2_result["X_sr"]
y_P2_sr = P2_result["y_sr"]

X_P2_r = P2_result["X_rejected"]
y_P2_r = P2_result["y_rejected"]

X0_P2_2 = P2_result["X_original_retained"]
y0_P2_2 = P2_result["y_original_retained"]

X0_P2_2r = P2_result["X_original_rejected"]
y0_P2_2r = P2_result["y_original_rejected"]


# ------------------------------------------------------------
# 9E. P1
# ------------------------------------------------------------

P1_result = process_partition(
    "P1",
    [X_P2_sr, X_P3_sr, X_P4_sr, X_P5_sr],
    [y_P2_sr, y_P3_sr, y_P4_sr, y_P5_sr],
    X_P1,
    y_P1,
    X0_P1,
    y0_P1
)

X_P1_sr = P1_result["X_sr"]
y_P1_sr = P1_result["y_sr"]

X_P1_r = P1_result["X_rejected"]
y_P1_r = P1_result["y_rejected"]

X0_P1_2 = P1_result["X_original_retained"]
y0_P1_2 = P1_result["y_original_retained"]

X0_P1_2r = P1_result["X_original_rejected"]
y0_P1_2r = P1_result["y_original_rejected"]


# ============================================================
# 10. FINAL RETAINED DATASET
# ============================================================

print("\n" + "=" * 75)
print("FINAL RETAINED DATASET")
print("=" * 75)

# Embedded/reconstructed feature representation.
X_merge = np.concatenate(
    [
        X_P1_sr,
        X_P2_sr,
        X_P3_sr,
        X_P4_sr,
        X_P5_sr
    ],
    axis=0
)

y_merge = np.concatenate(
    [
        y_P1_sr,
        y_P2_sr,
        y_P3_sr,
        y_P4_sr,
        y_P5_sr
    ],
    axis=0
)

# Original/reduced predictor dataframe.
X0_merge = pd.concat(
    [
        X0_P1_2,
        X0_P2_2,
        X0_P3_2,
        X0_P4_2,
        X0_P5_2
    ],
    ignore_index=True
)

y0_merge = pd.concat(
    [
        y0_P1_2,
        y0_P2_2,
        y0_P3_2,
        y0_P4_2,
        y0_P5_2
    ],
    ignore_index=True
)

# Original/reduced rejected dataframe.
X0_rejected = pd.concat(
    [
        X0_P1_2r,
        X0_P2_2r,
        X0_P3_2r,
        X0_P4_2r,
        X0_P5_2r
    ],
    ignore_index=True
)

y0_rejected = pd.concat(
    [
        y0_P1_2r,
        y0_P2_2r,
        y0_P3_2r,
        y0_P4_2r,
        y0_P5_2r
    ],
    ignore_index=True
)


print("\nFinal retained embedded X:", X_merge.shape)
print("Final retained y:", y_merge.shape)

print("\nFinal retained original/reduced X:", X0_merge.shape)
print("Final retained original/reduced y:", y0_merge.shape)

print("\nFinal rejected X:", X0_rejected.shape)
print("Final rejected y:", y0_rejected.shape)

print(
    "\nTotal rejected:",
    len(y0_rejected)
)

print(
    "Total retained:",
    len(y0_merge)
)

print(
    "Original observations:",
    len(y00)
)

# Consistency check.
assert len(y0_merge) + len(y0_rejected) == len(y00), (
    "Retained + rejected observations do not equal the original "
    "number of observations."
)


# ============================================================
# 11. REJECTION SUMMARY
# ============================================================

rejection_summary = pd.DataFrame([
    {
        "Partition": "P5",
        "Original": len(y_P5),
        "Retained": len(y_P5_sr),
        "Rejected": len(y_P5_r),
        "Threshold": P5_result["threshold"]
    },
    {
        "Partition": "P4",
        "Original": len(y_P4),
        "Retained": len(y_P4_sr),
        "Rejected": len(y_P4_r),
        "Threshold": P4_result["threshold"]
    },
    {
        "Partition": "P3",
        "Original": len(y_P3),
        "Retained": len(y_P3_sr),
        "Rejected": len(y_P3_r),
        "Threshold": P3_result["threshold"]
    },
    {
        "Partition": "P2",
        "Original": len(y_P2),
        "Retained": len(y_P2_sr),
        "Rejected": len(y_P2_r),
        "Threshold": P2_result["threshold"]
    },
    {
        "Partition": "P1",
        "Original": len(y_P1),
        "Retained": len(y_P1_sr),
        "Rejected": len(y_P1_r),
        "Threshold": P1_result["threshold"]
    }
])

print("\nRejection summary:")
print(rejection_summary.to_string(index=False))


# ============================================================
# 12. SAVE CSV FILES
# ============================================================

RETAINED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Correlation_Reduced_Retained.csv"
)

REJECTED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Correlation_Reduced_Rejected_Rows.csv"
)

X_RETAINED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Retained_X.csv"
)

Y_RETAINED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Retained_y.csv"
)

X_EMBEDDED_RETAINED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Embedded_Retained_X.csv"
)

Y_EMBEDDED_RETAINED_CSV = OUTPUT_DIR / (
    "GlacierDataset_170Vars_Embedded_Retained_y.csv"
)

REJECTION_SUMMARY_CSV = OUTPUT_DIR / (
    "Sequential_Rejection_Summary.csv"
)

X0_merge.to_csv(
    RETAINED_CSV,
    index=False
)

pd.concat(
    [X0_rejected, y0_rejected.rename(y00.name)],
    axis=1
).to_csv(
    REJECTED_CSV,
    index=False
)

X0_merge.to_csv(
    X_RETAINED_CSV,
    index=False
)

y0_merge.to_csv(
    Y_RETAINED_CSV,
    index=False
)

pd.DataFrame(
    X_merge,
    columns=[
        f"Embedded_{i + 1}"
        for i in range(X_merge.shape[1])
    ]
).to_csv(
    X_EMBEDDED_RETAINED_CSV,
    index=False
)

pd.DataFrame({
    "y": y_merge
}).to_csv(
    Y_EMBEDDED_RETAINED_CSV,
    index=False
)

rejection_summary.to_csv(
    REJECTION_SUMMARY_CSV,
    index=False
)


# ============================================================
# 13. SAVE STAGE 3 CHECKPOINT
# ============================================================

stage3 = {
    # Carry all Stage 2 objects forward.
    **stage2,

    # Final embedded/reconstructed retained dataset.
    "X_merge": X_merge,
    "y_merge": y_merge,

    # Final original/reduced retained dataset.
    "X0_merge": X0_merge,
    "y0_merge": y0_merge,

    # Final original/reduced rejected dataset.
    "X0_rejected": X0_rejected,
    "y0_rejected": y0_rejected,

    # Partition-specific retained embedded datasets.
    "X_P1_sr": X_P1_sr,
    "y_P1_sr": y_P1_sr,
    "X_P2_sr": X_P2_sr,
    "y_P2_sr": y_P2_sr,
    "X_P3_sr": X_P3_sr,
    "y_P3_sr": y_P3_sr,
    "X_P4_sr": X_P4_sr,
    "y_P4_sr": y_P4_sr,
    "X_P5_sr": X_P5_sr,
    "y_P5_sr": y_P5_sr,

    # Partition-specific rejected embedded datasets.
    "X_P1_r": X_P1_r,
    "y_P1_r": y_P1_r,
    "X_P2_r": X_P2_r,
    "y_P2_r": y_P2_r,
    "X_P3_r": X_P3_r,
    "y_P3_r": y_P3_r,
    "X_P4_r": X_P4_r,
    "y_P4_r": y_P4_r,
    "X_P5_r": X_P5_r,
    "y_P5_r": y_P5_r,

    # Partition-specific original retained datasets.
    "X0_P1_2": X0_P1_2,
    "y0_P1_2": y0_P1_2,
    "X0_P2_2": X0_P2_2,
    "y0_P2_2": y0_P2_2,
    "X0_P3_2": X0_P3_2,
    "y0_P3_2": y0_P3_2,
    "X0_P4_2": X0_P4_2,
    "y0_P4_2": y0_P4_2,
    "X0_P5_2": X0_P5_2,
    "y0_P5_2": y0_P5_2,

    # Partition-specific original rejected datasets.
    "X0_P1_2r": X0_P1_2r,
    "y0_P1_2r": y0_P1_2r,
    "X0_P2_2r": X0_P2_2r,
    "y0_P2_2r": y0_P2_2r,
    "X0_P3_2r": X0_P3_2r,
    "y0_P3_2r": y0_P3_2r,
    "X0_P4_2r": X0_P4_2r,
    "y0_P4_2r": y0_P4_2r,
    "X0_P5_2r": X0_P5_2r,
    "y0_P5_2r": y0_P5_2r,

    # Rejection diagnostics.
    "P1_result": P1_result,
    "P2_result": P2_result,
    "P3_result": P3_result,
    "P4_result": P4_result,
    "P5_result": P5_result,
    "rejection_summary": rejection_summary,

    # Parameters actually used.
    "FINAL_GBR_PARAMS": FINAL_GBR_PARAMS,
    "FINAL_XGB_PARAMS": FINAL_XGB_PARAMS,
}

with open(
    STAGE3_FILE,
    "wb"
) as f:
    pickle.dump(
        stage3,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

required_stage3_objects = [
    "X_merge",
    "y_merge",
    "X0_merge",
    "y0_merge",
    "X0_rejected",
    "y0_rejected",
    "rejection_summary",
]

missing_stage3 = [
    name
    for name in required_stage3_objects
    if name not in stage3
]

if missing_stage3:
    raise RuntimeError(
        "Stage 3 checkpoint is missing: "
        + str(missing_stage3)
    )

print("\n" + "=" * 75)
print("STAGE 3 COMPLETE")
print("=" * 75)

print(
    "\nCheckpoint saved to:"
)
print(STAGE3_FILE)

print(
    "\nFinal retained observations:",
    len(y0_merge)
)

print(
    "Final rejected observations:",
    len(y0_rejected)
)

print(
    "Total observations:",
    len(y00)
)

print(
    "\nNext step:"
)
print(
    "Run 04_FinalCV_Analysis_Excel.py"
)

