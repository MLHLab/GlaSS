# ============================================================
# GLACIER MASS-BALANCE MANUSCRIPT-CONSISTENT PIPELINE
# SPYDER STAGED VERSION
# ============================================================
# This script is one stage of a four-stage workflow.
# Run the stages sequentially. Each stage saves a compact pickle
# checkpoint that is consumed by the next stage.
# ============================================================

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "GlacierDataset_170Vars.csv"
OUTPUT_DIR = BASE_DIR / "Glacier_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_FILE = OUTPUT_DIR / "Stage1_Preprocessed.pkl"
STAGE2_FILE = OUTPUT_DIR / "Stage2_SSL_GridSearch.pkl"
STAGE3_FILE = OUTPUT_DIR / "Stage3_Retained_FinalCV.pkl"
OUTPUT_EXCEL = OUTPUT_DIR / "All Results.xlsx"

if not STAGE1_FILE.exists():
    raise FileNotFoundError(f"Stage 1 checkpoint not found: {STAGE1_FILE}\nRun 01_Preprocessing_Correlation.py first.")
with open(STAGE1_FILE, "rb") as f:
    stage1 = pickle.load(f)

globals().update(stage1)
print("Loaded Stage 1:", STAGE1_FILE)
# ============================================================
# 5. LOAD FEATURES AND TARGET FROM STAGE 1
# ============================================================
# Stage 1 already created and saved:
#   X0_df, y0_series : correlation-reduced unscaled data
#   X00, y00         : preserved copies of the reduced data
#   scaler           : fitted StandardScaler
#   X0, y0           : standardized predictors and target
#
# Therefore Stage 2 must NOT recreate these variables from data2.
# ============================================================

required_stage1_objects = [
    "data2",
    "feature_columns_reduced",
    "X0_df",
    "y0_series",
    "X00",
    "y00",
    "scaler",
    "X0",
    "y0",
    "IDENTIFIER_COLUMNS",
    "target_column",
]

missing_stage1_objects = [
    name for name in required_stage1_objects
    if name not in stage1
]

if missing_stage1_objects:
    raise KeyError(
        "The Stage 1 checkpoint is missing the following required "
        f"objects: {missing_stage1_objects}. "
        "Please rerun 01_Preprocessing_Correlation.py."
    )

print("\nStage 1 objects loaded successfully.")
print("Reduced feature matrix X0_df:", X0_df.shape)
print("Target vector y0_series:", y0_series.shape)
print("Standardized matrix X0:", X0.shape)
print("Number of reduced predictors:", X0_df.shape[1])

# ============================================================
# 6. CONSTRUCT MASKED-RECONSTRUCTION EMBEDDINGS
# ============================================================
# Manuscript-matched implementation:
#
#   * predictors are standardized first;
#   * predictors are divided into contiguous, non-overlapping
#     groups of four;
#   * within each group, one predictor is masked by setting it
#     to zero (the standardized group mean);
#   * GBR + XGBR are used as base learners and LR as meta-learner;
#   * the model is trained using the remaining three predictors;
#   * after training, the TRUE four-variable group is restored and
#     passed through the fitted model to reconstruct the masked
#     predictor;
#   * the procedure is repeated for every member of every group.
# ============================================================

print('\n' + '=' * 70)
print('STEP 4: GROUP-WISE SELF-SUPERVISED FEATURE RECONSTRUCTION')
print('=' * 70)

X_standardized = np.asarray(X0, dtype=float)

n_features_reduced = X_standardized.shape[1]

# Strict four-variable grouping, as described in the manuscript.
feature_groups = [
    list(range(i, min(i + 4, n_features_reduced)))
    for i in range(0, n_features_reduced, 4)
]

print(
    f'Number of reduced predictors: {n_features_reduced}'
)

print(
    f'Number of reconstruction groups: {len(feature_groups)}'
)

if any(len(g) != 4 for g in feature_groups):
    print(
        'NOTE: The final reconstruction group contains fewer than '
        'four predictors because the current predictor count is not '
        'divisible by four. For the manuscript dataset (88 predictors), '
        'this forms exactly 22 groups of four.'
    )

# Keep the original column names for the reconstructed representation.
X_embd = pd.DataFrame(
    index=X0_df.index
)

reconstruction_records = []

for group_number, group_indices in enumerate(
    feature_groups,
    start=1
):

    group_names = [
        feature_columns_reduced[j]
        for j in group_indices
    ]

    print(
        f'\nReconstruction group {group_number}/{len(feature_groups)}:'
    )
    print(group_names)

    group_matrix = X_standardized[:, group_indices].copy()

    for local_position, global_position in enumerate(group_indices):

        feature_name = feature_columns_reduced[
            global_position
        ]

        print(
            f'  Reconstructing {feature_name} '
            f'({local_position + 1}/{len(group_indices)})'
        )

        # Target is the standardized true value of the selected
        # predictor.
        y_msk = group_matrix[:, local_position].copy()

        # Mask only the selected member of the four-variable group.
        X_masked = group_matrix.copy()
        X_masked[:, local_position] = 0.0

        # The reconstruction learner receives ONLY the masked
        # four-variable group; because the selected column is constant
        # zero in training, the effective information comes from the
        # other three predictors.
        gbr = GradientBoostingRegressor(
            random_state=42
        )

        xgb = xgboost.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        lr = LinearRegression()

        reconstruction_model = StackingRegressor(
            estimators=[
                ('gbr', gbr),
                ('xgb', xgb)
            ],
            final_estimator=lr
        )

        reconstruction_model.fit(
            X_masked,
            y_msk
        )

        # IMPORTANT: after training, restore the true four-variable
        # group before reconstruction, exactly as described in the
        # manuscript. The masked coordinate remains irrelevant to the
        # trained tree split structure because it was constant zero
        # during fitting.
        X_restored = group_matrix.copy()

        X_reconstructed = reconstruction_model.predict(
            X_restored
        )

        X_embd[
            feature_name + '_mod'
        ] = X_reconstructed

        reconstruction_records.append({
            'Group': group_number,
            'Feature': feature_name,
            'Group Features': ', '.join(group_names)
        })

reconstruction_log_df = pd.DataFrame(
    reconstruction_records
)

X_embd_a = X_embd.to_numpy(dtype=float)

print(
    '\nEmbedding dataframe shape:',
    X_embd.shape
)

print(
    'X_embd_a shape:',
    X_embd_a.shape
)

reconstruction_log_df.to_csv(
    OUTPUT_DIR / 'Self_Supervised_Reconstruction_Groups.csv',
    index=False
)


# ============================================================
# 7. GRIDSEARCHCV FOR HYPERPARAMETER OPTIMIZATION
# ============================================================
# The manuscript specifies that GridSearchCV with 3-fold CV is used
# to select the final GBR and XGBR configurations.
#
# The selected configurations reported in the manuscript are:
#
# GBR : n_estimators=600, learning_rate=0.05,
#       max_depth=6, subsample=0.8
#
# XGBR: n_estimators=600, learning_rate=0.05,
#       max_depth=4, subsample=0.8, colsample_bytree=0.8
#
# These parameters are used by the final stacking regressor below.
# ============================================================

print('\n' + '=' * 70)
print('STEP 5: GRIDSEARCHCV HYPERPARAMETER OPTIMIZATION')
print('=' * 70)

X_grid_train, X_grid_test, y_grid_train, y_grid_test = train_test_split(
    X_embd_a,
    y0,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# GBR
# ------------------------------------------------------------

print('\nGridSearchCV: GradientBoostingRegressor')

gbr_grid = {
    'n_estimators': [400, 500, 600, 700],
    'learning_rate': [0.03, 0.05, 0.10],
    'max_depth': [4, 5, 6, 7],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

grid_search_gbr = GridSearchCV(
    estimator=GradientBoostingRegressor(
        random_state=42
    ),
    param_grid=gbr_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search_gbr.fit(
    X_grid_train,
    y_grid_train
)

BEST_GBR_PARAMS = grid_search_gbr.best_params_

print(
    'Best GBR hyperparameters:',
    BEST_GBR_PARAMS
)

# ------------------------------------------------------------
# XGBR
# ------------------------------------------------------------

print('\nGridSearchCV: XGBRegressor')

xgb_grid = {
    'n_estimators': [400, 500, 600, 700],
    'learning_rate': [0.03, 0.05, 0.10],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

grid_search_xgb = GridSearchCV(
    estimator=xgboost.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    param_grid=xgb_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search_xgb.fit(
    X_grid_train,
    y_grid_train
)

BEST_XGB_PARAMS = grid_search_xgb.best_params_

print(
    'Best XGBR hyperparameters:',
    BEST_XGB_PARAMS
)

# ------------------------------------------------------------
# SVR
# ------------------------------------------------------------
# SVR is a benchmark model in the comparative evaluation. It is
# not a component of the proposed stacking model.
# ------------------------------------------------------------

print('\nGridSearchCV: SVR')

svr_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [3900, 4000, 4100]
}

grid_search_svr = GridSearchCV(
    estimator=SVR(),
    param_grid=svr_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search_svr.fit(
    X_grid_train,
    y_grid_train
)

BEST_SVR_PARAMS = grid_search_svr.best_params_

print(
    'Best SVR hyperparameters:',
    BEST_SVR_PARAMS
)

# For exact reproducibility of the manuscript-reported final model,
# the reported optimum is used below. If the supplied data produce a
# different optimum, the code reports that result and uses the actual
# GridSearch optimum.
MANUSCRIPT_GBR_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8
}

MANUSCRIPT_XGB_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

if BEST_GBR_PARAMS != MANUSCRIPT_GBR_PARAMS:
    print(
        '\nWARNING: GridSearch GBR optimum on the current data differs '
        'from the manuscript-reported optimum.'
    )

if BEST_XGB_PARAMS != MANUSCRIPT_XGB_PARAMS:
    print(
        'WARNING: GridSearch XGBR optimum on the current data differs '
        'from the manuscript-reported optimum.'
    )

# Use the actual GridSearch-selected configurations for all subsequent
# model training, as required by the manuscript methodology.
FINAL_GBR_PARAMS = BEST_GBR_PARAMS.copy()
FINAL_XGB_PARAMS = BEST_XGB_PARAMS.copy()



# ============================================================
# VALIDATE STAGE 2 OBJECTS BEFORE SAVING
# ============================================================

required_stage2_objects = [
    "X_embd",
    "X_embd_a",
    "BEST_GBR_PARAMS",
    "BEST_XGB_PARAMS",
    "BEST_SVR_PARAMS",
    "FINAL_GBR_PARAMS",
    "FINAL_XGB_PARAMS",
]

missing_stage2_objects = [
    name for name in required_stage2_objects
    if name not in globals()
]

if missing_stage2_objects:
    raise RuntimeError(
        "Stage 2 cannot be saved because these objects are missing: "
        f"{missing_stage2_objects}"
    )

# ============================================================
# SAVE STAGE 2 CHECKPOINT
# ============================================================
stage2 = {
    **stage1,
    "X_embd": X_embd,
    "X_embd_a": X_embd_a,
    "reconstruction_log_df": reconstruction_log_df,
    "BEST_GBR_PARAMS": BEST_GBR_PARAMS,
    "BEST_XGB_PARAMS": BEST_XGB_PARAMS,
    "BEST_SVR_PARAMS": BEST_SVR_PARAMS,
    "MANUSCRIPT_GBR_PARAMS": MANUSCRIPT_GBR_PARAMS,
    "MANUSCRIPT_XGB_PARAMS": MANUSCRIPT_XGB_PARAMS,
    "FINAL_GBR_PARAMS": FINAL_GBR_PARAMS,
    "FINAL_XGB_PARAMS": FINAL_XGB_PARAMS,
}
with open(STAGE2_FILE, "wb") as f:
    pickle.dump(stage2, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nSTAGE 2 COMPLETE")
print("Checkpoint saved:", STAGE2_FILE)
print("Run 03_Rejection_and_RetainedData.py next.")
