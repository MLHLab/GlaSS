GLACIER ANNUAL MASS-BALANCE PREDICTION PIPELINE
================================================

OVERVIEW
--------
This repository contains Python scripts for a staged machine-learning workflow to predict glacier annual mass balance from climate and spatial predictors. The workflow includes data preprocessing, correlation-based feature reduction, self-supervised feature reconstruction, hyperparameter search, sequential rejection of high-error observations, and final model evaluation.

IMPORTANT TO NOTE
-----------------
1. Use the scripts and dataset version that match the analysis being reproduced. Column names, row order, and the target variable must match the expected inputs.
2. The scripts may require adjustment of file paths for your computer. Keep the Python scripts and input CSV files together in one project directory where possible.
3. Do not upload confidential, restricted, or third-party data to GitHub unless you have permission to redistribute it. You may publish the code separately and explain how authorized users can obtain the data.
4. Pickle (.pkl) checkpoints are generated intermediate files. They may be large and should generally not be committed to GitHub unless there is a specific reason and their contents are suitable for public release.
5. Results may vary with software versions, data versions, and execution environment. Use the same data and package versions for close reproducibility.

MAIN FOUR-STAGE PIPELINE
------------------------

01_Preprocessing_Correlation.py
    Loads the glacier dataset, performs preprocessing and gap filling as implemented in the script, applies correlation-based feature reduction, prepares model inputs, and saves the Stage 1 checkpoint.

02_SSL_and_GridSearch.py
    Loads the Stage 1 checkpoint, performs self-supervised feature reconstruction using grouped predictors and a regression-based stacking approach, conducts GridSearchCV for specified regression models, and saves the Stage 2 checkpoint.

03_Rejection_and_RetainedData.py
    Loads the preceding checkpoint(s), performs sequential rejection of observations with high absolute prediction errors using the configured percentile threshold and partition sequence, and saves retained-data and rejection information in the Stage 3 checkpoint.

04_FinalCV_Analysis_Excel.py
    Loads the Stage 3 checkpoint and performs final five-fold cross-validation. It generates fold-wise actual-versus-predicted values, model performance comparisons, ablation results, feature-importance analyses, and an Excel workbook containing result tables.

INPUT DATA
----------

Depending on the script and analysis, the following CSV files may be required:

- GlacierDataset_170Vars.csv
  Main dataset expected by the four-stage preprocessing and modelling pipeline. Confirm the exact columns and dimensions against the version used for the analysis.

- GlacierDataset_IndTest_170Vars.csv
  Independent predictor-only dataset for row-wise prediction. It should not contain the target variable when used for independent prediction.

- GlacierDataset_172Vars.csv
  Original dataset used by the WGMS_ID export utility when identifiers and year information are required. Confirm that its row order corresponds exactly to the observations used to generate the Stage 3 rejection indices.

Do not assume similarly named datasets are interchangeable. Verify file version, columns, row count, row order, target variable, and identifier columns before running the workflow.

SOFTWARE REQUIREMENTS
---------------------

Use a supported Python 3 environment. The scripts use packages including:

- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- openpyxl

Install the required packages in the same environment used to run the scripts. For example, from a terminal:

    pip install numpy pandas scikit-learn xgboost matplotlib seaborn openpyxl

If using Anaconda/Spyder, ensure that packages are installed in the Python environment Spyder uses. Check package versions if compatibility issues occur.

The scripts may create the output directory when needed. Some scripts require checkpoints or result files from previous stages. Keep generated files in the expected output directory or update paths in the scripts.

HOW TO RUN THE MAIN PIPELINE
----------------------------

Run the four main scripts in this order:

1. Run 01_Preprocessing_Correlation.py
   - Check the input CSV path and column names.
   - Confirm preprocessing completes and the Stage 1 checkpoint is saved.

2. Run 02_SSL_and_GridSearch.py
   - Confirm the Stage 1 checkpoint is available.
   - Allow reconstruction and hyperparameter search to finish.
   - Confirm the Stage 2 checkpoint is saved.

3. Run 03_Rejection_and_RetainedData.py
   - Confirm the required preceding checkpoint is available.
   - Review the rejection summary and retained-data dimensions.
   - Confirm the Stage 3 checkpoint is saved.

4. Run 04_FinalCV_Analysis_Excel.py
   - Confirm the Stage 3 checkpoint is available.
   - Wait for cross-validation, comparison, ablation, and importance calculations to finish.
   - Review the generated Excel workbook and CSV prediction output.

METHODOLOGICAL SUMMARY
----------------------

The main pipeline implements the following general sequence:

1. Data preprocessing and gap filling according to the implementation.
2. Correlation-based reduction of predictor variables.
3. Standardization of predictors.
4. Self-supervised reconstruction of predictor features using groups of features and regression-based reconstruction.
5. Hyperparameter tuning for selected regression algorithms using GridSearchCV.
6. Sequential rejection of high-error observations based on a percentile threshold and a predefined partition order.
7. Final evaluation using five-fold cross-validation.
8. Comparative and ablation analyses.
9. Permutation-based feature-importance analysis.
10. Export of tables, predictions, figures, and identifier information.

OUTPUTS
-------

Depending on the scripts and settings, outputs may include:

- Stage1_Preprocessed.pkl
- Stage2_SSL_GridSearch.pkl
- Stage3_Retained_FinalCV.pkl
- All Results.xlsx
- Final_CV_Actual_vs_Predicted.csv

Output filenames and locations are controlled by the scripts and may need to be updated for your local setup.

REPRODUCIBILITY AND DATA INTEGRITY
----------------------------------

- Preserve original row order when mapping predictions or rejection indices back to WGMS_ID and year.
- Do not independently recompute feature selection on the independent test dataset. Apply the feature mapping and preprocessing defined by the training workflow.
- Keep training and independent datasets separate.
- Do not use the target variable as an input predictor.
- Review output summaries to confirm the intended number of observations and predictors is used at each stage.
- Record Python and package versions when preparing a reproducible experiment.
- Check generated numerical results against the manuscript before publication.

LIMITATIONS AND NOTES
---------------------

- The scripts are research code and should be reviewed and tested in the intended environment before use.
- Some analyses depend on intermediate checkpoint files generated by earlier stages.
- The topographical parameter comparison requires the exact additional predictor columns to be available and configured in the relevant script.
- Deep-learning benchmark architectures should not be considered reproduced unless their architecture and training details are explicitly implemented and documented.
- This repository does not guarantee that the input data may be publicly redistributed. Check the data provider's terms and permissions.
