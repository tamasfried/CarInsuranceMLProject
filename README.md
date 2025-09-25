# Car Insurance Premium Prediction

This project investigates how driver and vehicle attributes can be used to estimate car insurance premiums. It combines exploratory data analysis, a suite of regression experiments, a K-Means clustering study, and a command-line application that serves the best-performing regression model as an interactive premium predictor.

## Quick Start

1. Ensure Python 3.10+ is installed.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The bundled requirements file captures the full environment used during development. For a minimal setup install `pandas numpy scikit-learn matplotlib seaborn xgboost`.
3. Launch the interactive predictor:
   ```bash
   python Insurance_Price_Predictor.py
   ```

## Repository Structure

- `car_insurance.csv` – Kaggle dataset with 1,000 driver/vehicle records and premium labels.
- `Insurance_Price_Predictor.py` – CLI menu to inspect model metrics or predict a premium for new inputs with validation and feature contribution breakdowns.
- `car_insurance.ipynb` – Exploratory data analysis (data quality, distributions, correlations, regression fits).
- `insurance_analysis_MLR.ipynb` – Detailed multivariate linear regression workflow and diagnostics.
- `Clustering_Car_Insurance.ipynb` – Feature scaling and K-Means experiments to discover risk archetypes.
- `MLPRegressor_Car_Insurance.ipynb` & `XGBoost_Car_Insurance.ipynb` – Non-linear model experiments (neural network and gradient boosting).
- `actual_vs_predicted.png`, `correlation_matrix.png`, `residual_plot.png` – Key diagnostic visualisations exported from the notebooks.
- `LaTex/` & `PDF/` – Formal project report source and compiled submission.

## Dataset

- **Source:** [Car Insurance Premium Dataset](https://www.kaggle.com/datasets/govindaramsriram/car-insurance-premium-dataset) by Govindaram Sriram.
- **Size:** 1,000 rows, 7 numeric columns.
- **Features:**
  - `Driver Age`, `Driver Experience`, `Previous Accidents`, `Annual Mileage (x1000 km)`, `Car Manufacturing Year`, `Car Age`.
  - Target: `Insurance Premium ($)`.
- The data contains no missing values and appears to be synthetic, which explains the near-perfect fit achieved by simple models. Feature scaling (StandardScaler) was applied before distance-based or neural models.

## Exploratory Analysis Highlights

- Correlation analysis shows premiums decreasing with age/experience and increasing with accident count; age and experience remain strongly negatively correlated (≈ -0.8).
- Distribution and box plots confirm the absence of outliers after scaling.
- Simple linear regression on `Driver Age` alone yields an R² of ~0.57, signalling that multiple features are needed for accurate pricing.
- Visual diagnostics (correlation matrix, residual plots, actual vs. predicted scatter) are available as PNG exports in the repository.

## Modelling Summary

| Model | Description | MSE | R² | Notes |
| --- | --- | --- | --- | --- |
| Simple Linear Regression | Single feature (`Driver Age`) baseline | – | 0.57 | Illustrates diminishing premiums with age but leaves large residuals. |
| Multivariate Linear Regression | All engineered features, 80/20 train/test split | 0.00 | 1.00 | Perfect fit on held-out set; likely inflated by synthetic data. |
| K-Nearest Neighbours | k=5, scaled features | 1.35 | 0.96 | Captures local structure; sensitive to feature scaling. |
| MLPRegressor | Two hidden layers (100, 50), scaled features | 0.53 | 0.98 | Convergence warning after 2,000 iterations; still strong performance. |
| XGBoost Regressor | Gradient boosting via `xgboost.train` | – | – | Notebook focuses on feature importance and residual diagnostics; formal metrics pending. |

All regression experiments use an 80/20 train/test split with `random_state=42` for reproducibility.

## Clustering Experiment

`Clustering_Car_Insurance.ipynb` standardises the features and applies K-Means. The elbow method suggests 3–4 segments; k=4 was selected to prototype risk archetypes. The resulting clusters separate high-mileage/low-experience drivers from low-risk cohorts and can be used to support risk profiling alongside regression predictions.

## Interactive Premium Predictor

`Insurance_Price_Predictor.py` exposes the multivariate linear regression model via a terminal UI:

- **Analyse dataset** prints test-set Mean Squared Error, R², and feature coefficients so the user can interpret the model.
- **Predict insurance premium** prompts for validated driver/vehicle details, shows the learned base premium, and quantifies each feature’s contribution to the quote.
- Input guards enforce realistic values (e.g., minimum driving age 17, non-negative mileage/accidents).

## Working with the Notebooks

Launch JupyterLab or Notebook after installing dependencies:

```bash
jupyter lab
```

Each notebook is self-contained and can be rerun top-to-bottom. For repeatability, ensure the virtual environment uses Python 3.10+ and install the libraries listed above.

## Reports and Deliverables

- `LaTex/main.tex` contains the full written report with methodology, results discussion, and references.
- `PDF/HCS502_Assessment_AI_ML_project_2310190.pdf` is the compiled submission.
- Supporting graphics (e.g., `actual_vs_predicted.png`, `residual_plot.png`) are available for integration into presentations or documentation.

## Limitations and Next Steps

- The current dataset is synthetic; validation on real-world policies is required before deployment.
- Perfect R² scores hint at potential overfitting; k-fold cross validation and regularisation should be added.
- XGBoost experiments need explicit evaluation metrics and hyperparameter tuning.
- Additional features (vehicle power, drivetrain, postcode, telematics data) could improve generalisability.
- Bias/fairness checks are recommended before production use.

## Acknowledgements

- Dataset: Govindaram Sriram, *Car Insurance Premium Dataset* (Kaggle, 2025).
- Tooling: scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost.
