![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# Mosses - Model Assessment Toolkit

## Description
`Mosses` is a library that provides a set of functionalities to assess molecular property prediction models, e.g., QSAR/QSPR models. The library currently includes:

- **Predictive Validity Module** (`predictive_validity.py`) - Built on top of the concept of *predictive validity* described by Scannell et al. Nat Rev Drug Discov. 2022;21(12):915-931. [doi:10.1038/s41573-022-00552-x](https://www.nature.com/articles/s41573-022-00552-x). The function `predictive_validity.evaluate_pv()` allows the analysis of the quality of predictions on a given data set (e.g., a prospective test set of compounds), according to a desired threshold.

- **Heatmap Module** (`heatmap.py`) - Summarises the information from the validation using *predictive validity*. The heatmap shows in one table, for each series in the data and according to the selected experimental threshold (SET), what the PPV and FOR percentages are, the recommended thresholds and resulting optimised PPV and FOR percentages, as well as, a qualitative label indicating whether the model is Good, Medium, or Bad.

- **Multi-Parameter Optimization (MPO) Module** (`mpo.py`) - Provides a comprehensive toolkit for computing and optimizing MPO scores. MPO combines multiple molecular properties into a single score using sigmoid-based desirability functions.

## Software Requirements
The library is written in Python and requires a version >= 3.10 for runtime. The dependencies required by the library are defined in `pyproject.toml` and are automatically installed when installing the library.

## How to Install `mosses`
You can install the library using `pip install mosses`, or you can clone this repository then run `make build && make install`.

## Examples of Usage
Jupyter notebooks with examples can be found in the folder `examples`. We recommend following those to adapt your data, configs, and code to work with the modules in `mosses`.

---

## Multi-Parameter Optimization (MPO) Module

The `mosses.mpo` module provides a high-level API for Multi-Parameter Optimization analysis of compound data. It is commonly used in drug discovery to combine multiple ADMET properties into a single desirability score.

### Key Features

- **Sigmoid-based scoring functions** for transforming raw values to 0-1 scores
- **Multiple optimization algorithms** for weight optimization
- **ML-based weight estimation** using Random Forest, Ridge, and Logistic classifiers
- **Feature importance analysis** via mutual information
- **Enrichment and correlation statistics**
- **Visualization tools** for analysis and comparison

### Quick Start

```python
from mosses import mpo
import pandas as pd

# Load your compound data
df = pd.read_csv("compounds.csv")

# Define parameter configurations
config = {
    "LogD": {
        "preference": "middle",      # Optimal range preferred
        "threshold": (0.0, 3.0),     # Values in this range score highest
        "weight": 1.0,
    },
    "Solubility": {
        "preference": "maximize",    # Higher is better
        "threshold": 50.0,           # Values > 50 score high
        "weight": 1.5,
    },
    "Clearance": {
        "preference": "minimize",    # Lower is better
        "threshold": 50.0,           # Values < 50 score high
        "weight": 1.0,
    },
}

# Compute MPO scores
result = mpo.compute_scores(df, config, return_intermediate=True)
print(result[["Compound Name", "MPO_Score"]].head())
```

### Preference Types

The module supports three optimization preferences that determine how raw values are transformed into scores:

| Preference | Description | Threshold | Scoring Function |
|------------|-------------|-----------|------------------|
| `maximize` | Higher values are better | Single value (inflection point) | `sigmoid()` |
| `minimize` | Lower values are better | Single value (inflection point) | `reverse_sigmoid()` |
| `middle` | Optimal range preferred | Tuple `(lower, upper)` | `double_sigmoid()` |

#### Visualizing Scoring Functions

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 6, 200)

# Maximize: values above threshold score high
maximize_scores = mpo.sigmoid(x, threshold=2.0, steepness=2.0)

# Minimize: values below threshold score high
minimize_scores = mpo.reverse_sigmoid(x, threshold=3.0, steepness=2.0)

# Middle: values in range score high
middle_scores = mpo.double_sigmoid(x, lower_threshold=1.0, upper_threshold=4.0, steepness=3.0)
```

### Weight Optimization

Optimize weights to match experimental target data using various algorithms:

```python
# First compute individual parameter scores
result = mpo.compute_scores(df, config, return_intermediate=True)
df_with_scores = df.merge(result, on="Compound Name")

# Define score columns
score_columns = ["LogD_score", "Solubility_score", "Clearance_score"]

# Optimize weights against experimental activity
optimized_weights, opt_result = mpo.optimize_mpo_weights(
    df_with_scores,
    score_columns,
    target_column="Activity",
    method="differential_evolution",  # or "least_squares", "minimize", "powell"
    verbose=True,
)

print("Optimized weights:", optimized_weights)
```

#### Available Optimization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `least_squares` | Linear least squares | Fast, good baseline |
| `minimize` | Scipy minimize (L-BFGS-B) | General purpose |
| `differential_evolution` | Global evolutionary algorithm | Robust, handles local minima |
| `dual_annealing` | Simulated annealing variant | Complex landscapes |
| `powell` | Powell's method | Derivative-free |
| `pygad` | Genetic algorithm (optional) | Highly customizable |

### ML-Based Weight Estimation

Use machine learning to estimate feature importance as weights:

```python
# Random Forest regression
ml_result = mpo.rf_regression(
    df_with_scores,
    score_columns,
    reference_col="Activity",
)

print("Feature importance:", ml_result.weights)
print(f"R² Score: {ml_result.metrics['test_r2']:.3f}")

# Other estimators available:
# mpo.rf_classifier() - Random Forest classification
# mpo.logistic_classifier() - Logistic regression
# mpo.ridge_classifier() - Ridge classifier
```

### Evaluation Metrics

Evaluate MPO performance against experimental data:

```python
stats = mpo.evaluate_mpo(
    df_with_scores,
    mpo_column="MPO_Score",
    reference_column="Activity",
    top_percent=10.0,  # Analyze top 10% of compounds
)

# Access metrics
print(f"Enrichment: {stats.enrichment:.2f}")
print(f"Spearman correlation: {stats.spearman_correlation:.3f}")
print(f"F1 score: {stats.f1_score:.3f}")
print(f"RMSE: {stats.rmse:.3f}")
```

### Feature Importance Analysis

Analyze which parameters contribute most to the target:

```python
importance_result = mpo.analyze_feature_importance(
    df_with_scores,
    score_columns,
    reference_col="Activity",
)

# Visualize
mpo.plot_mutual_info(importance_result, title="Feature Importance")
```

### Complete Pipeline

For end-to-end analysis with automatic threshold detection and optional weight optimization:

```python
result = mpo.build_mpo_pipeline(
    df,
    experimental_columns=["LogD", "Solubility", "Clearance", "Permeability"],
    target_column="Activity",
    preferences={
        "LogD": "middle",
        "Solubility": "maximize",
        "Clearance": "minimize",
        "Permeability": "maximize",
    },
    auto_threshold=True,  # Calculate thresholds from data
    optimize_weights_method="least_squares",  # Optional: optimize weights
)

# Result contains individual scores and final MPO_Score
print(result[["Compound Name", "MPO_Score"]].head())
```

### Visualization Functions

The module provides several plotting utilities:

```python
# Score distribution histogram
mpo.plot_mpo_histogram(result["MPO_Score"], title="MPO Score Distribution")

# Scatter plot with regression line
mpo.plot_best_fit_scatter(
    result["Activity"],
    result["MPO_Score"],
    label="MPO vs Activity"
)

# Correlation matrix
mpo.plot_parameter_correlation_matrix(
    df,
    columns=["LogD", "Solubility", "Clearance"],
    title="Parameter Correlations",
)

# Compare multiple methods
mpo.plot_comparison(
    df_with_scores,
    method_columns=["MPO_Score", "Optimized_MPO"],
    reference_column="Activity"
)
```

### API Reference

#### Main Functions

| Function | Description |
|----------|-------------|
| `compute_scores(df, config)` | Compute MPO scores from parameter configuration |
| `optimize_mpo_weights(df, score_cols, target)` | Optimize weights against target column |
| `evaluate_mpo(df, mpo_col, ref_col)` | Compute enrichment and correlation statistics |
| `build_mpo_pipeline(df, columns, ...)` | End-to-end MPO workflow |

#### Scoring Functions

| Function | Description |
|----------|-------------|
| `sigmoid(x, threshold, steepness)` | Standard sigmoid for maximization |
| `reverse_sigmoid(x, threshold, steepness)` | Reversed sigmoid for minimization |
| `double_sigmoid(x, lower, upper, steepness)` | Double sigmoid for middle preference |

#### Statistics Functions

| Function | Description |
|----------|-------------|
| `calculate_enrichment(percent_top, df, ref_col, method_col)` | Enrichment factor calculation |
| `calculate_spearman_correlation(df, col1, col2)` | Spearman rank correlation |
| `find_top_n_percent_ids(percent_top, df, score_col)` | Get IDs of top N% compounds |
| `collect_stats(...)` | Comprehensive statistics collection |

#### Plotting Functions

| Function | Description |
|----------|-------------|
| `plot_mpo_histogram(scores)` | Distribution of MPO scores |
| `plot_best_fit_scatter(x, y)` | Scatter plot with regression |
| `plot_parameter_correlation_matrix(df, columns)` | Correlation heatmap |
| `plot_experimental_correlation_matrix(df, cols)` | Experimental parameter correlations |
| `plot_predicted_correlation_matrix(df, cols)` | Predicted parameter correlations |
| `plot_mutual_info(importance)` | Feature importance bar chart |
| `plot_comparison(df, methods, ref)` | Side-by-side method comparison |
| `plot_scoring_curves(config)` | Visualize sigmoid functions |

### Example Notebook

See `examples/mpo_example.ipynb` for a complete walkthrough including:
1. Loading and exploring compound data
2. Configuring parameters with different preferences
3. Computing and visualizing MPO scores
4. Optimizing weights against experimental data
5. Evaluating MPO performance
6. Using ML-based weight estimation
7. Feature importance analysis
8. Building complete pipelines

---

## License
See [LICENSE.md](LICENSE.md) for details.
