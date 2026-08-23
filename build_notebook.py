from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "CKD_data_preprocessing_assignment.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

nb["cells"] = [
    md(
        r"""
# CKD data preprocessing and exploratory analysis

This notebook completes the **Data preprocessing** assignment using the Chronic Kidney Disease (CKD) dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease).

The analysis is reproducible from the local source file in `data/chronic_kidney_disease.csv`. The notebook:

1. loads and describes the source data;
2. selects and renames the required columns;
3. converts hemoglobin from g/dL to g/L and recodes the class;
4. removes rows with three or more missing values;
5. splits affected and control individuals;
6. calculates descriptive statistics and reviews histograms/outliers; and
7. calculates and visualizes a correlation matrix for each group.

**Data citation:** Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020 (CC BY 4.0).
"""
    ),
    md(
        r"""
## 1. Setup and display options

Only standard data-analysis libraries are used. Pandas provides the data-frame pipeline, while Matplotlib and Seaborn provide the visualizations.
"""
    ),
    code(
        r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")

# This works both when the notebook is launched from its own folder and from
# the parent workspace folder.
DATA_CANDIDATES = [
    Path("data/chronic_kidney_disease.csv"),
    Path("completed_data_preprocessing/data/chronic_kidney_disease.csv"),
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find data/chronic_kidney_disease.csv. "
        "Run this notebook from the assignment folder or its parent."
    )

DATA_PATH
"""
    ),
    md(
        r"""
## 2. Load and describe the original dataset

UCI describes this as a multivariate classification dataset with **400 observations**, **24 features**, one binary target (`class`), and missing values. The target labels are `ckd` and `notckd`. The source file represents missing values as `NaN`.

The fields needed here include age; blood pressure; specific gravity; albumin; sugar; several blood measurements; and class. According to the assignment, the source hemoglobin unit is g/dL; it will be converted to g/L later by multiplying by 10.
"""
    ),
    code(
        r"""
raw = pd.read_csv(DATA_PATH)

print(f"Original shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
display(raw.head())
display(
    pd.DataFrame(
        {
            "dtype": raw.dtypes.astype(str),
            "missing": raw.isna().sum(),
            "missing_%": raw.isna().mean().mul(100).round(1),
            "unique_non_missing": raw.nunique(dropna=True),
        }
    )
)
"""
    ),
    md(
        r"""
## 3. Construct the preprocessing pipeline

The assignment uses full, readable column names, while the source uses abbreviations. The mapping below selects **exactly** the 14 requested columns and renames them.

The pipeline also:

- coerces the 13 measurement columns to numeric values (invalid text would become missing);
- strips whitespace from the source class values (the raw file contains both `ckd` and `ckd\t`);
- maps `ckd` to `a` (affected) and `notckd` to `c` (control);
- multiplies hemoglobin by 10 to convert g/dL to g/L; and
- keeps only rows containing fewer than three missing values across the selected 14 columns.
"""
    ),
    code(
        r"""
COLUMN_MAP = {
    "age": "age",
    "bp": "blood pressure",
    "sg": "specific gravity",
    "al": "albumin",
    "su": "sugar",
    "bgr": "blood glucose random",
    "bu": "blood urea",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "hemoglobin",
    "pcv": "packed cell volume",
    "wbcc": "white blood cell count",
    "rbcc": "red blood cell count",
    "class": "class",
}

NUMERIC_COLUMNS = [name for name in COLUMN_MAP.values() if name != "class"]


def preprocess_ckd(frame: pd.DataFrame):
    '''Return the assignment-ready frame and a row-level cleaning audit.'''
    prepared = frame.loc[:, list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
    prepared[NUMERIC_COLUMNS] = prepared[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    prepared["class"] = (
        prepared["class"]
        .astype("string")
        .str.strip()
        .map({"ckd": "a", "notckd": "c"})
    )
    prepared["hemoglobin"] = prepared["hemoglobin"] * 10

    missing_count = prepared.isna().sum(axis=1)
    audit = pd.DataFrame(
        {
            "missing_values": missing_count,
            "kept": missing_count < 3,
        },
        index=prepared.index,
    )
    cleaned = prepared.loc[audit["kept"]].copy()
    return cleaned, audit


ckd, cleaning_audit = preprocess_ckd(raw)

assert ckd.columns.tolist() == list(COLUMN_MAP.values())
assert ckd.shape[1] == 14
assert set(ckd["class"].dropna().unique()) == {"a", "c"}
assert ckd.isna().sum(axis=1).lt(3).all()

print("Rows by number of missing values before filtering:")
display(
    cleaning_audit["missing_values"]
    .value_counts()
    .sort_index()
    .rename_axis("missing values in selected columns")
    .to_frame("rows")
)
print(f"Rows removed (3 or more missing values): {(~cleaning_audit['kept']).sum()}")
print(f"Rows left in the modified data frame: {len(ckd)}")
"""
    ),
    code(
        r"""
print("Missing values remaining by column (one or two per retained row are allowed):")
display(ckd.isna().sum().to_frame("missing"))

print("Modified data frame:")
display(ckd)
"""
    ),
    md(
        r"""
### Cleaning result

The rule removes **135 rows** and leaves **265 rows**. Missing values are not otherwise imputed because the assignment asks only to remove rows with three or more missing values. Retaining the remaining missing values also avoids inventing measurements before group-level exploration; Pandas' descriptive statistics and pairwise correlations ignore them where appropriate.
"""
    ),
    md(
        r"""
## 4. Split affected and control individuals

`a` means affected (original class `ckd`) and `c` means control (original class `notckd`). Each displayed data frame includes the same 14 required columns.
"""
    ),
    code(
        r"""
affected = ckd.loc[ckd["class"].eq("a")].copy()
control = ckd.loc[ckd["class"].eq("c")].copy()

assert len(affected) + len(control) == len(ckd)

print(f"Affected individuals: {len(affected)} rows")
display(affected)

print(f"Control individuals: {len(control)} rows")
display(control)
"""
    ),
    md(
        r"""
### Split result

- **Affected:** 126 rows
- **Control:** 139 rows

The split is fairly balanced after applying the missing-value rule.
"""
    ),
    md(
        r"""
## 5. Basic statistics

The tables report count, mean, standard deviation, minimum, quartiles, and maximum for every numerical column. The class column is constant within each split by construction, so its frequency is shown separately.
"""
    ),
    code(
        r"""
def show_basic_statistics(frame: pd.DataFrame, group_name: str):
    print(f"{group_name}: numerical statistics")
    display(frame[NUMERIC_COLUMNS].describe().T)
    print(f"{group_name}: class statistics")
    display(frame[["class"]].describe().T)


show_basic_statistics(affected, "Affected")
show_basic_statistics(control, "Control")
"""
    ),
    md(
        r"""
### Statistical comparison

The affected group is visibly more heterogeneous. It has lower central values for specific gravity, hemoglobin, packed cell volume, and red blood cell count, while blood glucose and blood urea are higher and more variable. These are descriptive associations only; they do not establish causal effects, and the retained missing values mean that column counts differ slightly.
"""
    ),
    md(
        r"""
## 6. Histograms and outlier review

All 13 retained measurement columns are numeric and are plotted below. Each subplot uses its own horizontal scale, and missing observations are automatically omitted. Albumin and sugar are ordinal test grades, and specific gravity takes a small fixed set of values, so their histogram shapes should not be read as continuous bell-shaped distributions.
"""
    ),
    code(
        r"""
def plot_histograms(frame: pd.DataFrame, group_name: str):
    axes = frame[NUMERIC_COLUMNS].hist(
        bins=15,
        figsize=(16, 13),
        layout=(4, 4),
        color="#2C7FB8",
        edgecolor="white",
        linewidth=0.7,
    )
    axes = np.asarray(axes).ravel()
    for ax in axes:
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[len(NUMERIC_COLUMNS):]:
        ax.set_visible(False)
    plt.suptitle(f"{group_name}: distributions of numerical columns", fontsize=18, y=1.01)
    plt.tight_layout()
    plt.show()


plot_histograms(affected, "Affected")
plot_histograms(control, "Control")
"""
    ),
    md(
        r"""
### IQR outlier screen

To make the visual review reproducible, the next table applies the common 1.5 × IQR rule separately within each group. This is a **screening rule**, not an automatic deletion rule. It is especially limited for discrete variables such as albumin, sugar, and specific gravity.
"""
    ),
    code(
        r"""
def iqr_outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    q1 = frame[NUMERIC_COLUMNS].quantile(0.25)
    q3 = frame[NUMERIC_COLUMNS].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = frame[NUMERIC_COLUMNS].lt(lower) | frame[NUMERIC_COLUMNS].gt(upper)
    return pd.DataFrame(
        {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower fence": lower,
            "upper fence": upper,
            "flagged values": flagged.sum(),
        }
    )


print("Affected: IQR outlier screen")
affected_outliers = iqr_outlier_summary(affected)
display(affected_outliers)

print("Control: IQR outlier screen")
control_outliers = iqr_outlier_summary(control)
display(control_outliers)
"""
    ),
    md(
        r"""
### Outlier interpretation and handling

The affected histograms show clear right tails for blood glucose random and blood urea, plus a few unusually low or high values in measurements such as sodium, hemoglobin, packed cell volume, and cell counts. The IQR screen also flags several affected observations. The control group is much tighter and has no values beyond its group-specific 1.5 × IQR fences.

I would **not remove the affected values automatically**: extreme laboratory measurements may be genuine manifestations of kidney disease, and deleting them could erase clinically important cases. A suitable handling sequence would be:

1. check the source record, unit, and plausible physiological range;
2. correct or mark as missing only values shown to be data-entry errors;
3. retain plausible extremes for this descriptive analysis; and
4. for a later model sensitive to scale, consider robust scaling, a log transform for strongly right-skewed positive variables, or winsorization documented as a sensitivity analysis.
"""
    ),
    md(
        r"""
## 7. Correlation matrices

Pearson correlations are calculated separately for affected and control individuals. Pandas uses pairwise complete observations, so each coefficient uses the rows where both variables are present. Correlations involving the discrete albumin, sugar, and specific-gravity fields should be interpreted cautiously.
"""
    ),
    code(
        r"""
def plot_correlation_matrix(frame: pd.DataFrame, group_name: str) -> pd.DataFrame:
    corr = frame[NUMERIC_COLUMNS].corr(method="pearson")
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title(f"{group_name}: correlation matrix", fontsize=17, pad=16)
    plt.xticks(rotation=55, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return corr


affected_corr = plot_correlation_matrix(affected, "Affected")
control_corr = plot_correlation_matrix(control, "Control")
"""
    ),
    code(
        r"""
def strongest_unique_correlations(corr: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    upper_triangle = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(upper_triangle).stack().rename("correlation").reset_index()
    pairs.columns = ["variable 1", "variable 2", "correlation"]
    pairs["absolute correlation"] = pairs["correlation"].abs()
    return pairs.sort_values("absolute correlation", ascending=False).head(n)


print("Affected: strongest absolute correlations")
display(strongest_unique_correlations(affected_corr))

print("Control: strongest absolute correlations")
display(strongest_unique_correlations(control_corr))
"""
    ),
    md(
        r"""
### Correlation interpretation

**Affected individuals.** The strongest relationship is the positive correlation between hemoglobin and packed cell volume (about **0.95**). Red blood cell count is also strongly positively related to packed cell volume (about **0.80**) and hemoglobin (about **0.78**). These variables describe related aspects of red-cell status, so their co-movement is expected. Sugar and random blood glucose have a moderately strong positive correlation (about **0.69**). Blood urea is negatively associated with hemoglobin and packed cell volume (absolute correlations around **0.61** and **0.57**, respectively). This pattern describes co-occurring measurements within the affected sample; it does not show that one measurement causes another.

**Control individuals.** Correlations are much weaker. The largest absolute coefficient is only about **0.34** (sodium with white blood cell count), followed by random blood glucose with packed cell volume (about **−0.26**). The restricted range of relatively healthy control measurements can naturally attenuate correlations, and the sample size and remaining missingness add uncertainty.

**Comparison.** The affected group contains several pronounced clusters—especially the red-cell measurements and sugar/glucose—whereas the control heatmap is mostly near zero. Because many pairs were inspected and no uncertainty intervals or significance tests were calculated, small isolated coefficients should not be overinterpreted.
"""
    ),
    md(
        r"""
## 8. Final conclusions

- The required preprocessing pipeline produced a 14-column data frame with hemoglobin in g/L and class coded as `a`/`c`.
- Removing rows with at least three missing selected values left **265 observations**: **126 affected** and **139 control**.
- Affected individuals show broader and more skewed measurement distributions. Plausible extreme laboratory values should be retained unless source validation identifies an error.
- The affected group has strong relationships among hemoglobin, packed cell volume, and red blood cell count, plus a sugar/glucose association. The control group has no similarly strong linear relationships.
- The findings are exploratory and associational, not causal.
"""
    ),
]

nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT}")
