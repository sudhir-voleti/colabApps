# ols_complete.py
# Self-contained ipywidgets port of the ols-shinyapp R/Shiny application.
# Run in Colab directly, or from GitHub using requests + exec.

import io
import os
import tempfile
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.formula.api as smf
from IPython.display import Markdown, clear_output, display

sns.set_theme(style="whitegrid")

TRAIN_DF = None
PRED_DF = None
MODEL = None
FITTED_DF = None
PREDICTED_DF = None

# Raw sample files from the original R application repository.
SAMPLE_TRAIN_URL = "https://raw.githubusercontent.com/sudhir-voleti/ols-shinyapp/master/data/beer%20data.csv"
SAMPLE_PRED_URL = "https://raw.githubusercontent.com/sudhir-voleti/ols-shinyapp/master/data/beer%20data%20-%20prediction%20sample.csv"


def uploaded_bytes(upload_widget):
    """Read FileUpload content in both ipywidgets 7 and 8."""
    value = upload_widget.value
    if not value:
        return None
    if isinstance(value, dict):
        return next(iter(value.values()))["content"]
    return value[0]["content"]


def read_upload(upload_widget, label):
    content = uploaded_bytes(upload_widget)
    if content is None:
        raise ValueError(f"Upload a {label} CSV first.")
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        raise ValueError(f"The {label} CSV is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("CSV column names must be unique.")
    return df


def q(name):
    """Quote column names for patsy formulas."""
    return f'Q("{name}")'


def coefficient_table(result):
    ci = result.conf_int(alpha=0.05)
    out = pd.DataFrame(
        {
            "Term": result.params.index,
            "Estimate": result.params.values,
            "Std. Error": result.bse.values,
            "t value": result.tvalues.values,
            "p value": result.pvalues.values,
            "CI 2.5%": ci.iloc[:, 0].values,
            "CI 97.5%": ci.iloc[:, 1].values,
        }
    )
    return out.round(5)


def model_metrics(result):
    out = pd.DataFrame(
        [
            ("Observations used", result.nobs),
            ("R-squared", result.rsquared),
            ("Adjusted R-squared", result.rsquared_adj),
            ("F-statistic", result.fvalue),
            ("F p-value", result.f_pvalue),
            ("AIC", result.aic),
            ("BIC", result.bic),
            ("Residual SE", np.sqrt(result.mse_resid)),
        ],
        columns=["Metric", "Value"],
    )
    return out.round(5)


def numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T
    out.insert(1, "Missing", numeric.isna().sum())
    out.insert(0, "Variable", out.index)
    return out.reset_index(drop=True).round(4)


def categorical_summary(df):
    categorical = df.select_dtypes(exclude=np.number)
    rows = []
    for col in categorical.columns:
        s = categorical[col]
        top = s.mode(dropna=True)
        rows.append(
            {
                "Variable": col,
                "Count": s.notna().sum(),
                "Missing": s.isna().sum(),
                "Unique": s.nunique(dropna=True),
                "Most frequent": "" if top.empty else top.iloc[0],
                "Frequency": 0 if top.empty else (s == top.iloc[0]).sum(),
            }
        )
    return pd.DataFrame(rows)


def download_dataframe(df, filename, label):
    button = widgets.Button(description=label, icon="download", button_style="primary")
    message = widgets.HTML()

    def on_click(_):
        directory = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
        path = directory / filename
        df.to_csv(path, index=False)
        try:
            from google.colab import files

            files.download(str(path))
            message.value = f"Download started: <code>{filename}</code>"
        except Exception:
            message.value = f"Saved to: <code>{path}</code>"

    button.on_click(on_click)
    return widgets.VBox([button, message])


def download_url(url, filename, label):
    button = widgets.Button(description=label, icon="download")
    message = widgets.HTML()

    def on_click(_):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            directory = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
            path = directory / filename
            path.write_bytes(response.content)
            try:
                from google.colab import files

                files.download(str(path))
                message.value = f"Download started: <code>{filename}</code>"
            except Exception:
                message.value = f"Saved to: <code>{path}</code>"
        except Exception as exc:
            message.value = f"Download failed: {exc}"

    button.on_click(on_click)
    return widgets.VBox([button, message])


# Sidebar-style controls
train_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Training CSV")
pred_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Prediction CSV")
y_widget = widgets.Dropdown(description="Y variable", style={"description_width": "initial"})
x_widget = widgets.SelectMultiple(
    description="X variables",
    rows=8,
    style={"description_width": "initial"},
    layout=widgets.Layout(width="95%"),
)
factor_widget = widgets.SelectMultiple(
    description="Categorical X",
    rows=5,
    style={"description_width": "initial"},
    layout=widgets.Layout(width="95%"),
)
run_button = widgets.Button(description="Run OLS", button_style="success", icon="play")
control_status = widgets.HTML("Upload the training CSV.")

# One output area per original Shiny tab.
overview_out = widgets.Output()
summary_out = widgets.Output()
correlation_out = widgets.Output()
ols_out = widgets.Output()
plots_out = widgets.Output()
fitted_out = widgets.Output()
prediction_out = widgets.Output()

tabs = widgets.Tab(
    children=[overview_out, summary_out, correlation_out, ols_out, plots_out, fitted_out, prediction_out]
)
for i, title in enumerate(
    ["Overview", "Summary Stats", "Correlation", "Summary OLS", "Residuals Plot", "Data with predicted Y", "Prediction"]
):
    tabs.set_title(i, title)


def update_x_choices(change=None):
    if TRAIN_DF is None or y_widget.value is None:
        return
    choices = [c for c in TRAIN_DF.columns if c != y_widget.value]
    old_x = [c for c in x_widget.value if c in choices]
    x_widget.options = choices
    x_widget.value = tuple(old_x or choices)
    factor_widget.options = list(x_widget.value)
    factor_widget.value = tuple(c for c in factor_widget.value if c in x_widget.value)


def on_factor_selection(change):
    # Categorical choices must be selected predictors.
    valid = tuple(c for c in change["new"] if c in x_widget.value)
    if valid != change["new"]:
        factor_widget.value = valid


def on_x_selection(change):
    factor_widget.options = list(change["new"])
    factor_widget.value = tuple(c for c in factor_widget.value if c in change["new"])


def on_train_upload(change):
    global TRAIN_DF
    try:
        TRAIN_DF = read_upload(train_upload, "training")
        y_widget.options = list(TRAIN_DF.columns)
        y_widget.value = TRAIN_DF.columns[0]
        update_x_choices()
        control_status.value = f"Training data loaded: <b>{TRAIN_DF.shape[0]:,} rows</b> × <b>{TRAIN_DF.shape[1]:,} columns</b>."
    except Exception as exc:
        control_status.value = f"Training-file error: {exc}"


def on_pred_upload(change):
    global PRED_DF
    try:
        PRED_DF = read_upload(pred_upload, "prediction")
        control_status.value = f"Prediction data loaded: <b>{PRED_DF.shape[0]:,} rows</b> × <b>{PRED_DF.shape[1]:,} columns</b>."
    except Exception as exc:
        control_status.value = f"Prediction-file error: {exc}"


def render_overview():
    with overview_out:
        clear_output(wait=True)
        display(Markdown("## OLS App"))
        display(
            Markdown(
                "Upload comma-separated CSV files with headers. Select one numeric outcome (Y), "
                "one or more predictors (X), and any categorical/non-metric predictors. "
                "The prediction CSV must contain every selected X column."
            )
        )
        display(Markdown("### Sample input files"))
        display(widgets.HBox([
            download_url(SAMPLE_TRAIN_URL, "beer data.csv", "Training sample"),
            download_url(SAMPLE_PRED_URL, "beer data - prediction sample.csv", "Prediction sample"),
        ]))


def show_correlation(model_df):
    numeric = model_df.select_dtypes(include=np.number)
    corr = numeric.corr() if numeric.shape[1] >= 2 else pd.DataFrame()
    display(Markdown("### Pairwise-complete Pearson correlation"))
    if corr.empty:
        print("At least two numeric variables are required.")
        return
    display(corr.round(4))
    size = max(5, min(12, 0.55 * len(corr.columns) + 3))
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=len(corr) <= 12, fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap")
    plt.tight_layout()
    plt.show()


def show_residual_plots(result, y):
    fitted = np.asarray(result.fittedvalues)
    residuals = np.asarray(result.resid)
    observed = np.asarray(result.model.endog)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].scatter(fitted, residuals, alpha=0.65)
    axes[0].axhline(0, color="crimson", linestyle="--")
    axes[0].set(title="Fitted values vs residuals", xlabel="Fitted values", ylabel="Residuals")

    axes[1].scatter(observed, fitted, alpha=0.65)
    limits = [min(observed.min(), fitted.min()), max(observed.max(), fitted.max())]
    axes[1].plot(limits, limits, color="crimson", linestyle="--")
    axes[1].set(title=f"Observed {y} vs fitted {y}", xlabel=f"Observed {y}", ylabel=f"Fitted {y}")

    axes[2].plot(np.arange(1, len(residuals) + 1), residuals, marker="o", markersize=3, linewidth=0.8)
    axes[2].axhline(0, color="crimson", linestyle="--")
    axes[2].set(title="Residuals by observation", xlabel="Observation", ylabel="Residual")

    plt.tight_layout()
    plt.show()


def run_ols(_):
    global MODEL, FITTED_DF, PREDICTED_DF

    try:
        if TRAIN_DF is None:
            raise ValueError("Upload a training CSV.")
        y = y_widget.value
        xs = list(x_widget.value)
        factors = set(factor_widget.value)
        if not y:
            raise ValueError("Select a Y variable.")
        if not xs:
            raise ValueError("Select at least one X variable.")
        if not pd.api.types.is_numeric_dtype(TRAIN_DF[y]):
            raise ValueError(f"Y variable '{y}' must be numeric.")

        selected = TRAIN_DF[[y] + xs].copy()
        for col in factors:
            selected[col] = selected[col].astype("category")

        formula = q(y) + " ~ " + " + ".join(f"C({q(c)})" if c in factors else q(c) for c in xs)
        MODEL = smf.ols(formula, data=selected).fit()

        # Same convention as server.R: numeric X standardized; raw Y and factors unchanged.
        standardized = selected.copy()
        for col in xs:
            if col not in factors and pd.api.types.is_numeric_dtype(standardized[col]):
                sd = standardized[col].std(ddof=1)
                standardized[col] = 0.0 if pd.isna(sd) or sd == 0 else (standardized[col] - standardized[col].mean()) / sd
        standardized_model = smf.ols(formula, data=standardized).fit()

        used = selected.loc[MODEL.fittedvalues.index].copy()
        used.insert(0, "Yhat", MODEL.fittedvalues)
        FITTED_DF = used

        with summary_out:
            clear_output(wait=True)
            display(Markdown(f"### Model data\n`{formula}`"))
            display(Markdown("#### Numeric data"))
            display(numeric_summary(selected))
            display(Markdown("#### Categorical/non-metric data"))
            display(categorical_summary(selected))

        with correlation_out:
            clear_output(wait=True)
            show_correlation(selected)

        with ols_out:
            clear_output(wait=True)
            display(Markdown("### Model-fit statistics"))
            display(model_metrics(MODEL))
            display(Markdown("### Summary OLS model"))
            display(coefficient_table(MODEL))
            display(Markdown("### Summary OLS standardized model"))
            display(Markdown("Numeric X variables are standardized; Y and categorical variables are unchanged."))
            display(coefficient_table(standardized_model))

        with plots_out:
            clear_output(wait=True)
            show_residual_plots(MODEL, y)

        with fitted_out:
            clear_output(wait=True)
            display(Markdown("### Data with predicted Y (preview)"))
            display(FITTED_DF.head(20))
            display(download_dataframe(FITTED_DF, "Data with predicted Y.csv", "Download fitted data"))

        with prediction_out:
            clear_output(wait=True)
            if PRED_DF is None:
                display(Markdown("Upload a prediction CSV and run OLS again to generate predictions."))
                PREDICTED_DF = None
            else:
                missing = [c for c in xs if c not in PRED_DF.columns]
                if missing:
                    raise ValueError(f"Prediction CSV is missing selected X columns: {missing}")
                predictors = PRED_DF[xs].copy()
                for col in factors:
                    training_levels = set(selected[col].dropna().astype(str))
                    unseen = sorted(set(predictors[col].dropna().astype(str)) - training_levels)
                    if unseen:
                        raise ValueError(f"Column '{col}' has unseen prediction categories: {unseen}")
                    predictors[col] = predictors[col].astype("category")

                complete = predictors.notna().all(axis=1)
                yhat = pd.Series(np.nan, index=PRED_DF.index)
                if complete.any():
                    yhat.loc[complete] = MODEL.predict(predictors.loc[complete])
                PREDICTED_DF = PRED_DF.copy()
                PREDICTED_DF.insert(0, "Yhat", yhat)

                display(Markdown("### First 10 rows of predicted data"))
                print('"Yhat" is the predicted value.')
                display(PREDICTED_DF.head(10))
                display(download_dataframe(PREDICTED_DF, "Predicted Data.csv", "Download predicted data"))

        control_status.value = (
            f"OLS fitted successfully with <b>{int(MODEL.nobs):,}</b> observations; "
            f"R² = <b>{MODEL.rsquared:.3f}</b>."
        )
        tabs.selected_index = 3

    except Exception as exc:
        control_status.value = f"OLS error: {exc}"
        with ols_out:
            clear_output(wait=True)
            display(Markdown(f"**Error:** {exc}"))


train_upload.observe(on_train_upload, names="value")
pred_upload.observe(on_pred_upload, names="value")
y_widget.observe(update_x_choices, names="value")
x_widget.observe(on_x_selection, names="value")
factor_widget.observe(on_factor_selection, names="value")
run_button.on_click(run_ols)

render_overview()

controls = widgets.VBox(
    [
        widgets.HTML("<h3>Data input</h3>"),
        train_upload,
        pred_upload,
        widgets.HTML("<h3>Data selection</h3>"),
        y_widget,
        x_widget,
        factor_widget,
        run_button,
        control_status,
    ],
    layout=widgets.Layout(width="30%", min_width="270px"),
)
results = widgets.Box([tabs], layout=widgets.Layout(width="70%"))
app = widgets.HBox([controls, results])
display(app)
