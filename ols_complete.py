# ols_complete.py
# Colab-native ipywidgets OLS app for metric conjoint demonstrations.
# Run from GitHub in Colab using requests.get(raw_url).text + exec(...).

import io
import json
import tempfile
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from IPython.display import HTML, Markdown, clear_output, display

TRAIN_DF = None
PRED_DF = None
MODEL = None
STD_MODEL = None
PREDICTED_DF = None


def uploaded_content(upload_widget):
    """Support ipywidgets FileUpload in versions 7 and 8."""
    value = upload_widget.value
    if not value:
        return None
    if isinstance(value, dict):
        return next(iter(value.values()))["content"]
    return value[0]["content"]


def read_upload(upload_widget, label):
    content = uploaded_content(upload_widget)
    if content is None:
        raise ValueError(f"Upload a {label} CSV first.")
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        raise ValueError(f"The {label} CSV has no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"The {label} CSV has duplicate column names.")
    return df


def q(name):
    return f'Q("{name}")'


def sorted_levels(series):
    values = pd.Series(series.dropna().unique())
    return sorted(values.tolist(), key=lambda x: str(x))


def selected_data():
    if TRAIN_DF is None:
        raise ValueError("Upload the training CSV in the Data tab.")
    y = y_widget.value
    xs = list(x_widget.value)
    categorical = set(factor_widget.value)

    if not y:
        raise ValueError("Select the Y variable.")
    if not xs:
        raise ValueError("Select at least one X variable.")
    if not pd.api.types.is_numeric_dtype(TRAIN_DF[y]):
        raise ValueError(f"Y variable '{y}' must be numeric for OLS.")

    missing = [c for c in [y] + xs if c not in TRAIN_DF.columns]
    if missing:
        raise ValueError(f"Missing selected columns: {missing}")

    df = TRAIN_DF[[y] + xs].copy()
    nonnumeric = [c for c in xs if c not in categorical and not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric:
        raise ValueError(f"Mark these as categorical X variables: {nonnumeric}")

    for col in categorical:
        df[col] = pd.Categorical(df[col], categories=sorted_levels(df[col]))
    return df, y, xs, categorical


def make_formula(df, y, xs, categorical, first_baseline=True):
    terms = []
    for col in xs:
        if col in categorical:
            levels = list(df[col].cat.categories)
            if first_baseline:
                terms.append(f"C({q(col)})")
            else:
                reference = json.dumps(str(levels[-1]))
                terms.append(f"C({q(col)}, Treatment(reference={reference}))")
        else:
            terms.append(q(col))
    return f"{q(y)} ~ " + " + ".join(terms)


def html_table(df, index=False):
    return HTML(
        df.to_html(
            index=index,
            border=0,
            classes="table table-striped table-hover table-sm",
            justify="center",
        )
    )


def coefficient_table(result):
    ci = result.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "Term": result.params.index,
            "Coefficient": result.params.values,
            "Std. Error": result.bse.values,
            "t": result.tvalues.values,
            "p-value": result.pvalues.values,
            "CI 2.5%": ci.iloc[:, 0].values,
            "CI 97.5%": ci.iloc[:, 1].values,
        }
    )
    return table.round(5)


def fit_table(result):
    table = pd.DataFrame(
        {
            "Metric": [
                "Observations",
                "R-squared",
                "Adjusted R-squared",
                "F-statistic",
                "F p-value",
                "Residual degrees of freedom",
                "AIC",
                "BIC",
            ],
            "Value": [
                int(result.nobs),
                result.rsquared,
                result.rsquared_adj,
                result.fvalue,
                result.f_pvalue,
                int(result.df_resid),
                result.aic,
                result.bic,
            ],
        }
    )
    return table.round(5)


def numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame({"Message": ["No numeric variables selected."]})
    out = numeric.describe().T
    out.insert(0, "Variable", out.index)
    out.insert(2, "Missing", numeric.isna().sum().values)
    return out.reset_index(drop=True).round(4)


def categorical_summary(df):
    categorical = df.select_dtypes(include=["category", "object", "string", "bool"])
    rows = []
    for col in categorical.columns:
        s = categorical[col]
        counts = s.value_counts(dropna=False)
        levels = sorted_levels(s)
        rows.append(
            {
                "Variable": col,
                "N": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "Levels": int(s.nunique(dropna=True)),
                "Baseline": "" if not levels else str(levels[0]),
                "Most frequent": "" if counts.empty else str(counts.index[0]),
                "Frequency": 0 if counts.empty else int(counts.iloc[0]),
            }
        )
    if not rows:
        return pd.DataFrame({"Message": ["No categorical variables selected."]})
    return pd.DataFrame(rows)


def standardized_data(df, xs, categorical):
    out = df.copy()
    for col in xs:
        if col in categorical:
            continue
        sd = out[col].std(ddof=1)
        out[col] = 0.0 if pd.isna(sd) or sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def download_button(df, filename, label):
    button = widgets.Button(description=label, button_style="primary", icon="download")
    status = widgets.HTML()

    def on_click(_):
        directory = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
        path = directory / filename
        df.to_csv(path, index=False)
        try:
            from google.colab import files

            files.download(str(path))
            status.value = f"Download started: <code>{filename}</code>"
        except Exception:
            status.value = f"Saved to: <code>{path}</code>"

    button.on_click(on_click)
    return widgets.VBox([button, status])


# Widgets
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
baseline_widget = widgets.Checkbox(
    value=True,
    description="Use first option as baseline (unchecked: last option)",
    indent=False,
    style={"description_width": "initial"},
)
run_button = widgets.Button(description="Fit OLS model", button_style="success", icon="play")
formula_html = widgets.HTML("Select variables to preview the OLS equation.")

data_preview_out = widgets.Output()
selection_status_out = widgets.Output()
summary_out = widgets.Output()
ols_out = widgets.Output()
std_ols_out = widgets.Output()
prediction_out = widgets.Output()


def refresh_x_options(change=None):
    if TRAIN_DF is None or y_widget.value is None:
        return
    choices = [c for c in TRAIN_DF.columns if c != y_widget.value]
    retained = [c for c in x_widget.value if c in choices]
    x_widget.options = choices
    x_widget.value = tuple(retained or choices)
    refresh_factor_options()
    refresh_formula_preview()


def refresh_factor_options(change=None):
    selected = list(x_widget.value)
    retained = [c for c in factor_widget.value if c in selected]
    factor_widget.options = selected
    factor_widget.value = tuple(retained)
    refresh_formula_preview()


def refresh_formula_preview(change=None):
    try:
        df, y, xs, categorical = selected_data()
        formula = make_formula(df, y, xs, categorical, baseline_widget.value)
        first_last = "first" if baseline_widget.value else "last"
        formula_html.value = (
            "<b>OLS equation:</b><br>"
            f"<code>{formula}</code><br>"
            f"<small>Categorical baseline: {first_last} level.</small>"
        )
    except Exception:
        formula_html.value = "Select variables to preview the OLS equation."


def preview_data(change=None):
    with data_preview_out:
        clear_output(wait=True)
        if TRAIN_DF is None and PRED_DF is None:
            print("Upload CSV files above.")
            return
        if TRAIN_DF is not None:
            display(Markdown(f"### Training data preview — {TRAIN_DF.shape[0]:,} rows × {TRAIN_DF.shape[1]:,} columns"))
            display(TRAIN_DF.head(10))
        if PRED_DF is not None:
            display(Markdown(f"### Prediction data preview — {PRED_DF.shape[0]:,} rows × {PRED_DF.shape[1]:,} columns"))
            display(PRED_DF.head(10))


def on_train_upload(change):
    global TRAIN_DF
    with data_preview_out:
        clear_output(wait=True)
    try:
        TRAIN_DF = read_upload(train_upload, "training")
        y_widget.options = list(TRAIN_DF.columns)
        y_widget.value = TRAIN_DF.columns[0]
        refresh_x_options()
        preview_data()
    except Exception as exc:
        with data_preview_out:
            display(Markdown(f"**Training-file error:** {exc}"))


def on_pred_upload(change):
    global PRED_DF
    try:
        PRED_DF = read_upload(pred_upload, "prediction")
        preview_data()
    except Exception as exc:
        with data_preview_out:
            display(Markdown(f"**Prediction-file error:** {exc}"))


def render_summary(df):
    with summary_out:
        clear_output(wait=True)
        display(Markdown("## Data summary"))
        display(Markdown(f"Selected data: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**"))
        display(Markdown("### Numeric variables"))
        display(html_table(numeric_summary(df)))
        display(Markdown("### Categorical variables"))
        display(html_table(categorical_summary(df)))


def render_regular_ols(result):
    with ols_out:
        clear_output(wait=True)
        display(Markdown("## Regular OLS results"))
        display(Markdown("### Fit statistics"))
        display(html_table(fit_table(result)))
        display(Markdown("### Coefficients"))
        display(html_table(coefficient_table(result)))


def render_standardized_ols(result):
    with std_ols_out:
        clear_output(wait=True)
        display(Markdown("## Standardized OLS results"))
        display(Markdown("Numeric X variables are z-standardized; Y remains in its original units."))
        display(Markdown("### Fit statistics"))
        display(html_table(fit_table(result)))
        display(Markdown("### Standardized-X coefficients"))
        display(html_table(coefficient_table(result)))


def render_prediction(result, df, xs, categorical):
    global PREDICTED_DF
    with prediction_out:
        clear_output(wait=True)
        display(Markdown("## Prediction results"))
        if PRED_DF is None:
            display(Markdown("Upload a prediction CSV in the Data tab, then fit the model again."))
            PREDICTED_DF = None
            return

        missing = [c for c in xs if c not in PRED_DF.columns]
        if missing:
            raise ValueError(f"Prediction CSV is missing selected X columns: {missing}")

        predictors = PRED_DF[xs].copy()
        for col in categorical:
            training_levels = set(df[col].dropna().astype(str))
            unseen = sorted(set(predictors[col].dropna().astype(str)) - training_levels)
            if unseen:
                raise ValueError(f"Column '{col}' contains unseen prediction categories: {unseen}")
            predictors[col] = pd.Categorical(predictors[col], categories=list(df[col].cat.categories))

        complete = predictors.notna().all(axis=1)
        yhat = pd.Series(np.nan, index=PRED_DF.index, name="Yhat")
        if complete.any():
            yhat.loc[complete] = result.predict(predictors.loc[complete])

        PREDICTED_DF = PRED_DF.copy()
        PREDICTED_DF.insert(0, "Yhat", yhat)
        display(Markdown("### First 10 predicted rows"))
        display(PREDICTED_DF.head(10))
        display(download_button(PREDICTED_DF, "Predicted Data.csv", "Download predicted CSV"))


def fit_ols(_):
    global MODEL, STD_MODEL
    try:
        df, y, xs, categorical = selected_data()
        first_baseline = baseline_widget.value
        formula = make_formula(df, y, xs, categorical, first_baseline)

        MODEL = smf.ols(formula, data=df).fit()
        std_df = standardized_data(df, xs, categorical)
        STD_MODEL = smf.ols(formula, data=std_df).fit()

        refresh_formula_preview()
        render_summary(df)
        render_regular_ols(MODEL)
        render_standardized_ols(STD_MODEL)
        render_prediction(MODEL, df, xs, categorical)

        with selection_status_out:
            clear_output(wait=True)
            display(
                Markdown(
                    f"**Model fitted.** Observations: **{int(MODEL.nobs):,}**; "
                    f"R²: **{MODEL.rsquared:.3f}**; F-statistic: **{MODEL.fvalue:.3f}**."
                )
            )
        tabs.selected_index = 3

    except Exception as exc:
        with selection_status_out:
            clear_output(wait=True)
            display(Markdown(f"**OLS error:** {exc}"))


train_upload.observe(on_train_upload, names="value")
pred_upload.observe(on_pred_upload, names="value")
y_widget.observe(refresh_x_options, names="value")
x_widget.observe(refresh_factor_options, names="value")
factor_widget.observe(lambda change: refresh_formula_preview(), names="value")
baseline_widget.observe(refresh_formula_preview, names="value")
run_button.on_click(fit_ols)

# Six Colab-oriented tabs.
tab0 = widgets.VBox(
    [
        widgets.HTML("<h2>0. Data load and preview</h2>"),
        train_upload,
        pred_upload,
        widgets.HTML("<small>Prediction CSV is optional and should contain all selected X columns.</small>"),
        data_preview_out,
    ]
)
tab1 = widgets.VBox(
    [
        widgets.HTML("<h2>1. Variable selection and OLS equation</h2>"),
        y_widget,
        x_widget,
        factor_widget,
        baseline_widget,
        formula_html,
        run_button,
        selection_status_out,
    ]
)
tabs = widgets.Tab(children=[tab0, tab1, summary_out, ols_out, std_ols_out, prediction_out])
for index, title in enumerate(
    ["0. Data", "1. Variables", "2. Summary", "3. OLS", "4. Standardized OLS", "5. Prediction"]
):
    tabs.set_title(index, title)

display(widgets.HTML("<h1>OLS App for Colab</h1>"))
display(tabs)
preview_data()
