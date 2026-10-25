import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
import ipywidgets as widgets
from IPython.display import display, clear_output

# Expected CSV: one row per conjoint profile;
# numeric attribute columns + one numeric preference/rating column.

upload = widgets.FileUpload(accept=".csv", multiple=False, description="Upload CSV")
y_box = widgets.Dropdown(description="Rating (Y)", style={"description_width": "initial"})
x_box = widgets.SelectMultiple(
    description="Attributes (X)",
    rows=8,
    style={"description_width": "initial"},
    layout=widgets.Layout(width="60%")
)
load_btn = widgets.Button(description="Load data", button_style="info")
run_btn = widgets.Button(description="Run conjoint OLS", button_style="success")
output = widgets.Output()

df = None


def get_uploaded_bytes():
    """Handle both ipywidgets 7 and 8 upload formats."""
    value = upload.value
    if isinstance(value, dict):
        return next(iter(value.values()))["content"]
    return value[0]["content"]


def load_data(_):
    global df
    with output:
        clear_output()
        try:
            df = pd.read_csv(io.BytesIO(get_uploaded_bytes()))
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(numeric_cols) < 2:
                raise ValueError("Need at least two numeric columns: attributes and rating.")

            y_box.options = numeric_cols
            y_box.value = numeric_cols[-1]  # assumes rating is the last numeric column
            x_box.options = [c for c in numeric_cols if c != y_box.value]
            x_box.value = tuple(x_box.options)

            print(f"Loaded {df.shape[0]} profiles and {df.shape[1]} columns.")
            display(df.head())
        except Exception as e:
            print("Error:", e)


def update_x_options(change):
    if df is None:
        return
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    choices = [c for c in numeric_cols if c != change["new"]]
    x_box.options = choices
    x_box.value = tuple(choices)


def run_ols(_):
    with output:
        clear_output()
        try:
            xs = list(x_box.value)
            if not xs:
                raise ValueError("Select at least one attribute.")

            # Complete-case OLS
            data = df[[y_box.value] + xs].dropna()
            X = sm.add_constant(data[xs])
            model = sm.OLS(data[y_box.value], X).fit()

            print(f"Observations used: {int(model.nobs)}")
            print(f"R² = {model.rsquared:.3f}; Adjusted R² = {model.rsquared_adj:.3f}\n")
            print(model.summary())

            # Metric conjoint attribute importance:
            # absolute coefficient × observed attribute range
            ranges = data[xs].max() - data[xs].min()
            importance = model.params[xs].abs() * ranges
            importance = 100 * importance / importance.sum()

            results = pd.DataFrame({
                "Coefficient": model.params[xs],
                "p-value": model.pvalues[xs],
                "Importance (%)": importance
            })

            print("\nConjoint results:")
            display(results.round(3))

        except Exception as e:
            print("Error:", e)


load_btn.on_click(load_data)
run_btn.on_click(run_ols)
y_box.observe(update_x_options, names="value")

display(
    widgets.VBox([
        widgets.HTML("<h3>Metric Conjoint OLS Demo</h3>"),
        upload,
        load_btn,
        y_box,
        x_box,
        run_btn
    ]),
    output
)
