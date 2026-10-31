# logit_cbc.py
# Generic Colab/ip ywidgets conditional-logit app for choice-based conjoint data.
# Expected input: long-format CSV, one row per alternative, one choice per respondent-task.

import io
import tempfile
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, clear_output, display
from scipy.stats import chi2
from statsmodels.discrete.conditional_models import ConditionalLogit

NONE = "(none)"
DATA = None
SCENARIO_DATA = None
MODEL_RESULT = None
DESIGN_SPEC = None
TRAIN_X = None
PREDICTION_OUTPUT = None
DROPPED_TERMS = []


def uploaded_content(upload_widget):
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


def html_table(df, index=False):
    return HTML(
        df.to_html(
            index=index,
            border=0,
            classes="table table-striped table-hover table-sm",
            justify="center",
        )
    )


def sorted_levels(values):
    return sorted(pd.Series(values).dropna().unique().tolist(), key=lambda x: str(x))


def find_default(columns, keywords, fallback_position):
    lower = {str(c).lower(): c for c in columns}
    for name, column in lower.items():
        if any(k in name for k in keywords):
            return column
    return columns[min(fallback_position, len(columns) - 1)]


def make_group_key(df, respondent_col, task_col):
    return df[respondent_col].astype(str) + " || " + df[task_col].astype(str)


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


def validate_choice_structure(df, respondent_col, task_col, choice_col):
    if choice_col not in df.columns:
        raise ValueError("Select a chosen-indicator column.")
    chosen = pd.to_numeric(df[choice_col], errors="coerce")
    if chosen.isna().any() or not set(chosen.unique()).issubset({0, 1}):
        raise ValueError("The chosen indicator must contain only 0 and 1 values.")

    groups = make_group_key(df, respondent_col, task_col)
    structure = pd.DataFrame({"group": groups, "choice": chosen.astype(int)})
    group_stats = structure.groupby("group")["choice"].agg(["size", "sum"])
    bad_count = group_stats[group_stats["size"] < 2]
    bad_choice = group_stats[group_stats["sum"] != 1]
    if not bad_count.empty:
        examples = bad_count.index[:5].tolist()
        raise ValueError(f"Every choice set needs at least two alternatives. Examples: {examples}")
    if not bad_choice.empty:
        examples = bad_choice.index[:5].tolist()
        raise ValueError(f"Every choice set needs exactly one chosen row. Examples: {examples}")
    return groups, chosen.astype(int)


def prepare_training_data():
    if DATA is None:
        raise ValueError("Upload CBC training data in Tab 0.")

    respondent = respondent_widget.value
    task = task_widget.value
    alternative = alternative_widget.value
    choice = choice_widget.value
    xs = list(x_widget.value)
    categorical = set(factor_widget.value)

    if not respondent or not task or not choice:
        raise ValueError("Select respondent, task/choice-set, and chosen-indicator columns.")
    if not xs:
        raise ValueError("Select at least one attribute X variable.")
    if len({respondent, task, choice}) < 3:
        raise ValueError("Respondent, task, and choice columns must be different.")

    required = [respondent, task, choice] + xs
    if alternative != NONE:
        required.append(alternative)
    missing = [c for c in required if c not in DATA.columns]
    if missing:
        raise ValueError(f"Training data is missing selected columns: {missing}")

    df = DATA[required].copy()
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Selected training columns contain missing values: {bad}")

    for col in xs:
        if col in categorical:
            levels = sorted_levels(df[col])
            df[col] = pd.Categorical(df[col], categories=levels)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Mark non-numeric attribute '{col}' as categorical.")

    groups, chosen = validate_choice_structure(df, respondent, task, choice)
    df["_group"] = groups
    df["_chosen"] = chosen

    spec = {
        "respondent": respondent,
        "task": task,
        "alternative": alternative,
        "choice": choice,
        "xs": xs,
        "categorical": sorted(categorical),
        "levels": {c: sorted_levels(df[c]) for c in categorical},
        "first_baseline": bool(baseline_widget.value),
        "include_asc": bool(asc_widget.value and alternative != NONE),
        "alt_levels": sorted_levels(df[alternative]) if alternative != NONE else [],
    }
    return df, spec


def build_design(df, spec, training=False):
    xs = spec["xs"]
    categorical = set(spec["categorical"])
    parts = []

    for col in xs:
        if col in categorical:
            if training:
                levels = spec["levels"][col]
                values = pd.Categorical(df[col], categories=levels)
            else:
                levels = spec["levels"][col]
                unseen = sorted(set(df[col].dropna().astype(str)) - {str(v) for v in levels})
                if unseen:
                    raise ValueError(f"Column '{col}' contains unseen levels: {unseen}")
                values = pd.Categorical(df[col], categories=levels)
            dummies = pd.get_dummies(values, prefix=col, prefix_sep=" = ", dtype=float)
            dummies.index = df.index
            dummies = dummies.iloc[:, 1:] if spec["first_baseline"] else dummies.iloc[:, :-1]
            parts.append(dummies)
        else:
            numeric = pd.to_numeric(df[col], errors="raise").astype(float)
            parts.append(numeric.rename(col).to_frame())

    if spec["include_asc"]:
        alt_col = spec["alternative"]
        levels = spec["alt_levels"]
        if not training:
            unseen = sorted(set(df[alt_col].dropna().astype(str)) - {str(v) for v in levels})
            if unseen:
                raise ValueError(f"Alternative column contains unseen labels: {unseen}")
        values = pd.Categorical(df[alt_col], categories=levels)
        dummies = pd.get_dummies(values, prefix="Alternative", prefix_sep=" = ", dtype=float)
        dummies.index = df.index
        dummies = dummies.iloc[:, 1:] if spec["first_baseline"] else dummies.iloc[:, :-1]
        parts.append(dummies)

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if X.empty:
        raise ValueError("The model needs at least one estimable attribute or alternative-specific constant.")
    return X


def drop_collinear_columns(X):
    """Drop zero-variance or exactly redundant columns while preserving order."""
    kept = []
    dropped = []
    rank = 0
    for col in X.columns:
        candidate = X[kept + [col]].to_numpy(dtype=float)
        if np.allclose(candidate[:, -1], candidate[0, -1]):
            dropped.append(col)
            continue
        new_rank = np.linalg.matrix_rank(candidate)
        if new_rank > rank:
            kept.append(col)
            rank = new_rank
        else:
            dropped.append(col)
    if not kept:
        raise ValueError("All model terms are constant or perfectly collinear.")
    return X[kept].copy(), dropped


class ScaledResult:
    """ConditionalLogit result rescaled from standardized back to original units.

    Numeric attributes are z-standardized before fitting for numerical
    stability. Coefficients, standard errors and confidence intervals are
    divided by the attribute standard deviation, so all reported values are
    in original units (e.g., per rupee, per mAh). z, p and log-likelihood
    are invariant to this linear rescaling.
    """

    def __init__(self, result, scales):
        self.llf = result.llf
        s = pd.Series(1.0, index=result.params.index)
        for col, value in scales.items():
            if col in s.index:
                s[col] = value
        self.params = result.params / s
        self.bse = result.bse / s
        self.tvalues = self.params / self.bse
        self.pvalues = result.pvalues
        self._ci = result.conf_int(alpha=0.05).div(s, axis=0)

    def conf_int(self, alpha=0.05):
        return self._ci


def prepare_scenario_data(spec):
    if SCENARIO_DATA is None:
        return None
    required = [spec["respondent"], spec["task"]] + spec["xs"]
    if spec["alternative"] != NONE:
        required.append(spec["alternative"])
    missing = [c for c in required if c not in SCENARIO_DATA.columns]
    if missing:
        raise ValueError(f"Scenario CSV is missing selected columns: {missing}")
    df = SCENARIO_DATA[required].copy()
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Selected scenario columns contain missing values: {bad}")
    df["_group"] = make_group_key(df, spec["respondent"], spec["task"])
    if df["_group"].nunique() == 0:
        raise ValueError("The scenario CSV has no choice sets.")
    return df


def fit_statistics(result, groups, k):
    sizes = pd.Series(groups).value_counts()
    ll_null = float(-np.log(sizes).sum())
    ll = float(result.llf)
    lr = 2 * (ll - ll_null)
    df_model = int(k)
    table = pd.DataFrame(
        {
            "Metric": [
                "Choice sets",
                "Alternative rows",
                "Parameters",
                "Log-likelihood",
                "Null log-likelihood",
                "McFadden pseudo R²",
                "Likelihood-ratio χ²",
                "LR p-value",
                "AIC",
                "BIC",
            ],
            "Value": [
                int(sizes.size),
                int(len(groups)),
                k,
                ll,
                ll_null,
                1 - ll / ll_null,
                lr,
                chi2.sf(lr, df_model),
                -2 * ll + 2 * k,
                -2 * ll + np.log(sizes.size) * k,
            ],
        }
    )
    return table.round(5)


def coefficient_table(result):
    ci = result.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "Term": result.params.index,
            "Coefficient": result.params.values,
            "Std. Error": result.bse.values,
            "z": result.tvalues.values,
            "p-value": result.pvalues.values,
            "CI 2.5%": ci.iloc[:, 0].values,
            "CI 97.5%": ci.iloc[:, 1].values,
            "exp(Coefficient)": np.exp(result.params.values),
        }
    )
    return table.round(5)


def grouped_probabilities(utilities, groups):
    utility = pd.Series(utilities, index=pd.RangeIndex(len(utilities)))
    group_series = pd.Series(groups, index=utility.index)
    probabilities = pd.Series(np.nan, index=utility.index, dtype=float)
    for _, idx in group_series.groupby(group_series).groups.items():
        values = utility.loc[idx].astype(float)
        shifted = values - values.max()
        probabilities.loc[idx] = np.exp(shifted) / np.exp(shifted).sum()
    return probabilities.to_numpy()


def build_prediction_output(source_df, X, result, spec, include_choice):
    utilities = X.to_numpy(dtype=float) @ result.params.to_numpy(dtype=float)
    probabilities = grouped_probabilities(utilities, source_df["_group"].to_numpy())
    out = pd.DataFrame(
        {
            "Respondent": source_df[spec["respondent"]].values,
            "Task": source_df[spec["task"]].values,
        }
    )
    if spec["alternative"] != NONE:
        out["Alternative"] = source_df[spec["alternative"]].values
    else:
        out["Alternative"] = source_df.groupby("_group").cumcount() + 1
    if include_choice and spec["choice"] in source_df.columns:
        out["Chosen"] = source_df["_chosen"].values
    out["Utility"] = utilities
    out["Choice probability"] = probabilities
    out["Predicted chosen"] = out.groupby(["Respondent", "Task"])["Choice probability"].transform(
        lambda s: s.eq(s.max()).astype(int)
    )
    return out.round(6)


def partworth_table(df, X, result, spec):
    rows = []
    params = result.params
    for col in spec["xs"]:
        if col in spec["categorical"]:
            levels = spec["levels"][col]
            baseline = levels[0] if spec["first_baseline"] else levels[-1]
            rows.append({"Attribute": col, "Level": baseline, "Utility": 0.0, "Type": "Baseline"})
            for level in levels:
                term = f"{col} = {level}"
                if term in params.index:
                    rows.append({"Attribute": col, "Level": level, "Utility": params[term], "Type": "Categorical"})
        else:
            if col in params.index:
                coefficient = params[col]
                rows.append({"Attribute": col, "Level": "Per unit", "Utility": coefficient, "Type": "Numeric"})
            else:
                rows.append({"Attribute": col, "Level": "Per unit", "Utility": np.nan, "Type": "Dropped: collinear"})

    if spec["include_asc"]:
        levels = spec["alt_levels"]
        baseline = levels[0] if spec["first_baseline"] else levels[-1]
        rows.append({"Attribute": "Alternative", "Level": baseline, "Utility": 0.0, "Type": "ASC baseline"})
        for level in levels:
            term = f"Alternative = {level}"
            if term in params.index:
                rows.append({"Attribute": "Alternative", "Level": level, "Utility": params[term], "Type": "ASC"})

    partworths = pd.DataFrame(rows)

    importance_rows = []
    for col in spec["xs"]:
        if col in spec["categorical"]:
            utilities = partworths.loc[partworths["Attribute"].eq(col), "Utility"]
            attribute_range = utilities.max() - utilities.min()
        else:
            if col in params.index:
                coefficient = params[col]
                attribute_range = abs(coefficient) * (df[col].max() - df[col].min())
            else:
                attribute_range = 0.0
        importance_rows.append({"Attribute": col, "Utility range": attribute_range})
    importance = pd.DataFrame(importance_rows)
    total = importance["Utility range"].sum()
    importance["Importance (%)"] = 0.0 if total == 0 else 100 * importance["Utility range"] / total
    return partworths.round(5), importance.sort_values("Importance (%)", ascending=False).round(3)


# Widgets
train_upload = widgets.FileUpload(accept=".csv", multiple=False, description="CBC training CSV")
scenario_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Scenario CSV (optional)")

respondent_widget = widgets.Dropdown(description="Respondent ID", style={"description_width": "initial"})
task_widget = widgets.Dropdown(description="Task / choice set", style={"description_width": "initial"})
alternative_widget = widgets.Dropdown(description="Alternative label", style={"description_width": "initial"})
choice_widget = widgets.Dropdown(description="Chosen (0/1)", style={"description_width": "initial"})
x_widget = widgets.SelectMultiple(
    description="Attribute X variables", rows=8, style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
factor_widget = widgets.SelectMultiple(
    description="Categorical X", rows=5, style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
baseline_widget = widgets.Checkbox(
    value=True,
    description="Use first option as baseline (unchecked: last option)",
    indent=False,
    style={"description_width": "initial"},
)
asc_widget = widgets.Checkbox(
    value=False,
    description="Include alternative-specific constants",
    indent=False,
    style={"description_width": "initial"},
)
fit_button = widgets.Button(description="Fit CBC logit", button_style="success", icon="play")
formula_html = widgets.HTML("Select variables to preview the choice model.")

data_preview_out = widgets.Output()
setup_status_out = widgets.Output()
summary_out = widgets.Output()
logit_out = widgets.Output()
insights_out = widgets.Output()
prediction_out = widgets.Output()


def role_columns():
    roles = [respondent_widget.value, task_widget.value, choice_widget.value]
    if alternative_widget.value != NONE:
        roles.append(alternative_widget.value)
    return [r for r in roles if r]


def refresh_formula_preview(change=None):
    try:
        if DATA is None or not x_widget.value:
            formula_html.value = "Select variables to preview the choice model."
            return
        xs = list(x_widget.value)
        cats = set(factor_widget.value)
        terms = [f"C({c})" if c in cats else c for c in xs]
        if asc_widget.value and alternative_widget.value != NONE:
            terms.append("alternative-specific constants")
        first_last = "first" if baseline_widget.value else "last"
        formula_html.value = (
            "<b>Conditional choice model:</b><br>"
            f"<code>choice | respondent × task ~ {' + '.join(terms)}</code><br>"
            f"<small>Categorical baseline: {first_last} level. No overall intercept is estimated.</small>"
        )
    except Exception:
        formula_html.value = "Select variables to preview the choice model."


def refresh_factor_options(change=None):
    selected = list(x_widget.value)
    factor_widget.options = selected
    factor_widget.value = tuple(c for c in factor_widget.value if c in selected)
    refresh_formula_preview()


def refresh_x_options(change=None):
    if DATA is None:
        return
    excluded = set(role_columns())
    choices = [c for c in DATA.columns if c not in excluded]
    retained = [c for c in x_widget.value if c in choices]
    x_widget.options = choices
    x_widget.value = tuple(retained or choices)
    refresh_factor_options()


def preview_data():
    with data_preview_out:
        clear_output(wait=True)
        if DATA is None and SCENARIO_DATA is None:
            display(Markdown("Upload CBC training data. Scenario data is optional."))
            return
        if DATA is not None:
            display(Markdown(f"### CBC training preview — {DATA.shape[0]:,} rows × {DATA.shape[1]:,} columns"))
            display(DATA.head(10))
        if SCENARIO_DATA is not None:
            display(Markdown(f"### Scenario preview — {SCENARIO_DATA.shape[0]:,} rows × {SCENARIO_DATA.shape[1]:,} columns"))
            display(SCENARIO_DATA.head(10))


def on_train_upload(change):
    global DATA
    try:
        DATA = read_upload(train_upload, "CBC training")
        columns = list(DATA.columns)
        respondent_widget.options = columns
        task_widget.options = columns
        alternative_widget.options = [NONE] + columns
        choice_widget.options = columns

        respondent_widget.value = find_default(columns, ["respondent", "resp", "participant"], 0)
        task_widget.value = find_default(columns, ["task", "set", "question", "trial"], 1)
        alternative_widget.value = find_default(columns, ["alternative", "alt", "option", "profile"], 2)
        choice_widget.value = find_default(columns, ["chosen", "choice", "selected", "pick"], 3)
        refresh_x_options()
        preview_data()
    except Exception as exc:
        with data_preview_out:
            clear_output(wait=True)
            display(Markdown(f"**Training-file error:** {exc}"))


def on_scenario_upload(change):
    global SCENARIO_DATA
    try:
        SCENARIO_DATA = read_upload(scenario_upload, "scenario")
        preview_data()
    except Exception as exc:
        with data_preview_out:
            clear_output(wait=True)
            display(Markdown(f"**Scenario-file error:** {exc}"))


def numeric_summary(df, xs, categorical):
    numeric = [c for c in xs if c not in categorical]
    if not numeric:
        return pd.DataFrame({"Message": ["No numeric attributes selected."]})
    out = df[numeric].describe().T
    out.insert(0, "Attribute", out.index)
    return out.reset_index(drop=True).round(4)


def categorical_choice_summary(df, xs, categorical):
    rows = []
    for col in [c for c in xs if c in categorical]:
        for level, sub in df.groupby(col, observed=False):
            rows.append(
                {
                    "Attribute": col,
                    "Level": level,
                    "Alternatives": len(sub),
                    "Times chosen": int(sub["_chosen"].sum()),
                    "Choice share": sub["_chosen"].mean(),
                }
            )
    if not rows:
        return pd.DataFrame({"Message": ["No categorical attributes selected."]})
    return pd.DataFrame(rows).round(4)


def render_summary(df, spec):
    groups = df["_group"]
    group_sizes = groups.value_counts()
    overview = pd.DataFrame(
        {
            "Metric": [
                "Respondents",
                "Task/choice-set values",
                "Unique respondent-task sets",
                "Alternative rows",
                "Minimum alternatives per set",
                "Maximum alternatives per set",
                "Observed choice rate",
            ],
            "Value": [
                df[spec["respondent"]].nunique(),
                df[spec["task"]].nunique(),
                groups.nunique(),
                len(df),
                int(group_sizes.min()),
                int(group_sizes.max()),
                df["_chosen"].mean(),
            ],
        }
    ).round(4)

    with summary_out:
        clear_output(wait=True)
        display(Markdown("## Choice data summary"))
        display(html_table(overview))
        if spec["alternative"] != NONE:
            alt = df.groupby(spec["alternative"])["_chosen"].agg(Alternatives="size", Chosen="sum", Choice_share="mean").reset_index()
            display(Markdown("### Choice share by alternative"))
            display(html_table(alt.round(4)))
        display(Markdown("### Numeric attributes"))
        display(html_table(numeric_summary(df, spec["xs"], set(spec["categorical"]))))
        display(Markdown("### Categorical attributes and choices"))
        display(html_table(categorical_choice_summary(df, spec["xs"], set(spec["categorical"]))))


def render_results(result, groups, k):
    with logit_out:
        clear_output(wait=True)
        display(Markdown("## CBC conditional-logit results"))
        display(Markdown("### Fit statistics"))
        display(html_table(fit_statistics(result, groups, k)))
        display(Markdown("### Utility coefficients"))
        display(html_table(coefficient_table(result)))


def render_insights(df, X, result, spec):
    partworths, importance = partworth_table(df, X, result, spec)
    with insights_out:
        clear_output(wait=True)
        display(Markdown("## Attribute insights"))
        display(Markdown("### Part-worth utilities"))
        display(html_table(partworths))
        display(Markdown("### Relative attribute importance"))
        display(html_table(importance))
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(importance) + 1.5)))
        ax.barh(importance["Attribute"], importance["Importance (%)"], color="#4C78A8")
        ax.invert_yaxis()
        ax.set_xlabel("Relative importance (%)")
        ax.set_title("Attribute importance based on utility ranges")
        plt.tight_layout()
        plt.show()


def render_predictions(train_df, train_X, result, spec):
    global PREDICTION_OUTPUT
    scenario_df = prepare_scenario_data(spec)
    if scenario_df is None:
        source = train_df
        X = train_X
        title = "In-sample choice probabilities"
        include_choice = True
    else:
        source = scenario_df
        X = build_design(scenario_df, spec, training=False)
        X = X.reindex(columns=spec["model_columns"], fill_value=0.0)
        title = "Scenario choice probabilities"
        include_choice = spec["choice"] in scenario_df.columns

    PREDICTION_OUTPUT = build_prediction_output(source, X, result, spec, include_choice)
    with prediction_out:
        clear_output(wait=True)
        display(Markdown(f"## {title}"))
        display(Markdown("Probabilities sum to 1 within each respondent-task choice set."))
        display(PREDICTION_OUTPUT.head(20))
        display(download_button(PREDICTION_OUTPUT, "cbc_choice_probabilities.csv", "Download prediction CSV"))


def fit_cbc_model(_):
    global MODEL_RESULT, DESIGN_SPEC, TRAIN_X, DROPPED_TERMS
    try:
        df, spec = prepare_training_data()
        X_full = build_design(df, spec, training=True)
        X, DROPPED_TERMS = drop_collinear_columns(X_full)
        spec["model_columns"] = list(X.columns)

        # Z-standardize numeric attributes for numerical stability; the raw
        # optimizer can stall at the zero vector when attributes are on large
        # scales (e.g., price in rupees). Estimates are rescaled afterwards.
        numeric_cols = [c for c in spec["xs"] if c not in spec["categorical"] and c in X.columns]
        scales = X[numeric_cols].std(ddof=0).replace(0, 1.0) if numeric_cols else pd.Series(dtype=float)
        X_fit = X.copy()
        if numeric_cols:
            X_fit[numeric_cols] = (X_fit[numeric_cols] - X_fit[numeric_cols].mean()) / scales

        raw_result = ConditionalLogit(
            endog=df["_chosen"].to_numpy(dtype=float),
            exog=X_fit,
            groups=df["_group"].to_numpy(),
        ).fit(disp=False, method="bfgs", maxiter=2000)
        result = ScaledResult(raw_result, scales)

        if not np.isfinite(result.params).all() or np.allclose(result.params.to_numpy(dtype=float), 0):
            raise ValueError(
                "Optimization returned an all-zero solution, which is a numerical failure, "
                "not a model result. Check that chosen is 0/1, every choice set has exactly "
                "one chosen alternative, and attributes vary within choice sets."
            )

        MODEL_RESULT = result
        DESIGN_SPEC = spec
        TRAIN_X = X

        render_summary(df, spec)
        render_results(result, df["_group"], X.shape[1])
        render_insights(df, X, result, spec)
        render_predictions(df, X, result, spec)

        stats = fit_statistics(result, df["_group"], X.shape[1])
        pseudo_r2 = stats.loc[stats["Metric"].eq("McFadden pseudo R²"), "Value"].iloc[0]
        warning = f"<br><small>Dropped constant/collinear terms: {', '.join(DROPPED_TERMS)}</small>" if DROPPED_TERMS else ""
        with setup_status_out:
            clear_output(wait=True)
            display(
                Markdown(
                    f"**Model fitted.** Choice sets: **{df['_group'].nunique():,}**; "
                    f"McFadden pseudo-R²: **{pseudo_r2:.3f}**.{warning}"
                )
            )
        tabs.selected_index = 3

    except Exception as exc:
        with setup_status_out:
            clear_output(wait=True)
            display(Markdown(f"**CBC logit error:** {exc}"))


train_upload.observe(on_train_upload, names="value")
scenario_upload.observe(on_scenario_upload, names="value")
for widget in [respondent_widget, task_widget, alternative_widget, choice_widget]:
    widget.observe(refresh_x_options, names="value")
x_widget.observe(refresh_factor_options, names="value")
factor_widget.observe(lambda change: refresh_formula_preview(), names="value")
baseline_widget.observe(refresh_formula_preview, names="value")
asc_widget.observe(refresh_formula_preview, names="value")
fit_button.on_click(fit_cbc_model)

case_text = """
**Required long format:** one row per alternative. Required columns are respondent ID,
task/choice-set ID, a 0/1 chosen indicator, and one or more attribute columns.
Every respondent-task choice set must contain at least two alternatives and exactly one chosen row.
Column names are selected after upload, so any CSV following this structure can be used.
"""

tab0 = widgets.VBox(
    [
        widgets.HTML("<h2>0. Case & Data</h2>"),
        widgets.HTML(case_text),
        train_upload,
        scenario_upload,
        widgets.HTML("<small>The scenario CSV is optional. It uses the same respondent, task, alternative, and attribute columns.</small>"),
        data_preview_out,
    ]
)
tab1 = widgets.VBox(
    [
        widgets.HTML("<h2>1. Choice Setup</h2>"),
        respondent_widget,
        task_widget,
        alternative_widget,
        choice_widget,
        x_widget,
        factor_widget,
        baseline_widget,
        asc_widget,
        formula_html,
        fit_button,
        setup_status_out,
    ]
)
tabs = widgets.Tab(children=[tab0, tab1, summary_out, logit_out, insights_out, prediction_out])
for index, title in enumerate(
    ["0. Case & Data", "1. Choice Setup", "2. Choice Summary", "3. CBC Logit", "4. Attribute Insights", "5. Predictions"]
):
    tabs.set_title(index, title)

display(widgets.HTML("<h1>CBC Conditional Logit App for Colab</h1>"))
display(tabs)
preview_data()
