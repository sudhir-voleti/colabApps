# logit_cbc.py
# Generic Colab/ipywidgets conditional-logit app for choice-based conjoint data.
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
PRED_SOURCE_DF = None
TRAIN_DF = None
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
        "values": {
            c: sorted(pd.to_numeric(df[c]).dropna().unique().tolist())
            for c in xs
            if c not in categorical
        },
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
        raise ValueError(f"Prediction-data CSV is missing selected columns: {missing}")
    df = SCENARIO_DATA[required].copy()
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Selected prediction-data columns contain missing values: {bad}")
    df["_group"] = make_group_key(df, spec["respondent"], spec["task"])
    if df["_group"].nunique() == 0:
        raise ValueError("The prediction-data CSV has no choice sets.")
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
            "Odds ratio": np.exp(result.params.values),
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
scenario_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Prediction data CSV (optional)")

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
summary_button = widgets.Button(description="View summary", button_style="info", icon="table")
fit_button = widgets.Button(description="Fit CBC logit", button_style="success", icon="play")
price_widget = widgets.Dropdown(
    description="Price variable (for WTP)",
    options=[NONE],
    value=NONE,
    style={"description_width": "initial"},
)
wtp_button = widgets.Button(description="Calculate WTP", button_style="warning", icon="calculator")
formula_html = widgets.HTML("Select variables to preview the choice model.")

sim_set_widget = widgets.Dropdown(
    description="Reference lineup (choice set)", style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
sim_mode_widget = widgets.Dropdown(description="Entry mode", style={"description_width": "initial"})
sim_profile_box = widgets.VBox()
SIM_ATTR_WIDGETS = {}
sim_button = widgets.Button(description="Simulate product entry", button_style="info", icon="magic")
sim_out = widgets.Output()

data_preview_out = widgets.Output()
setup_status_out = widgets.Output()
summary_status_out = widgets.Output()
summary_out = widgets.Output()
logit_out = widgets.Output()
insights_out = widgets.Output()
wtp_out = widgets.Output()
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
        if DATA is None:
            display(Markdown("Upload CBC training data. Prediction data is optional (Tab 6)."))
            return
        display(Markdown(f"### CBC training preview — {DATA.shape[0]:,} rows × {DATA.shape[1]:,} columns"))
        display(DATA.head(10))


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
        SCENARIO_DATA = read_upload(scenario_upload, "prediction data")
        if MODEL_RESULT is not None and DESIGN_SPEC is not None and TRAIN_DF is not None:
            # Model already fitted: (re)render predictions on the new data right away.
            render_predictions(TRAIN_DF, TRAIN_X, MODEL_RESULT, DESIGN_SPEC)
        else:
            with prediction_out:
                clear_output(wait=True)
                display(Markdown("Prediction data loaded. Fit the model (Tab 2) to see predictions."))
    except Exception as exc:
        with prediction_out:
            clear_output(wait=True)
            display(Markdown(f"**Prediction-file error:** {exc}"))


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
        display(
            Markdown(
                "<small>Odds ratio = exp(β): the multiplicative effect on choice odds — versus the baseline "
                "level for categorical terms, per unit for numeric terms. Values farther from 1 mean "
                "stronger effects; 1 means no effect.</small>"
            )
        )


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
    global PREDICTION_OUTPUT, PRED_SOURCE_DF
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
        title = "Prediction-data choice probabilities"
        include_choice = spec["choice"] in scenario_df.columns

    PRED_SOURCE_DF = source
    refresh_sim_options()
    PREDICTION_OUTPUT = build_prediction_output(source, X, result, spec, include_choice)
    with prediction_out:
        clear_output(wait=True)
        display(Markdown(f"## {title}"))
        display(Markdown("Probabilities sum to 1 within each respondent-task choice set."))
        display(PREDICTION_OUTPUT.head(20))
        plot_choice_shares(PREDICTION_OUTPUT)
        display(download_button(PREDICTION_OUTPUT, "cbc_choice_probabilities.csv", "Download prediction CSV"))


def on_view_summary(_):
    """Render descriptive statistics for the selected variables, no model fit."""
    try:
        df, spec = prepare_training_data()
        render_summary(df, spec)
        with summary_status_out:
            clear_output(wait=True)
            display(
                Markdown(
                    f"**Summary ready.** Choice sets: **{df['_group'].nunique():,}**; "
                    f"rows: **{len(df):,}**. Review Tab 2, then click **Fit CBC logit** there."
                )
            )
        tabs.selected_index = 2
    except Exception as exc:
        with summary_status_out:
            clear_output(wait=True)
            display(Markdown(f"**Summary error:** {exc}"))


def compute_wtp(_):
    """WTP for each term = -beta / beta_price, in money units of the price variable."""
    try:
        if MODEL_RESULT is None or DESIGN_SPEC is None:
            raise ValueError("Fit the CBC logit model first (Tab 2).")
        price = price_widget.value
        if price == NONE:
            raise ValueError("Select the price variable in the dropdown above.")
        if price not in MODEL_RESULT.params.index:
            raise ValueError(f"'{price}' has no estimated coefficient in this model.")
        beta_price = MODEL_RESULT.params[price]
        if beta_price == 0 or not np.isfinite(beta_price):
            raise ValueError("Price coefficient is zero or non-finite; WTP is undefined.")

        params = MODEL_RESULT.params
        spec = DESIGN_SPEC
        rows = []
        covered = set()
        for col in spec["xs"]:
            if col == price:
                continue
            if col in spec["categorical"]:
                levels = spec["levels"][col]
                baseline = levels[0] if spec["first_baseline"] else levels[-1]
                ordered = [l for l in levels if l != baseline] + [baseline]
                for level in ordered:
                    term = f"{col} = {level}"
                    if level == baseline:
                        rows.append({"Term": term, "Coefficient": 0.0, "WTP (money)": 0.0, "Note": "baseline level"})
                    elif term in params.index:
                        covered.add(term)
                        rows.append({"Term": term, "Coefficient": params[term], "WTP (money)": -params[term] / beta_price, "Note": ""})
            elif col in params.index:
                covered.add(col)
                rows.append({"Term": col, "Coefficient": params[col], "WTP (money)": -params[col] / beta_price, "Note": "per unit"})
        for term, coef in params.items():
            if term == price or term in covered:
                continue
            rows.append({"Term": term, "Coefficient": coef, "WTP (money)": -coef / beta_price, "Note": ""})
        wtp = pd.DataFrame(rows).round(4)
        with wtp_out:
            clear_output(wait=True)
            display(Markdown(f"## Willingness to pay (WTP = −β / β_price; β_price = {beta_price:.3g})"))
            display(Markdown("Positive WTP = money value of that level/unit versus its baseline. Baseline levels are shown with zero coefficient and zero WTP by construction."))
            display(html_table(wtp))
        tabs.selected_index = 5
    except Exception as exc:
        with wtp_out:
            clear_output(wait=True)
            display(Markdown(f"**WTP error:** {exc}"))


def plot_choice_shares(out, max_sets=8):
    """Small-multiple bar charts of predicted shares, one panel per choice set."""
    grouped = list(out.groupby(["Respondent", "Task"], sort=False))[:max_sets]
    n = len(grouped)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0), sharey=True)
    for ax, ((resp, task), sub) in zip(np.atleast_1d(axes), grouped):
        ax.bar(sub["Alternative"].astype(str), sub["Choice probability"], color="#4C78A8")
        ax.set_title(f"R{resp} · T{task}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Predicted share")
    fig.suptitle("Predicted choice shares by choice set", y=1.03)
    plt.tight_layout()
    plt.show()


def predict_utilities(df, spec):
    """Deterministic utility for each row of a prediction/training dataframe."""
    X = build_design(df, spec, training=False)
    X = X.reindex(columns=spec["model_columns"], fill_value=0.0)
    return X.to_numpy(dtype=float) @ MODEL_RESULT.params.to_numpy(dtype=float)


def predict_probabilities(df, spec):
    """Grouped softmax probabilities for a prediction/training dataframe."""
    utilities = predict_utilities(df, spec)
    return grouped_probabilities(utilities, df["_group"].to_numpy())


def alternative_labels(sub, spec):
    if spec["alternative"] != NONE:
        return [str(v) for v in sub[spec["alternative"]].tolist()]
    return [str(i + 1) for i in range(len(sub))]


def refresh_sim_options():
    """Populate the Profile Simulator controls after a model fit/prediction run."""
    if PRED_SOURCE_DF is None or DESIGN_SPEC is None:
        return
    spec = DESIGN_SPEC
    sets = PRED_SOURCE_DF["_group"].drop_duplicates().tolist()
    sim_set_widget.options = sets
    if sets:
        sim_set_widget.value = sets[0]

    SIM_ATTR_WIDGETS.clear()
    children = []
    for col in spec["xs"]:
        if col in spec["categorical"]:
            options = list(spec["levels"][col])
        else:
            options = list(spec["values"].get(col, []))
        dd = widgets.Dropdown(description=col, options=options, style={"description_width": "initial"})
        SIM_ATTR_WIDGETS[col] = dd
        children.append(dd)
    sim_profile_box.children = children
    refresh_sim_mode_options()


def refresh_sim_mode_options(change=None):
    """Entry mode: add as a new option, or replace one existing alternative."""
    if PRED_SOURCE_DF is None or DESIGN_SPEC is None or sim_set_widget.value is None:
        return
    sub = PRED_SOURCE_DF[PRED_SOURCE_DF["_group"].eq(sim_set_widget.value)]
    labels = alternative_labels(sub, DESIGN_SPEC)
    options = [("Add as a new option", ("add", None))]
    options += [(f"Replace: {lab}", ("replace", lab)) for lab in labels]
    sim_mode_widget.options = options
    sim_mode_widget.value = ("add", None)


def compute_profile_sim(_):
    """Build a product profile from dropdowns, compute its net utility, and show
    predicted shares of the reference lineup before vs after its entry."""
    try:
        if MODEL_RESULT is None or DESIGN_SPEC is None:
            raise ValueError("Fit the CBC logit model first (Tab 2).")
        if PRED_SOURCE_DF is None:
            raise ValueError("No prediction data available.")
        if not SIM_ATTR_WIDGETS:
            raise ValueError("Profile controls are not ready; refit the model.")
        spec = DESIGN_SPEC
        set_key = sim_set_widget.value
        if set_key is None:
            raise ValueError("Select a reference lineup (choice set).")

        base = PRED_SOURCE_DF[PRED_SOURCE_DF["_group"].eq(set_key)].copy()
        if len(base) < 2:
            raise ValueError("The selected lineup has fewer than two alternatives.")
        labels = alternative_labels(base, spec)

        focal = base.iloc[[0]].copy()
        for col, dd in SIM_ATTR_WIDGETS.items():
            focal[col] = dd.value
        profile_text = ", ".join(f"{col} = {dd.value}" for col, dd in SIM_ATTR_WIDGETS.items())

        mode, target = sim_mode_widget.value
        if mode == "add":
            combined = pd.concat([base, focal], ignore_index=True)
            after_labels = labels + ["Your product"]
            focal_pos = len(labels)
        else:
            pos = labels.index(target)
            combined = base.copy()
            for col, dd in SIM_ATTR_WIDGETS.items():
                combined.loc[combined.index[pos], col] = dd.value
            after_labels = ["Your product" if i == pos else lab for i, lab in enumerate(labels)]
            focal_pos = pos

        u_before = predict_utilities(base, spec)
        p_before = grouped_probabilities(u_before, base["_group"].to_numpy())
        u_after = predict_utilities(combined, spec)
        p_after = grouped_probabilities(u_after, combined["_group"].to_numpy())

        focal_u = float(u_after[focal_pos])
        share_before = list(p_before) + [np.nan] if mode == "add" else list(p_before)
        change_pp = [
            round(100 * (p_after[i] - p_before[i]), 2) if i < len(p_before) else np.nan
            for i in range(len(after_labels))
        ]
        table = pd.DataFrame(
            {
                "Option": after_labels,
                "Utility": np.round(u_after, 4),
                "Share before": np.round(share_before, 4),
                "Share after": np.round(p_after, 4),
                "Change (pp)": change_pp,
            }
        )

        focal_share = float(p_after[focal_pos])
        losses = [(after_labels[i], p_after[i] - p_before[i]) for i in range(len(p_before))]
        loser, loss = min(losses, key=lambda t: t[1])
        interpretation = (
            f"Your product takes a **{100 * focal_share:.1f}%** share of this lineup. "
            f"Biggest loser: **{loser}** ({100 * loss:+.1f} pp). "
            "These shares are specific to this lineup; the same product would fare differently "
            "against a different competitive set."
        )

        with sim_out:
            clear_output(wait=True)
            display(Markdown(f"## Your product: {profile_text}"))
            display(Markdown(f"**Net utility: {focal_u:.4f}** (higher = more attractive; compare across options)"))
            display(Markdown(interpretation))
            display(html_table(table))
            x = np.arange(len(after_labels))
            width = 0.38
            before_plot = [p_before[i] if i < len(p_before) else 0.0 for i in range(len(after_labels))]
            fig, ax = plt.subplots(figsize=(max(4.5, 1.4 * len(after_labels)), 3.2))
            ax.bar(x - width / 2, before_plot, width, label="Before entry", color="#4C78A8")
            ax.bar(x + width / 2, p_after, width, label="After entry", color="#F58518")
            ax.set_xticks(x)
            ax.set_xticklabels(after_labels)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Predicted share")
            ax.set_title(f"Lineup: {set_key}")
            ax.legend()
            plt.tight_layout()
            plt.show()
        tabs.selected_index = 7
    except Exception as exc:
        with sim_out:
            clear_output(wait=True)
            display(Markdown(f"**Simulator error:** {exc}"))


def fit_cbc_model(_):
    global MODEL_RESULT, DESIGN_SPEC, TRAIN_X, TRAIN_DF, DROPPED_TERMS
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
        TRAIN_DF = df

        # Refresh the WTP price dropdown with the fitted numeric attributes.
        numeric_xs = [c for c in spec["xs"] if c not in spec["categorical"] and c in result.params.index]
        price_widget.options = [NONE] + numeric_xs
        price_widget.value = NONE
        wtp_out.clear_output()
        sim_out.clear_output()

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
summary_button.on_click(on_view_summary)
fit_button.on_click(fit_cbc_model)
wtp_button.on_click(compute_wtp)
sim_set_widget.observe(refresh_sim_mode_options, names="value")
sim_button.on_click(compute_profile_sim)

case_text = """
**Required long format:** one row per alternative. Required columns are respondent ID,
task/choice-set ID, a 0/1 chosen indicator, and one or more attribute columns.
Every respondent-task choice set must contain at least two alternatives and exactly one chosen row.
Column names are selected after upload, so any CSV following this structure can be used.
An optional prediction-data CSV (same respondent, task, and attribute columns; no chosen
indicator needed) can be uploaded in Tab 6 to drive the Predictions and Profile Simulator tabs.
"""

tab0 = widgets.VBox(
    [
        widgets.HTML("<h2>0. Case & Data</h2>"),
        widgets.HTML(case_text),
        train_upload,
        widgets.HTML("<small>The optional prediction-data CSV is uploaded separately in Tab 6 (Predictions).</small>"),
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
        summary_button,
        summary_status_out,
    ]
)
tab2 = widgets.VBox(
    [
        widgets.HTML("<h2>2. Choice Summary</h2>"),
        summary_out,
        fit_button,
        setup_status_out,
    ]
)
tab3 = widgets.VBox([logit_out])
tab4 = widgets.VBox([insights_out])
tab5 = widgets.VBox(
    [
        widgets.HTML("<h2>5. Willingness to Pay (WTP)</h2>"),
        widgets.HTML(
            "<small>WTP = −β / β_price: the money amount that exactly compensates a one-unit "
            "(numeric) or one-level (categorical) change in that attribute, holding utility constant. "
            "It lives in utility space, so it does not depend on any particular competitive lineup. "
            "Requires price to enter the model as a numeric attribute.</small>"
        ),
        price_widget,
        wtp_button,
        wtp_out,
    ]
)
tab7 = widgets.VBox(
    [
        widgets.HTML("<h2>7. Profile Simulator</h2>"),
        widgets.HTML(
            "<small>Pick a reference lineup, build your product by choosing one level per attribute, "
            "and simulate its entry — either added as a new option or replacing an existing one. "
            "You get your product's net utility and the lineup's predicted shares before vs after entry. "
            "Shares are specific to this lineup, unlike WTP (Tab 5), which is lineup-invariant.</small>"
        ),
        sim_set_widget,
        sim_mode_widget,
        widgets.HTML("<b>Build your product:</b>"),
        sim_profile_box,
        sim_button,
        sim_out,
    ]
)
tab6 = widgets.VBox(
    [
        widgets.HTML("<h2>6. Predictions</h2>"),
        widgets.HTML(
            "<small>Optionally upload a prediction-data CSV (same respondent, task, and attribute "
            "columns; no chosen indicator needed). If the model is already fitted, predictions render "
            "immediately on upload; without it, predictions are shown in-sample on the training data.</small>"
        ),
        scenario_upload,
        prediction_out,
    ]
)
tabs = widgets.Tab(children=[tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7])
for index, title in enumerate(
    [
        "0. Case & Data",
        "1. Choice Setup",
        "2. Choice Summary",
        "3. CBC Logit",
        "4. Attribute Insights",
        "5. WTP",
        "6. Predictions",
        "7. Profile Simulator",
    ]
):
    tabs.set_title(index, title)

display(widgets.HTML("<h1>CBC Conditional Logit App for Colab</h1>"))
display(tabs)
preview_data()
