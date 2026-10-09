# ============================================================================
# Session 01 - Descriptive Statistics App (Generic CSV Upload Engine, v3.1)
# Repo : github.com/sudhir-voleti/colabApps
# Launch code in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/desc_stats_app.py").text)
#   launch_app()
# ============================================================================

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

SAMPLE_URL = ("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/"
              "vrs_retail_weekly.csv")
TIME_NAMES = {"week", "month", "date", "day", "period", "yr", "year", "quarter"}
ID_PATTERNS = ("id", "code", "number", "no_", "_no", "num")


def _is_time_or_id(col):
    c = col.strip().lower()
    if c in TIME_NAMES:
        return True
    return any(p in c for p in ID_PATTERNS) and not any(
        k in c for k in ("sales", "limit", "qty", "count", "days", "dso"))


def launch_app():
    upload_widget = widgets.FileUpload(
        accept=".csv", multiple=False, description="Upload CSV",
        button_style="primary", icon="upload")
    sample_btn = widgets.Button(description="Load sample data (VRS)",
                                button_style="info", icon="table")
    out_main = widgets.Output()

    print("=" * 64)
    print("  Session 01: Interactive Descriptive Statistics Engine")
    print("=" * 64)
    print("Upload your CSV (from the LMS) or load the sample dataset.\n")
    display(widgets.HBox([upload_widget, sample_btn]))
    display(out_main)

    def run_analysis(df, label):
        with out_main:
            clear_output()
            print(f"✅ Loaded: {label}  |  {df.shape[0]} rows × {df.shape[1]} columns\n")

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            # Recast low-cardinality numerics (<= 5 unique values) to categorical
            for col in num_cols[:]:
                if df[col].nunique() <= 5:
                    num_cols.remove(col)
                    if col not in cat_cols:
                        cat_cols.append(col)

            # Detect time & metric hygiene
            time_col = next((c for c in num_cols if c.strip().lower() in TIME_NAMES), None)
            metric_cols = [c for c in num_cols if not _is_time_or_id(c)]
            if time_col and time_col in metric_cols:
                metric_cols.remove(time_col)

            if not metric_cols and not cat_cols:
                print("❌ No analyzable columns found. Check the file.")
                return

            excl = {"from": None, "to": None}   # Wedding-experiment window

            def filtered(sub):
                if time_col and excl["from"] is not None and excl["from"] <= excl["to"]:
                    return sub[~sub[time_col].between(excl["from"], excl["to"])]
                return sub

            # ---------------- TAB 1: DATA PREVIEW (ALWAYS FIRST) ----------------
            out1 = widgets.Output()
            def render_tab1():
                with out1:
                    clear_output()
                    print("--- Top 10 Rows Preview ---")
                    display(df.head(10))
                    
                    print("\n--- Dataset Schema & Type Auto-Classification ---")
                    diag_df = pd.DataFrame({
                        "Column Name": df.columns,
                        "Data Type": [str(df[c].dtype) for c in df.columns],
                        "Non-Null Count": [f"{df[c].count()} / {len(df)}" for c in df.columns],
                        "Missing Values": [df[c].isnull().sum() for c in df.columns],
                        "Unique Values": [df[c].nunique() for c in df.columns],
                        "Classification": [
                            "Time Axis" if c == time_col else
                            "Metric Variable" if c in metric_cols else
                            "Categorical" if c in cat_cols else "ID / Label"
                            for c in df.columns
                        ]
                    })
                    display(diag_df)

            # ---------------- TAB 2: NUMERIC ANALYTICS ----------------
            out2 = widgets.Output()
            dd_metric = widgets.Dropdown(options=metric_cols or ["(none)"], description="Metric:")

            def render_tab2(change=None):
                with out2:
                    clear_output()
                    if not metric_cols:
                        print("No metric columns detected.")
                        return
                    col = dd_metric.value
                    s = filtered(df)[col].dropna()
                    if len(s) == 0:
                        print("Nothing left after exclusion window - widen slider.")
                        return
                    mean_v, med_v, sd_v = s.mean(), s.median(), s.std()
                    cv_v = sd_v / mean_v if mean_v else np.nan
                    mode_v = s.mode()[0] if not s.mode().empty else np.nan
                    summary = pd.DataFrame({
                        "Metric": [col],
                        "Mean (average)": [round(mean_v, 2)],
                        "Median (typical)": [round(med_v, 2)],
                        "Mode (most common)": [round(mode_v, 2)],
                        "SD (wobble)": [round(sd_v, 2)],
                        "CV (wobble per rupee)": [round(cv_v, 3)],
                        "Min": [round(s.min(), 2)], "Max": [round(s.max(), 2)],
                        "Count": [int(s.count())]})
                    if time_col and excl["from"] is not None and excl["from"] <= excl["to"]:
                        print(f"EXCLUDING {time_col} {excl['from']}-{excl['to']} "
                              f"({int(s.count())} rows remain)")
                    display(summary)

                    fig, ax = plt.subplots(figsize=(7.5, 3.6))
                    if time_col is not None:
                        t = filtered(df)[[time_col, col]].dropna().sort_values(time_col)
                        ax.plot(t[time_col], t[col], color="#64748B", lw=1.2, marker="o", ms=2.5)
                        ax.axhline(mean_v, color="#DC2626", ls="--", lw=1.8, label=f"mean = {mean_v:.2f}")
                        ax.axhline(med_v, color="#059669", ls=":", lw=1.8, label=f"median = {med_v:.2f}")
                        ax.set_xlabel(time_col)
                        ax.set_title(f"{col} over {time_col} - watch the gap between mean and median", fontweight="bold")
                        ax.legend()
                    else:
                        s.plot(kind="hist", bins=20, alpha=0.55, color="#003366", ax=ax)
                        ax.axvline(mean_v, color="#DC2626", ls="--", lw=1.8, label=f"mean = {mean_v:.2f}")
                        ax.axvline(med_v, color="#059669", ls=":", lw=1.8, label=f"median = {med_v:.2f}")
                        ax.set_title(f"Distribution of {col}", fontweight="bold")
                        ax.legend()
                    plt.tight_layout()
                    plt.show()

            # ---------------- TAB 3: CATEGORICAL ANALYTICS ----------------
            out3 = widgets.Output()
            dd_cat = widgets.Dropdown(options=cat_cols or ["(none)"], description="Category:")

            def render_tab3(change=None):
                with out3:
                    clear_output()
                    if not cat_cols:
                        print("No categorical columns detected.")
                        return
                    col = dd_cat.value
                    s = df[col].astype(str)
                    tab = pd.DataFrame({
                        "Count": s.value_counts(),
                        "Proportion (%)": (s.value_counts(normalize=True) * 100).round(1)})
                    print(f"Categorical breakdown: {col}  (labels are counted, not averaged)")
                    display(tab)
                    fig, ax = plt.subplots(figsize=(7, 3.4))
                    tab["Proportion (%)"].plot(kind="bar", color="#003366", ax=ax)
                    ax.set_ylabel("% of rows")
                    ax.set_title(f"Proportions: {col}", fontweight="bold")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()

            # ---------------- TAB 4: GROUPED BREAKDOWN & TRENDS ----------------
            out4 = widgets.Output()
            dd_group = widgets.Dropdown(options=cat_cols or ["(none)"], description="Group:")
            dd_target = widgets.Dropdown(options=(metric_cols + cat_cols) or ["(none)"], description="Target:")

            def render_tab4(change=None):
                with out4:
                    clear_output()
                    if not cat_cols:
                        print("Grouping needs a categorical column.")
                        return
                    gvar, tvar = dd_group.value, dd_target.value
                    if tvar in metric_cols:
                        print(f"{tvar} summarized within each {gvar} (mean vs median gap = skew signal):")
                        res = filtered(df).groupby(gvar)[tvar].agg(
                            Count="count", Mean="mean", Median="median", SD="std").round(2)
                        display(res)
                        if time_col is not None:
                            tr = filtered(df).groupby([time_col, gvar])[tvar].mean().unstack()
                            if tr.shape[0] > 2:
                                fig, ax = plt.subplots(figsize=(7.5, 3.4))
                                tr.plot(ax=ax, marker="o", ms=3)
                                ax.set_title(f"Mean {tvar} over {time_col}, by {gvar} - is anything drifting?", fontweight="bold")
                                ax.set_ylabel(f"mean {tvar}")
                                plt.tight_layout()
                                plt.show()
                    else:
                        print(f"Row-% crosstab: {gvar} vs {tvar} (shares, never averages, for labels):")
                        ct = pd.crosstab(df[gvar], df[tvar], normalize="index") * 100
                        display(ct.round(1))

            # ---------------- TAB 5: CO-MOVEMENT ----------------
            out5 = widgets.Output()
            dd_x = widgets.Dropdown(options=metric_cols or ["(none)"], description="X:")
            dd_y = widgets.Dropdown(options=(metric_cols[1:] + metric_cols[:1]) or ["(none)"], description="Y:")

            def render_tab5(change=None):
                with out5:
                    clear_output()
                    if len(metric_cols) < 2:
                        print("Need two metric columns for a scatter plot.")
                        return
                    x, y = dd_x.value, dd_y.value
                    sub = filtered(df)[[x, y]].dropna()
                    r = sub[x].corr(sub[y])
                    fig, ax = plt.subplots(figsize=(6.5, 4.2))
                    ax.scatter(sub[x], sub[y], alpha=0.35, color="#003366", s=18)
                    ax.set_xlabel(x); ax.set_ylabel(y)
                    ax.set_title(f"{y} vs {x}   |   r = {r:.3f}\n(observe only - Session 5 names this number)", fontweight="bold")
                    plt.tight_layout()
                    plt.show()
                    print(f"correlation r = {r:.3f}")

            # ---------------- Exclusion Controls ----------------
            excl_ui = None
            if time_col is not None:
                tmin, tmax = int(df[time_col].min()), int(df[time_col].max())
                sl_from = widgets.IntSlider(value=tmax + 1, min=tmin, max=tmax + 1, description="Exclude from:", continuous_update=False)
                sl_to = widgets.IntSlider(value=tmax, min=tmin - 1, max=tmax, description="to:", continuous_update=False)

                def on_excl(change):
                    excl["from"], excl["to"] = sl_from.value, sl_to.value
                    render_tab2(); render_tab4()

                sl_from.observe(on_excl, names="value")
                sl_to.observe(on_excl, names="value")
                excl_ui = widgets.VBox([
                    widgets.HTML(f"<b>Experiment:</b> exclude a window of {time_col}s and watch the summary change."),
                    widgets.HBox([sl_from, sl_to])])

            # ---------------- Assemble Tabs ----------------
            tab_ui = widgets.Tab()
            t2 = [dd_metric] + ([excl_ui] if excl_ui else []) + [out2]
            
            tab_ui.children = [
                widgets.VBox([out1]),                                       # TAB 1: DATA PREVIEW
                widgets.VBox(t2),                                           # TAB 2: NUMERIC
                widgets.VBox([dd_cat, out3]),                               # TAB 3: CATEGORICAL
                widgets.VBox([widgets.HBox([dd_group, dd_target]), out4]),  # TAB 4: GROUPED/TREND
                widgets.VBox([widgets.HBox([dd_x, dd_y]), out5])            # TAB 5: CO-MOVEMENT
            ]
            
            tab_ui.set_title(0, "📋 Data Preview")
            tab_ui.set_title(1, "📊 Numeric Analytics")
            tab_ui.set_title(2, "🏷️ Categorical Analytics")
            tab_ui.set_title(3, "📈 Grouped Breakdown & Trends")
            tab_ui.set_title(4, "🔗 Co-movement (Observe)")
            
            display(tab_ui)

            dd_metric.observe(render_tab2, names="value")
            dd_cat.observe(render_tab3, names="value")
            dd_group.observe(render_tab4, names="value")
            dd_target.observe(render_tab4, names="value")
            dd_x.observe(render_tab5, names="value")
            dd_y.observe(render_tab5, names="value")
            
            # Trigger initial renders
            render_tab1()
            render_tab2()
            render_tab3()
            render_tab4()
            render_tab5()

    def on_upload(change):
        if not upload_widget.value:
            return
        f = (list(upload_widget.value.values())[0]
             if isinstance(upload_widget.value, dict) else upload_widget.value[0])
        content = f["content"] if isinstance(f, dict) else f.content
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            with out_main:
                clear_output()
                print(f"❌ Error reading CSV: {e}")
            return
        name = (f.get("name", "uploaded.csv") if isinstance(f, dict)
                else getattr(f, "name", "uploaded.csv"))
        run_analysis(df, name)

    def on_sample(btn):
        with out_main:
            clear_output()
            print("Fetching sample dataset from GitHub...")
        try:
            import requests
            df = pd.read_csv(io.BytesIO(requests.get(SAMPLE_URL).content))
        except Exception as e:
            with out_main:
                print(f"❌ Sample load failed ({e}). Please upload your CSV instead.")
            return
        run_analysis(df, "vrs_retail_weekly.csv (sample)")

    upload_widget.observe(on_upload, names="value")
    sample_btn.on_click(on_sample)
