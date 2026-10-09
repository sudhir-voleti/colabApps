# ============================================================================
# Session 01 - Descriptive Statistics App (Generic CSV Upload Engine, v5)
# Repo : github.com/sudhir-voleti/colabApps
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/desc_stats_app.py").text)
#   launch_app()
#
# THE STANDARD TEMPLATE for all course Colab apps:
#   header -> upload -> sequenced numbered tabs -> type-detection helpers.
# Tabs: 0. Data Preview | 1. One Variable | 2. Two Variables | 3. Pivot Table
# Fully generic: no column names, no tuned defaults, no dataset vocabulary.
# ============================================================================

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

TIME_NAMES = {"week", "month", "date", "day", "period", "yr", "year", "quarter"}
ID_PATTERNS = ("id", "code", "number", "no_", "_no", "num")


def _is_time_or_id(col):
    c = col.strip().lower()
    if c in TIME_NAMES:
        return True
    return any(p in c for p in ID_PATTERNS) and not any(
        k in c for k in ("sales", "limit", "qty", "count", "days", "amount", "value"))


def launch_app():
    upload_widget = widgets.FileUpload(
        accept=".csv", multiple=False, description="Upload CSV",
        button_style="primary", icon="upload")
    out_main = widgets.Output()

    print("=" * 64)
    print("  Descriptive Statistics: Preview -> Describe -> Relate -> Pivot")
    print("=" * 64)
    print("Upload any CSV to begin.\n")
    display(upload_widget)
    display(out_main)

    def run_analysis(df, label):
        with out_main:
            clear_output()
            print(f"Loaded: {label}  |  {df.shape[0]} rows x {df.shape[1]} columns\n")

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            for col in num_cols[:]:
                if df[col].nunique() <= 5:
                    num_cols.remove(col)
                    if col not in cat_cols:
                        cat_cols.append(col)

            time_col = next((c for c in num_cols if c.strip().lower() in TIME_NAMES), None)
            metric_cols = [c for c in num_cols if not _is_time_or_id(c)]
            if time_col and time_col in metric_cols:
                metric_cols.remove(time_col)
            uni_cols = metric_cols + cat_cols
            all_cols = ([time_col] if time_col else []) + uni_cols

            if not uni_cols:
                print("No analyzable columns found. Check the file.")
                return

            excl = {"from": None, "to": None}

            def filtered(sub):
                if time_col and excl["from"] is not None and excl["from"] <= excl["to"]:
                    return sub[~sub[time_col].between(excl["from"], excl["to"])]
                return sub

            # ---------------- TAB 0: PREVIEW ----------------
            out0 = widgets.Output()

            def render_tab0():
                with out0:
                    clear_output()
                    print(f"First 10 of {df.shape[0]} rows (what the machine sees):")
                    display(df.head(10))
                    print(f"Columns: {', '.join(df.columns)}")

            # ---------------- TAB 1: ONE VARIABLE ----------------
            out1 = widgets.Output()
            dd_uni = widgets.Dropdown(options=uni_cols, description="Variable:")
            excl_ui = None
            if time_col is not None:
                tmin, tmax = int(df[time_col].min()), int(df[time_col].max())
                sl_from = widgets.IntSlider(value=tmax + 1, min=tmin, max=tmax + 1,
                                            description="Exclude from:",
                                            continuous_update=False)
                sl_to = widgets.IntSlider(value=tmax, min=tmin - 1, max=tmax,
                                          description="to:", continuous_update=False)

                def on_excl(change):
                    excl["from"], excl["to"] = sl_from.value, sl_to.value
                    render_tab1(); render_tab2()

                sl_from.observe(on_excl, names="value")
                sl_to.observe(on_excl, names="value")
                excl_ui = widgets.VBox([
                    widgets.HTML(f"<b>Range control:</b> set aside rows where "
                                 f"<code>{time_col}</code> falls in a window - "
                                 "watch every number below update."),
                    widgets.HBox([sl_from, sl_to])])

            def render_tab1(change=None):
                with out1:
                    clear_output()
                    col = dd_uni.value
                    d = filtered(df)
                    active = (time_col and excl["from"] is not None
                              and excl["from"] <= excl["to"])
                    if col in metric_cols:
                        s = d[col].dropna()
                        if len(s) == 0:
                            print("No rows left in this range - widen the window.")
                            return
                        mean_v, med_v, sd_v = s.mean(), s.median(), s.std()
                        cv_v = sd_v / mean_v if mean_v else np.nan
                        mode_v = s.mode()[0] if not s.mode().empty else np.nan
                        if active:
                            print(f"[range set aside: {time_col} "
                                  f"{excl['from']}-{excl['to']} | {len(s)} rows]")
                        display(pd.DataFrame({
                            "Variable": [col],
                            "Count": [int(s.count())],
                            "Mean": [round(mean_v, 2)],
                            "Median": [round(med_v, 2)],
                            "Mode": [round(mode_v, 2)],
                            "SD": [round(sd_v, 2)],
                            "CV (SD/Mean)": [round(cv_v, 3)],
                            "Min": [round(s.min(), 2)],
                            "Max": [round(s.max(), 2)]}))
                        fig, ax = plt.subplots(figsize=(7.5, 3.6))
                        if time_col is not None:
                            t = d[[time_col, col]].dropna().sort_values(time_col)
                            ax.plot(t[time_col], t[col], color="#64748B",
                                    lw=1.1, marker="o", ms=2.5)
                            ax.set_xlabel(time_col)
                            ax.set_title(f"{col} over {time_col}",
                                         fontweight="bold")
                        else:
                            s.plot(kind="hist", bins=20, alpha=0.55,
                                   color="#003366", ax=ax)
                            ax.set_title(f"Distribution of {col}", fontweight="bold")
                        ax.axhline(mean_v, color="#DC2626", ls="--", lw=1.8,
                                   label=f"mean = {mean_v:.2f}")
                        ax.axhline(med_v, color="#059669", ls=":", lw=1.8,
                                   label=f"median = {med_v:.2f}")
                        ax.legend()
                        plt.tight_layout(); plt.show()
                    else:
                        s = df[col].astype(str)
                        tab = pd.DataFrame({
                            "Count": s.value_counts(),
                            "Proportion (%)": (s.value_counts(normalize=True) * 100).round(1)})
                        print(f"{col}: a label - count it, take shares, never average it.")
                        display(tab)
                        fig, ax = plt.subplots(figsize=(7, 3.4))
                        tab["Proportion (%)"].plot(kind="bar", color="#003366", ax=ax)
                        ax.set_ylabel("% of rows")
                        ax.set_title(f"Proportions: {col}", fontweight="bold")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout(); plt.show()

            # ---------------- TAB 2: TWO VARIABLES ----------------
            out2 = widgets.Output()
            dd_x = widgets.Dropdown(options=all_cols, description="X:")
            dd_y = widgets.Dropdown(options=uni_cols, description="Y:")

            def render_tab2(change=None):
                with out2:
                    clear_output()
                    x, y = dd_x.value, dd_y.value
                    if not x or not y or x == y:
                        print("Pick two different variables.")
                        return
                    d = filtered(df)
                    x_num, y_num = x in metric_cols, y in metric_cols

                    if x == time_col and y_num:
                        t = d[[x, y]].dropna().sort_values(x)
                        fig, ax = plt.subplots(figsize=(7.5, 3.6))
                        ax.plot(t[x], t[y], color="#64748B", lw=1.1,
                                marker="o", ms=2.5)
                        ax.axhline(t[y].mean(), color="#DC2626", ls="--", lw=1.6,
                                   label=f"overall mean = {t[y].mean():.2f}")
                        ax.set_title(f"{y} over {x}", fontweight="bold")
                        ax.legend()
                        plt.tight_layout(); plt.show()

                    elif x_num and y_num:
                        sub = d[[x, y]].dropna()
                        r = sub[x].corr(sub[y])
                        fig, ax = plt.subplots(figsize=(6.5, 4.2))
                        ax.scatter(sub[x], sub[y], alpha=0.35,
                                   color="#003366", s=18)
                        ax.set_xlabel(x); ax.set_ylabel(y)
                        ax.set_title(f"{y} vs {x}   |   r = {r:.3f}",
                                     fontweight="bold")
                        plt.tight_layout(); plt.show()
                        print(f"correlation r = {r:.3f}")

                    elif x_num != y_num:
                        nv, gv = (x, y) if x_num else (y, x)
                        print(f"{nv} summarized within each level of {gv}:")
                        res = d.groupby(gv)[nv].agg(
                            Count="count", Mean="mean", Median="median",
                            SD="std").round(2)
                        display(res)

                    else:
                        print(f"Crosstab: {x} (rows) vs {y} (columns)")
                        print("Counts:")
                        display(pd.crosstab(df[x], df[y]))
                        print("Row % (each row sums to 100):")
                        display((pd.crosstab(df[x], df[y], normalize="index") * 100).round(1))

            # ---------------- TAB 3: PIVOT TABLE ----------------
            out3 = widgets.Output()
            dd_p_rows = widgets.Dropdown(options=uni_cols, description="Rows:")
            dd_p_cols = widgets.Dropdown(options=["(none)"] + uni_cols,
                                         description="Columns:", value="(none)")
            dd_p_vals = widgets.Dropdown(options=metric_cols, description="Values:")
            dd_p_agg = widgets.Dropdown(
                options=["mean", "count", "median", "sum", "std"],
                value="mean", description="Summarize by:")
            ch_margins = widgets.Checkbox(value=True, description="Show totals")

            def render_tab3(change=None):
                with out3:
                    clear_output()
                    r, c, v = dd_p_rows.value, dd_p_cols.value, dd_p_vals.value
                    agg = dd_p_agg.value
                    if not r or not v:
                        print("Pick Rows and Values.")
                        return
                    use_cols = None if c == "(none)" else c
                    margins = ch_margins.value and use_cols is not None
                    try:
                        pv = filtered(df).pivot_table(
                            index=r, columns=use_cols, values=v,
                            aggfunc=agg, margins=margins,
                            margins_name="All", observed=True)
                    except Exception as e:
                        print(f"Pivot failed: {e}")
                        return
                    print(f"Pivot: {agg} of {v} by {r}" +
                          (f" and {c}" if use_cols else "") +
                          "  (a crosstab is this with count)")
                    display(pv.round(2))

            # sensible neutral defaults: first of each type
            dd_uni.value = uni_cols[0]
            dd_x.value = time_col if time_col else all_cols[0]
            y_default = metric_cols[0] if metric_cols else uni_cols[-1]
            dd_y.options = [c for c in uni_cols if c != dd_x.value] or uni_cols
            dd_y.value = y_default if y_default in dd_y.options else dd_y.options[0]
            dd_p_rows.value = cat_cols[0] if cat_cols else uni_cols[0]
            if metric_cols:
                dd_p_vals.value = metric_cols[0]

            # ---------------- assemble ----------------
            tab_ui = widgets.Tab()
            t1 = [dd_uni] + ([excl_ui] if excl_ui else []) + [out1]
            tab_ui.children = [
                widgets.VBox([out0]),
                widgets.VBox(t1),
                widgets.VBox([widgets.HBox([dd_x, dd_y]), out2]),
                widgets.VBox([widgets.VBox([dd_p_rows, dd_p_cols,
                                            dd_p_vals, dd_p_agg, ch_margins]),
                              out3])]
            tab_ui.set_title(0, "0. Data Preview")
            tab_ui.set_title(1, "1. One Variable")
            tab_ui.set_title(2, "2. Two Variables")
            tab_ui.set_title(3, "3. Pivot Table")
            display(tab_ui)

            dd_uni.observe(render_tab1, names="value")
            dd_x.observe(render_tab2, names="value")
            dd_y.observe(render_tab2, names="value")
            for w_ in (dd_p_rows, dd_p_cols, dd_p_vals, dd_p_agg, ch_margins):
                w_.observe(render_tab3, names="value")
            render_tab0(); render_tab1(); render_tab2(); render_tab3()

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
                print(f"Error reading CSV: {e}")
            return
        name = (f.get("name", "uploaded.csv") if isinstance(f, dict)
                else getattr(f, "name", "uploaded.csv"))
        run_analysis(df, name)

    upload_widget.observe(on_upload, names="value")
