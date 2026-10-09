# ============================================================================
# Imputation App (Generic) -- STANDARD TEMPLATE
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/impute_app.py").text)
#   launch_app()
#
# Tabs: 0 Preview | 1 Missing Overview | 2 Impute (mean/median/mode/kNN + k)
# The compare panel shows BEFORE/AFTER - the judgment is the lesson.
# ============================================================================

import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NAN", "NONE", "-", "?"}


def _as_missing_aware(s):
    """Return numeric series with disguised strings -> NaN (NaN if not numeric)."""
    if s.dtype == object:
        cleaned = s.astype(str).str.strip().str.upper()
        cleaned = cleaned.replace({k: np.nan for k in MISSING_STRINGS})
        return pd.to_numeric(cleaned.astype(str).str.replace(",", "", regex=False),
                             errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 64)
    print("  Imputation: every guess must be visible, comparable, confessed")
    print("=" * 64)
    print("Upload any CSV (from the LMS) to begin.\n")
    display(upload)

    def on_upload(change):
        if not upload.value:
            return
        f = (list(upload.value.values())[0]
             if isinstance(upload.value, dict) else upload.value[0])
        content = f["content"] if isinstance(f, dict) else f.content
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        _build(df)

    upload.observe(on_upload, names="value")


def _build(df):
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    # ---------------- Tab 1: missing overview ----------------
    out_miss = widgets.Output()
    with out_miss:
        rows = []
        for c in df.columns:
            if df[c].dtype == object:
                dis = df[c].astype(str).str.strip().str.upper() \
                         .isin(list(MISSING_STRINGS)).sum()
            else:
                dis = 0
            true_miss = int(df[c].isna().sum())
            if true_miss or dis:
                num_aware = _as_missing_aware(df[c])
                kind = ("numeric" if num_aware.notna().sum() > 0
                        else "categorical")
                rows.append([c, kind, true_miss, int(dis)])
        if rows:
            display(pd.DataFrame(rows, columns=["Column", "Type",
                                                "true missing",
                                                "blank/'NA'-like"]).set_index("Column"))
            print("Only columns with missing values are listed.")
        else:
            print("GREEN LIGHT: no missing values detected in this file.")

    # ---------------- Tab 2: impute ----------------
    cols_with_missing = [r[0] for r in rows] if rows else []
    dd_col = widgets.Dropdown(options=cols_with_missing or ["(none)"],
                              description="Column:")
    dd_strat = widgets.Dropdown(options=[], description="Strategy:")
    sl_k = widgets.IntSlider(value=5, min=1, max=15, description="k (neighbors):",
                             style={"description_width": "initial"})
    btn = widgets.Button(description="Impute & compare", button_style="success",
                         icon="play")
    chk_flag = widgets.Checkbox(value=True, description="add was_missing flag")
    btn_dl = widgets.Button(description="Download imputed CSV",
                            button_style="info", icon="download")
    out_work = widgets.Output()
    out_dl = widgets.Output()

    def _refresh_strategies(*args):
        c = dd_col.value
        if c == "(none)" or c not in df.columns:
            dd_strat.options = []
            sl_k.layout.display = "none"
            return
        aware = _as_missing_aware(df[c])
        if aware.notna().sum() > 0:   # numeric
            dd_strat.options = ["median", "mean", "kNN (ask the neighbors)"]
            dd_strat.value = "median"
        else:                          # categorical -> mode only
            dd_strat.options = ["mode (most frequent label)"]
            dd_strat.value = dd_strat.options[0]
        _toggle_k()

    def _toggle_k(*args):
        sl_k.layout.display = ("flex" if "kNN" in dd_strat.value else "none")

    def _summ(s, label):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return {"panel": label, "count": 0, "mean": np.nan,
                    "median": np.nan, "SD": np.nan}
        return {"panel": label, "count": int(s.count()),
                "mean": round(s.mean(), 2), "median": round(s.median(), 2),
                "SD": round(s.std(), 2)}

    state = {"df": df.copy()}

    def do_impute(btn_):
        with out_work:
            out_work.clear_output()
            c = dd_col.value
            if c not in df.columns:
                print("Pick a column with missing values.")
                return
            work = state["df"].copy()
            flag = (work[c].astype(str).str.strip().str.upper()
                    .isin(list(MISSING_STRINGS)) | work[c].isna())
            if flag.sum() == 0:
                print("This column has no missing values left.")
                return
            before = work[c]

            if "kNN" in dd_strat.value:
                from sklearn.impute import KNNImputer
                num_cols = [x for x in work.columns
                            if _as_missing_aware(work[x]).notna().sum() > 0]
                X = pd.DataFrame({x: _as_missing_aware(work[x])
                                  for x in num_cols})
                imp = KNNImputer(n_neighbors=sl_k.value, weights="distance")
                Xt = pd.DataFrame(imp.fit_transform(X), columns=num_cols,
                                  index=work.index)
                work[c] = Xt[c]
                note = (f"kNN with k={sl_k.value}: neighbors = the rows most "
                        f"similar on the other numeric columns")
            elif "mode" in dd_strat.value:
                fill = work[c].astype(str).value_counts().idxmax()
                work[c] = work[c].astype(str).replace(
                    {k: fill for k in MISSING_STRINGS})
                work.loc[flag, c] = fill
                note = f"mode = most frequent label ({fill})"
            else:
                aware = _as_missing_aware(work[c])
                fill = aware.median() if dd_strat.value == "median" \
                       else aware.mean()
                work[c] = aware
                work.loc[flag, c] = fill
                note = f"{dd_strat.value} fill = {round(fill, 2)}"

            cmp_tbl = pd.DataFrame([
                _summ(before, "BEFORE (missing dropped)"),
                _summ(work[c], "AFTER (imputed)")]).set_index("panel")
            display(cmp_tbl)
            moved = abs(cmp_tbl.loc["BEFORE (missing dropped)", "mean"]
                        - cmp_tbl.loc["AFTER (imputed)", "mean"])
            print(f"\nStrategy: {note}")
            print(f"Rows filled: {int(flag.sum())} | the mean moved by "
                  f"{round(moved, 2)}")
            print("Flip the strategy and watch what moves. Which fill would")
            print("survive the board's questions?")
            if chk_flag.value:
                work[c + "_was_missing"] = flag.astype(int)
                print(f"Flag column added: {c}_was_missing")
            state["df"] = work

    def do_download(btn_):
        with out_dl:
            out_dl.clear_output()
            state["df"].to_csv("imputed.csv", index=False)
            print("Download starting: imputed.csv")
            try:
                from google.colab import files
                files.download("imputed.csv")
            except Exception:
                pass

    dd_col.observe(_refresh_strategies, names="value")
    dd_strat.observe(_toggle_k, names="value")
    btn.on_click(do_impute)
    btn_dl.on_click(do_download)
    _refresh_strategies()

    tab = widgets.Tab()
    tab.children = [out_pre, out_miss,
                    widgets.VBox([dd_col, dd_strat, sl_k,
                                  widgets.HBox([btn, chk_flag]), out_work,
                                  btn_dl, out_dl])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Missing Overview")
    tab.set_title(2, "2. Impute & Compare")
    display(tab)
