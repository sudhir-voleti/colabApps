# ============================================================================
# Fix & Fill App (Generic) -- STANDARD TEMPLATE  (v2: was impute_app)
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/impute_app.py").text)
#   launch_app()
#
# Two DIFFERENT acts, in sequence:
#   Tab 1 Detect  - what is each column: number/date/label, missing where?
#   Tab 2 Repair  - numbers stored as TEXT -> real numbers (no values invented;
#                   unparseable cells become MISSING, confessed)
#   Tab 3 Impute  - fill the holes: median/mean/kNN(+k) for numbers,
#                   MODE for labels. Every fill can wear a was_missing flag.
# ============================================================================

import re
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NAN", "NONE", "-", "?"}


def _missing_mask(s):
    if s.dtype == object:
        return (s.isna()
                | s.astype(str).str.strip().str.upper()
                 .isin(list(MISSING_STRINGS)))
    return s.isna()


def _parse_numeric(s, rupees_to_lakh=False):
    """Text -> numeric. Costumes stripped; unparseable -> NaN (confessed)."""
    raw = s.astype(str).str.strip()
    neg = raw.str.startswith("-")
    t = raw.str.lstrip("+-")
    t = t.str.replace(r"(?i)(rs|usd)", "", regex=True)
    t = t.str.replace("[₹$]", "", regex=True)
    t = t.str.replace(" ", "")
    t = t.str.replace(r"(?i)(lakh|lac|l|%)$", "", regex=True)
    comma = t.str.contains(",", regex=False) & ~t.str.contains(r"\.", regex=False)
    stripped = pd.to_numeric(t.str.replace(",", "", regex=False), errors="coerce")
    # comma-grouped values >= 100,000 are almost certainly RUPEES
    # (small comma'd numbers like "1,574" are just thousands separators)
    big_rupees = comma & (stripped.abs() >= 100000)
    val = stripped.where(~(rupees_to_lakh & big_rupees),
                         stripped / 100000.0)
    val = val.where(~neg, -val.abs())
    return val


def _classify(df):
    rows = []
    for c in df.columns:
        s = df[c]
        miss = int(_missing_mask(s).sum())
        if pd.api.types.is_numeric_dtype(s):
            kind = "number"
            action = "impute if missing" if miss else "ok"
        elif s.dtype == object:
            core = s.astype(str).str.strip()
            core = core[~core.isin(list(MISSING_STRINGS))]
            dparse = pd.to_datetime(core, errors="coerce", format="mixed")
            nparse = _parse_numeric(pd.Series(core), rupees_to_lakh=False)
            if len(core) and dparse.notna().mean() > 0.8:
                kind, action = "dates as text", "date_fix app"
            elif len(core) and nparse.notna().mean() > 0.6:
                kind = "numbers as text"
                action = "Repair (Tab 2), then impute"
            else:
                kind = "label"
                action = "impute (mode) if missing" if miss else "ok"
        else:
            kind, action = "label", "impute (mode) if missing" if miss else "ok"
        rows.append([c, kind, miss, action])
    return pd.DataFrame(rows, columns=["Column", "The app sees",
                                       "missing (incl. blanks/'NA')",
                                       "Suggested next"]).set_index("Column")


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 64)
    print("  Fix & Fill: REPAIR how values are read, then FILL the holes")
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
    state = {"df": df.copy()}
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    out_det = widgets.Output()
    with out_det:
        display(_classify(df))
        print("Numbers repair first, dates go to the date_fix app, "
              "then the holes get filled.")

    # ---------------- Tab 2: repair ----------------
    det = _classify(df).reset_index()
    disguise = det[det["The app sees"] == "numbers as text"]["Column"].tolist()
    dd_rep = widgets.Dropdown(options=disguise or ["(none)"], description="Column:")
    chk_rupees = widgets.Checkbox(
        value=False,
        description="money is in RUPEES - convert big amounts to lakh")
    btn_rep = widgets.Button(description="Repair to numeric",
                             button_style="success", icon="wrench")
    out_rep = widgets.Output()

    def do_repair(btn_):
        with out_rep:
            out_rep.clear_output()
            c = dd_rep.value
            if c not in state["df"].columns:
                print("Pick a 'numbers as text' column (see Tab 1).")
                return
            old = state["df"][c]
            new = _parse_numeric(old, rupees_to_lakh=chk_rupees.value)
            unparsed = int(new.isna().sum() - old.isna().sum()
                           - _missing_mask(old).sum())
            state["df"][c] = new
            show = pd.DataFrame({"before": old.astype(str).head(10),
                                 "after": new.head(10)})
            display(show)
            print(f"Unparseable cells -> MISSING (confessed): {max(unparsed, 0)}")
            print("They will wait in Tab 3 for an imputation decision.")
            refresh_impute()

    btn_rep.on_click(do_repair)

    # ---------------- Tab 3: impute ----------------
    dd_col = widgets.Dropdown(options=[], description="Column:")
    dd_strat = widgets.Dropdown(options=[], description="Strategy:")
    sl_k = widgets.IntSlider(value=5, min=1, max=15, description="k (neighbors):",
                             style={"description_width": "initial"})
    btn = widgets.Button(description="Impute & compare", button_style="success",
                         icon="play")
    chk_flag = widgets.Checkbox(value=True, description="add was_missing flag")
    btn_dl = widgets.Button(description="Download fixed CSV",
                            button_style="info", icon="download")
    out_work = widgets.Output()
    out_dl = widgets.Output()

    def refresh_impute(*args):
        cols = [c for c in state["df"].columns if int(_missing_mask(state["df"][c]).sum()) > 0]
        dd_col.options = cols or ["(none)"]
        if cols:
            dd_col.value = cols[0]
            _refresh_strategies()

    def _refresh_strategies(*args):
        c = dd_col.value
        if c not in state["df"].columns:
            dd_strat.options = []
            sl_k.layout.display = "none"
            return
        if pd.api.types.is_numeric_dtype(state["df"][c]):
            dd_strat.options = ["median", "mean", "kNN (ask the neighbors)"]
            dd_strat.value = "median"
        else:
            dd_strat.options = ["mode (most frequent label)"]
            dd_strat.value = dd_strat.options[0]
        sl_k.layout.display = "flex" if "kNN" in dd_strat.value else "none"

    def _toggle_k(*args):
        sl_k.layout.display = "flex" if "kNN" in dd_strat.value else "none"

    def _summ(s, label):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return {"panel": label, "count": 0, "mean": np.nan,
                    "median": np.nan, "SD": np.nan}
        return {"panel": label, "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "SD": round(float(s.std()), 2)}

    def do_impute(btn_):
        with out_work:
            out_work.clear_output()
            c = dd_col.value
            if c not in state["df"].columns:
                print("No column with missing values left.")
                return
            work = state["df"].copy()
            flag = _missing_mask(work[c])
            if flag.sum() == 0:
                print("This column has no missing values.")
                return
            before = work[c]

            if "kNN" in dd_strat.value:
                from sklearn.impute import KNNImputer
                num_cols = [x for x in work.columns
                            if pd.api.types.is_numeric_dtype(work[x])]
                X = work[num_cols]
                imp = KNNImputer(n_neighbors=sl_k.value, weights="distance")
                Xt = pd.DataFrame(imp.fit_transform(X), columns=num_cols,
                                  index=work.index)
                work[c] = Xt[c]
                note = (f"kNN, k={sl_k.value}: neighbors = rows most similar "
                        f"on the other numeric columns")
            elif "mode" in dd_strat.value:
                fill = work[c].astype(str).value_counts().idxmax()
                work[c] = work[c].where(~flag, fill)
                note = f"mode = most frequent label ({fill})"
            else:
                s = pd.to_numeric(work[c], errors="coerce")
                fill = s.median() if dd_strat.value == "median" else s.mean()
                work[c] = work[c].where(~flag, fill)
                note = f"{dd_strat.value} fill = {round(float(fill), 2)}"

            tbl = pd.DataFrame([
                _summ(before, "BEFORE (missing dropped)"),
                _summ(work[c], "AFTER (imputed)")]).set_index("panel")
            display(tbl)
            moved = abs(tbl.loc["BEFORE (missing dropped)", "mean"]
                        - tbl.loc["AFTER (imputed)", "mean"])
            print(f"\nStrategy: {note}")
            print(f"Rows filled: {int(flag.sum())} | the mean moved by "
                  f"{round(moved, 2)}")
            print("Flip the strategy. Which fill survives the board's questions?")
            if chk_flag.value:
                work[c + "_was_missing"] = flag.astype(int)
                print(f"Flag added: {c}_was_missing")
            state["df"] = work
            refresh_impute()

    def do_download(btn_):
        with out_dl:
            out_dl.clear_output()
            state["df"].to_csv("fixed_and_filled.csv", index=False)
            print("Download starting: fixed_and_filled.csv")
            try:
                from google.colab import files
                files.download("fixed_and_filled.csv")
            except Exception:
                pass

    dd_col.observe(_refresh_strategies, names="value")
    dd_strat.observe(_toggle_k, names="value")
    btn.on_click(do_impute)
    btn_dl.on_click(do_download)
    refresh_impute()

    tab = widgets.Tab()
    tab.children = [out_pre, out_det,
                    widgets.VBox([dd_rep, chk_rupees, btn_rep, out_rep]),
                    widgets.VBox([dd_col, dd_strat, sl_k,
                                  widgets.HBox([btn, chk_flag]), out_work,
                                  btn_dl, out_dl])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Detect")
    tab.set_title(2, "2. Repair to Numeric")
    tab.set_title(3, "3. Impute & Compare")
    display(tab)
