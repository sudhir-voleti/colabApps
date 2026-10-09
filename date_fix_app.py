# ============================================================================
# Date-Fix App (Generic) -- STANDARD TEMPLATE
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/date_fix_app.py").text)
#   launch_app()
#
# Tabs: 0 Preview | 1 Detect (format signatures) | 2 Convert (HUMAN declares
# each dialect -> one clean new column; original never touched)
# Principle: the machine detects, the HUMAN decides. 01-04-2026 cannot be
# resolved by software - that is the lesson.
# ============================================================================

import re
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


def _signature(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "YYYY-MM-DD"
    if re.fullmatch(r"\d{8}", s):
        return "YYYYMMDD"
    if re.search(r"[A-Za-z]", s):
        return "Mon DD, YYYY"
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):
        return "DD/MM vs MM/DD (ambiguous)"
    return "other"


INTERPRETATIONS = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYYMMDD": "%Y%m%d",
    "Mon DD, YYYY": "%b %d, %Y",
    "DD/MM vs MM/DD (ambiguous) - DAY first": "%d-%m-%Y",
    "DD/MM vs MM/DD (ambiguous) - MONTH first": "%m-%d-%Y",
}
TARGETS = {"ISO  (2026-04-01)": "%Y-%m-%d",
           "Indian  (01-04-2026)": "%d-%m-%Y",
           "US  (04/01/2026)": "%m/%d/%Y"}


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 64)
    print("  Date-Fix: the machine detects, the HUMAN decides")
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


def _date_like_cols(df):
    out = {}
    for c in df.columns:
        if df[c].dtype == object:
            core = df[c].astype(str).str.strip()
            core = core[~core.isin(["", "NA", "N/A", "NULL", "NAN", "NONE", "-"])]
            if len(core) == 0:
                continue
            parsed = pd.to_datetime(core, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                out[c] = core.map(_signature).value_counts().to_dict()
    return out


def _build(df):
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    like = _date_like_cols(df)

    out_det = widgets.Output()
    with out_det:
        if not like:
            print("No text column looks date-like.")
        for c, sigs in like.items():
            print(f"--- {c} ---")
            for k, v in sigs.items():
                tag = "   <-- AMBIGUOUS: a human must decide" \
                      if "ambiguous" in k else ""
                print(f"   {k}: {v} rows{tag}")
            print()

    # ---------------- Tab 2: convert ----------------
    dd_col = widgets.Dropdown(options=list(like.keys()) or ["(none)"],
                              description="Column:")
    dd_target = widgets.Dropdown(options=list(TARGETS.keys()),
                                 value="ISO  (2026-04-01)",
                                 description="Output as:")
    interp_box = widgets.VBox([])
    btn = widgets.Button(description="Convert", button_style="success",
                         icon="calendar")
    btn_dl = widgets.Button(description="Download with clean column",
                            button_style="info", icon="download")
    out_work = widgets.Output()
    out_dl = widgets.Output()
    state = {"df": df.copy(), "widgets": {}}

    def _refresh_interps(*args):
        c = dd_col.value
        interp_box.children = []
        state["widgets"] = {}
        if c not in like:
            return
        for sig in like[c]:
            if "ambiguous" in sig:
                opts = [k for k in INTERPRETATIONS if k.startswith("DD/MM")]
                val = opts[0]
            else:
                opts = [sig]
                val = sig
            dd = widgets.Dropdown(options=opts, value=val,
                                  description=f"{sig} means:",
                                  style={"description_width": "initial"},
                                  layout=widgets.Layout(width="480px"))
            state["widgets"][sig] = dd
            interp_box.children = tuple(list(interp_box.children) + [dd])

    def do_convert(btn_):
        with out_work:
            out_work.clear_output()
            c = dd_col.value
            if c not in like:
                print("Pick a date-like column (Tab 1 shows candidates).")
                return
            sig_of = df[c].astype(str).str.strip().map(_signature)
            new_col = c + "_clean"
            out = pd.Series(pd.NaT, index=df.index)
            n_unparsed = 0
            for sig, dd in state["widgets"].items():
                fmt = INTERPRETATIONS[dd.value]
                mask = sig_of == sig
                parsed = pd.to_datetime(df.loc[mask, c].astype(str).str.strip(),
                                        format=fmt, errors="coerce")
                out.loc[mask] = parsed
                n_unparsed += int(parsed.isna().sum())
            tgt = TARGETS[dd_target.value]
            work = state["df"].copy()
            work[new_col] = out.dt.strftime(tgt)
            print(f"New column created: {new_col}   (original '{c}' untouched)")
            print(f"Output format: {dd_target.value}")
            print(f"Rows that failed to parse: {n_unparsed} -> left blank\n")
            show = pd.DataFrame({c: df[c], new_col: work[new_col]}).head(12)
            display(show)
            state["df"] = work

    def do_download(btn_):
        with out_dl:
            out_dl.clear_output()
            state["df"].to_csv("with_clean_dates.csv", index=False)
            print("Download starting: with_clean_dates.csv")
            try:
                from google.colab import files
                files.download("with_clean_dates.csv")
            except Exception:
                pass

    dd_col.observe(_refresh_interps, names="value")
    btn.on_click(do_convert)
    btn_dl.on_click(do_download)
    _refresh_interps()

    tab = widgets.Tab()
    tab.children = [out_pre, out_det,
                    widgets.VBox([dd_col, interp_box, dd_target, btn,
                                  out_work, btn_dl, out_dl])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Detect (signatures)")
    tab.set_title(2, "2. Convert (you decide)")
    display(tab)
