# ============================================================================
# Data Audit App (Generic CSV Profiler)  --  STANDARD TEMPLATE
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/audit_app.py").text)
#   launch_app()
#
# Tabs: 0 Preview | 1 Structure (df.info) | 2 Text & Labels | 3 Numbers
#       | 4 Dates | 5 Verdict (green light / red flags) | 6 Copy-for-LLM
# The Verdict tab runs deterministic checks and marks ONLY what needs attention.
# ============================================================================

import re
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NAN", "NONE", "-", "?"}


def _date_signature(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "YYYY-MM-DD"
    if re.fullmatch(r"\d{8}", s):
        return "YYYYMMDD"
    if re.search(r"[A-Za-z]", s):
        return "Mon DD, YYYY"
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):
        return "DD-MM-YYYY (ambiguous)"
    return "other"


def _info_html(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return "<pre style='font-size:12px'>" + buf.getvalue() + "</pre>"


def _verdict(df):
    """Deterministic health check. Returns (rows, all_clear)."""
    rows, all_clear = [], True
    dup_n = int(df.duplicated().sum())
    if dup_n:
        all_clear = False
        rows.append(("<b>Whole table</b>", "ATTENTION",
                     f"{dup_n} exact duplicate rows"))
    n = len(df)
    for c in df.columns:
        s = df[c]
        notes = []
        if s.dtype == object:
            sv = s.astype(str)
            dis = sv.isin(list(MISSING_STRINGS)).sum()
            if dis:
                notes.append(f"{dis} blank/'NA'-like strings")
            if s.isna().sum():
                notes.append(f"{int(s.isna().sum())} true missing")
            nu_raw = s.nunique(dropna=True)
            nu_norm = sv.str.strip().str.upper().nunique(dropna=True)
            if nu_norm < nu_raw:
                notes.append(f"inconsistent casing/spacing "
                             f"({nu_raw} spellings -> {nu_norm} labels)")
            core = sv[~sv.isin(list(MISSING_STRINGS))]
            dparse = pd.to_datetime(core, errors="coerce", format="mixed")
            nparse = pd.to_numeric(core.str.replace(",", "", regex=False),
                                   errors="coerce")
            if len(core) and dparse.notna().mean() > 0.8:
                amb = (core.map(_date_signature) == "DD-MM-YYYY (ambiguous)").sum()
                notes.append(f"dates stored as text; "
                             f"{int(core.map(_date_signature).nunique())} formats"
                             + (f", {amb} ambiguous" if amb else ""))
            elif len(core) and nparse.notna().mean() > 0.8:
                notes.append("numbers stored as text")
            if nu_raw > 0.9 * n and (len(core) and nparse.notna().mean() > 0.5):
                notes.append("looks like an ID - never summarize")
        else:
            if s.isna().sum():
                notes.append(f"{int(s.isna().sum())} missing")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            out_n = int(((s < lo) | (s > hi)).sum())
            if out_n:
                notes.append(f"{out_n} outlier value(s) beyond the "
                             f"boxplot whiskers")
        if notes:
            all_clear = False
            rows.append((c, "ATTENTION", "; ".join(notes)))
    return rows, all_clear


def _llm_block(df):
    lines = [f"shape: {df.shape[0]} rows x {df.shape[1]} columns"]
    for c in df.columns:
        s = df[c]
        nu = s.nunique(dropna=True)
        lines.append(f"\n[{c}] dtype={s.dtype}, non-null {s.notna().sum()}/"
                     f"{len(df)}, distinct {nu}")
        if s.dtype == object or nu <= 15:
            vc = s.astype(str).value_counts()
            dis = sum(vc.get(k, 0) for k in MISSING_STRINGS if k in vc)
            if dis:
                lines.append(f"  !! {dis} rows hold blank/'NA'-like strings")
            lines.append("  labels: " + str({k: int(v)
                                             for k, v in vc.head(8).items()}))
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if s.dtype == object and parsed.notna().mean() > 0.8:
            sigs = s.astype(str).map(_date_signature).value_counts()
            lines.append(f"  DATE-LIKE, range {parsed.min().date()} to "
                         f"{parsed.max().date()}")
            lines.append(f"  formats: "
                         f"{ {k: int(v) for k, v in sigs.items()} }")
        elif pd.api.types.is_numeric_dtype(s):
            lines.append(f"  numeric min {s.min()}, max {s.max()}, "
                         f"mean {round(s.mean(), 2)}")
    return "\n".join(lines)


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 64)
    print("  Data Audit: look at the file before you trust it")
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
        _build_tabs(df)

    upload.observe(on_upload, names="value")


def _build_tabs(df):
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    out_info = widgets.Output()
    with out_info:
        display(HTML(_info_html(df)))

    out_text = widgets.Output()
    with out_text:
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].astype(str)
                print(f"--- {c} ---")
                print(f"distinct labels: {s.nunique()} | "
                      f"blank/'NA'-like: {s.isin(list(MISSING_STRINGS)).sum()}")
                print(s.value_counts().head(8).to_string())
                print()

    out_num = widgets.Output()
    with out_num:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            display(df[num_cols].describe().round(2))
            print()
            for c in num_cols:
                s = df[c].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                print(f"{c}: outliers beyond whiskers = "
                      f"{int(((s < lo) | (s > hi)).sum())} "
                      f"(min {s.min()}, max {s.max()})")
        else:
            print("No true numeric columns.")
        print("\nColumns that LOOK numeric but are stored as text:")
        for c in df.columns:
            if df[c].dtype == object:
                core = df[c].astype(str)
                core = core[~core.isin(list(MISSING_STRINGS))]
                pn = pd.to_numeric(core.str.replace(",", "", regex=False),
                                   errors="coerce")
                if len(core) and pn.notna().mean() > 0.8:
                    print(f"  {c}  ({int(pn.notna().mean()*100)}% numeric-in-disguise)")

    out_date = widgets.Output()
    with out_date:
        found = False
        for c in df.columns:
            s = df[c]
            if s.dtype == object:
                core = s.astype(str)
                core = core[~core.isin(list(MISSING_STRINGS))]
                parsed = pd.to_datetime(core, errors="coerce", format="mixed")
                if len(core) and parsed.notna().mean() > 0.8:
                    found = True
                    sigs = core.map(_date_signature).value_counts()
                    print(f"--- {c} is DATE-LIKE ---")
                    print(f"parsed range: {parsed.min().date()} to "
                          f"{parsed.max().date()}")
                    print("format signatures:")
                    for k, v in sigs.items():
                        print(f"   {k}: {int(v)} rows")
                    print()

        if not found:
            print("No text column parses as dates.")

    out_verdict = widgets.Output()
    with out_verdict:
        rows, ok = _verdict(df)
        if ok:
            display(HTML("<div style='background:#d1fae5;border:2px solid "
                         "#059669;padding:12px;border-radius:8px'>"
                         "<b>GREEN LIGHT</b> - all checks passed. "
                         "This file needs no rescue.</div>"))
        else:
            html = ["<div style='background:#fee2e2;border:2px solid #DC2626;"
                    "padding:12px;border-radius:8px'><b>ATTENTION NEEDED</b> - "
                    "columns marked below:</div>",
                    "<table style='border-collapse:collapse;margin-top:10px'>",
                    "<tr><th style='border:1px solid #999;padding:6px'>Column</th>"
                    "<th style='border:1px solid #999;padding:6px'>Issue</th></tr>"]
            for col, _, note in rows:
                html.append(f"<tr><td style='border:1px solid #999;padding:6px'>"
                            f"<b>{col}</b></td>"
                            f"<td style='border:1px solid #999;padding:6px'>"
                            f"{note}</td></tr>")
            html.append("</table>")
            display(HTML("".join(html)))

    out_llm = widgets.Output()
    llm_btn = widgets.Button(description="Print copy-block for the LLM audit",
                             button_style="warning", icon="copy")
    def dump_llm(btn):
        with out_llm:
            out_llm.clear_output()
            print("========== COPY BELOW INTO YOUR LLM ==========")
            print('Prompt: "You are a data-quality auditor. Here is a CSV '
                  'profile. List every data-quality problem you can infer -')
            print('numbered, ordered by risk, with column and evidence. '
                  'Do not write code."')
            print()
            print(_llm_block(df))
            print("========== END COPY BLOCK ==========")
    llm_btn.on_click(dump_llm)

    tab = widgets.Tab()
    tab.children = [out_pre, out_info, out_text, out_num, out_date,
                    out_verdict, widgets.VBox([llm_btn, out_llm])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Structure (df.info)")
    tab.set_title(2, "2. Text & Labels")
    tab.set_title(3, "3. Numbers")
    tab.set_title(4, "4. Dates")
    tab.set_title(5, "5. Verdict")
    tab.set_title(6, "6. For the LLM")
    display(tab)
