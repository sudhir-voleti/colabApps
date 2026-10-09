# ============================================================================
# Apex Raw Ledger - Rescue Scaffold (Session 2)
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY - no CSVs in repo)
# Data : apex_ledger_raw.csv via LMS; students upload it below.
# Launch in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/apex_cleaning_scaffold.py").text)
#
# STEP 0 prints a full profile (dtypes, missing, label counts, date
# detection) to COPY into any LLM for the diagnostic plan.
# All interaction is via BUTTONS (input() does not work in callbacks).
# ============================================================================

import re
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NAN", "NONE", "-"}

def hr(t):
    print("\n" + "=" * 64 + "\n  " + t + "\n" + "=" * 64)

def log(msg):
    print("   >> " + msg)

def _date_signature(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return "YYYY-MM-DD"
    if re.fullmatch(r"\d{8}", s): return "YYYYMMDD"
    if re.search(r"[A-Za-z]", s): return "Mon DD, YYYY"
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s): return "DD-MM-YYYY (ambiguous!)"
    return "other"

def profile_block(df):
    """The dense, copy-paste-able profile for the LLM audit."""
    lines = []
    buf = io.StringIO()
    df.info(buf=buf)                       # (a) dtypes + non-null, pandas-style
    lines.append("=== df.info() ===")
    lines.append(buf.getvalue())
    lines.append("\n=== per-column detail ===")
    for c in df.columns:
        s = df[c]
        nn, nmiss, nu = s.notna().sum(), s.isna().sum(), s.nunique(dropna=True)
        lines.append(f"\n[{c}]  non-null {nn}/{len(df)}  missing {nmiss}  "
                     f"distinct {nu}")
        if s.dtype == object or nu <= 15:
            vc = s.astype(str).value_counts()
            disguises = sum(vc.get(k, 0) for k in MISSING_STRINGS if k in vc)
            if disguises:
                lines.append(f"  !! {disguises} rows hold blank/'NA'-like strings")
            lines.append("  label counts: " +
                         str({k: int(v) for k, v in vc.head(8).items()}))
            if nu > 8:
                lines.append(f"  ... ({nu} distinct labels total)")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if s.dtype == object and parsed.notna().mean() > 0.8:
            sigs = s.map(_date_signature).value_counts()
            lines.append(f"  ** DATE-LIKE: {parsed.notna().sum()} rows parse as dates, "
                         f"range {parsed.min().date()} to {parsed.max().date()}")
            lines.append(f"     format signatures: "
                         f"{ {k: int(v) for k, v in sigs.items()} }")
        elif pd.api.types.is_numeric_dtype(s):
            lines.append(f"  numeric: min {s.min()}, max {s.max()}, "
                         f"mean {round(s.mean(), 2)}")
    return "\n".join(lines)

def run_rescue(df):
    hr("THE RAW LEDGER")
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Expected: 100 dealers x 12 months = 1,200.")
    print("Ask: where are the rest, and why do some rows appear twice?")

    # --------------------------------------------------- STEP 0: PROFILE ----
    hr("STEP 0 - DATA PROFILE  (COPY THE BLOCK INTO YOUR LLM)")
    print('Prompt: "You are a data-quality auditor. Here is a CSV profile.')
    print("List every data-quality problem you can infer - numbered, ordered")
    print('by risk, with column and evidence. Do not write code."')
    print("\n========== COPY: EVERYTHING BETWEEN THE LINES ==========")
    print(profile_block(df))
    print("========== END COPY BLOCK ==========")
    print("(optional: paste the first 30 raw rows too, for a sharper audit)")

    gate = widgets.Button(description="Plan ready - start the rescue",
                          button_style="success", icon="check")
    display(gate)

    # --------------------------------------------------- gate -> steps 1-4 --
    def stage_mechanical(btn):
        gate.disabled = True
        steps_1_to_4(df, stage_judgment)

    gate.on_click(stage_mechanical)

def steps_1_to_4(df, next_stage):
    # --------------------------------------------------- STEP 1: LABELS -----
    hr("STEP 1 - STANDARDIZE THE LABELS")
    before = df["tier"].astype(str).value_counts().to_dict()
    df["dealer_id"] = df["dealer_id"].astype(str).str.strip().str.upper()
    tier_map = {"T1": "Tier 1", "TIER 1": "Tier 1", "T-1": "Tier 1",
                "T2": "Tier 2", "TIER 2": "Tier 2",
                "T3": "Tier 3", "TIER 3": "Tier 3", "GRADE-3": "Tier 3"}
    df["tier"] = (df["tier"].astype(str).str.strip().str.upper()
                  .map(lambda t: tier_map.get(t, t)))
    city_map = {"BENGALURU": "Bangalore", "BLR": "Bangalore",
                "HYDERABAD": "Hyderabad", "HYD": "Hyderabad",
                "MUMBAI": "Mumbai", "MUM": "Mumbai",
                "PUNE": "Pune", "PNQ": "Pune",
                "CHENNAI": "Chennai", "MADRAS": "Chennai", "CHN": "Chennai",
                "DELHI": "Delhi", "DEL": "Delhi", "NEW DELHI": "Delhi"}
    n_city_before = df["city"].nunique()
    df["city"] = (df["city"].astype(str).str.strip().str.upper()
                  .map(lambda c: city_map.get(c, c)))
    log(f"tier: {before} -> {df['tier'].value_counts().to_dict()}")
    log(f"city: {n_city_before} spellings -> {df['city'].nunique()} cities")

    # --------------------------------------------------- STEP 2: DATES ------
    hr("STEP 2 - PARSE THE DATES (we TELL the machine the dialect)")
    def parse_mixed(s):
        s = str(s).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return pd.to_datetime(s, format="%Y-%m-%d")
        if re.fullmatch(r"\d{8}", s):
            return pd.to_datetime(s, format="%Y%m%d")
        if re.search(r"[A-Za-z]", s):
            return pd.to_datetime(s, format="%b %d, %Y")
        return pd.to_datetime(s, format="%d-%m-%Y")   # ambiguous: dd-mm
    df["transaction_date"] = df["transaction_date"].astype(str).map(parse_mixed)
    df["month"] = df["transaction_date"].dt.month
    log(f"parsed {df['transaction_date'].notna().sum()} dates; "
        f"month column recreated")

    # --------------------------------------------------- STEP 3: MONEY ------
    hr("STEP 3 - PARSE THE MONEY AND THE NUMBERS")
    def parse_money(s):
        s = str(s).strip()
        neg = s.startswith("-")
        s = s.lstrip("-").replace("Rs", "").replace("rs", "")
        s = s.replace("lakh", "").replace("L", "").replace(" ", "")
        if "," in s and "." not in s:
            val = float(s.replace(",", "")) / 100000.0
        else:
            val = float(s.replace(",", ""))
        return -val if neg else val
    df["sales_lakh"] = df["sales_lakh"].astype(str).map(parse_money)
    log("sales_lakh: text -> numbers")
    print(f"\n   !! DECISION: sales_lakh max = {df['sales_lakh'].max():.0f} "
          "- ten times any real month. Decimal slip -> divide by 10.")
    df.loc[df["sales_lakh"] > 200, "sales_lakh"] = \
        df.loc[df["sales_lakh"] > 200, "sales_lakh"] / 10.0
    log("950 -> 95.0 recorded; negatives kept (genuine returns)")
    df["discount_pct"] = pd.to_numeric(
        df["discount_pct"].astype(str).str.replace("%", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    df["order_qty"] = pd.to_numeric(
        df["order_qty"].astype(str).str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    df["dso_days"] = pd.to_numeric(
        df["dso_days"].astype(str).str.strip().str.upper()
        .replace({"": np.nan, "NA": np.nan, "NAN": np.nan}))
    print(f"   !! DECISION: dso_days max = {df['dso_days'].max():.0f} "
          "- no collector waits that long. Refuse it -> set missing.")
    df.loc[df["dso_days"] > 200, "dso_days"] = np.nan
    log("450 -> missing, flagged for Step 5")

    # --------------------------------------------------- STEP 4: DUPLICATES -
    hr("STEP 4 - REMOVE DUPLICATES")
    dup_n = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log(f"{dup_n} exact duplicate rows dropped -> {len(df)} rows remain")

    next_stage(df)

def stage_judgment(df):
    # --------------------------------------------------- STEP 5: MISSING ----
    hr("STEP 5 - THE MISSING VALUES (the judgment call)")
    print("Missing counts:")
    print(df[["dso_days", "order_qty", "discount_pct"]].isna().sum().to_string())
    print("\nVote: what should fill the missing dso_days?")
    opts = [("a", "drop the rows"), ("b", "fill with 0"),
            ("c", "dealer's own average"), ("d", "the tier's median"),
            ("e", "phone the salesperson")]
    btns = [widgets.Button(description=f"{k}) {v}", button_style="info")
            for k, v in opts]
    display(widgets.HBox(btns))

    def vote(btn):
        for b in btns:
            b.disabled = True
        choice = btn.description
        print(f"\nGROUP VOTE: {choice}")
        if not choice.startswith("d"):
            print("Noted - now watch what the scaffold does and WHY, then")
            print("decide if your vote would survive the board's questions.")
        print("\nScaffold's choice: (d) the tier's median.")
        print("Why median, not the mean? These columns are skewed - a few")
        print("huge values live in them. The median is what a typical")
        print("dealer looks like; the mean is what the pullers look like.")
        df["was_missing"] = ((df["dso_days"].isna())
                             | (df["order_qty"].isna())
                             | (df["discount_pct"].isna())).astype(int)
        tier_medians = df.groupby("tier")[["dso_days", "order_qty",
                                           "discount_pct"]].median()
        for col in ["dso_days", "order_qty", "discount_pct"]:
            df[col] = df[col].fillna(df["tier"].map(tier_medians[col]))
        log(f"was_missing flag: {int(df['was_missing'].sum())} confessed "
            f"guesses ({100 * df['was_missing'].mean():.1f}% of rows)")
        steps_6_validate(df)

    for b in btns:
        b.on_click(vote)

def steps_6_validate(df):
    hr("STEP 6 - VALIDATE AND RECONCILE")
    assert df["dealer_id"].nunique() == 100, "dealer count wrong"
    assert df["month"].between(1, 12).all(), "month out of range"
    assert df[["sales_lakh", "dso_days", "order_qty"]].notna().all().all()
    print("Checks passed: 100 dealers | months 1-12 | no gaps in key columns.")
    print("\nMean DSO by tier - the number the CFO will quote:")
    print(df.groupby("tier")["dso_days"].agg(["count", "mean", "median"])
          .round(1).to_string())
    print("\n(The ORDER Tier1 < Tier2 < Tier3 must hold.)")
    df.to_csv("apex_trade_credit_cleaned.csv", index=False)
    print("\nCleaned file written: apex_trade_credit_cleaned.csv")
    print("NEXT: upload it into the Session-01 app and re-run the CFO pivot.")
    try:
        from google.colab import files
        files.download("apex_trade_credit_cleaned.csv")
    except Exception:
        pass
    hr("RESCUE COMPLETE")

upload_widget = widgets.FileUpload(
    accept=".csv", multiple=False, description="Upload apex_ledger_raw.csv",
    button_style="primary", icon="upload")
print("Upload apex_ledger_raw.csv (from the LMS) to begin the rescue.\n")
display(upload_widget)

def on_upload(change):
    if not upload_widget.value:
        return
    f = (list(upload_widget.value.values())[0]
         if isinstance(upload_widget.value, dict) else upload_widget.value[0])
    content = f["content"] if isinstance(f, dict) else f.content
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    run_rescue(df)

upload_widget.observe(on_upload, names="value")
