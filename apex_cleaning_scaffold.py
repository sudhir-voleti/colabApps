# ============================================================================
# Apex Raw Ledger - Rescue Scaffold (Session 2)
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY - no CSVs in repo)
# Data : apex_ledger_raw.csv via LMS; students upload it below.
# Launch in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/apex_cleaning_scaffold.py").text)
#
# STEP 0 prints a profile block to COPY into any LLM for the diagnostic.
# STEPS 1-6 run the rescue; every repair prints a change-log.
# The scaffold PAUSES twice: type anything + Enter in the input box that
# appears under the launcher cell.
# ============================================================================

import re
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def hr(t):
    print("\n" + "=" * 64 + "\n  " + t + "\n" + "=" * 64)

def log(msg):
    print("   >> " + msg)

def run_rescue(df):
    n0 = len(df)
    hr("THE RAW LEDGER")
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Expected: 100 dealers x 12 months = 1,200.")
    print("Ask: where are the rest, and why do some rows appear twice?")

    # --------------------------------------------------- STEP 0: PROFILE ----
    hr("STEP 0 - DATA PROFILE  (COPY THE BLOCK BELOW INTO YOUR LLM)")
    print('Prompt to use: "You are a data-quality auditor. Here is a CSV')
    print('profile. List every data-quality problem you can infer - numbered,')
    print('ordered by risk, with the column and the evidence. Do not write code."')
    print("\n========== COPY EVERYTHING BETWEEN THE LINES ==========")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nnon-null and distinct counts:")
    for c in df.columns:
        nn = df[c].notna().sum()
        nu = df[c].nunique(dropna=True)
        line = f"  {c}: non-null {nn}/{len(df)}, distinct {nu}"
        if df[c].dtype == object or nu <= 15:
            line += " | top values: " + str(df[c].astype(str).value_counts().head(6).to_dict())
        else:
            s_num = pd.to_numeric(df[c], errors="coerce")
            if s_num.notna().sum() > 0:
                line += f" | numeric range {s_num.min()} to {s_num.max()}"
        print(line)
    print("========== END OF COPY BLOCK ==========")
    print("(optional: add the first 30 raw rows for a sharper audit)")
    input("\n>>> Press Enter once your group has the LLM's plan. Rescue starts.")

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
    cities = (df["city"].astype(str).str.strip().str.upper()
              .map(lambda c: city_map.get(c, c)))
    n_city_before = df["city"].nunique()
    df["city"] = cities
    log(f"tier: {before} -> {df['tier'].value_counts().to_dict()}")
    log(f"city: {n_city_before} spellings -> {df['city'].nunique()} cities: "
        f"{sorted(df['city'].unique())}")

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
    log(f"parsed {df['transaction_date'].notna().sum()} dates: "
        f"{df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")
    log("month column recreated from the parsed dates")

    # --------------------------------------------------- STEP 3: MONEY ------
    hr("STEP 3 - PARSE THE MONEY AND THE NUMBERS")
    def parse_money(s):
        s = str(s).strip()
        neg = s.startswith("-")
        s = s.lstrip("-").replace("Rs", "").replace("rs", "")
        s = s.replace("lakh", "").replace("L", "").replace(" ", "")
        if "," in s and "." not in s:
            val = float(s.replace(",", "")) / 100000.0   # rupee string -> lakh
        else:
            val = float(s.replace(",", ""))
        return -val if neg else val
    df["sales_lakh"] = df["sales_lakh"].astype(str).map(parse_money)
    log(f"sales_lakh: text -> numbers (min {df['sales_lakh'].min():.2f}, "
        f"max {df['sales_lakh'].max():.2f})")
    print(f"\n   !! DECISION: sales_lakh max = {df['sales_lakh'].max():.0f} "
          "- ten times any real month. Decimal slip -> divide by 10.")
    df.loc[df["sales_lakh"] > 200, "sales_lakh"] = \
        df.loc[df["sales_lakh"] > 200, "sales_lakh"] / 10.0
    log("950 -> 95.0 recorded; negative values kept (genuine returns)")

    df["discount_pct"] = pd.to_numeric(
        df["discount_pct"].astype(str).str.replace("%", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    df["order_qty"] = pd.to_numeric(
        df["order_qty"].astype(str).str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    df["dso_days"] = pd.to_numeric(
        df["dso_days"].astype(str).str.strip().str.upper()
        .replace({"": np.nan, "NA": np.nan, "NAN": np.nan}))
    log(f"dso_days: blanks/'NA' -> missing (range {df['dso_days'].min():.0f} "
        f"to {df['dso_days'].max():.0f})")
    print(f"\n   !! DECISION: dso_days max = {df['dso_days'].max():.0f} "
          "- no collector waits that long. Refuse it -> set missing.")
    df.loc[df["dso_days"] > 200, "dso_days"] = np.nan
    log("450 -> missing, flagged for Step 5")

    # --------------------------------------------------- STEP 4: DUPLICATES -
    hr("STEP 4 - REMOVE DUPLICATES")
    dup_n = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log(f"{dup_n} exact duplicate rows dropped -> {len(df)} rows remain")

    # --------------------------------------------------- STEP 5: MISSING ----
    hr("STEP 5 - THE MISSING VALUES (the judgment call)")
    print("Missing counts:")
    print(df[["dso_days", "order_qty", "discount_pct"]].isna().sum().to_string())
    print("\nOptions for filling dso_days:")
    print("  (a) drop the rows   (b) fill with 0   (c) dealer's own average")
    print("  (d) the TIER's median   (e) phone the salesperson")
    input("\n>>> 60 seconds, vote in the Forms poll, then press Enter "
          "to see the scaffold's choice and its reason.")

    df["was_missing"] = ((df["dso_days"].isna()) | (df["order_qty"].isna())
                         | (df["discount_pct"].isna())).astype(int)
    tier_medians = df.groupby("tier")[["dso_days", "order_qty",
                                       "discount_pct"]].median()
    for col in ["dso_days", "order_qty", "discount_pct"]:
        df[col] = df[col].fillna(df["tier"].map(tier_medians[col]))
    log("filled from the TIER MEDIAN - why median, not the mean?")
    log("  (skewed data - a few huge values live in these columns)")
    log(f"was_missing flag created: {int(df['was_missing'].sum())} confessed "
        f"guesses ({100 * df['was_missing'].mean():.1f}% of rows)")

    # --------------------------------------------------- STEP 6: VALIDATE ---
    hr("STEP 6 - VALIDATE AND RECONCILE")
    assert df["dealer_id"].nunique() == 100, "dealer count wrong"
    assert df["month"].between(1, 12).all(), "month out of range"
    key_ok = df[["sales_lakh", "dso_days", "order_qty"]].notna().all().all()
    assert key_ok, "missing values remain in key columns"
    print("Checks passed: 100 dealers | months 1-12 | no gaps in key columns.")
    print("\nMean DSO by tier - the number the CFO will quote:")
    print(df.groupby("tier")["dso_days"].agg(["count", "mean", "median"])
          .round(1).to_string())
    print("\n(The ORDER Tier1 < Tier2 < Tier3 must hold - roughly the")
    print(" low-50s to high-70s depending on the draw.)")

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
