# ============================================================================
# Apex Raw Ledger - Rescue Scaffold (Session 2)
# Repo : github.com/sudhir-voleti/colabApps  (needs apex_ledger_raw.csv there too)
# Launch in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/apex_cleaning_scaffold.py").text)
#
# Flow: STEP 0 prints a profile block to COPY into any LLM for the diagnostic.
#       STEPS 1-6 execute the rescue with a change-log after every repair.
#       The scaffold PAUSES once, at the missing-values decision.
# ============================================================================

import re
import io
import requests
import numpy as np
import pandas as pd

RAW_URL = ("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/"
           "apex_ledger_raw.csv")

def hr(t):
    print("\n" + "=" * 64 + "\n  " + t + "\n" + "=" * 64)

def log(msg):
    print("   >> " + msg)

# ------------------------------------------------------------------ LOAD ----
hr("LOADING THE RAW LEDGER")
df = pd.read_csv(io.BytesIO(requests.get(RAW_URL).content))
n0 = len(df)
print(f"Loaded apex_ledger_raw.csv: {df.shape[0]} rows x {df.shape[1]} columns")
print("Expected: 100 dealers x 12 months = 1,200. Ask: where are the rest,")
print("and why do some rows appear twice?")

# ------------------------------------------------------- STEP 0: PROFILE ----
hr("STEP 0 - DATA PROFILE  (COPY THE BLOCK BELOW INTO YOUR LLM)")
print("Paste it under this prompt:")
print('    "You are a data-quality auditor. Here is a CSV profile. List every')
print('     data-quality problem you can infer - numbered, ordered by risk,')
print('     with the column and the evidence. Do not write code."')
print("\n========== COPY EVERYTHING BETWEEN THE LINES ==========")
print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\ndtypes and non-null counts:")
print(df.info())
print("\ncolumn profile:")
for c in df.columns:
    nn = df[c].notna().sum()
    nu = df[c].nunique(dropna=True)
    print(f"  {c}: non-null {nn}/{len(df)}, distinct {nu}")
    if df[c].dtype == object or nu <= 15:
        print("     top values:", df[c].astype(str).value_counts().head(6).to_dict())
    else:
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.notna().sum() > 0:
            print(f"     numeric range: {s_num.min()} to {s_num.max()}")
print("========== END OF COPY BLOCK ==========")
print("\n(optional) also paste the first 30 raw rows if you want the audit sharper.")

input("\n>>> Press Enter once your group has the LLM's plan and we start the rescue.")

# -------------------------------------------------- STEP 1: LABELS ----------
hr("STEP 1 - STANDARDIZE THE LABELS")
before = df["tier"].astype(str).value_counts().to_dict()
df["dealer_id"] = df["dealer_id"].astype(str).str.strip().str.upper()
tier_map = {"T1": "Tier 1", "TIER 1": "Tier 1", "T-1": "Tier 1",
            "T2": "Tier 2", "TIER 2": "Tier 2",
            "T3": "Tier 3", "TIER 3": "Tier 3", "GRADE-3": "Tier 3"}
df["tier"] = (df["tier"].astype(str).str.strip().str.upper()
              .map(lambda t: tier_map.get(t, t)))
city_map = {"BENGALURU": "Bangalore", "BLR": "Bangalore", "BANGALORE": "Bangalore",
            "HYDERABAD": "Hyderabad", "HYD": "Hyderabad",
            "MUMBAI": "Mumbai", "MUM": "Mumbai", "MUMBAI ": "Mumbai",
            "PUNE": "Pune", "PNQ": "Pune",
            "CHENNAI": "Chennai", "MADRAS": "Chennai", "CHN": "Chennai",
            "DELHI": "Delhi", "DEL": "Delhi", "NEW DELHI": "Delhi"}
df["city"] = (df["city"].astype(str).str.strip().str.upper()
              .map(lambda c: city_map.get(c, city_map.get(c.strip(), c))))
log(f"tier labels: {before} -> {df['tier'].value_counts().to_dict()}")
log(f"city spellings now: {sorted(df['city'].unique())}")

# -------------------------------------------------- STEP 2: DATES -----------
hr("STEP 2 - PARSE THE DATES (we TELL the machine which dialect)")
def parse_mixed(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return pd.to_datetime(s, format="%Y-%m-%d")
    if re.fullmatch(r"\d{8}", s):
        return pd.to_datetime(s, format="%Y%m%d")
    if re.search(r"[A-Za-z]", s):
        return pd.to_datetime(s, format="%b %d, %Y")
    return pd.to_datetime(s, format="%d-%m-%Y")   # the ambiguous one: dd-mm
raw_dates = df["transaction_date"].astype(str)
df["transaction_date"] = raw_dates.map(parse_mixed)
df["month"] = df["transaction_date"].dt.month
log(f"dates parsed: {df['transaction_date'].min().date()} to "
    f"{df['transaction_date'].max().date()}; month column recreated")

# -------------------------------------------------- STEP 3: MONEY -----------
hr("STEP 3 - PARSE THE MONEY AND THE NUMBERS")
def parse_money(s):
    s = str(s).strip()
    neg = s.startswith("-")
    s = s.lstrip("-").replace("Rs", "").replace("rs", "")
    s = s.replace("lakh", "").replace("L", "").replace(" ", "")
    if "," in s and "." not in s:          # rupee-grouped string -> lakh
        val = float(s.replace(",", "")) / 100000.0
    else:
        val = float(s.replace(",", ""))
    return -val if neg else val

b = df["sales_lakh"].astype(str)
df["sales_lakh"] = b.map(parse_money)
log(f"sales_lakh: text -> numbers; min {df['sales_lakh'].min():.2f}, "
    f"max {df['sales_lakh'].max():.2f}")

print("\n   !! DECISION POINT: sales_lakh max is "
      f"{df['sales_lakh'].max():.0f} - ten times larger than any real month.")
print("      Documented fix: treat as a decimal slip (950 -> 95.0).")
df.loc[df["sales_lakh"] > 200, "sales_lakh"] = (
    df.loc[df["sales_lakh"] > 200, "sales_lakh"] / 10.0)
log("950 -> 95.0 recorded in the change-log; negatives kept as genuine returns")

df["discount_pct"] = pd.to_numeric(
    df["discount_pct"].astype(str).str.replace("%", "", regex=False)
                       .replace({"": np.nan, "nan": np.nan}))
df["order_qty"] = pd.to_numeric(
    df["order_qty"].astype(str).str.replace(",", "", regex=False)
                   .replace({"": np.nan, "nan": np.nan}))
dso_raw = df["dso_days"].astype(str).str.strip().str.upper()
df["dso_days"] = pd.to_numeric(dso_raw.replace({"": np.nan, "NA": np.nan,
                                                "NAN": np.nan}))
log(f"dso_days: blanks/'NA' -> missing; range {df['dso_days'].min():.0f} to "
    f"{df['dso_days'].max():.0f}")
print("\n   !! DECISION POINT: dso_days max is "
      f"{df['dso_days'].max():.0f} - no collector waits that long.")
print("      Documented fix: refuse to believe it -> set missing, let Step 5 decide.")
df.loc[df["dso_days"] > 200, "dso_days"] = np.nan
log("450 -> missing, flagged for Step 5")

# -------------------------------------------------- STEP 4: DUPLICATES ------
hr("STEP 4 - REMOVE DUPLICATES")
dup_n = df.duplicated().sum()
df = df.drop_duplicates()
log(f"{dup_n} exact duplicate rows dropped -> {len(df)} rows")

# -------------------------------------------------- STEP 5: MISSING VALUES --
hr("STEP 5 - THE MISSING VALUES (the judgment call)")
miss = df[["dso_days", "order_qty", "discount_pct"]].isna().sum()
print("Missing values on the table:\n", miss.to_string())
print("\nYour options for filling dso_days:")
print("  (a) drop those rows      (b) fill with 0")
print("  (c) fill with the dealer's own average")
print("  (d) fill with the TIER's MEDIAN   (e) phone the salesperson")
input("\n>>> Discuss 60 seconds, vote in the Forms poll, then press Enter "
      "to see the scaffold's choice and its reason.")

df["was_missing"] = ((df["dso_days"].isna()) | (df["order_qty"].isna())
                     | (df["discount_pct"].isna())).astype(int)
fill_stats = df.groupby("tier")[["dso_days", "order_qty", "discount_pct"]].median()
for col in ["dso_days", "order_qty", "discount_pct"]:
    df[col] = df[col].fillna(df["tier"].map(fill_stats[col]))
log(f"filled from the TIER MEDIAN - why median? think back to mean vs median "
    f"for skewed data")
log(f"was_missing flag created: {df['was_missing'].sum()} rows carry a "
    f"confessed guess ({100*df['was_missing'].mean():.1f}%)")

# -------------------------------------------------- STEP 6: VALIDATE --------
hr("STEP 6 - VALIDATE AND RECONCILE")
assert df["dealer_id"].nunique() == 100, "dealer count wrong!"
assert df["month"].between(1, 12).all(), "month out of range!"
assert df[["sales_lakh", "dso_days", "order_qty"]].notna().all().all()
print("Checks passed: 100 dealers, months 1-12, no missing in key columns.")
print("\nMean DSO by tier - the number the CFO will quote:")
print(df.groupby("tier")["dso_days"].agg(["count", "mean", "median"])
        .round(1).to_string())
print("\n(Expect the order T1 < T2 < T3, roughly the low-50s to high-70s")
print(" depending on the draw. If the order holds, the rescue worked.)")

df.to_csv("apex_trade_credit_cleaned.csv", index=False)
print("\nCleaned file written: apex_trade_credit_cleaned.csv")
print("NEXT: re-upload it into the Session-01 app and re-run the CFO's pivot.")
try:
    from google.colab import files
    files.download("apex_trade_credit_cleaned.csv")
    print("(download should have started - use it for the app re-upload)")
except Exception:
    pass

hr("RESCUE COMPLETE")
