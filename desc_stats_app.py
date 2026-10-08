# ============================================================================
# Session 01 - Descriptive Statistics App
# Repo : github.com/sudhir-voleti/colabApps  (path: session01/desc_stats_app.py)
# Data : vrs_retail_weekly.csv + apex_trade_credit.csv in the SAME repo folder
# Pull : one launcher cell in Colab -
#        import requests
#        exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/session01/desc_stats_app.py").text)
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/session01"

# instructor guard: the app must NOT leak the Session-5 confound
SHOW_WITHIN_TIER_CORR = False

def hr(title):
    print("\n" + "=" * 64 + "\n  " + title + "\n" + "=" * 64)

# ---------------------------------------------------------------- LOAD ----
vrs  = pd.read_csv(BASE + "/vrs_retail_weekly.csv")     # 104 rows
apex = pd.read_csv(BASE + "/apex_trade_credit.csv")     # 1200 rows
assert vrs.shape[0] == 104,   "vrs_retail_weekly.csv looks wrong - tell the instructor"
assert apex.shape[0] == 1200, "apex_trade_credit.csv looks wrong - tell the instructor"
print("Both datasets loaded. vrs:", vrs.shape, "| apex:", apex.shape)

# =========================================================== PART A =======
hr("PART A - Micro-case 1: VRS Retail (we do this together)")

print("A1. What the machine sees (first 8 rows):")
display(vrs.head(8))

print("A2. The raw readout - more numbers than we need. That is normal.")
display(vrs.groupby("branch")["sales"].describe().round(2))

print("A3. The friendly summary - the same numbers, in board language:")
friendly = vrs.groupby("branch")["sales"].agg(
    average_week="mean", middle_week="median",
    most_common_week=lambda s: s.mode()[0],
    wobble="std", best_week="max", weakest_week="min").round(2)
display(friendly)

print("A4. Wobble per rupee - the coefficient of variation (a RATIO):")
g = vrs.groupby("branch")["sales"].agg(["mean", "std"])
g["wobble_per_rupee (CV)"] = (g["std"] / g["mean"]).round(3)
display(g.rename(columns={"std": "wobble (SD)"}))

print("A5. The wedding-season experiment:")
b    = vrs[vrs.branch == "Branch B - Karan"]
run  = b[(b.week >= 45) & (b.week <= 50)]
rest = b.drop(run.index)
print("  Karan's mean WITH the wedding run   :", round(b.sales.mean(), 2))
print("  Karan's mean WITHOUT the wedding run:", round(rest.sales.mean(), 2))
print("  The run's share of his entire year  :", round(100 * run.sales.sum() / b.sales.sum(), 1), "%")

# =========================================================== PART B =======
hr("PART B - Micro-case 2: Apex Distributors (your group drives)")

print("B1. The dealer ledger (first 8 of 1,200 rows):")
display(apex.head(8))

print("B2. T1 - DSO by dealer tier: wherever the mean sits above the median,")
print("    a minority of slow payers is dragging the average up.")
display(apex.groupby("tier")["dso_days"].agg(
    accounts_months="count", mean_DSO="mean",
    median_DSO="median", wobble="std").round(1))

print("B3. T2 - tier x overdue bucket (row % = each tier's own dealer-months):")
ct = pd.crosstab(apex["tier"], apex["overdue_bucket"], normalize="index") * 100
display(ct.round(1))

print("B4. T3 - the CFO's paradox: DSO climbing while sales stay flat")
dso_trend = apex.pivot_table(index="month", columns="tier",
                             values="dso_days", aggfunc="mean").round(1)
sales_by_month = apex.groupby("month")["sales_lakh"].sum().round(1)
display(dso_trend)
print("Total monthly sales (Rs lakh) - watch how FLAT this stays:")
display(sales_by_month)

print("B5. T4 - correlation teaser (observe, do not conclude - Session 5 owns this):")
colors = apex["tier"].map({"Tier 1": "#1f77b4", "Tier 2": "#ff7f0e", "Tier 3": "#2ca02c"})
ax = apex.plot.scatter(x="discount_pct", y="order_qty", c=colors, alpha=0.35)
ax.set_title("Discount % vs Order Quantity - all 1,200 dealer-months")
plt.show()
print("correlation (discount, order_qty):",
      round(apex["discount_pct"].corr(apex["order_qty"]), 3))
if SHOW_WITHIN_TIER_CORR:
    t1 = apex[apex.tier == "Tier 1"]
    print("within Tier 1 only:",
          round(t1["discount_pct"].corr(t1["order_qty"]), 3))

hr("Done. Now answer T5: the CFO's one-pager, in exactly two numbers.")
