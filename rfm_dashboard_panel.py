# ================================================================
# RFM ACCOUNT DASHBOARD -- B2B trade book (4 tabs, Panel, Colab)
# ----------------------------------------------------------------
# RUN IN COLAB with ONE line:
#   import requests; exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/rfm_dashboard_panel.py").text)
#
# CSV CONTRACT (printed on the Data tab; uploads are validated against it):
#   account_id | date | amount_inr      -- one row per INVOICE
#   account_id: text (e.g. AC-0007)     date: dd-mm-yyyy or yyyy-mm-dd
#   amount_inr: number, rupees          gaps/duplicates tolerated, logged
#
# Tabs: 1 Data (upload + health) | 2 Scores (who is each account)
#       3 Grid (R x F picture)   | 4 Action (the call list, in rupees)
# Live modular build: PHASE 0 = Data tab + engine, PHASE 1 = Scores,
# PHASE 2 = Grid, PHASE 3 = Action. Emergency code = this whole file.
# Default data = embedded synthetic ledger; upload replaces it (batch refresh).
# ================================================================
import subprocess, sys, io
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "panel"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
pn.extension()

CONTRACT_MD = """
**This tool expects one row per INVOICE:**

| Column | Meaning | Example |
|---|---|---|
| `account_id` | Account code (text) | AC-0007 |
| `date` | Invoice date | 14-03-2026 |
| `amount_inr` | Invoice value in rupees | 48500 |
"""

# ------------------ PHASE 0 : ENGINE + DATA IN ------------------
def sample_transactions(seed=42):
    """Synthetic B2B distributor ledger: one row per invoice."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-09-30")
    arche = [("whale", 12, (26, 38), (0, 40), (240000, 0.35)),
             ("loyal", 25, (13, 22), (5, 55), (65000, 0.45)),
             ("new",   30, (1, 3),  (0, 30), (32000, 0.50)),
             ("spor",  50, (2, 6),  (20, 110), (28000, 0.55)),
             ("risk",  35, (10, 18), (150, 250), (55000, 0.50)),
             ("dorm",  28, (3, 8),  (280, 560), (40000, 0.55))]
    rows, acct = [], 0
    for name, n, fr, gap, (mu, sig) in arche:
        for _ in range(n):
            acct += 1
            aid = f"AC-{acct:04d}"
            k_inv = int(rng.integers(fr[0], fr[1] + 1))
            gap_d = int(rng.integers(gap[0], gap[1] + 1))
            last = end - pd.Timedelta(days=gap_d)
            first = last - pd.Timedelta(days=730)
            dates = pd.to_datetime(rng.choice(pd.date_range(first, last, freq="D"),
                                              size=k_inv, replace=False)).sort_values()
            amts = np.minimum(np.round(rng.lognormal(np.log(mu) - 0.5 * sig ** 2, sig, k_inv)),
                              1e7).astype(int)
            rows += [(aid, d, a) for d, a in zip(dates, amts)]
    df = pd.DataFrame(rows, columns=["account_id", "date", "amount_inr"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def ingest(file_bytes):
    """Single source of truth. Upload > synthetic. Validates the contract,
    logs what it fixed/dropped, returns (df, note, health_md)."""
    if file_bytes:
        try:
            up = pd.read_csv(io.BytesIO(file_bytes))
            up.columns = [str(c).strip().lower() for c in up.columns]
            need = {"account_id", "date", "amount_inr"}
            missing = need - set(up.columns)
            if missing:
                raise ValueError("missing columns: %s" % ", ".join(sorted(missing)))
            raw_n, dup_n = len(up), int(up.duplicated().sum())
            up["date"] = pd.to_datetime(up["date"], dayfirst=True, errors="coerce")
            up["amount_inr"] = pd.to_numeric(up["amount_inr"], errors="coerce")
            up = up.dropna(subset=["account_id", "date", "amount_inr"])
            if len(up) < 10:
                raise ValueError("too few valid rows")
            dropped = raw_n - len(up)
            df = up
            note = "Uploaded ledger in use (%d invoice rows)." % len(df)
        except Exception as e:
            df = sample_transactions()
            note = "Upload rejected (%s). Embedded synthetic ledger in use." % e
            raw_n = dup_n = dropped = 0
    else:
        df = sample_transactions()
        note = "Embedded synthetic ledger in use. Upload your own CSV to replace."
        raw_n, dup_n, dropped = len(df), int(df.duplicated().sum()), 0
    health = ("**%s** invoice rows | **%d** accounts | %s to %s | snapshot **%s** | "
              "%d duplicates | %d rows dropped in cleaning"
              % ("{:,}".format(len(df)), df["account_id"].nunique(),
                 df["date"].min().date(), df["date"].max().date(),
                 df["date"].max().date(), dup_n, dropped))
    return df, note, health

def compute_rfm(df):
    """One row per account. Recency measured from the LATEST date in the file."""
    snap = df["date"].max()
    g = df.groupby("account_id")
    rfm = pd.DataFrame({
        "recency_days": (snap - g["date"].max()).dt.days,
        "frequency": g.size().astype(int),
        "monetary": g["amount_inr"].sum().astype(float)})
    return rfm.reset_index(), snap

def _pct_score(s, k, low_is_good=False):
    # rank-based equal-frequency bins; 'first' breaks ties so scores never collapse
    r = s.rank(method="first", ascending=not low_is_good, pct=True)
    return np.ceil(r * k).astype(int)

def score_bins(rfm, k=4):
    out = rfm.copy()
    out["R"] = _pct_score(out["recency_days"], k, low_is_good=True)
    out["F"] = _pct_score(out["frequency"], k)
    out["M"] = _pct_score(out["monetary"], k)
    return out

ACTIONS = {
    "Core Partners": "Protect & grow: key-account manager, volume pricing",
    "New / Warming": "Onboard: second-order incentive, credit terms",
    "AT-RISK": "CALL THIS WEEK: senior visit + service review",
    "Dormant": "Cheap reactivation only: one WhatsApp/email nudge",
}
SEG_ORDER = ["Core Partners", "New / Warming", "AT-RISK", "Dormant"]

def add_segments(scored, k):
    hi_r = scored["R"] > k / 2
    hi_f = scored["F"] > k / 2
    seg = np.where(hi_r & hi_f, "Core Partners",
          np.where(hi_r & ~hi_f, "New / Warming",
          np.where(~hi_r & hi_f, "AT-RISK", "Dormant")))
    scored = scored.copy()
    scored["segment"] = seg
    scored["action"] = scored["segment"].map(ACTIONS)
    scored["revenue_lakh"] = (scored["monetary"] / 1e5).round(1)
    return scored

def pipeline(file_bytes, k):
    df, note, health = ingest(file_bytes)
    rfm, snap = compute_rfm(df)
    return add_segments(score_bins(rfm, k), k), df, note, health, snap

# ------------------ WIDGETS ------------------
w_file = pn.widgets.FileInput(name="Upload ledger CSV (account_id, date, amount_inr)", accept=".csv")
w_bins = pn.widgets.IntSlider(name="RFM tiers per dimension", start=3, end=5, value=4)
w_topn = pn.widgets.IntSlider(name="Call-list size", start=5, end=50, value=20)

# ------------------ PHASE 0 : TAB 1 - DATA ------------------
@pn.depends(w_file.param.value)
def data_tab(fb):
    df, note, health = ingest(fb)
    return pn.Column(
        pn.pane.Markdown("### The CSV contract"), pn.pane.Markdown(CONTRACT_MD),
        pn.pane.Markdown("*%s*" % note),
        pn.pane.Markdown(health),
        pn.pane.Markdown("**Preview (first rows, as the machine sees them)**"),
        pn.widgets.Tabulator(df.head(12), disabled=True, height=300, sizing_mode="stretch_width"),
    )

# ------------------ PHASE 1 : TAB 2 - SCORES ------------------
@pn.depends(w_file.param.value, w_bins.param.value)
def scores_tab(fb, k):
    scored, df, note, health, snap = pipeline(fb, k)
    detail = (scored.sort_values("monetary", ascending=False)
              [["account_id", "recency_days", "frequency", "revenue_lakh", "R", "F", "M",
                "segment", "action"]].reset_index(drop=True))
    return pn.Column(
        pn.pane.Markdown("### One row per account -- sorted by revenue (click any header to sort)"),
        pn.widgets.Tabulator(detail, disabled=True, height=440, sizing_mode="stretch_width"),
        pn.widgets.FileDownload(file=lambda: detail.to_csv(index=False).encode("utf-8"),
                                filename="rfm_scored_accounts.csv",
                                label="Download scored accounts CSV"),
    )

# ------------------ PHASE 2 : TAB 3 - GRID ------------------
@pn.depends(w_file.param.value, w_bins.param.value)
def grid_tab(fb, k):
    scored, df, note, health, snap = pipeline(fb, k)
    piv = scored.pivot_table(index="R", columns="F", values="account_id",
                             aggfunc="count", fill_value=0)
    piv = piv.reindex(index=list(range(k, 0, -1)), columns=list(range(1, k + 1)), fill_value=0)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.imshow(piv.values, cmap="Blues", vmin=0)
    vmax = max(piv.values.max(), 1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax * 0.6 else "#1b3a5c")
    ax.set_xticks(range(k)); ax.set_xticklabels(range(1, k + 1))
    ax.set_yticks(range(k)); ax.set_yticklabels(range(k, 0, -1))
    ax.set_xlabel("Frequency score (higher = more frequent)")
    ax.set_ylabel("Recency score (top = most recent)")
    ax.set_title("Accounts by Recency x Frequency")
    fig.tight_layout()
    ss = (scored.groupby("segment")
          .agg(accounts=("account_id", "count"), revenue_lakh=("revenue_lakh", "sum"))
          .reset_index())
    ss["rev_share_%"] = (100 * ss["revenue_lakh"] / ss["revenue_lakh"].sum()).round(1)
    ss["o"] = ss["segment"].map({s: i for i, s in enumerate(SEG_ORDER)})
    ss = ss.sort_values("o").drop(columns="o")
    return pn.Column(
        pn.Row(w_bins),
        pn.Row(pn.pane.Matplotlib(fig, tight=True, format="png", width=520),
               pn.Column(pn.pane.Markdown("### Segment summary"),
                         pn.widgets.Tabulator(ss, disabled=True, width=440, height=300))),
    )

# ------------------ PHASE 3 : TAB 4 - ACTION ------------------
@pn.depends(w_file.param.value, w_bins.param.value, w_topn.param.value)
def action_tab(fb, k, n):
    scored, df, note, health, snap = pipeline(fb, k)
    rev = scored["monetary"].sum() / 1e5
    risk_seg = scored[scored["segment"].isin(["AT-RISK", "Dormant"])]
    ar, arn = risk_seg["monetary"].sum() / 1e5, len(risk_seg)
    kpis = ("**Accounts:** %d &nbsp;|&nbsp; **Revenue:** ₹ %.1f L &nbsp;|&nbsp; "
            "**At-risk:** ₹ %.1f L across %d accounts"
            % (len(scored), rev, ar, arn))
    sat = (scored.groupby("segment")
           .agg(accounts=("account_id", "count"), revenue_lakh=("revenue_lakh", "sum"))
           .reset_index())
    sat["action"] = sat["segment"].map(ACTIONS)
    sat["o"] = sat["segment"].map({s: i for i, s in enumerate(SEG_ORDER)})
    sat = sat.sort_values("o").drop(columns="o")
    call = (risk_seg.sort_values("monetary", ascending=False)
            [["account_id", "recency_days", "frequency", "revenue_lakh", "segment", "action"]]
            .head(n).reset_index(drop=True))
    return pn.Column(
        pn.pane.Markdown(kpis, sizing_mode="stretch_width"),
        pn.pane.Markdown("### What each segment gets"),
        pn.widgets.Tabulator(sat[["segment", "accounts", "revenue_lakh", "action"]],
                             disabled=True, width=760, height=210),
        pn.pane.Markdown("### The call list (top %d by revenue, at-risk + dormant)" % n),
        pn.Row(w_topn),
        pn.widgets.Tabulator(call, disabled=True, height=300, sizing_mode="stretch_width"),
        pn.widgets.FileDownload(file=lambda: call.to_csv(index=False).encode("utf-8"),
                                filename="rfm_call_list.csv", label="Download call list CSV"),
    )

# ------------------ APP SHELL ------------------
app = pn.Column(
    pn.pane.Markdown("# RFM Account Dashboard -- B2B Trade Book"),
    pn.pane.Markdown("*Three questions per account: when last? how often? how much? "
                     "One dataset feeds every tab -- upload a new ledger anytime and all tabs refresh. "
                     "No streams; refresh is a re-upload.*"),
    pn.Row(w_file),
    pn.Tabs(
        ("1 | Data",   pn.panel(data_tab)),
        ("2 | Scores", pn.panel(scores_tab)),
        ("3 | Grid",   pn.panel(grid_tab)),
        ("4 | Action", pn.panel(action_tab)),
    ),
    max_width=1120, sizing_mode="stretch_width",
)

from IPython.display import display
display(app)
