# ================================================================
# RFM ACCOUNT DASHBOARD (B2B trade book) -- Panel, Colab-ready
# ----------------------------------------------------------------
# RUN IN COLAB with ONE line:
#   import requests; exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/rfm_dashboard_panel.py").text)
#
# Default data = embedded synthetic B2B ledger (180 accounts, 20 months).
# Upload option: CSV with columns  account_id, date, amount_inr
#                (one row per invoice; dd-mm-yyyy or yyyy-mm-dd dates).
# Modular teaching blocks: BLOCK 1 = DATA IN, BLOCK 2 = RFM ENGINE,
# BLOCK 3 = VIEWS. Each block is self-contained; build them stepwise live
# by prompting the AI for one block at a time (prompts in session notes).
# ================================================================
import subprocess, sys, io
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "panel"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
pn.extension()

# ------------------------- BLOCK 1 : DATA IN -------------------------
def sample_transactions(seed=42):
    """Synthetic B2B distributor ledger: one row per invoice."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-09-30")
    # (count, invoices, last-gap days, span months, amount ~ lognormal(mean mu, sigma))
    arche = [("whale", 12, (26, 38), (0, 40), 18, (240000, 0.35)),
             ("loyal", 25, (13, 22), (5, 55), 18, (65000, 0.45)),
             ("new",   30, (1, 3),  (0, 30),  6, (32000, 0.50)),
             ("spor",  50, (2, 6),  (20, 110), 18, (28000, 0.55)),
             ("risk",  35, (10, 18), (150, 250), 18, (55000, 0.50)),
             ("dorm",  28, (3, 8),  (280, 560), 18, (40000, 0.55))]
    rows, acct = [], 0
    for name, n, fr, gap, span, (mu, sig) in arche:
        for _ in range(n):
            acct += 1
            aid = f"AC-{acct:04d}"
            k_inv = int(rng.integers(fr[0], fr[1] + 1))
            gap_d = int(rng.integers(gap[0], gap[1] + 1))
            span_d = int(span * 30.4)
            win_start = end - pd.Timedelta(days=gap_d + span_d)
            win_end = end - pd.Timedelta(days=gap_d)
            dates = pd.to_datetime(rng.choice(pd.date_range(win_start, win_end, freq="D"),
                                              size=k_inv, replace=False)).sort_values()
            amts = np.round(rng.lognormal(np.log(mu) - 0.5 * sig ** 2, sig, k_inv)).astype(int)
            rows += [(aid, d, a) for d, a in zip(dates, amts)]
    df = pd.DataFrame(rows, columns=["account_id", "date", "amount_inr"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def load_transactions(file_bytes):
    """Upload takes priority; validated; falls back to the synthetic ledger."""
    if file_bytes:
        try:
            up = pd.read_csv(io.BytesIO(file_bytes))
            up.columns = [str(c).strip().lower() for c in up.columns]
            need = {"account_id", "date", "amount_inr"}
            if not need.issubset(up.columns):
                raise ValueError("needs columns account_id, date, amount_inr")
            up["date"] = pd.to_datetime(up["date"], dayfirst=True, errors="coerce")
            up["amount_inr"] = pd.to_numeric(up["amount_inr"], errors="coerce")
            up = up.dropna(subset=["account_id", "date", "amount_inr"])
            if len(up) < 10:
                raise ValueError("too few valid rows")
            return up, "Using uploaded ledger: %d invoice rows." % len(up)
        except Exception as e:
            return sample_transactions(), ("Upload rejected (%s). Using embedded synthetic ledger." % e)
    return sample_transactions(), ("Embedded synthetic ledger: 180 accounts, Sep 2024 - Sep 2026. "
                                   "Upload your own CSV (account_id, date, amount_inr) to replace.")

# ------------------------- BLOCK 2 : RFM ENGINE -------------------------
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
    out["R"] = _pct_score(out["recency_days"], k, low_is_good=True)   # smaller gap = higher score
    out["F"] = _pct_score(out["frequency"], k)
    out["M"] = _pct_score(out["monetary"], k)
    return out

ACTIONS = {
    "Core Partners": "Protect & grow: key-account manager, volume pricing",
    "New / Warming": "Onboard: second-order incentive, credit terms",
    "AT-RISK": "CALL THIS WEEK: senior visit + service review",
    "Dormant": "Cheap reactivation only: one WhatsApp/email nudge",
}
def add_segments(scored, k):
    hi_r = scored["R"] > k / 2
    hi_f = scored["F"] > k / 2
    seg = np.where(hi_r & hi_f, "Core Partners",
          np.where(hi_r & ~hi_f, "New / Warming",
          np.where(~hi_r & hi_f, "AT-RISK", "Dormant")))
    scored = scored.copy()
    scored["segment"] = seg
    scored["action"] = scored["segment"].map(ACTIONS)
    return scored

# ------------------------- BLOCK 3 : VIEWS -------------------------
w_bins = pn.widgets.IntSlider(name="RFM tiers per dimension", start=3, end=5, value=4)
w_file = pn.widgets.FileInput(name="Upload ledger CSV", accept=".csv")

@pn.depends(w_bins.param.value, w_file.param.value)
def views(k, file_bytes):
    df, note = load_transactions(file_bytes)
    rfm, snap = compute_rfm(df)
    scored = add_segments(score_bins(rfm, k), k)
    scored["revenue_lakh"] = (scored["monetary"] / 1e5).round(1)

    n = len(scored)
    rev = scored["monetary"].sum() / 1e5
    ar = scored.loc[scored["segment"].isin(["AT-RISK", "Dormant"]), "monetary"].sum() / 1e5
    arn = int(scored["segment"].isin(["AT-RISK", "Dormant"]).sum())
    kpis = ("**Accounts:** %d &nbsp;|&nbsp; **Revenue:** ₹ %.1f L &nbsp;|&nbsp; "
            "**At-risk:** ₹ %.1f L across %d accounts -- the call list."
            % (n, rev, ar, arn))

    # R x F grid: account counts
    piv = scored.pivot_table(index="R", columns="F", values="account_id",
                             aggfunc="count", fill_value=0)
    piv = piv.reindex(index=list(range(k, 0, -1)), columns=list(range(1, k + 1)), fill_value=0)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.imshow(piv.values, cmap="Blues", vmin=0)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > piv.values.max() * 0.6 else "#1b3a5c")
    ax.set_xticks(range(k)); ax.set_xticklabels(range(1, k + 1))
    ax.set_yticks(range(k)); ax.set_yticklabels(range(k, 0, -1))
    ax.set_xlabel("Frequency score (higher = more frequent)")
    ax.set_ylabel("Recency score (top = most recent)")
    ax.set_title("Accounts by Recency x Frequency")
    fig.tight_layout()

    # segment summary
    ss = (scored.groupby("segment")
          .agg(accounts=("account_id", "count"), revenue_lakh=("revenue_lakh", "sum"))
          .reset_index())
    ss["rev_share_%"] = (100 * ss["revenue_lakh"] / ss["revenue_lakh"].sum()).round(1)
    order = ["Core Partners", "New / Warming", "AT-RISK", "Dormant"]
    ss["o"] = ss["segment"].map({s: i for i, s in enumerate(order)})
    ss = ss.sort_values("o").drop(columns="o")

    detail = (scored.sort_values("monetary", ascending=False)
              [["account_id", "recency_days", "frequency", "revenue_lakh", "R", "F", "M",
                "segment", "action"]].reset_index(drop=True))

    return pn.Column(
        pn.pane.Markdown("*%s &nbsp; Snapshot: %s*" % (note, snap.date())),
        pn.pane.Markdown(kpis, sizing_mode="stretch_width"),
        pn.Row(pn.pane.Matplotlib(fig, tight=True, format="png", width=520),
               pn.widgets.Tabulator(ss, disabled=True, width=430, height=360)),
        pn.widgets.Tabulator(detail, disabled=True, height=380, sizing_mode="stretch_width"),
        pn.widgets.FileDownload(file=lambda: detail.to_csv(index=False).encode("utf-8"),
                                filename="rfm_scored_accounts.csv", label="Download scored accounts CSV"),
    )

app = pn.Column(
    pn.pane.Markdown("# RFM Account Dashboard -- B2B Trade Book"),
    pn.pane.Markdown("*Three questions per account: when last? how often? how much? "
                     "Data refreshes in batch -- re-upload to update. No streams required.*"),
    pn.Row(w_bins, pn.Spacer(width=20), w_file),
    pn.panel(views),
    max_width=1100, sizing_mode="stretch_width",
)

from IPython.display import display
display(app)
