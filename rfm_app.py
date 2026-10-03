
"""
=====================================================================
 rfm_app.py — RFM Segmentation App for Google Colab (PCPDM)
 Repo: https://github.com/sudhir-voleti/colabApps/
---------------------------------------------------------------------
 Colab setup cell (paste as-is; bump ?v= after each repo update):

   import requests
   URL = ("https://raw.githubusercontent.com/sudhir-voleti/colabApps/"
          "main/rfm_app.py?v=1")
   exec(requests.get(URL).text)
   launch_app()

 Built for line-item transaction CSVs (e.g. Kaggle UK e-commerce
 excerpt): user maps customer / invoice / date / amount columns (or
 Quantity x UnitPrice), app aggregates to invoice level, computes
 RFM, scores into 3-8 bins, and shows plots, a segment-action table
 (4 bins), and a DYNAMIC SEGMENTATION demo (migration matrix between
 two snapshot dates).
=====================================================================
"""
import io
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

try:
    from itables import show as _show_df
except Exception:
    try:
        import sys as _sys
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                        "itables"], check=True, capture_output=True)
        from itables import show as _show_df
    except Exception:
        _show_df = None

_S = {}
_W = {}


def _itable(df):
    if _show_df is not None:
        try:
            _show_df(df)
            return
        except Exception:
            pass
    display(df)


# ------------------------------------------------------------------ #
# Engine                                                            #
# ------------------------------------------------------------------ #

def guess_col(cols, *keys):
    for k in keys:
        for c in cols:
            if k in str(c).lower():
                return c
    return None


def parse_dates(series, fmt=None):
    if fmt:
        try:
            return pd.to_datetime(series, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def build_transactions(df, cust, inv, dates, amt, drop_nonpos=True):
    """Line-item -> invoice level: one row per (customer, invoice)."""
    t = pd.DataFrame({"cust": df[cust].astype(str),
                      "inv": df[inv].astype(str),
                      "date": dates, "amt": pd.to_numeric(amt, errors="coerce")}
                     ).dropna(subset=["date", "amt"])
    if drop_nonpos:
        t = t[t["amt"] > 0]
    if t.empty:
        raise ValueError("No valid rows after cleaning — check column mapping.")
    return (t.groupby(["cust", "inv"])
             .agg(date=("date", "max"), amt=("amt", "sum"))
             .reset_index())


def score_bins(s, bins, reverse=False):
    """Quantile-bin a series into 1..bins (rank-based; robust to ties)."""
    s = s.astype(float)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    n = len(edges) - 1                     # effective #bins after dedup
    if n < 1:
        return pd.Series(np.ones(len(s), dtype=int), index=s.index)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n - 1)
    sc = (n - idx) if reverse else (idx + 1)   # R reversed: recent = high
    return pd.Series(sc, index=s.index, dtype=int)


def compute_rfm(txn, bins=4, snapshot=None):
    """Customer-level RFM as of `snapshot` (default: max date in txn)."""
    snap = pd.Timestamp(snapshot) if snapshot is not None else txn["date"].max()
    g = txn[txn["date"] <= snap].groupby("cust")
    rfm = pd.DataFrame({
        "recency_days": (snap - g["date"].max()).dt.days,
        "txn_count": g["inv"].nunique(),
        "spend": g["amt"].sum()}).reset_index().rename(columns={"cust": "customer_id"})
    rfm["R"] = score_bins(rfm["recency_days"], bins, reverse=True)
    rfm["F"] = score_bins(rfm["txn_count"], bins)
    rfm["M"] = score_bins(rfm["spend"], bins)
    rfm["rfm_score"] = (rfm["R"].astype(str) + rfm["F"].astype(str)
                        + rfm["M"].astype(str))
    return rfm.round(2), snap


SEG4 = {   # deck's segment-action mapping (bins = 4)
    "444": ("Best Customers", "Loyalty rewards, VIP benefits, early access, referral perks"),
    "241": ("Price-Sensitive Regulars", "Bundles, cross-sell, personalized promos to lift basket size"),
    "413": ("Occasional High-Value", "Personalized follow-ups, premium recommendations, loyalty incentives"),
    "132": ("At-Risk Customers", "Win-back campaign, reminders, limited-time offers before churn"),
    "324": ("High-Value Growth", "Recommendations + loyalty perks to raise purchase frequency"),
    "111": ("Lost / Low-Value", "Low-cost reactivation only; suppress expensive campaigns"),
}


def segment_tables(rfm):
    sizes = (rfm["rfm_score"].value_counts().rename_axis("rfm_score")
             .reset_index(name="customers"))
    sizes["pct"] = (sizes["customers"] / len(rfm) * 100).round(1)
    actions = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in SEG4.items()
         if k in set(rfm["rfm_score"])],
        columns=["rfm_score", "segment", "recommended_action"])
    return sizes, actions


def seg_copy_text(rfm, sizes, actions, bins, snap):
    lines = [f"RFM segmentation | bins = {bins} | customers = {len(rfm)} "
             f"| snapshot = {pd.Timestamp(snap).date()}", "",
             "SEGMENT SIZES (rfm_score = R,F,M with 4=highest):",
             sizes.to_string(index=False), "",
             "SEGMENT -> ACTION MAPPING:"]
    for _, r in actions.iterrows():
        lines.append(f"  {r['rfm_score']} {r['segment']}: "
                     f"{r['recommended_action']}")
    lines += ["", "Task: interpret the sizes, flag the segments that need "
              "urgent action, and justify the targeting priority."]
    return "\n".join(lines)


def month_ends(txn):
    p = pd.PeriodIndex(pd.to_datetime(txn["date"]).unique(), freq="M")
    return sorted(p.to_timestamp(how="end").normalize().unique())


def migration_matrix(rfm_a, rfm_b):
    """from-segment (as of A) x to-segment (as of B), customer counts."""
    m = rfm_a[["customer_id", "rfm_score"]].merge(
        rfm_b[["customer_id", "rfm_score"]],
        on="customer_id", suffixes=("_A", "_B"))
    ct = pd.crosstab(m["rfm_score_A"], m["rfm_score_B"])
    ct.index.name = "from_A"
    return ct.reset_index(), m


# ------------------------------------------------------------------ #
# Plots                                                             #
# ------------------------------------------------------------------ #

def fig_hist3(rfm):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, ttl in zip(axes, ["recency_days", "txn_count", "spend"],
                            ["Recency (days)", "Frequency (invoices)",
                             "Monetary (total spend)"]):
        ax.hist(rfm[col], bins=25, color="steelblue", edgecolor="white")
        ax.set(title=ttl)
    fig.tight_layout()
    return fig


def fig_rm(rfm):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(rfm["recency_days"], rfm["spend"], s=25, alpha=.6)
    ax.set(xlabel="Recency (days since last purchase)",
           ylabel="Monetary (total spend)", title="Recency vs Monetary")
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_fm(rfm):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(rfm["txn_count"], rfm["spend"], s=25, alpha=.6, color="darkgreen")
    ax.set(xlabel="Frequency (invoices)", ylabel="Monetary (total spend)",
           title="Frequency vs Monetary")
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_heat(rfm):
    piv = rfm.pivot_table(index="R", columns="F", values="spend",
                          aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"F={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"R={r}" for r in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.85, label="mean spend")
    ax.set(title="Heat map: mean Monetary by R x F score")
    fig.tight_layout()
    return fig


def fig_bar(rfm, top=15):
    vc = rfm["rfm_score"].value_counts().head(top)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(vc.index[::-1], vc.values[::-1], color="slateblue")
    ax.set(xlabel="customers", title=f"Top {top} RFM cells by size")
    fig.tight_layout()
    return fig


PLOTS = {"RFM histograms (3-panel)": fig_hist3,
         "Recency vs Monetary (scatter)": fig_rm,
         "Frequency vs Monetary (scatter)": fig_fm,
         "R x F heat map (mean Monetary)": fig_heat,
         "RFM cell sizes (bar)": fig_bar}


# ------------------------------------------------------------------ #
# Handlers                                                          #
# ------------------------------------------------------------------ #

def _status(msg, color="#333"):
    _W["status"].value = f"<div style='color:{color};font-size:13px'>{msg}</div>"


def _on_load(btn):
    if not _W["upload"].value:
        _status("Choose a CSV first.", "crimson"); return
    item = (list(_W["upload"].value.values())[0] if isinstance(_W["upload"].value, dict)
            else _W["upload"].value[0])
    try:
        _S["df"] = pd.read_csv(io.BytesIO(item["content"]))
    except Exception as e:
        _status(f"Could not read CSV: {e}", "crimson"); return
    df = _S["df"]
    cols = list(df.columns)
    for key, guesses in [("cust", ("customer", "cust", "client")),
                         ("inv", ("invoice", "txn", "transaction", "order")),
                         ("date", ("date",)),
                         ("amt", ("totrev", "revenue", "amount", "sales", "total")),
                         ("qty", ("quantity", "qty")),
                         ("price", ("unitprice", "unit_price", "price", "rate"))]:
        c = guess_col(cols, *guesses)
        if c is not None and key in ("cust", "inv", "date", "amt", "qty", "price"):
            _W[key].options = cols
            _W[key].value = c
    mode = "amount" if _W["amt"].value else "qtyprice"
    _W["amt_mode"].value = mode
    with _W["data_out"]:
        clear_output(wait=True)
        print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        display(df.head(8))
    _status("CSV loaded. Check the column mapping below, then press "
            "<b>Prepare RFM data</b>.", "seagreen")


def _on_mode_change(change):
    _W["amt_box"].layout.display = "" if change["new"] == "amount" else "none"
    _W["qp_box"].layout.display = "" if change["new"] == "qtyprice" else "none"


def _on_prepare(btn):
    df = _S.get("df")
    if df is None:
        _status("Upload a CSV first.", "crimson"); return
    try:
        dates = parse_dates(df[_W["date"].value], _W["fmt"].value.strip() or None)
        if _W["amt_mode"].value == "amount":
            amt = df[_W["amt"].value]
        else:
            amt = (pd.to_numeric(df[_W["qty"].value], errors="coerce")
                   * pd.to_numeric(df[_W["price"].value], errors="coerce"))
        txn = build_transactions(df, _W["cust"].value, _W["inv"].value,
                                 dates, amt, _W["clean"].value)
    except Exception as e:
        _status(f"Prepare failed: {e}", "crimson"); return
    _S["txn"] = txn
    with _W["prep_out"]:
        clear_output(wait=True)
        print(f"Invoice-level data: {txn.shape[0]} invoices from "
              f"{txn['cust'].nunique()} customers")
        print(f"Date range: {txn['date'].min().date()} to "
              f"{txn['date'].max().date()}")
        print(f"Total revenue: {txn['amt'].sum():,.2f}")
        print("Go to the RFM tab and press Calculate RFM.")
    me = month_ends(txn)
    opts = [(d.strftime("%Y-%m-%d"), d) for d in me]
    _W["snapA"].options = opts
    _W["snapB"].options = opts
    if len(me) >= 2:
        _W["snapA"].value = me[-2]
        _W["snapB"].value = me[-1]
    _status("Data prepared.", "seagreen")


def _on_rfm(btn):
    if "txn" not in _S:
        _status("Prepare data first (Data tab).", "crimson"); return
    bins = _W["bins"].value
    rfm, snap = compute_rfm(_S["txn"], bins=bins)
    _S.update(rfm=rfm, bins=bins, snap=snap)
    sizes, actions = segment_tables(rfm)
    with _W["rfm_out"]:
        clear_output(wait=True)
        print(f"RFM as of {snap.date()} | bins = {bins} | "
              f"customers = {len(rfm)}")
        _itable(rfm.sort_values(["R", "F", "M"], ascending=False))
        print("\nScore boundaries (quartile cut-points):")
        for col, s in [("R", "recency_days"), ("F", "txn_count"),
                       ("M", "spend")]:
            q = rfm[s].quantile([0, .25, .5, .75, 1]).round(1).tolist()
            print(f"  {col} ({s}): min={q[0]}, q1={q[1]}, q2={q[2]}, "
                  f"q3={q[3]}, max={q[4]}")
    with _W["seg_out"]:
        clear_output(wait=True)
        print("SEGMENT SIZES:")
        _itable(sizes)
        if bins == 4:
            print("\nSEGMENT -> ACTION MAPPING (4 bins):")
            _itable(actions)
        else:
            print(f"\n(segment -> action mapping applies at bins = 4; "
                  f"currently bins = {bins})")
    _W["seg_copy"].value = seg_copy_text(rfm, sizes, actions, bins, snap)
    _S["sizes"], _S["actions"] = sizes, actions
    _status("RFM calculated. Check RFM, Plots, Segments tabs.", "seagreen")


def _on_plot(btn):
    if "rfm" not in _S:
        _status("Calculate RFM first.", "crimson"); return
    fn = PLOTS[_W["plot_sel"].value]
    with _W["plot_out"]:
        clear_output(wait=True); plt.close("all")
        display(fn(_S["rfm"]))
    _status("Plot rendered.", "seagreen")


def _on_migrate(btn):
    if "txn" not in _S:
        _status("Prepare data first (Data tab).", "crimson"); return
    bins = _S.get("bins", 4)
    A, B = _W["snapA"].value, _W["snapB"].value
    if pd.Timestamp(A) >= pd.Timestamp(B):
        _status("Snapshot A must be BEFORE snapshot B.", "crimson"); return
    rfm_a, _ = compute_rfm(_S["txn"], bins=bins, snapshot=A)
    rfm_b, _ = compute_rfm(_S["txn"], bins=bins, snapshot=B)
    ct, m = migration_matrix(rfm_a, rfm_b)
    _S["mig"] = (ct, m)
    diag = np.diag(pd.crosstab(m["rfm_score_A"], m["rfm_score_B"])
                   .reindex(index=ct["from_A"], columns=ct["from_A"])
                   .fillna(0).values).sum() if len(m) else 0
    with _W["mig_out"]:
        clear_output(wait=True)
        print(f"DYNAMIC SEGMENTATION: migration between {pd.Timestamp(A).date()} "
              f"and {pd.Timestamp(B).date()} (bins = {bins})")
        print(f"Customers observed at A: {len(rfm_a)} | still active by B: "
              f"{len(m)} | stayed in same segment: {int(diag)} "
              f"({diag/len(m)*100 if len(m) else 0:.1f}%)")
        print("\nMIGRATION MATRIX (rows = segment at A, columns = at B):")
        _itable(ct)
    lines = [f"RFM migration {pd.Timestamp(A).date()} -> "
             f"{pd.Timestamp(B).date()} (bins={bins})", "",
             "FROM_A x TO_B customer counts:",
             ct.to_string(index=False), "",
             "Task: identify the biggest migration flows, explain the "
             "business reasons, and recommend one action per major flow."]
    _W["mig_copy"].value = "\n".join(lines)
    _status("Migration computed.", "seagreen")


def _on_download(btn):
    if "rfm" not in _S:
        _status("Calculate RFM first.", "crimson"); return
    _S["rfm"].to_csv("rfm_scores.csv", index=False)
    try:
        from google.colab import files
        files.download("rfm_scores.csv")
    except Exception:
        _status("Saved 'rfm_scores.csv' in the Colab file pane.", "seagreen")


# ------------------------------------------------------------------ #
# UI                                                                #
# ------------------------------------------------------------------ #

def launch_app():
    style = {"description_width": "170px"}
    try:
        from itables import init_notebook_mode
        init_notebook_mode(all_interactive=False)
    except Exception:
        pass

    _W["upload"] = widgets.FileUpload(accept=".csv", multiple=False,
                                      description="CSV")
    _W["btn_load"] = widgets.Button(description="Load CSV",
                                    button_style="info", icon="upload")
    _W["btn_load"].on_click(_on_load)
    _W["data_out"] = widgets.Output()

    dd = lambda: widgets.Dropdown(options=[], style=style,
                                  layout=widgets.Layout(width="420px"))
    for key, ttl in [("cust", "Customer ID column"),
                     ("inv", "Transaction/Invoice ID column"),
                     ("date", "Transaction date column")]:
        _W[key] = dd()
        _W[key].description = ttl
    _W["amt"] = dd(); _W["amt"].description = "Amount column (line revenue)"
    _W["qty"] = dd(); _W["qty"].description = "Quantity column"
    _W["price"] = dd(); _W["price"].description = "Unit price column"

    _W["amt_mode"] = widgets.ToggleButtons(
        options=[("Amount column", "amount"), ("Quantity x Price", "qtyprice")],
        value="amount", description="Revenue from:")
    _W["amt_mode"].observe(_on_mode_change, names="value")
    _W["amt_box"] = widgets.VBox([_W["amt"]])
    _W["qp_box"] = widgets.VBox([_W["qty"], _W["price"]])
    _W["fmt"] = widgets.Text(value="%m/%d/%Y %H:%M",
                             description="Date format (optional)",
                             style=style, layout=widgets.Layout(width="420px"))
    _W["clean"] = widgets.Checkbox(
        value=True, indent=False, style=style,
        description="Drop rows with amount <= 0 (cancellations/returns)")
    _W["btn_prep"] = widgets.Button(description="Prepare RFM data",
                                    button_style="info")
    _W["btn_prep"].on_click(_on_prepare)
    _W["prep_out"] = widgets.Output()

    tab_data = widgets.VBox([
        widgets.HTML("<b>1.</b> Upload + inspect:"),
        widgets.HBox([_W["upload"], _W["btn_load"]]),
        _W["data_out"],
        widgets.HTML("<b>2.</b> Map columns (auto-guessed — verify!): "
                     "line-item rows are aggregated to one row per "
                     "(customer, invoice):"),
        _W["cust"], _W["inv"], _W["date"],
        _W["amt_mode"], _W["amt_box"], _W["qp_box"], _W["fmt"], _W["clean"],
        _W["btn_prep"], _W["prep_out"]])

    _W["bins"] = widgets.IntSlider(value=4, min=3, max=8, style=style,
                                   description="RFM bins (3-8)")
    _W["btn_rfm"] = widgets.Button(description="Calculate RFM",
                                   button_style="success")
    _W["btn_rfm"].on_click(_on_rfm)
    _W["rfm_out"] = widgets.Output()
    _W["btn_dl"] = widgets.Button(description="⬇ Download RFM scores CSV")
    _W["btn_dl"].on_click(_on_download)
    tab_rfm = widgets.VBox([
        widgets.HTML("<b>3.</b> RFM scores per customer (higher = better on "
                     "each of R, F, M; click headers to sort):"),
        _W["bins"], _W["btn_rfm"], _W["rfm_out"], _W["btn_dl"]])

    _W["plot_sel"] = widgets.Dropdown(options=list(PLOTS.keys()),
                                      description="RFM plot", style=style,
                                      layout=widgets.Layout(width="420px"))
    _W["btn_plot"] = widgets.Button(description="Render plot",
                                    button_style="info")
    _W["btn_plot"].on_click(_on_plot)
    _W["plot_out"] = widgets.Output()
    tab_plots = widgets.VBox([
        widgets.HTML("<b>4.</b> Exploratory plots of the RFM table:"),
        widgets.HBox([_W["plot_sel"], _W["btn_plot"]]), _W["plot_out"]])

    _W["seg_out"] = widgets.Output()
    _W["seg_copy"] = widgets.Textarea(
        value="Calculate RFM to generate the segment summary.",
        layout=widgets.Layout(width="95%", height="240px"))
    tab_seg = widgets.VBox([
        widgets.HTML("<b>5.</b> Segment sizes and the segment -> action "
                     "mapping (deck activity). Copy block for the AI "
                     "interpretation exercise:"),
        _W["seg_out"],
        widgets.HTML("<small>Click in the box, Ctrl/Cmd-A, copy:</small>"),
        _W["seg_copy"]])

    _W["snapA"] = widgets.Dropdown(options=[], description="Snapshot A (from)",
                                   style=style, layout=widgets.Layout(width="360px"))
    _W["snapB"] = widgets.Dropdown(options=[], description="Snapshot B (to)",
                                   style=style, layout=widgets.Layout(width="360px"))
    _W["btn_mig"] = widgets.Button(description="Compute migration",
                                   button_style="success")
    _W["btn_mig"].on_click(_on_migrate)
    _W["mig_out"] = widgets.Output()
    _W["mig_copy"] = widgets.Textarea(
        value="Pick two snapshot dates to generate the migration matrix.",
        layout=widgets.Layout(width="95%", height="220px"))
    tab_dyn = widgets.VBox([
        widgets.HTML("<b>6.</b> DYNAMIC SEGMENTATION demo: RFM re-scored "
                     "at two snapshot dates; the matrix shows how customers "
                     "MIGRATE between segments (how real CDPs do 'dynamic' "
                     "with batch re-scoring). Rows = segment at A, columns = "
                     "at B:"),
        _W["snapA"], _W["snapB"], _W["btn_mig"], _W["mig_out"],
        widgets.HTML("<small>Copy block for the AI exercise:</small>"),
        _W["mig_copy"]])

    tabs = widgets.Tab(children=[tab_data, tab_rfm, tab_plots, tab_seg,
                                 tab_dyn])
    for i, t in enumerate(["1 · Data", "2 · RFM", "3 · Plots",
                           "4 · Segments", "5 · Dynamic"]):
        tabs.set_title(i, t)

    _W["status"] = widgets.HTML("<i>Upload a transactions CSV to begin.</i>")
    display(widgets.VBox([
        widgets.HTML("<h2>RFM Segmentation App</h2>"),
        tabs, widgets.HTML("<hr>"), _W["status"]]))
