
"""
=====================================================================
 segmentation_app.py  —  Generic K-Means Segmentation App (Colab)
 Repo: https://github.com/sudhir-voleti/colabApps/
---------------------------------------------------------------------
 Colab setup cell (paste as-is):

   import requests
   URL = ("https://raw.githubusercontent.com/sudhir-voleti/colabApps/"
          "main/segmentation_app.py?v=1")   # bump ?v= after edits
   exec(requests.get(URL).text)
   launch_app()

 Flow:  upload CSV -> pick metric basis vars + categorical vars
        (auto one-hot encoded) -> scree plot -> choose K manually
        -> K-Means -> full raw centroid table + z-centroid table
        -> simplified maxima/minima table (copy-paste ready for an LLM)
=====================================================================
"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

_S = {}      # state: df, num_cols, cat_cols, Xraw, Xz, features, labels ...
_W = {}      # widget handles


# ------------------------------------------------------------------ #
# Engine (widget-free)                                              #
# ------------------------------------------------------------------ #

def parse_upload(upload_value):
    """Return DataFrame from a widgets.FileUpload value."""
    item = (list(upload_value.values())[0] if isinstance(upload_value, dict)
            else upload_value[0])
    return pd.read_csv(io.BytesIO(item["content"]))


def build_design(df, num_cols, cat_cols):
    """Raw design matrix: numeric columns as-is + one-hot dummies
    (all levels kept, no drop-first). Returns DataFrame."""
    parts, names = [], []
    if num_cols:
        parts.append(df[list(num_cols)].astype(float).values)
        names += list(num_cols)
    if cat_cols:
        d = pd.get_dummies(df[list(cat_cols)].astype(str),
                           columns=list(cat_cols), drop_first=False,
                           dtype=float)
        parts.append(d.values)
        names += list(d.columns)
    if not parts:
        raise ValueError("Select at least one basis variable.")
    X = parts[0] if len(parts) == 1 else np.hstack(parts)
    return pd.DataFrame(X, columns=names, index=df.index)


def scree_inertias(Xz, kmax=10):
    kmax = int(min(kmax, len(Xz) - 1))
    return [KMeans(n_clusters=k, n_init=10, random_state=42)
            .fit(Xz).inertia_ for k in range(1, kmax + 1)]


def fig_scree(inertias):
    ks = range(1, len(inertias) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "o-")
    ax.set(xlabel="K (number of clusters)",
           ylabel="Inertia (within-cluster SS)",
           title="Scree plot — pick K at the sharpest bend")
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fit_kmeans(Xz, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(Xz), km


def raw_profile(df, num_cols, labels):
    """Cluster means on the original scale: numeric means + size/pct."""
    prof = (df[list(num_cols)].astype(float).groupby(labels).mean().round(2)
            if num_cols else pd.DataFrame(index=sorted(set(labels))))
    prof.insert(0, "pct", (np.bincount(labels) / len(labels) * 100).round(1))
    prof.insert(0, "size", np.bincount(labels))
    return prof.reset_index(names="segment")


def z_centroids(Xz_df, labels):
    zc = Xz_df.groupby(labels).mean().round(2)
    return zc.reset_index(names="segment")


def simplified_table(zc, size_map):
    """3-column LLM-ready table: where each segment attains the
    column-wise max / min of the z-score centroids."""
    feats = [c for c in zc.columns if c != "segment"]
    mx, mn = zc[feats].max(), zc[feats].min()
    rows = []
    for _, r in zc.iterrows():
        maxima = ", ".join(f for f in feats if r[f] >= mx[f] - 1e-9)
        minima = ", ".join(f for f in feats if r[f] <= mn[f] + 1e-9)
        rows.append((int(r["segment"]), int(size_map[r["segment"]]),
                     maxima, minima))
    return pd.DataFrame(rows, columns=["segment_num", "size",
                                       "maxima_basis", "minima_basis"])


def cat_shares(df, cat_cols, labels):
    """Level shares within each segment (for profiling context)."""
    tabs = []
    for c in cat_cols:
        t = (pd.crosstab(labels, df[c].astype(str), normalize="index") * 100
             ).round(1)
        t.insert(0, "segment", t.index)
        t.insert(1, "variable", c)
        tabs.append(t.reset_index(drop=True))
    return pd.concat(tabs, ignore_index=True) if tabs else None


def _render_centroids(change=None):
    """Toggle the centroid table between raw units and z-score units.
    Same clusters either way: z*sigma + mu == raw cluster mean."""
    mode = _W["units"].value
    with _W["cen_out"]:
        clear_output(wait=True)
        if "prof" not in _S:
            print("Run K-Means to see centroids."); return
        if mode == "raw":
            print("CENTROID TABLE — original units (means per segment):")
            display(_S["prof"])
        else:
            print("CENTROID TABLE — z-score units (the scale K-Means used):")
            display(_S["zc"])


def copy_text(k, n, simp):
    lines = [f"K-Means segmentation output | K = {k} | n = {n}",
             "Column legend: MAXIMA = basis variables where this segment "
             "scores HIGHEST across segments; MINIMA = where it scores "
             "LOWEST (centroids are z-standardized).", ""]
    for _, r in simp.iterrows():
        lines.append(f"Segment {r['segment_num']} (n={r['size']}): "
                     f"MAXIMA: {r['maxima_basis']} | "
                     f"MINIMA: {r['minima_basis']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Widget handlers                                                   #
# ------------------------------------------------------------------ #

def _status(msg, color="#333"):
    _W["status"].value = f"<div style='color:{color};font-size:13px'>{msg}</div>"


def _on_load(btn):
    if not _W["upload"].value:
        _status("Choose a CSV file first.", "crimson"); return
    try:
        _S["df"] = parse_upload(_W["upload"].value)
    except Exception as e:
        _status(f"Could not read CSV: {e}", "crimson"); return
    df = _S["df"]
    _S["num_cols"] = list(df.select_dtypes(include=np.number).columns)
    _S["cat_cols"] = [c for c in df.columns if c not in _S["num_cols"]]
    # ID-like columns (one level per row) must NOT become dummies
    low_card, high_card = [], []
    for c in _S["cat_cols"]:
        if df[c].nunique() <= min(10, max(2, len(df) // 5)):
            low_card.append(c)
        else:
            high_card.append(c)
    _S["low_card"], _S["high_card"] = low_card, high_card
    # one checkbox per variable; high-cardinality NOT pre-ticked
    _W["num_boxes"] = [widgets.Checkbox(value=True, description=str(c),
                                        indent=False,
                                        layout=widgets.Layout(width="99%"))
                       for c in _S["num_cols"]]
    _W["cat_boxes"] = [widgets.Checkbox(value=(c in low_card), description=str(c),
                                        indent=False,
                                        layout=widgets.Layout(width="99%"))
                       for c in _S["cat_cols"]]
    _W["num_sel_box"].children = (widgets.VBox(
        _W["num_boxes"],
        layout=widgets.Layout(max_height="240px", overflow_y="auto")),)
    _W["cat_sel_box"].children = (widgets.VBox(
        _W["cat_boxes"],
        layout=widgets.Layout(max_height="160px", overflow_y="auto")),)
    with _W["data_out"]:
        clear_output(wait=True)
        print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Numeric columns ({len(_S['num_cols'])}): "
              f"{', '.join(_S['num_cols'])}")
        print(f"Text/categorical columns ({len(_S['cat_cols'])}): "
              f"{', '.join(_S['cat_cols']) or 'none'}")
        if high_card:
            print(f"! NOT pre-selected (ID-like, one level per row -> would "
                  f"create {sum(df[c].nunique() for c in high_card)} dummy "
                  f"columns and break clustering): {', '.join(high_card)}")
        display(df.head(8))
    _status("CSV loaded. Review selections below, then press "
            "<b>Prepare data</b>.", "seagreen")


def _on_prepare(btn):
    num = [b.description for b in _W.get("num_boxes", []) if b.value]
    cat = [b.description for b in _W.get("cat_boxes", []) if b.value]
    if not (num or cat):
        _status("Pick at least one basis variable.", "crimson"); return
    try:
        Xraw = build_design(_S["df"], num, cat)
    except ValueError as e:
        _status(str(e), "crimson"); return
    _S.update(num_cols=num, cat_cols=cat, Xraw=Xraw,
              features=list(Xraw.columns))
    _S["Xz"] = (StandardScaler().fit_transform(Xraw.values)
                if _W["std"].value else Xraw.values)
    with _W["prep_out"]:
        clear_output(wait=True)
        print(f"Design matrix ready: {Xraw.shape[0]} sessions x "
              f"{Xraw.shape[1]} features")
        print(f"  metric basis vars ({len(num)}): {', '.join(num)}")
        if cat:
            lv = [c for c in Xraw.columns if c not in num]
            print(f"  one-hot encoded ({len(lv)}): {', '.join(lv)}")
        print(f"  standardized: {_W['std'].value}")
        print("Go to the 'Scree & K' tab.")
    _status("Data prepared. Next: show the scree plot and pick K.",
            "seagreen")


def _on_scree(btn):
    if "Xz" not in _S:
        _status("Prepare data first (Data tab).", "crimson"); return
    inert = scree_inertias(_S["Xz"])
    _S["inertias"] = inert
    with _W["scree_out"]:
        clear_output(wait=True); plt.close("all")
        display(fig_scree(inert)); plt.show()
    _status("Read the sharpest bend to choose K, then press "
            "<b>Run K-Means</b>.", "seagreen")


def _on_run(btn):
    if "Xz" not in _S:
        _status("Prepare data first (Data tab).", "crimson"); return
    k = _W["k"].value
    if k >= len(_S["Xz"]):
        _status("K must be smaller than the number of rows.", "crimson")
        return
    labels, km = fit_kmeans(_S["Xz"], k)
    _S["labels"] = labels
    Xz_df = pd.DataFrame(_S["Xz"], columns=_S["features"])
    prof = raw_profile(_S["df"], _S["num_cols"], labels)
    zc = z_centroids(Xz_df, labels)
    simp = simplified_table(zc.set_index("segment").reset_index(),
                            dict(zip(zc["segment"], prof["size"])))
    _S["simp"] = simp
    shares = cat_shares(_S["df"], _S["cat_cols"], labels)

    _S.update(prof=prof, zc=zc, simp=simp)
    _render_centroids()
    with _W["res_out"]:
        clear_output(wait=True)
        print(f"K-Means | K = {k} | n = {len(labels)} "
              f"| inertia = {km.inertia_:.1f}")
        print("\nSIMPLIFIED TABLE — maxima/minima basis per segment "
              "(LLM-ready):")
        display(simp)
        if shares is not None:
            print("\nCATEGORICAL LEVEL SHARES WITHIN SEGMENT (%):")
            display(shares)
    _W["copy"].value = copy_text(k, len(labels), simp)
    _status("Done. Copy the text box below into your AI for the "
            "interpretation exercise.", "seagreen")


def _on_download(btn):
    if "labels" not in _S:
        _status("Run K-Means first.", "crimson"); return
    out = _S["df"].copy(); out["segment"] = _S["labels"]
    out.to_csv("segment_assignments.csv", index=False)
    try:
        from google.colab import files
        files.download("segment_assignments.csv")
    except Exception:
        _status("Saved 'segment_assignments.csv' in the Colab file pane.",
                "seagreen")


# ------------------------------------------------------------------ #
# UI                                                                #
# ------------------------------------------------------------------ #

def launch_app():
    style = {"description_width": "180px"}

    _W["upload"] = widgets.FileUpload(accept=".csv", multiple=False,
                                      description="CSV")
    _W["btn_load"] = widgets.Button(description="Load CSV",
                                    button_style="info", icon="upload")
    _W["btn_load"].on_click(_on_load)

    _W["data_out"] = widgets.Output()
    _W["num_sel_box"] = widgets.VBox()   # checkbox containers, filled on load
    _W["cat_sel_box"] = widgets.VBox()

    def _set_boxes(boxes, v):
        for b in boxes:
            b.value = v

    def _mk_btns(key):
        ba = widgets.Button(description="Select all",
                            layout=widgets.Layout(width="95px"))
        bc = widgets.Button(description="Clear",
                            layout=widgets.Layout(width="70px"))
        ba.on_click(lambda b: _set_boxes(_W.get(key, []), True))
        bc.on_click(lambda b: _set_boxes(_W.get(key, []), False))
        return widgets.HBox([ba, bc])
    _W["num_btns"] = _mk_btns("num_boxes")
    _W["cat_btns"] = _mk_btns("cat_boxes")
    _W["units"] = widgets.ToggleButtons(
        options=[("Raw units", "raw"), ("Z-scores", "z")],
        value="raw", description="Centroid units:")
    _W["units"].observe(_render_centroids, names="value")
    _W["cen_out"] = widgets.Output()
    _W["std"] = widgets.Checkbox(
        value=True, indent=False, style=style,
        description="Standardize before clustering (recommended)")
    _W["btn_prep"] = widgets.Button(description="Prepare data",
                                    button_style="info")
    _W["btn_prep"].on_click(_on_prepare)
    _W["prep_out"] = widgets.Output()

    tab_data = widgets.VBox([
        widgets.HTML("<b>1.</b> Upload + inspect:"),
        widgets.HBox([_W["upload"], _W["btn_load"]]),
        _W["data_out"],
        widgets.HTML("<b>2.</b> Select basis variables. Metric vars enter "
                     "the distance as numbers; categorical vars are "
                     "one-hot encoded as 0/1 dummies (all levels kept)."),
        widgets.HTML("<b>Metric basis variables</b> — tick to include in "
                     "clustering:"),
        _W["num_sel_box"], _W["num_btns"],
        widgets.HTML("<b>Categorical variables</b> — ticked ones are one-hot "
                     "encoded as 0/1 dummies:"),
        _W["cat_sel_box"], _W["cat_btns"],
        _W["std"], _W["btn_prep"],
        _W["prep_out"]])

    _W["btn_scree"] = widgets.Button(description="Show scree plot",
                                     button_style="info")
    _W["btn_scree"].on_click(_on_scree)
    _W["scree_out"] = widgets.Output()
    _W["k"] = widgets.IntSlider(value=3, min=2, max=10, style=style,
                                description="K (num clusters)")
    _W["btn_run"] = widgets.Button(description="▶ Run K-Means",
                                   button_style="success")
    _W["btn_run"].on_click(_on_run)
    tab_scree = widgets.VBox([
        widgets.HTML("<b>3.</b> Scree plot — look for the sharpest bend "
                     "(elbow) in inertia:"),
        _W["btn_scree"], _W["scree_out"],
        widgets.HTML("<b>4.</b> Set K from the elbow:"),
        _W["k"], _W["btn_run"]])

    _W["res_out"] = widgets.Output()
    _W["copy"] = widgets.Textarea(
        value="Run K-Means to generate the copy-paste summary.",
        layout=widgets.Layout(width="95%", height="160px"))
    _W["btn_dl"] = widgets.Button(description="⬇ segment assignments CSV")
    _W["btn_dl"].on_click(_on_download)
    tab_res = widgets.VBox([
        widgets.HTML("<b>5.</b> Centroid table — toggle units (same clusters "
                     "either way; Raw = z &times; SD + mean):"),
        _W["units"], _W["cen_out"],
        widgets.HTML("<b>6.</b> Simplified maxima/minima table + categorical "
                     "shares:"),
        _W["res_out"],
        widgets.HTML("<b>Copy-paste block for your AI interpretation "
                     "exercise</b> (click in, Ctrl/Cmd-A, copy):"),
        _W["copy"], _W["btn_dl"]])

    tabs = widgets.Tab(children=[tab_data, tab_scree, tab_res])
    tabs.set_title(0, "1 · Data")
    tabs.set_title(1, "2 · Scree & K")
    tabs.set_title(2, "3 · Results")

    _W["status"] = widgets.HTML("<i>Upload a CSV to begin.</i>")
    display(widgets.VBox([
        widgets.HTML("<h2>Segmentation App — K-Means Cluster Analysis</h2>"),
        tabs, widgets.HTML("<hr>"), _W["status"]]))
