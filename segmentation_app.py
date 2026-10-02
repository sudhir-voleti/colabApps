
"""
=====================================================================
 segmentation_app.py  —  Generic K-Means Segmentation App (Colab)
 Repo: https://github.com/sudhir-voleti/colabApps/
---------------------------------------------------------------------
 Colab setup cell (paste as-is; bump ?v= after each repo update):

   import requests
   URL = ("https://raw.githubusercontent.com/sudhir-voleti/colabApps/"
          "main/segmentation_app.py?v=5")
   exec(requests.get(URL).text)
   launch_app()

 Flow:  upload CSV -> tick metric basis vars + categorical vars
        (auto one-hot) -> scree + silhouette -> choose K manually
        -> K-Means -> transposed centroid table (segments = columns;
        sortable) -> text-format simplified maxima/minima table
        (copy-paste ready) -> 10-row assignment preview + CSV download
=====================================================================
"""
import io
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# sortable/searchable tables (R DT::datatable-style); self-installs in Colab
try:
    from itables import show as _show_df
except Exception:
    try:
        import sys as _sys
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                        "itables"], check=True, capture_output=True)
        from itables import show as _show_df
    except Exception:
        _show_df = None          # fallback: plain pandas display

_S = {}
_W = {}


def _itable(df):
    """Display a DataFrame sortable/searchable; fallback to plain."""
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

def parse_upload(upload_value):
    item = (list(upload_value.values())[0] if isinstance(upload_value, dict)
            else upload_value[0])
    return pd.read_csv(io.BytesIO(item["content"]))


def build_design(df, num_cols, cat_cols):
    """Raw design matrix: numerics as-is + one-hot dummies (all levels)."""
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


def scree_stats(Xz, kmax=10):
    """Inertia for k=1..kmax and silhouette for k=2..kmax."""
    kmax = int(min(kmax, len(Xz) - 1))
    inert, sil = [], []
    for k in range(1, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xz)
        inert.append(km.inertia_)
        if 1 < k < len(Xz):
            sil.append(silhouette_score(Xz, km.labels_))
    return inert, sil


def fig_scree(inertias, sils=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = range(1, len(inertias) + 1)
    axes[0].plot(ks, inertias, "o-")
    axes[0].set(xlabel="K (number of clusters)",
                ylabel="Inertia (within-cluster SS)",
                title="Scree / elbow — sharpest bend")
    axes[0].grid(alpha=.3)
    if sils:
        axes[1].plot(range(2, 2 + len(sils)), sils, "s-", color="green")
        axes[1].set(xlabel="K", ylabel="Silhouette (higher = cleaner)",
                    title="Silhouette — cross-check for K")
        axes[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_biplot(Xz, labels, features):
    """PCA 2-D map of the clustering: dots = sessions (by segment),
    arrows = basis variables (direction/strength of association)."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xz)
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c in np.unique(labels):
        ax.scatter(Z[labels == c, 0], Z[labels == c, 1], s=30, alpha=.8,
                   label=f"segment {c}")
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    lim = float(np.abs(Z).max())
    scale = 0.85 * lim / float(np.abs(load).max())
    for j, name in enumerate(features):
        ax.arrow(0, 0, load[j, 0] * scale, load[j, 1] * scale,
                 head_width=.025 * scale, color="k", alpha=.6)
        ax.text(load[j, 0] * scale * 1.08, load[j, 1] * scale * 1.08,
                name, fontsize=8)
    ax.set(xlabel=f"PC1 ({ev[0]:.0%} of variance)",
           ylabel=f"PC2 ({ev[1]:.0%} of variance)",
           title="PCA biplot — 2-D projection of the K-Means solution")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
    m = lim * 1.15
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    fig.tight_layout()
    return fig


def fit_kmeans(Xz, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(Xz), km


def raw_profile(df, num_cols, labels):
    prof = (df[list(num_cols)].astype(float).groupby(labels).mean().round(2)
            if num_cols else pd.DataFrame(index=sorted(set(labels))))
    prof.insert(0, "pct", (np.bincount(labels) / len(labels) * 100).round(1))
    prof.insert(0, "size", np.bincount(labels))
    return prof.reset_index(names="segment")


def z_centroids(Xz_df, labels):
    return Xz_df.groupby(labels).mean().round(2).reset_index(names="segment")


def simplified_table(zc, size_map):
    """Where each segment attains the column-wise max / min (z-scale)."""
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
    tabs = []
    for c in cat_cols:
        t = (pd.crosstab(labels, df[c].astype(str), normalize="index") * 100
             ).round(1)
        t.insert(0, "segment", t.index)
        t.insert(1, "variable", c)
        tabs.append(t.reset_index(drop=True))
    return pd.concat(tabs, ignore_index=True) if tabs else None


def simplified_text(k, n, simp):
    """Plain-text maxima/minima 'table' — nothing truncated, copy-friendly."""
    lines = [f"K-Means segmentation output | K = {k} | n = {n}",
             "MAXIMA = basis variables where this segment scores HIGHEST "
             "across segments; MINIMA = where it scores LOWEST "
             "(centroids are z-standardized).", ""]
    for _, r in simp.iterrows():
        lines.append(f"SEGMENT {r['segment_num']} (n={r['size']})")
        lines.append(f"  MAXIMA: {r['maxima_basis']}")
        lines.append(f"  MINIMA: {r['minima_basis']}")
        lines.append("")
    return "\n".join(lines)


def transpose_centroids(tab, name):
    """segments -> columns, basis variables -> rows (fits the screen)."""
    t = tab.set_index("segment").T
    t.columns = [f"segment_{c}" for c in t.columns]
    t.index.name = name
    return t.reset_index()


# ------------------------------------------------------------------ #
# Handlers                                                          #
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
    low_card, high_card = [], []
    for c in _S["cat_cols"]:
        if df[c].nunique() <= min(10, max(2, len(df) // 5)):
            low_card.append(c)
        else:
            high_card.append(c)
    _S["low_card"], _S["high_card"] = low_card, high_card
    _W["num_boxes"] = [widgets.Checkbox(value=True, description=str(c),
                                        indent=False,
                                        layout=widgets.Layout(width="99%"))
                       for c in _S["num_cols"]]
    _W["cat_boxes"] = [widgets.Checkbox(value=(c in low_card),
                                        description=str(c), indent=False,
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
            print(f"! NOT pre-ticked (ID-like, one level per row -> would "
                  f"create {sum(df[c].nunique() for c in high_card)} dummy "
                  f"columns and break clustering): {', '.join(high_card)}")
        display(df.head(8))
    _status("CSV loaded. Review ticks below, then press "
            "<b>Prepare data</b>.", "seagreen")


def _on_prepare(btn):
    num = [b.description for b in _W.get("num_boxes", []) if b.value]
    cat = [b.description for b in _W.get("cat_boxes", []) if b.value]
    if not (num or cat):
        _status("Tick at least one basis variable.", "crimson"); return
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
    inert, sils = scree_stats(_S["Xz"])
    _S["inertias"] = inert
    with _W["scree_out"]:
        clear_output(wait=True); plt.close("all")
        display(fig_scree(inert, sils))
    _status("Read the sharpest bend (cross-check the silhouette panel), "
            "set K, then press <b>Run K-Means</b>.", "seagreen")


def _render_centroids(change=None):
    """Toggle the transposed centroid table between raw and z units.
    Same clusters either way: z*sigma + mu == raw cluster mean."""
    mode = _W["units"].value
    with _W["cen_out"]:
        clear_output(wait=True)
        if "prof_t" not in _S:
            print("Run K-Means to see centroids."); return
        unit_txt = ("original units" if mode == "raw"
                    else "z-score units, the scale K-Means used")
        print(f"CENTROID TABLE — segments are COLUMNS, basis variables are "
              f"rows ({unit_txt}):")
        _itable(_S["prof_t"] if mode == "raw" else _S["zc_t"])


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
    simp = simplified_table(zc, dict(zip(zc["segment"], prof["size"])))
    _S["prof_t"] = transpose_centroids(prof, "basis_variable")
    _S["zc_t"] = transpose_centroids(zc, "basis_variable")
    shares = cat_shares(_S["df"], _S["cat_cols"], labels)
    _S["result_df"] = None

    _render_centroids()
    with _W["shares_out"]:
        clear_output(wait=True)
        if shares is not None:
            print("CATEGORICAL LEVEL SHARES WITHIN SEGMENT (%):")
            _itable(shares)
        else:
            print("(no categorical variables selected)")
    _W["simp_box"].value = simplified_text(k, len(labels), simp)
    preview = _S["df"].head(10).copy()
    preview.insert(0, "segment", labels[:10])
    counts = (pd.Series(labels, name="segment").value_counts()
              .rename_axis("segment").reset_index(name="rows"))
    with _W["asg_out"]:
        clear_output(wait=True)
        print(f"K-Means | K = {k} | n = {len(labels)} "
              f"| inertia = {km.inertia_:.1f}")
        print("\nRows per segment:")
        _itable(counts)
        print("\nAssignment preview (first 10 rows):")
        _itable(preview)
    _S["result_df"] = _S["df"].copy()
    _S["result_df"]["segment"] = labels
    _status("Done. Check Centroids, Simplified, and Assignments tabs.",
            "seagreen")


def _on_biplot(btn):
    if "labels" not in _S:
        _status("Run K-Means first (Scree & K tab).", "crimson"); return
    with _W["biplot_out"]:
        clear_output(wait=True); plt.close("all")
        display(fig_biplot(_S["Xz"], _S["labels"], _S["features"]))
    _status("Biplot rendered (PCA view of the same solution).", "seagreen")


def _on_download(btn):
    if "result_df" not in _S or _S["result_df"] is None:
        _status("Run K-Means first.", "crimson"); return
    _S["result_df"].to_csv("segment_assignments.csv", index=False)
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
    try:                                   # sortable tables (itables)
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
        widgets.HTML("<b>2.</b> Select basis variables:"),
        widgets.HTML("<b>Metric basis variables</b> — tick to include in "
                     "clustering:"),
        _W["num_sel_box"], _W["num_btns"],
        widgets.HTML("<b>Categorical variables</b> — ticked ones are one-hot "
                     "encoded as 0/1 dummies:"),
        _W["cat_sel_box"], _W["cat_btns"],
        widgets.HTML("<small>Tip: untick ID-like columns (session_id) and any "
                     "variable that is not a basis for similarity.</small>"),
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
                     "(elbow) in inertia, cross-checked by the silhouette "
                     "panel (its peak often corrects the elbow's bias "
                     "toward K=2):"),
        _W["btn_scree"], _W["scree_out"],
        widgets.HTML("<b>4.</b> Set K from the plots:"),
        _W["k"], _W["btn_run"]])

    _W["units"] = widgets.ToggleButtons(
        options=[("Raw units", "raw"), ("Z-scores", "z")],
        value="raw", description="Centroid units:")
    _W["units"].observe(_render_centroids, names="value")
    _W["cen_out"] = widgets.Output()
    _W["shares_out"] = widgets.Output()
    tab_cent = widgets.VBox([
        widgets.HTML("<b>5.</b> Centroid table — segments are columns, "
                     "basis variables are rows (click any column header to "
                     "sort). Toggle units — same clusters either way; "
                     "Raw = z &times; SD + mean:"),
        _W["units"], _W["cen_out"],
        widgets.HTML("<hr>"), _W["shares_out"]])

    _W["simp_box"] = widgets.Textarea(
        value="Run K-Means to generate the simplified maxima/minima table.",
        layout=widgets.Layout(width="95%", height="300px"))
    tab_simp = widgets.VBox([
        widgets.HTML("<b>6.</b> Simplified table (text format — nothing "
                     "truncated). Click in the box, Ctrl/Cmd-A to select "
                     "all, copy, and paste into your AI interpretation "
                     "prompt:"),
        _W["simp_box"]])

    _W["btn_biplot"] = widgets.Button(description="Show biplot",
                                      button_style="info")
    _W["btn_biplot"].on_click(_on_biplot)
    _W["biplot_out"] = widgets.Output()
    tab_bip = widgets.VBox([
        widgets.HTML("PCA biplot — a 2-D projection of the SAME clustering "
                     "(K-Means ran in full feature space; this is just an "
                     "eyeball view). Dots = sessions, colored by segment; "
                     "arrows = basis variables, longer = more separation "
                     "along that direction:"),
        _W["btn_biplot"], _W["biplot_out"]])

    _W["asg_out"] = widgets.Output()
    _W["btn_dl"] = widgets.Button(description="⬇ Download full assignments CSV",
                                  button_style="success")
    _W["btn_dl"].on_click(_on_download)
    tab_asg = widgets.VBox([
        widgets.HTML("<b>7.</b> Segment assignments — row counts per segment "
                     "and a 10-row preview (click headers to sort). Full "
                     "data via the download button:"),
        _W["asg_out"], _W["btn_dl"]])

    tabs = widgets.Tab(children=[tab_data, tab_scree, tab_cent, tab_bip,
                                 tab_simp, tab_asg])
    for i, t in enumerate(["1 · Data", "2 · Scree & K", "3 · Centroids",
                           "4 · Biplot", "5 · Simplified",
                           "6 · Assignments"]):
        tabs.set_title(i, t)

    _W["status"] = widgets.HTML("<i>Upload a CSV to begin.</i>")
    display(widgets.VBox([
        widgets.HTML("<h2>Segmentation App — K-Means Cluster Analysis</h2>"),
        tabs, widgets.HTML("<hr>"), _W["status"]]))
