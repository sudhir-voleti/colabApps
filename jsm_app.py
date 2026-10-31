# jsm_app.py
# Colab/ipywidgets port of the Joint Space Map (JSM) R Shiny app.
# Perceptual CSV: rows = attributes, columns = firms (first column = attribute names).
# Preference CSV: rows = users, columns = firms (first column = user names).
# Deploy: exec(compile(requests.get(<raw github url>).text, <url>, "exec"), globals())

import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, clear_output, display

PERC = None  # perceptual data: attributes (rows) x firms (columns)
PREF = None  # preference data: users (rows) x firms (columns)


# ---------------------------------------------------------------- helpers
def uploaded_content(upload_widget):
    value = upload_widget.value
    if not value:
        return None
    if isinstance(value, dict):
        return next(iter(value.values()))["content"]
    return value[0]["content"]


def read_upload(upload_widget, label):
    content = uploaded_content(upload_widget)
    if content is None:
        raise ValueError(f"Upload a {label} CSV first.")
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        raise ValueError(f"The {label} CSV has no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"The {label} CSV has duplicate column names.")
    # First column holds the row names (attributes or users).
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    try:
        df = df.astype(float)
    except ValueError:
        raise ValueError(f"All {label} values (after the first name column) must be numeric.")
    if df.isna().any().any():
        raise ValueError(f"The {label} CSV contains missing values.")
    return df


def html_table(df, index=True):
    return HTML(
        df.to_html(
            index=index,
            border=0,
            classes="table table-striped table-hover table-sm",
            justify="center",
        )
    )


# ---------------------------------------------------------------- widgets
perc_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Perceptual CSV")
pref_upload = widgets.FileUpload(accept=".csv", multiple=False, description="Preference CSV")

attr_widget = widgets.SelectMultiple(
    description="Attributes (min 3)", rows=8, style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
firm_widget = widgets.SelectMultiple(
    description="Firms (min 2, spider chart)", rows=6, style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
user_widget = widgets.SelectMultiple(
    description="Users (min 1)", rows=6, style={"description_width": "initial"}, layout=widgets.Layout(width="95%")
)
k0_widget = widgets.FloatText(value=1.5, description="Scaling factor entities (k0)", style={"description_width": "initial"})
k1_widget = widgets.FloatText(value=1.5, description="Scaling factor users (k1)", style={"description_width": "initial"})
render_button = widgets.Button(description="Render plots", button_style="success", icon="play")

data_preview_out = widgets.Output()
setup_status_out = widgets.Output()
pca_out = widgets.Output()
jsm_out = widgets.Output()
spider_out = widgets.Output()
data_out = widgets.Output()


# ---------------------------------------------------------------- upload handlers
def preview_data():
    with data_preview_out:
        clear_output(wait=True)
        if PERC is None and PREF is None:
            display(Markdown("Upload the perceptual CSV (attributes in rows, firms in columns). Preference CSV is optional."))
            return
        if PERC is not None:
            display(Markdown(f"### Perceptual preview — {PERC.shape[0]} attributes × {PERC.shape[1]} firms"))
            display(PERC.head(10))
        if PREF is not None:
            display(Markdown(f"### Preference preview — {PREF.shape[0]} users × {PREF.shape[1]} firms"))
            display(PREF.head(10))


def on_perc_upload(change):
    global PERC
    try:
        PERC = read_upload(perc_upload, "perceptual")
        attr_widget.options = list(PERC.index)
        attr_widget.value = tuple(PERC.index)
        firm_widget.options = list(PERC.columns)
        firm_widget.value = tuple(PERC.columns)
        preview_data()
    except Exception as exc:
        with data_preview_out:
            clear_output(wait=True)
            display(Markdown(f"**Perceptual-file error:** {exc}"))


def on_pref_upload(change):
    global PREF
    try:
        PREF = read_upload(pref_upload, "preference")
        user_widget.options = list(PREF.index)
        user_widget.value = tuple(PREF.index[:1])  # R default: first user only
        preview_data()
    except Exception as exc:
        with data_preview_out:
            clear_output(wait=True)
            display(Markdown(f"**Preference-file error:** {exc}"))


# ---------------------------------------------------------------- analysis
def selected_matrices():
    """Replicates the R subsetting: firms x attributes matrix + preference matrix."""
    if PERC is None:
        raise ValueError("Upload the perceptual CSV in Tab 0.")
    attrs = list(attr_widget.value)
    if len(attrs) < 3:
        raise ValueError("Select at least 3 attributes.")
    mydata = PERC.loc[attrs].T  # firms x attributes (the R t(mydata))

    pref = None
    if PREF is not None:
        users = list(user_widget.value)
        if len(users) < 1:
            raise ValueError("Select at least 1 user.")
        missing = [f for f in mydata.index if f not in PREF.columns]
        if missing:
            raise ValueError(f"Preference CSV is missing firm columns: {missing}")
        pref = PREF.loc[users, mydata.index]  # users x firms, aligned to firm order
    return mydata, pref


def pca_fit(X, scale):
    """prcomp equivalent: returns sdev, rotation (variables x components), scores."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    if scale:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sdev = S / np.sqrt(X.shape[0] - 1)
    rotation = Vt.T
    scores = U * S
    return sdev, rotation, scores


def plot_pca_variance(mydata):
    """Scree plot, mirroring R's plot(prcomp(mydata2)) — unscaled PCA."""
    sdev, _, _ = pca_fit(mydata.to_numpy(), scale=False)
    variances = sdev**2
    with pca_out:
        clear_output(wait=True)
        display(Markdown("## PCA variance plot"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([f"PC{i+1}" for i in range(len(variances))], variances, color="#4C78A8")
        ax.set_ylabel("Variance")
        ax.set_title("Variance of PCA")
        plt.tight_layout()
        plt.show()


def plot_jsm(mydata, pref, k0, k1):
    """Joint space map: attribute arrows, firm points, user preference vectors."""
    sdev, rotation, scores = pca_fit(mydata.to_numpy(), scale=True)
    # R: fit1$x[,j] = fit$x[,j] / sum(abs(fit$x[,j]))
    scores_n = scores / np.abs(scores).sum(axis=0)

    with jsm_out:
        clear_output(wait=True)
        display(Markdown("## Joint space map"))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Joint Space map")

        # Attribute vectors (loadings x sdev), blue arrows.
        for i, attrib in enumerate(mydata.columns):
            x1 = rotation[i, 0] * sdev[0]
            y1 = rotation[i, 1] * sdev[1]
            ax.annotate("", xy=(x1, y1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
            ax.text(x1, y1, attrib, color="blue", fontsize=11)

        # Firm positions (normalized scores x k0), red points.
        fx = scores_n[:, 0] * k0
        fy = scores_n[:, 1] * k0
        ax.scatter(fx, fy, color="red", zorder=3)
        for x, y, name in zip(fx, fy, mydata.index):
            ax.text(x, y, name, color="black", fontsize=11)

        # User preference vectors (pref %*% normalized scores), R's maroon2 = #EE30A7.
        if pref is not None:
            pref1 = pref.to_numpy(dtype=float) @ scores_n[:, :2]
            for (x, y), name in zip(pref1 * k1, pref.index):
                ax.plot([0, x], [0, y], color="#EE30A7", linewidth=1.25)
                ax.scatter([x], [y], color="#EE30A7", zorder=3)
                ax.text(x, y, name, color="#EE30A7", fontsize=11, ha="center", va="center")
        plt.tight_layout()
        plt.show()


def plot_spider(mydata):
    """fmsb::radarchart equivalent: per-axis max/min scaling, one polygon per firm."""
    firms = list(firm_widget.value)
    if len(firms) < 2:
        raise ValueError("Select at least 2 firms for the spider chart.")
    sub = mydata.loc[firms]  # selected firms x attributes
    attrs = list(sub.columns)
    vmin = sub.min(axis=0).to_numpy()
    vmax = sub.max(axis=0).to_numpy()
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)

    angles = np.linspace(0, 2 * np.pi, len(attrs), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    with spider_out:
        clear_output(wait=True)
        display(Markdown("## Spider chart"))
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10.colors
        for j, firm in enumerate(sub.index):
            vals = ((sub.loc[firm].to_numpy() - vmin) / span).tolist()
            vals += vals[:1]
            ax.plot(angles_closed, vals, color=colors[j % 10], linewidth=1.5, label=firm)
            ax.fill(angles_closed, vals, color=colors[j % 10], alpha=0.08)
        ax.set_xticks(angles)
        ax.set_xticklabels(attrs)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([])
        ax.set_title("Spider Chart", pad=20)
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
        plt.tight_layout()
        plt.show()


def render_data_tables(mydata, pref):
    with data_out:
        clear_output(wait=True)
        display(Markdown("### Perceptual data (selected attributes)"))
        d = PERC.loc[list(attr_widget.value)].copy()
        d.insert(0, "Attributes", d.index)
        display(html_table(d, index=False))
        if pref is not None:
            display(Markdown("### Preference data (selected users)"))
            d1 = pref.copy()
            d1.insert(0, "User", d1.index)
            display(html_table(d1, index=False))
        else:
            display(Markdown("### Preference data\nNo preference CSV uploaded."))


def render_all(_):
    try:
        mydata, pref = selected_matrices()
        plot_pca_variance(mydata)
        plot_jsm(mydata, pref, float(k0_widget.value), float(k1_widget.value))
        plot_spider(mydata)
        render_data_tables(mydata, pref)
        with setup_status_out:
            clear_output(wait=True)
            n_users = 0 if pref is None else pref.shape[0]
            display(
                Markdown(
                    f"**Plots rendered.** Firms: **{mydata.shape[0]}**, attributes: **{mydata.shape[1]}**, "
                    f"users: **{n_users}**. See Tabs 2–5."
                )
            )
        tabs.selected_index = 2
    except Exception as exc:
        with setup_status_out:
            clear_output(wait=True)
            display(Markdown(f"**Render error:** {exc}"))


perc_upload.observe(on_perc_upload, names="value")
pref_upload.observe(on_pref_upload, names="value")
render_button.on_click(render_all)

intro_text = """
**Joint Space Map** — perceptual mapping with preference vectors.

- **Perceptual CSV:** top row = firm names, first column = attribute names (attributes in rows, firms in columns). All values numeric.
- **Preference CSV:** top row = firm names (same firms), first column = user names; each cell is that user's preference rating for that firm. Optional.
- Upload both files, select attributes / firms / users in Tab 1, then click **Render plots**.
"""

tab0 = widgets.VBox(
    [
        widgets.HTML("<h2>0. Case & Data</h2>"),
        widgets.HTML(intro_text),
        perc_upload,
        pref_upload,
        data_preview_out,
    ]
)
tab1 = widgets.VBox(
    [
        widgets.HTML("<h2>1. Setup</h2>"),
        attr_widget,
        firm_widget,
        user_widget,
        k0_widget,
        k1_widget,
        render_button,
        setup_status_out,
    ]
)
tabs = widgets.Tab(children=[tab0, tab1, pca_out, jsm_out, spider_out, data_out])
for index, title in enumerate(
    ["0. Case & Data", "1. Setup", "2. PCA Variance", "3. JSM Plot", "4. Spider Chart", "5. Data"]
):
    tabs.set_title(index, title)

display(widgets.HTML("<h1>Joint Space Map App for Colab</h1>"))
display(tabs)
preview_data()
