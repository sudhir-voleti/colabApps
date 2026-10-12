# ============================================================================
# Multivariate Lens (The Wall + the Reveal) -- STANDARD TEMPLATE, Day 3
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/multivar_app.py").text)
#   launch_app()
#
# TAB 1 - The Wall: every numeric pair's co-movement, one heatmap.
# TAB 2 - The Reveal: pick X, pick Y, pick a split-by variable -> watch the
#         raw r collapse (or flip) inside groups. That gap is the lesson.
# ============================================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 68)
    print("  The Multivariate Lens: the Wall, and the Reveal")
    print("=" * 68)
    print("Upload any CSV (from the LMS) to begin.\n")
    display(upload)

    def on_upload(change):
        if not upload.value:
            return
        f = (list(upload.value.values())[0]
             if isinstance(upload.value, dict) else upload.value[0])
        content = f["content"] if isinstance(f, dict) else f.content
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        _build(df)

    upload.observe(on_upload, names="value")


def _build(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    # ---------------- TAB 1: The Wall ----------------
    out_wall = widgets.Output()
    with out_wall:
        if len(num_cols) < 2:
            print("Need at least two numeric columns.")
        else:
            corr = df[num_cols].corr().round(2)
            print("How strongly does every pair move together "
                  "(-1 to +1)? Everything whispers with everything -")
            print("the question is which whispers survive a closer look.\n")
            fig, ax = plt.subplots(figsize=(7.5, 6))
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=45, ha="right")
            ax.set_yticks(range(len(num_cols)))
            ax.set_yticklabels(num_cols)
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=9)
            fig.colorbar(im, shrink=0.8)
            ax.set_title("Pairwise co-movement of numeric columns",
                         fontweight="bold")
            plt.tight_layout()
            plt.show()

    # ---------------- TAB 2: The Reveal ----------------
    dd_x = widgets.Dropdown(options=num_cols, description="X:")
    dd_y = widgets.Dropdown(options=[c for c in num_cols if c != num_cols[0]]
                            or num_cols, description="Y:")
    dd_s = widgets.Dropdown(options=["(no split)"] + cat_cols + num_cols,
                            value="(no split)", description="Split by:")
    btn = widgets.Button(description="Show", button_style="success",
                         icon="search")
    out_rev = widgets.Output()

    def show_split(btn_=None):
        with out_rev:
            out_rev.clear_output(wait=True)
            x, y, s = dd_x.value, dd_y.value, dd_s.value
            sub = df[[x, y]].dropna()
            if len(sub) < 10:
                print("Not enough complete rows for this pair.")
                return
            r_raw = sub[x].corr(sub[y])
            print(f"RAW:  r({x}, {y}) = {r_raw:+.2f}   "
                  f"(computed on {len(sub)} rows)\n")

            fig, ax = plt.subplots(figsize=(8, 4.6))
            groups = {}
            if s != "(no split)":
                sv = df.loc[sub.index, s]
                if s in num_cols:      # numeric -> split at the median
                    cut = sv.median()
                    groups = {f"{s} below {cut:.1f}": sv <= cut,
                              f"{s} above {cut:.1f}": sv > cut}
                else:
                    for lvl in sv.unique()[:4]:
                        groups[f"{s} = {lvl}"] = sv == lvl
            colors = ["#DC2626", "#2563EB", "#059669", "#D97706"]
            for i, (name, mask) in enumerate(groups.items()):
                g = sub[mask]
                if len(g) < 8:
                    continue
                r_g = g[x].corr(g[y])
                ax.scatter(g[x], g[y], alpha=0.55, s=22,
                           color=colors[i % 4], label=f"{name}  (r={r_g:+.2f})")
                print(f"WITHIN {name}: r = {r_g:+.2f}  (n={len(g)})")
            if not groups:
                ax.scatter(sub[x], sub[y], alpha=0.45, s=22,
                           color="#003366")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title("Same two columns, split by a third",
                         fontweight="bold")
            if groups:
                ax.legend()
            ax.grid(True, ls=":", alpha=0.5)
            plt.tight_layout()
            plt.show()
            if groups:
                print("\nIf the number changed when you split, the raw pair")
                print("was carrying a third variable's shadow. That is the")
                print("whole lesson of today.")

    btn.on_click(show_split)
    if len(num_cols) >= 2:
        dd_y.value = num_cols[1]

    tab = widgets.Tab()
    tab.children = [out_pre, out_wall,
                    widgets.VBox([widgets.HBox([dd_x, dd_y]),
                                  dd_s, btn, out_rev])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Correln Tbl")
    tab.set_title(2, "2. ScatterPlot")
    display(tab)
