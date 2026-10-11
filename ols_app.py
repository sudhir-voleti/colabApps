# ============================================================================
# OLS App (The Fairness Machine) -- STANDARD TEMPLATE, Day 3
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/ols_app.py").text)
#   launch_app()
#
# TAB 1 - Pick the outcome (Y) and the drivers (any columns).
# TAB 2 - The Driver Table: one row per driver, effect in plain words,
#         the band, and a trust verdict. R2 as 'share of wobble explained'.
# TAB 3 - Elasticity: log-log of volume on price -> the 1%-1% number.
# Button-gated. No Greek, no p-values - bands and plain words only.
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
    print("  The Fairness Machine: one number per driver, others held still")
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
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))

    # ---------------- TAB 1: pick Y and drivers ----------------
    dd_y = widgets.Dropdown(options=num_cols, description="Outcome (Y):")
    sel_x = widgets.SelectMultiple(options=list(df.columns),
                                   description="Drivers:",
                                   rows=8,
                                   layout=widgets.Layout(width="340px"))
    btn = widgets.Button(description="Run the model", button_style="success",
                         icon="play")
    out_model = widgets.Output()
    state = {"fitted": False}

    def run_model(btn_=None):
        with out_model:
            out_model.clear_output(wait=True)
            y = dd_y.value
            drivers = list(sel_x.value)
            if not drivers:
                print("Pick at least one driver (hold Ctrl for several).")
                return
            import statsmodels.formula.api as smf
            terms = [f"C({d})" if df[d].dtype == object else d
                     for d in drivers]
            f = f"{y} ~ " + " + ".join(terms)
            try:
                m = smf.ols(f, data=df).fit()
            except Exception as e:
                print(f"Model failed: {e}")
                return
            state["fitted"] = True

            rows = []
            ci = m.conf_int()
            for name in m.params.index:
                if name == "Intercept":
                    continue
                b, t = m.params[name], m.tvalues[name]
                lo, hi = ci.loc[name]
                if name.startswith("C("):
                    col = name[2:name.index(")")]
                    lvl = name.split("[T.")[1].rstrip("]")
                    plain = f"being {lvl} (vs the rest of {col})"
                else:
                    col = name
                    plain = (f"each +1 unit of {col}")
                verdict = ("TRUST IT - band stays on one side of zero"
                           if lo * hi > 0 else
                           "DON'T BET ON IT - band crosses zero")
                rows.append([col, plain, round(b, 3),
                             f"{lo:.2f} to {hi:.2f}", verdict])
            tbl = pd.DataFrame(rows, columns=["Driver", "Reading",
                                              "Effect on outcome",
                                              "Band (95%)", "Verdict"])
            tbl["|t|"] = np.abs(m.tvalues[[n for n in m.params.index
                                           if n != "Intercept"]].values)
            tbl = tbl.sort_values("|t|", ascending=False).drop(columns="|t|")
            display(tbl.reset_index(drop=True))
            print(f"\nTogether, these drivers explain "
                  f"{100*m.rsquared:.0f}% of the wobble in {y}.")
            print("Read each row as: that effect happens WHILE the other")
            print("drivers are held still. That fairness is the whole point.")

    btn.on_click(run_model)

    # ---------------- TAB 2: elasticity ----------------
    dd_price = widgets.Dropdown(options=num_cols, description="Price column:")
    dd_vol = widgets.Dropdown(options=[c for c in num_cols if c != num_cols[0]]
                              or num_cols, description="Volume column:")
    btn2 = widgets.Button(description="Get the elasticity",
                          button_style="warning", icon="percent")
    out_el = widgets.Output()

    def run_elasticity(btn_=None):
        with out_el:
            out_el.clear_output(wait=True)
            p, v = dd_price.value, dd_vol.value
            sub = df[[p, v]].dropna()
            sub = sub[(sub[p] > 0) & (sub[v] > 0)]
            if len(sub) < 20:
                print("Need at least 20 positive rows for this pair.")
                return
            lp, lv = np.log(sub[p]), np.log(sub[v])
            beta = np.polyfit(lp, lv, 1)
            slope = beta[0]
            fig, ax = plt.subplots(figsize=(7, 4.4))
            ax.scatter(lp, lv, alpha=0.4, s=20, color="#003366")
            xs = np.linspace(lp.min(), lp.max(), 100)
            ax.plot(xs, np.polyval(beta, xs), color="#DC2626", lw=2)
            ax.set_xlabel(f"log of {p}")
            ax.set_ylabel(f"log of {v}")
            ax.set_title("The pricing-power line (log-log)",
                         fontweight="bold")
            plt.tight_layout()
            plt.show()
            print(f"ELASTICITY = {slope:.2f}")
            print(f"A 1% change in {p} moves {v} by {slope:.2f}%.")
            if slope < -1:
                print("Volume is elastic: a price CUT gains more % volume "
                      "than it loses in % price - cuts can pay.")
            elif slope < 0:
                print("Volume is inelastic: a price RAISE loses little "
                      "volume - you may be underpriced.")
            print(f"\nNow the owner question: is that trade worth it at "
                  f"YOUR margin?")

    btn2.on_click(run_elasticity)
    if len(num_cols) >= 2:
        dd_vol.value = num_cols[1]

    tab = widgets.Tab()
    tab.children = [
        out_pre,
        widgets.VBox([dd_y, sel_x, btn, out_model]),
        widgets.VBox([widgets.HBox([dd_price, dd_vol]), btn2, out_el])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Driver Table")
    tab.set_title(2, "2. Elasticity")
    display(tab)
