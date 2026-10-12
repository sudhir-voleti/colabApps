# ============================================================================
# OLS App v2 (The Fairness Machine) -- STANDARD TEMPLATE, generic CSV
# Repo : github.com/sudhir-voleti/colabApps  (CODE ONLY; CSVs via LMS upload)
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/ols_app.py").text)
#   launch_app()
#
# 0 Preview | 1 Select Variables (DV dropdown; IVs split metric/nonmetric;
#   live equation preview; lock on run) | 2 Results (raw + standardized
#   toggle) | 3 Price a Driver (delta x volume x periods -> Rs/yr)
#   | 4 Observed vs Predicted (fit plot + biggest errors)
#   | 5 Elasticity (log-log, bivariate)
# Button-gated. Plain words only: bands and verdicts, no p-values, no Greek.
# ============================================================================

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

TIME_NAMES = {"week", "month", "date", "day", "period", "yr", "year",
              "quarter"}
ID_PATTERNS = ("id", "code", "number", "no_", "_no", "num")


def _is_id(col):
    c = col.strip().lower()
    if c in TIME_NAMES:
        return True
    return any(p in c for p in ID_PATTERNS) and not any(
        k in c for k in ("sales", "limit", "qty", "count", "days",
                         "amount", "value", "margin", "profit", "revenue"))


def _inr(x):
    a = abs(x)
    if a >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"


def launch_app():
    upload = widgets.FileUpload(accept=".csv", multiple=False,
                                description="Upload CSV",
                                button_style="primary", icon="upload")
    print("=" * 68)
    print("  The Fairness Machine: one number per driver, "
          "others held still")
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
    num_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_all = [c for c in df.columns if c not in num_all]
    id_cols = [c for c in df.columns if _is_id(c)]
    dv_opts = [c for c in num_all if c not in id_cols]
    met_opts = [c for c in num_all if c not in id_cols]
    nonmet_opts = [c for c in cat_all if c not in id_cols]

    state = {"model": None, "fit": None, "y": None, "drivers": []}

    # ---------------- TAB 0 ----------------
    out_pre = widgets.Output()
    with out_pre:
        print(f"First 10 of {df.shape[0]} rows:")
        display(df.head(10))
        if id_cols:
            print(f"(treated as identifiers, not offered as variables: "
                  f"{', '.join(id_cols)})")

    # ---------------- TAB 1: select ----------------
    dd_y = widgets.Dropdown(options=dv_opts or ["(none)"],
                            description="Outcome (Y):")
    sel_met = widgets.SelectMultiple(options=met_opts, description="Metric IVs:",
                                     rows=7, layout=widgets.Layout(width="330px"))
    sel_non = widgets.SelectMultiple(options=nonmet_opts,
                                     description="Non-metric IVs:",
                                     rows=7,
                                     layout=widgets.Layout(width="330px"))
    btn_all = widgets.Button(description="Select all", button_style="info",
                             icon="check")
    btn_clr = widgets.Button(description="Clear")
    out_eq = widgets.Output()
    out_lock = widgets.Output()
    btn_run = widgets.Button(description="Lock & run the model",
                             button_style="success", icon="lock")

    def _equation():
        y = dd_y.value
        mets = list(sel_met.value)
        nons = list(sel_non.value)
        if y in ("(none)", None) or (not mets and not nons):
            return "(pick an outcome and at least one driver)"
        parts = mets + [f"C({n})" for n in nons]
        return f"{y} ~ " + " + ".join(parts)

    def show_equation(*args):
        with out_eq:
            out_eq.clear_output()
            nons = list(sel_non.value)
            print("The model you are about to run:")
            print("   ", _equation())
            if nons:
                print("   (", ", ".join(nons),
                      "will be expanded into 0/1 columns, one level at a "
                      "time )")

    def select_all(btn_):
        sel_met.value = tuple(m for m in met_opts if m != dd_y.value)
        sel_non.value = tuple(nonmet_opts)
        show_equation()

    def clear_all(btn_):
        sel_met.value = ()
        sel_non.value = ()
        show_equation()

    def run_model(btn_):
        with out_lock:
            out_lock.clear_output()
            y = dd_y.value
            mets = [m for m in sel_met.value if m != y]
            nons = list(sel_non.value)
            if y in ("(none)", None) or (not mets and not nons):
                print("Pick an outcome and at least one driver first.")
                return
            import statsmodels.formula.api as smf
            terms = mets + [f"C({n})" for n in nons]
            try:
                fit = smf.ols(f"{y} ~ " + " + ".join(terms),
                              data=df).fit()
            except Exception as e:
                print(f"Model failed: {e}")
                return
            state.update(model=fit, y=y,
                         drivers=mets + nons, met=mets, non=nons)
            dd_drv.options = mets
            for w in (dd_y, sel_met, sel_non, btn_all, btn_clr):
                w.disabled = True
            print("Model locked. See Tab 2 for the driver table.")
            print('Press "Unlock" there to change your picks.')
            render_results()

    btn_all.on_click(select_all)
    btn_clr.on_click(clear_all)
    btn_run.on_click(run_model)
    dd_y.observe(show_equation, names="value")
    sel_met.observe(show_equation, names="value")
    sel_non.observe(show_equation, names="value")
    select_all(None) if dv_opts else None
    show_equation()

    # ---------------- TAB 2: results ----------------
    tog = widgets.ToggleButtons(options=["Raw effects", "Rank by impact"],
                                description="View:")
    btn_unlock = widgets.Button(description="Unlock selections")
    out_res = widgets.Output()

    def render_results(*args):
        fit, y = state.get("model"), state.get("y")
        if fit is None:
            with out_res:
                out_res.clear_output()
                print("Run the model in Tab 1 first.")
            return
        ci = fit.conf_int()
        params, tvals = fit.params, fit.tvalues
        sd_y = df[y].std()
        rows_num, rows_dum = [], []
        for name in params.index:
            if name == "Intercept":
                continue
            b, lo, hi = params[name], ci.loc[name, 0], ci.loc[name, 1]
            verdict = ("TRUST IT - band stays one side of zero"
                       if lo * hi > 0 else
                       "DON'T BET ON IT - band crosses zero")
            if name.startswith("C("):
                col = name[2:name.index(")")]
                lvl = name.split("[T.")[1].rstrip("]")
                rows_dum.append([col, f"being {lvl} (vs the rest)",
                                 round(b, 3), f"{lo:.2f} to {hi:.2f}",
                                 verdict, np.nan])
            else:
                sd_x = df[name].std()
                std = b * sd_x / sd_y if sd_y else np.nan
                rows_num.append([name,
                                 f"each +1 unit of {name}",
                                 round(b, 3), f"{lo:.2f} to {hi:.2f}",
                                 verdict, round(std, 3)])
        with out_res:
            out_res.clear_output(wait=True)
            print(f"Outcome: {y}   |   drivers explain "
                  f"{100*fit.rsquared:.0f}% of the wobble\n")
            if tog.value == "Rank by impact" and rows_num:
                t = pd.DataFrame(rows_num,
                                 columns=["Driver", "Reading", "Effect",
                                          "Band (95%)", "Verdict",
                                          "Wobble-per-wobble"]).sort_values(
                    "Wobble-per-wobble",
                    key=lambda s: s.abs(), ascending=False)
                print("Numeric drivers, ranked by wobble-per-wobble "
                      "(if this driver moved by one of its own wobbles,")
                print("the outcome moves by this many of its wobbles - "
                      "unit-free, like CV on day 1):\n")
                display(t.drop(columns="Wobble-per-wobble").reset_index(
                    drop=True))
                display(t[["Driver", "Wobble-per-wobble"]].reset_index(
                    drop=True))
            else:
                if rows_num:
                    display(pd.DataFrame(
                        rows_num, columns=["Driver", "Reading", "Effect",
                                           "Band (95%)", "Verdict",
                                           "Wobble-per-wobble"]).drop(
                        columns="Wobble-per-wobble").reset_index(drop=True))
            if rows_dum:
                print("\nLabel drivers (read as 0/1 columns, raw effect):")
                display(pd.DataFrame(rows_dum,
                                     columns=["Driver", "Reading", "Effect",
                                              "Band (95%)", "Verdict",
                                              "Wobble-per-wobble"]).drop(
                    columns="Wobble-per-wobble").reset_index(drop=True))
            print("\nThe model, in plain form:")
            eq = f"{y} = {params['Intercept']:.1f}"
            for name in params.index:
                if name == "Intercept":
                    continue
                nm = name if not name.startswith("C(") else \
                    name.split("[T.")[1].rstrip("]")
                eq += f" {'+' if params[name] >= 0 else '-'} " \
                      f"{abs(params[name]):.2f}*{nm}"
            print("   ", eq)
            print("\nQuote the driver numbers, not the starting "
                  "number: the starting number is the cost of being in "
                  "business at all.")

    def unlock(btn_):
        for w in (dd_y, sel_met, sel_non, btn_all, btn_clr):
            w.disabled = False
        with out_lock:
            out_lock.clear_output()
            print("Unlocked. Adjust picks in Tab 1 and re-run.")

    tog.observe(render_results, names="value")
    btn_unlock.on_click(unlock)

    # ---------------- TAB 3: price a driver ----------------
    dd_drv = widgets.Dropdown(description="Driver:")
    ft_delta = widgets.FloatText(description="Change you could engineer "
                                            "(+/- units):")
    dd_vol = widgets.Dropdown(options=["(none)"] + dv_opts,
                              value="(none)", description="Volume column:")
    ft_per = widgets.FloatText(value=1.0, description="Periods per year:")
    btn_price = widgets.Button(description="Price it", button_style="warning",
                               icon="rupee")
    out_price = widgets.Output()

    def price(btn_):
        with out_price:
            out_price.clear_output()
            fit = state.get("model")
            if fit is None:
                print("Run the model in Tab 1 first.")
                return
            d = dd_drv.value
            if d not in fit.params.index:
                print("Pick a numeric driver from Tab 1's selection.")
                return
            delta = ft_delta.value
            if delta is None:
                print("Type the change (e.g. -10 for cutting overtime "
                      "by 10 hours).")
                return
            per_record = fit.params[d] * delta
            line = f"Each record: {d} by {delta:+.1f} -> outcome moves " \
                   f"by {per_record:+,.2f}"
            vol_txt = ""
            total = None
            if dd_vol.value != "(none)":
                vmean = df[dd_vol.value].mean()
                per = ft_per.value or 1
                total = per_record * vmean * per
                vol_txt = (f" x {dd_vol.value} (avg {vmean:,.0f}) "
                           f"x {per:.0f} periods")
            print(line)
            if total is not None:
                print(f"Per year{vol_txt}:")
                print(f"   IMPACT = {_inr(total)} per year")
            print("\nAssumption: the driver moves, everything else "
                  "stays still - which is exactly what the model "
                  "promises.")

    btn_price.on_click(price)

    # ---------------- TAB 4: observed vs predicted ----------------
    btn_fit = widgets.Button(description="Plot the fit",
                             button_style="success", icon="chart-line")
    out_fit = widgets.Output()

    def plot_fit(btn_):
        with out_fit:
            out_fit.clear_output(wait=True)
            fit, y = state.get("model"), state.get("y")
            if fit is None:
                print("Run the model in Tab 1 first.")
                return
            yh, res = fit.fittedvalues, fit.resid
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
            axes[0].scatter(yh, df[y], alpha=0.5, s=22, color="#003366")
            lo, hi = min(yh.min(), df[y].min()), max(yh.max(), df[y].max())
            axes[0].plot([lo, hi], [lo, hi], color="#DC2626", ls="--",
                         lw=1.8, label="perfect prediction line")
            axes[0].set_xlabel(f"predicted {y}")
            axes[0].set_ylabel(f"observed {y}")
            axes[0].set_title("Observed vs predicted: points ON the red "
                              "line = predicted perfectly",
                              fontweight="bold")
            axes[0].legend()
            axes[1].scatter(yh, res, alpha=0.5, s=22, color="#2563EB")
            axes[1].axhline(0, color="#DC2626", ls="--", lw=1.8)
            axes[1].set_xlabel(f"predicted {y}")
            axes[1].set_ylabel("error (observed - predicted)")
            axes[1].set_title("Errors vs predicted: the leftover wobble",
                              fontweight="bold")
            plt.tight_layout()
            plt.show()
            err = pd.DataFrame({"observed": df[y], "predicted":
                                np.round(yh, 2), "error":
                                np.round(res, 2)})
            err.index = df.index
            print("Biggest misses (where the model was most wrong):")
            display(err.reindex(err["error"].abs().sort_values(
                ascending=False).index).head(5))
            print("Prediction note: feed new driver values through the "
                  "plain-form equation in Tab 2 - that IS the predictor.")

    btn_fit.on_click(plot_fit)

    # ---------------- TAB 5: elasticity ----------------
    dd_p = widgets.Dropdown(options=dv_opts, description="Price column:")
    dd_v = widgets.Dropdown(options=[c for c in dv_opts
                                     if dv_opts and c != dv_opts[0]]
                            or dv_opts, description="Volume column:")
    btn_el = widgets.Button(description="Get the elasticity",
                            button_style="warning", icon="percent")
    out_el = widgets.Output()

    def elasticity(btn_):
        with out_el:
            out_el.clear_output(wait=True)
            p, v = dd_p.value, dd_v.value
            sub = df[[p, v]].dropna()
            sub = sub[(sub[p] > 0) & (sub[v] > 0)]
            if len(sub) < 20:
                print("Need at least 20 positive rows for this pair.")
                return
            lp, lv = np.log(sub[p]), np.log(sub[v])
            b = np.polyfit(lp, lv, 1)
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.scatter(lp, lv, alpha=0.4, s=20, color="#003366")
            xs = np.linspace(lp.min(), lp.max(), 100)
            ax.plot(xs, np.polyval(b, xs), color="#DC2626", lw=2)
            ax.set_title("The pricing-power line (log-log)",
                         fontweight="bold")
            plt.tight_layout()
            plt.show()
            print(f"ELASTICITY = {b[0]:.2f}")
            print(f"A 1% change in {p} moves {v} by {b[0]:.2f}%.")
            print("Warning: if a third variable drives both, this raw "
                  "number can lie - check Tab 2 with that variable held.")

    btn_el.on_click(elasticity)

    tab = widgets.Tab()
    tab.children = [
        out_pre,
        widgets.VBox([dd_y,
                      widgets.HBox([sel_met, sel_non]),
                      widgets.HBox([btn_all, btn_clr]),
                      out_eq, btn_run, out_lock]),
        widgets.VBox([tog, btn_unlock, out_res]),
        widgets.VBox([dd_drv, ft_delta, dd_vol, ft_per, btn_price,
                      out_price]),
        widgets.VBox([btn_fit, out_fit]),
        widgets.VBox([widgets.HBox([dd_p, dd_v]), btn_el, out_el])]
    tab.set_title(0, "0. Data Preview")
    tab.set_title(1, "1. Select Variables")
    tab.set_title(2, "2. Results")
    tab.set_title(3, "3. Price a Driver")
    tab.set_title(4, "4. Observed vs Predicted")
    tab.set_title(5, "5. Elasticity")
    display(tab)
