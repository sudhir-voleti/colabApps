# ============================================================================
# Risk Slide Rule v3 -- STANDARD TEMPLATE (two tabs, button-gated)
# Repo : github.com/sudhir-voleti/colabApps
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/risk_app.py").text)
#   launch_app()
#
# TAB 1 - Shape & Tail Risk: type your numbers, press UPDATE.
#   Normal / Student-t : Mean + Wobble(sigma)
#   Log-normal         : Mean + Median  (the two dashboard numbers;
#                        the app back-solves the shape from their gap)
# TAB 2 - Price the Risk: cost + margin + periods -> expected loss in Rs.
# Nothing renders until a button is pressed.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ipywidgets as widgets
from IPython.display import display, HTML


def _inr(x):
    a = abs(x)
    if a >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"


def _val(w):
    """FloatText -> float, or None if left empty."""
    try:
        return float(w.value)
    except (TypeError, ValueError):
        return None


def launch_app():
    state = {"p": None, "word": "", "thr": None}

    print("=" * 68)
    print("  The Risk Slide Rule: shape first, then the price of risk")
    print("=" * 68)
    print("Tab 1 finds the probability. Tab 2 prices it. "
          "Press UPDATE / PRICE IT to run.\n")

    # ---------------- TAB 1: shape & tail risk ----------------
    dd_kind = widgets.Dropdown(
        options=["Normal (thin, symmetric)",
                 "Log-normal (right-skewed: type mean + median)",
                 "Student-t (fat-tailed)"],
        description="Shape:", style={"description_width": "initial"})
    ft_mean = widgets.FloatText(description="Mean (average):")
    ft_sd = widgets.FloatText(description="Wobble (sigma):")
    ft_med = widgets.FloatText(description="Median (typical):")
    ft_thr = widgets.FloatText(description="Danger line:")
    dd_dir = widgets.Dropdown(
        options=["risk of EXCEEDING the line (overflow)",
                 "risk of FALLING BELOW the line (shortfall)"],
        description="Event:", style={"description_width": "initial"})
    btn1 = widgets.Button(description="Update", button_style="success",
                          icon="play")
    out1 = widgets.Output()

    def _toggle_shape(*args):
        log = dd_kind.value.startswith("Log-normal")
        ft_sd.layout.display = "none" if log else "flex"
        ft_med.layout.display = "flex" if log else "none"

    def render1(btn_=None):
        with out1:
            out1.clear_output(wait=True)
            mean = _val(ft_mean)
            thr = _val(ft_thr)
            if mean is None or thr is None:
                print("Fill in the Mean and the Danger line first.")
                return
            above = dd_dir.value.startswith("risk of EXCEEDING")
            word = "exceeds" if above else "falls below"
            note = ""

            if dd_kind.value.startswith("Log-normal"):
                med = _val(ft_med)
                if med is None:
                    print("Log-normal needs the Median too - "
                          "the two numbers off your dashboard.")
                    return
                if med <= 0 or mean <= med:
                    print("Log-normal draws only RIGHT skew: the mean must "
                          "sit ABOVE the median (and both above 0).")
                    print("Your numbers say mean <= median - that is left "
                          "skew or a typo. Pick Normal, or re-check.")
                    return
                s_log = np.sqrt(2.0 * np.log(mean / med))
                dist = stats.lognorm(s=s_log, scale=med)
                note = (f"back-solved from your two numbers: median anchors "
                        f"the curve, the gap implies wobble "
                        f"{_inr(dist.std()) if dist.std() >= 1e5 else round(dist.std(), 2)}")
            elif dd_kind.value.startswith("Student-t"):
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                df = 3
                dist = stats.t(df=df, loc=mean,
                               scale=sd * np.sqrt((df - 2.0) / df))
                note = "t (df=3), wobble-matched to what you typed"
            else:
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                dist = stats.norm(loc=mean, scale=sd)

            p_tail = dist.sf(thr) if above else dist.cdf(thr)
            state["p"] = float(p_tail)
            state["word"] = word
            state["thr"] = thr

            x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
            x_lo = min(x_lo, thr - (dist.std() or 1))
            x_hi = max(x_hi, thr + (dist.std() or 1))
            xs = np.linspace(x_lo, x_hi, 600)

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            ax.plot(xs, dist.pdf(xs), color="#003366", lw=2.2)
            mask = xs >= thr if above else xs <= thr
            ax.fill_between(xs[mask], dist.pdf(xs[mask]), color="#DC2626",
                            alpha=0.45)
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8)
            m, med = dist.mean(), dist.median()
            ax.axvline(m, color="#059669", ls="-", lw=1.6,
                       label=f"mean = {m:.1f}")
            if abs(med - m) > 0.05 * max(dist.std(), 1e-9):
                ax.axvline(med, color="#2563EB", ls=":", lw=1.8,
                           label=f"median = {med:.1f}")
            ax.set_title(f"{dd_kind.value.split(' (')[0]} | "
                         f"{word} {thr:.1f} shaded", fontweight="bold")
            ax.legend()
            ax.grid(True, ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

            pct = p_tail * 100
            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;"
                f"border-radius:8px;padding:14px;font-size:17px'>"
                f"P(demand {word} {thr:.0f}) = <b>{pct:.2f}%</b>"
                f"<br><span style='font-size:14px'>about {pct:.0f} periods "
                f"in every 100</span></div>"))
            if note:
                print("Note: " + note)
            print("Carried to Tab 2. Price it there when ready.")

    # ---------------- TAB 2: price the risk ----------------
    ft_cost = widgets.FloatText(description="Cost per bad event (Rs):")
    ft_marg = widgets.FloatText(
        description="Margin lost per event (Rs, 0 if none):")
    ft_per = widgets.FloatText(
        description="Periods per month (nights/days/orders):")
    btn2 = widgets.Button(description="Price it", button_style="warning",
                          icon="rupee")
    out2 = widgets.Output()

    def render2(btn_=None):
        with out2:
            out2.clear_output(wait=True)
            if state["p"] is None:
                print("Run Tab 1 first (press Update) - the probability "
                      "comes from there.")
                return
            cost, marg, per = _val(ft_cost), _val(ft_marg), _val(ft_per)
            if cost is None or per is None or per <= 0:
                print("Fill in cost per event and periods per month "
                      "(a positive number).")
                return
            marg = marg or 0.0
            price = cost + marg
            exp_p = state["p"] * price
            display(HTML(
                f"<div style='background:#fef3c7;border:2px solid #D97706;"
                f"border-radius:8px;padding:14px;font-size:16px'>"
                f"P = {state['p']*100:.2f}% | price per event = "
                f"{_inr(price)}<br>"
                f"EXPECTED LOSS per period = P x price = "
                f"<b>{_inr(exp_p)}</b><br>"
                f"<span style='font-size:14px'>per month: "
                f"{_inr(exp_p*per)} &nbsp;|&nbsp; per year: "
                f"{_inr(exp_p*per*12)}</span></div>"))
            print("Assumption: one bad event per period crossing the line.")
            print("Change the cost and press again - that re-pricing "
                  "conversation IS the CFO meeting.")

    dd_kind.observe(_toggle_shape, names="value")
    btn1.on_click(render1)
    btn2.on_click(render2)
    _toggle_shape()

    tab = widgets.Tab()
    tab.children = [
        widgets.VBox([dd_kind,
                      widgets.HBox([ft_mean, ft_sd]),
                      ft_med,
                      widgets.HBox([ft_thr, dd_dir]),
                      btn1, out1]),
        widgets.VBox([widgets.HTML("<i>The probability carries over from "
                                   "Tab 1.</i>"),
                      ft_cost, ft_marg, ft_per, btn2, out2])]
    tab.set_title(0, "1. Shape & Tail Risk")
    tab.set_title(1, "2. Price the Risk")
    display(tab)
