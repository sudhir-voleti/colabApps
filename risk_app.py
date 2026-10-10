# ============================================================================
# Session 02 - Interactive Risk & Financial Impact Engine (v3.2 Final)
# Repo : github.com/sudhir-voleti/colabApps
# Launch in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/risk_app.py").text)
#   launch_app()
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ipywidgets as widgets
from IPython.display import display, HTML


def _inr(x):
    """Format numbers cleanly into Rupees (Lakhs / Crores / Thousands)."""
    a = abs(x)
    if a >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"


def _val(w):
    """Safely fetch float values from widgets."""
    try:
        return float(w.value)
    except (TypeError, ValueError):
        return None


def launch_app():
    state = {"p": None, "word": "", "thr": None}

    print("=" * 68)
    print("  The Risk Slide Rule: Probability, Tail Risk & Financial Impact")
    print("=" * 68)
    print("Tab 1 finds the probability. Tab 2 prices it. Press UPDATE to run.\n")

    # ---------------- TAB 1: SHAPE & TAIL RISK ----------------
    dd_kind = widgets.Dropdown(
        options=["Normal (thin, symmetric)",
                 "Log-normal (right-skewed: type mean + median)",
                 "Left-skewed (low crashes: type mean + median or mean + wobble)",
                 "Student-t (fat-tailed: type mean + wobble)"],
        description="Shape:", style={"description_width": "initial"},
        layout=widgets.Layout(width="480px"))

    ft_mean = widgets.FloatText(description="Mean (average):")
    ft_sd = widgets.FloatText(description="Wobble (sigma):")
    ft_med = widgets.FloatText(description="Median (typical):")
    ft_thr = widgets.FloatText(description="Danger line:")
    
    dd_dir = widgets.Dropdown(
        options=["risk of EXCEEDING the line (overflow)",
                 "risk of FALLING BELOW the line (shortfall)"],
        description="Event:", style={"description_width": "initial"},
        layout=widgets.Layout(width="480px"))
    
    btn1 = widgets.Button(description="Update", button_style="success", icon="play")
    out1 = widgets.Output()

    def _toggle_shape(*args):
        kind = dd_kind.value
        if kind.startswith("Log-normal"):
            ft_sd.layout.display = "none"
            ft_med.layout.display = "flex"
        elif kind.startswith("Left-skewed"):
            ft_sd.layout.display = "flex"
            ft_med.layout.display = "flex"
        else:
            ft_sd.layout.display = "flex"
            ft_med.layout.display = "none"

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

            # 1. LOG-NORMAL (RIGHT SKEW)
            if dd_kind.value.startswith("Log-normal"):
                med = _val(ft_med)
                if med is None or med <= 0 or mean <= med:
                    print("Log-normal requires Mean > Median (and both > 0).")
                    print(f"Your input: Mean={mean}, Median={med}. If Mean < Median, pick 'Left-skewed'.")
                    return
                s_log = np.sqrt(2.0 * np.log(mean / med))
                s_log = min(s_log, 2.5)  # Cap for numerical plotting stability
                dist = stats.lognorm(s=s_log, scale=med)
                note = f"Right-skewed Log-normal: Solved from Mean ({mean}) & Median ({med})."

            # 2. LEFT-SKEWED (SKEW-NORMAL)
            elif dd_kind.value.startswith("Left-skewed"):
                med = _val(ft_med)
                sd = _val(ft_sd) or 10.0
                if med is not None and med > mean:
                    # Parameterized from Mean & Median
                    alpha = -6.0
                    delta = alpha / np.sqrt(1 + alpha**2)
                    scale_sn = (med - mean) / (np.sqrt(2/np.pi) * (1 - delta))
                    scale_sn = max(scale_sn, 0.1)
                    loc_sn = mean - scale_sn * delta * np.sqrt(2/np.pi)
                    dist = stats.skewnorm(a=alpha, loc=loc_sn, scale=scale_sn)
                    note = f"Left-skewed: Solved from Mean ({mean}) & Median ({med})."
                else:
                    # Fallback to Mean & SD with negative skew alpha
                    alpha = -5.0
                    dist = stats.skewnorm(a=alpha, loc=mean, scale=sd)
                    note = f"Left-skewed: Skew-Normal with Mean={mean}, Wobble={sd}."

            # 3. STUDENT-T (FAT TAILS)
            elif dd_kind.value.startswith("Student-t"):
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                df = 3
                dist = stats.t(df=df, loc=mean, scale=sd * np.sqrt((df - 2.0) / df))
                note = "Student-t (df=3): Fat-tailed shape with heavy extreme event risks."

            # 4. NORMAL
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

            # Plotting x-bounds calculation
            x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
            x_lo = min(x_lo, thr - abs(mean - thr))
            x_hi = max(x_hi, thr + abs(mean - thr))
            xs = np.linspace(x_lo, x_hi, 600)

            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.plot(xs, dist.pdf(xs), color="#003366", lw=2.2)
            mask = xs >= thr if above else xs <= thr
            ax.fill_between(xs[mask], dist.pdf(xs[mask]), color="#DC2626", alpha=0.45)
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8, label=f"Danger line = {thr:.1f}")
            
            m_val, med_val = dist.mean(), dist.median()
            ax.axvline(m_val, color="#059669", ls="-", lw=1.6, label=f"mean = {m_val:.1f}")
            if abs(med_val - m_val) > 0.02 * (abs(m_val) if m_val != 0 else 1):
                ax.axvline(med_val, color="#2563EB", ls=":", lw=1.8, label=f"median = {med_val:.1f}")

            ax.set_title(f"{dd_kind.value.split(' (')[0]} | {word} {thr:.1f} shaded", fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

            pct = p_tail * 100
            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;"
                f"border-radius:8px;padding:14px;font-size:16px'>"
                f"P(metric {word} {thr:.1f}) = <b>{pct:.2f}%</b>"
                f"<br><span style='font-size:14px'>about {pct:.0f} operating periods in every 100</span></div>"))
            if note:
                print("Note: " + note)
            print("Carried to Tab 2. Price it there when ready.")

    # ---------------- TAB 2: PRICE THE RISK ----------------
    ft_cost = widgets.FloatText(description="Cost per bad event (Rs):")
    ft_marg = widgets.FloatText(description="Margin lost per event (Rs, 0 if none):")
    ft_per = widgets.FloatText(value=30.0, description="Periods per month (days/nights):")
    btn2 = widgets.Button(description="Price it", button_style="warning", icon="rupee")
    out2 = widgets.Output()

    def render2(btn_=None):
        with out2:
            out2.clear_output(wait=True)
            if state["p"] is None:
                print("Run Tab 1 first (press Update) - the probability comes from there.")
                return
            cost, marg, per = _val(ft_cost), _val(ft_marg), _val(ft_per)
            if cost is None or per is None or per <= 0:
                print("Fill in cost per event and periods per month (a positive number).")
                return
            marg = marg or 0.0
            price = cost + marg
            exp_p = state["p"] * price
            
            display(HTML(
                f"<div style='background:#fef3c7;border:2px solid #D97706;"
                f"border-radius:8px;padding:14px;font-size:16px'>"
                f"P = {state['p']*100:.2f}% | Price per event = {_inr(price)}<br>"
                f"EXPECTED LOSS per period = P x price = <b>{_inr(exp_p)}</b><br>"
                f"<span style='font-size:14px'>Monthly Expected Loss ({per:.0f} periods): <b>{_inr(exp_p*per)}</b> &nbsp;|&nbsp; "
                f"Annual Expected Loss ({per*12:.0f} periods): <b>{_inr(exp_p*per*12)}</b></span></div>"))
            print("Assumption: one bad event per period crossing the line.")

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
        widgets.VBox([widgets.HTML("<i>The probability carries over from Tab 1.</i>"),
                      ft_cost, ft_marg, ft_per, btn2, out2])
    ]
    tab.set_title(0, "1. Shape & Tail Risk")
    tab.set_title(1, "2. Price the Risk")
    display(tab)
