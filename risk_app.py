# ============================================================================
# Session 02 - Interactive Risk & Financial Impact Engine (v3.4 Final)
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

    # ---------------- TAB 1 WIDGETS ----------------
    dd_kind = widgets.Dropdown(
        options=["Normal (thin, symmetric)",
                 "Log-normal (skewed: type mean + median)",
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
    
    btn1 = widgets.Button(description="Update Analysis", button_style="success", icon="play")
    out1 = widgets.Output()

    # ---------------- TAB 2 WIDGETS ----------------
    ft_cost = widgets.FloatText(description="Cost per bad event (Rs):")
    ft_marg = widgets.FloatText(description="Margin lost per event (Rs, 0 if none):")
    ft_per = widgets.FloatText(value=30.0, description="Periods per month (days/nights):")
    btn2 = widgets.Button(description="Price it", button_style="warning", icon="rupee")
    out2 = widgets.Output()

    def _toggle_shape(*args):
        kind = dd_kind.value
        if kind.startswith("Log-normal"):
            ft_sd.layout.display = "none"
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
            is_left_skew = False

            # 1. LOG-NORMAL (AUTOMATIC LEFT OR RIGHT SKEW FROM MEAN VS MEDIAN)
            if dd_kind.value.startswith("Log-normal"):
                med = _val(ft_med)
                if med is None or med <= 0 or mean <= 0 or mean == med:
                    print("Log-normal requires both Mean and Median to be positive non-equal numbers.")
                    return
                
                if mean > med:
                    # Standard Right Skew
                    s_log = np.sqrt(2.0 * np.log(mean / med))
                    s_log = min(s_log, 2.5)
                    dist = stats.lognorm(s=s_log, scale=med)
                    note = f"Right-skewed Log-normal (Mean > Median): solved from Mean ({mean}) & Median ({med})."
                else:
                    # Left Skew via inverted Log-normal
                    is_left_skew = True
                    gap = med - mean
                    upper_bound = med + 3.0 * gap
                    
                    mean_trans = upper_bound - mean
                    med_trans = upper_bound - med
                    
                    s_log = np.sqrt(2.0 * np.log(mean_trans / med_trans))
                    s_log = min(s_log, 2.5)
                    
                    # Inverted distribution mapping
                    base_dist = stats.lognorm(s=s_log, scale=med_trans)
                    dist = base_dist
                    note = f"Left-skewed Log-normal (Mean < Median): solved from Mean ({mean}) & Median ({med})."

            # 2. STUDENT-T (FAT TAILS)
            elif dd_kind.value.startswith("Student-t"):
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                df = 3
                dist = stats.t(df=df, loc=mean, scale=sd * np.sqrt((df - 2.0) / df))
                note = "Student-t (df=3): Fat-tailed shape with heavy extreme event risks."

            # 3. NORMAL
            else:
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                dist = stats.norm(loc=mean, scale=sd)

            # Tail risk calculation
            if is_left_skew:
                # Map threshold to inverted space
                thr_trans = upper_bound - thr
                p_tail = dist.cdf(thr_trans) if above else dist.sf(thr_trans)
            else:
                p_tail = dist.sf(thr) if above else dist.cdf(thr)

            state["p"] = float(p_tail)
            state["word"] = word
            state["thr"] = thr

            # Plotting x-bounds calculation
            if is_left_skew:
                x_lo_t, x_hi_t = dist.ppf(0.001), dist.ppf(0.999)
                x_lo, x_hi = upper_bound - x_hi_t, upper_bound - x_lo_t
            else:
                x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
            
            x_lo = min(x_lo, thr - abs(mean - thr))
            x_hi = max(x_hi, thr + abs(mean - thr))
            xs = np.linspace(x_lo, x_hi, 600)

            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            
            if is_left_skew:
                pdf_vals = dist.pdf(upper_bound - xs)
                mask = xs >= thr if above else xs <= thr
            else:
                pdf_vals = dist.pdf(xs)
                mask = xs >= thr if above else xs <= thr

            ax.plot(xs, pdf_vals, color="#003366", lw=2.2)
            ax.fill_between(xs[mask], pdf_vals[mask], color="#DC2626", alpha=0.45)
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8, label=f"Danger line = {thr:.1f}")
            
            if is_left_skew:
                m_val, med_val = mean, med
            else:
                m_val, med_val = dist.mean(), dist.median()

            ax.axvline(m_val, color="#059669", ls="-", lw=1.6, label=f"mean = {m_val:.1f}")
            if abs(med_val - m_val) > 0.01 * (abs(m_val) if m_val != 0 else 1):
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
            print("Carried to Tab 2. Click 'Price it' in Tab 2 when ready.")

        # Clear Tab 2 output until user clicks 'Price it'
        with out2:
            out2.clear_output(wait=True)
            print("Probability updated in Tab 1. Enter cost numbers above and click 'Price it'.")

    def render2(btn_=None):
        with out2:
            out2.clear_output(wait=True)
            if state["p"] is None:
                print("⚠️ Run Tab 1 first (press 'Update Analysis') — the probability comes from Tab 1.")
                return
            cost, marg, per = _val(ft_cost), _val(ft_marg), _val(ft_per)
            if cost is None or per is None or per <= 0:
                print("⚠️ Fill in cost per event and periods per month (a positive number).")
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

    # Initial load output messages (No pre-rendered charts)
    with out1:
        print("Set parameters above and click 'Update Analysis' to run Tab 1.")
    with out2:
        print("Run Tab 1 first, then fill in cost numbers above and click 'Price it'.")

    tab = widgets.Tab()
    tab.children = [
        widgets.VBox([dd_kind,
                      widgets.HBox([ft_mean, ft_sd]),
                      ft_med,
                      widgets.HBox([ft_thr, dd_dir]),
                      btn1, out1]),
        widgets.VBox([widgets.HTML("<b style='color:#003366;'>Financial Impact Inputs:</b><br>"
                                   "<i>The tail probability carries over automatically from Tab 1.</i><br><br>"),
                      ft_cost, ft_marg, ft_per, btn2, out2])
    ]
    tab.set_title(0, "1. Shape & Tail Risk")
    tab.set_title(1, "2. Price the Risk")
    display(tab)
