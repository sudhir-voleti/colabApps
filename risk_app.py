# ============================================================================
# Session 02 - Modular Dual-Risk Slide Rule & P&L Synthesis Engine (v4.0 Final)
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
    # Global state passed between tabs
    state = {
        "p_above": None,
        "p_below": None,
        "mean_overflow": 0.0,
        "mean_idle": 0.0,
        "loss_deficit_per_period": 0.0,
        "loss_idle_per_period": 0.0,
        "unit_margin": 0.0,
        "unit_penalty": 0.0,
        "unit_overhead": 0.0,
        "periods_per_month": 30.0,
        "thr": None
    }

    print("=" * 72)
    print("  The Risk Slide Rule: Generic Probability, Dual-Risk & P&L Engine")
    print("=" * 72)

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
    ft_thr = widgets.FloatText(description="Cutoff Limit:")
    
    dd_dir = widgets.Dropdown(
        options=["Primary Risk: EXCEEDING line (Overflow / Stockout)",
                 "Primary Risk: FALLING BELOW line (Shortfall / SLA Drop)"],
        description="Focus Event:", style={"description_width": "initial"},
        layout=widgets.Layout(width="480px"))
    
    btn1 = widgets.Button(description="Update Probability Model", button_style="success", icon="play")
    out1 = widgets.Output()

    # ---------------- TAB 2 WIDGETS ----------------
    ft_margin = widgets.FloatText(value=0.0, description="Lost Margin / Unit (Rs):", style={"description_width": "initial"})
    ft_penalty = widgets.FloatText(value=0.0, description="Penalty / Breach Cost / Unit (Rs):", style={"description_width": "initial"})
    btn2 = widgets.Button(description="Calculate Deficit Impact", button_style="warning", icon="rupee")
    out2 = widgets.Output()

    # ---------------- TAB 3 WIDGETS ----------------
    ft_overhead = widgets.FloatText(value=0.0, description="Fixed Overhead / Idle Cost / Unit (Rs):", style={"description_width": "initial"})
    btn3 = widgets.Button(description="Calculate Idle Impact", button_style="warning", icon="rupee")
    out3 = widgets.Output()

    # ---------------- TAB 4 WIDGETS ----------------
    ft_per = widgets.FloatText(value=30.0, description="Periods per month (days/nights):", style={"description_width": "initial"})
    btn4 = widgets.Button(description="Synthesize Executive P&L", button_style="danger", icon="calculator")
    out4 = widgets.Output()

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
                print("Fill in the Mean and the Cutoff Limit line first.")
                return

            above = dd_dir.value.startswith("Primary Risk: EXCEEDING")
            note = ""
            is_left_skew = False
            upper_bound = None

            # 1. LOG-NORMAL (AUTOMATIC SKEW DETECT)
            if dd_kind.value.startswith("Log-normal"):
                med = _val(ft_med)
                if med is None or med <= 0 or mean <= 0 or mean == med:
                    print("Log-normal requires both Mean and Median to be positive non-equal numbers.")
                    return
                
                if mean > med:
                    s_log = np.sqrt(2.0 * np.log(mean / med))
                    s_log = min(s_log, 2.5)
                    dist = stats.lognorm(s=s_log, scale=med)
                    note = f"Right-skewed Log-normal (Mean > Median): solved from Mean ({mean}) & Median ({med})."
                else:
                    is_left_skew = True
                    gap = med - mean
                    upper_bound = med + 3.0 * gap
                    mean_trans = upper_bound - mean
                    med_trans = upper_bound - med
                    s_log = np.sqrt(2.0 * np.log(mean_trans / med_trans))
                    s_log = min(s_log, 2.5)
                    dist = stats.lognorm(s=s_log, scale=med_trans)
                    note = f"Left-skewed Log-normal (Mean < Median): solved from Mean ({mean}) & Median ({med})."

            # 2. STUDENT-T
            elif dd_kind.value.startswith("Student-t"):
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                df = 3
                dist = stats.t(df=df, loc=mean, scale=sd * np.sqrt((df - 2.0) / df))
                note = "Student-t (df=3): Fat-tailed distribution with heavy extreme event risks."

            # 3. NORMAL
            else:
                sd = _val(ft_sd)
                if sd is None or sd <= 0:
                    print("Wobble (sigma) must be a positive number.")
                    return
                dist = stats.norm(loc=mean, scale=sd)

            # Probabilities
            if is_left_skew:
                thr_trans = upper_bound - thr
                p_above = dist.cdf(thr_trans)
                p_below = 1.0 - p_above
            else:
                p_above = dist.sf(thr)
                p_below = dist.cdf(thr)

            # Conditional Unit Gap Integration
            x_lo, x_hi = dist.ppf(0.0001), dist.ppf(0.9999)
            if is_left_skew:
                xs_grid = np.linspace(x_lo, x_hi, 2000)
                real_xs = upper_bound - xs_grid
                pdf_vals = dist.pdf(xs_grid)
                
                overflow_mask = real_xs > thr
                idle_mask = real_xs < thr
                
                mean_overflow = np.sum((real_xs[overflow_mask] - thr) * pdf_vals[overflow_mask]) / np.sum(pdf_vals) if np.sum(pdf_vals[overflow_mask]) > 0 else 0
                mean_idle = np.sum((thr - real_xs[idle_mask]) * pdf_vals[idle_mask]) / np.sum(pdf_vals) if np.sum(pdf_vals[idle_mask]) > 0 else 0
            else:
                xs_grid = np.linspace(x_lo, x_hi, 2000)
                pdf_vals = dist.pdf(xs_grid)
                
                overflow_mask = xs_grid > thr
                idle_mask = xs_grid < thr
                
                mean_overflow = np.sum((xs_grid[overflow_mask] - thr) * pdf_vals[overflow_mask]) / np.sum(pdf_vals) if np.sum(pdf_vals[overflow_mask]) > 0 else 0
                mean_idle = np.sum((thr - xs_grid[idle_mask]) * pdf_vals[idle_mask]) / np.sum(pdf_vals) if np.sum(pdf_vals[idle_mask]) > 0 else 0

            # Pass variables to global state
            state["p_above"] = float(p_above)
            state["p_below"] = float(p_below)
            state["mean_overflow"] = float(mean_overflow)
            state["mean_idle"] = float(mean_idle)
            state["thr"] = thr

            # Plotting
            if is_left_skew:
                x_lo_t, x_hi_t = dist.ppf(0.001), dist.ppf(0.999)
                x_plot_lo, x_plot_hi = upper_bound - x_hi_t, upper_bound - x_lo_t
            else:
                x_plot_lo, x_plot_hi = dist.ppf(0.001), dist.ppf(0.999)
            
            x_plot_lo = min(x_plot_lo, thr - abs(mean - thr))
            x_plot_hi = max(x_plot_hi, thr + abs(mean - thr))
            xs = np.linspace(x_plot_lo, x_plot_hi, 600)

            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            pdf_plot = dist.pdf(upper_bound - xs) if is_left_skew else dist.pdf(xs)

            ax.plot(xs, pdf_plot, color="#003366", lw=2.2)
            ax.fill_between(xs[xs >= thr], pdf_plot[xs >= thr], color="#DC2626", alpha=0.45, label=f"Overflow Area ({p_above*100:.1f}%)")
            ax.fill_between(xs[xs < thr], pdf_plot[xs < thr], color="#2563EB", alpha=0.15, label=f"Underutilization Area ({p_below*100:.1f}%)")
            
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8, label=f"Cutoff = {thr:.1f}")
            m_val = mean
            med_val = _val(ft_med) if dd_kind.value.startswith("Log-normal") else dist.median()
            
            ax.axvline(m_val, color="#059669", ls="-", lw=1.6, label=f"mean = {m_val:.1f}")
            if abs(med_val - m_val) > 0.01 * (abs(m_val) if m_val != 0 else 1):
                ax.axvline(med_val, color="#D97706", ls=":", lw=1.8, label=f"median = {med_val:.1f}")

            ax.set_title(f"{dd_kind.value.split(' (')[0]} | Dual-Tail Coexistence at Cutoff = {thr:.1f}", fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

            display(HTML(
                f"<div style='background:#f1f5f9;border-left:5px solid #003366;border-radius:6px;padding:12px;font-size:15px'>"
                f"• <b>P(Exceeding Cutoff > {thr:.1f}):</b> <b style='color:#dc2626'>{p_above*100:.2f}%</b> "
                f"<i>(Avg Deficit = {mean_overflow:.2f} units/period)</i><br>"
                f"• <b>P(Falling Below Cutoff < {thr:.1f}):</b> <b style='color:#2563eb'>{p_below*100:.2f}%</b> "
                f"<i>(Avg Unused = {mean_idle:.2f} units/period)</i></div>"
            ))
            if note:
                print("Note: " + note)
            print("Carried to Tabs 2, 3 & 4. Proceed when ready.")

        # Clear downstream tabs until executed
        with out2:
            out2.clear_output(wait=True)
            print("Probability model updated in Tab 1. Enter unit deficit economics above and click 'Calculate Deficit Impact'.")
        with out3:
            out3.clear_output(wait=True)
            print("Probability model updated in Tab 1. Enter idle overhead economics above and click 'Calculate Idle Impact'.")
        with out4:
            out4.clear_output(wait=True)
            print("Probability model updated in Tab 1. Click 'Synthesize Executive P&L' to generate master summary.")

    def render2(btn_=None):
        with out2:
            out2.clear_output(wait=True)
            if state["p_above"] is None:
                print("⚠️ Run Tab 1 first (press 'Update Probability Model') — probabilities originate from Tab 1.")
                return

            margin = _val(ft_margin) or 0.0
            penalty = _val(ft_penalty) or 0.0
            rate = margin + penalty
            loss = state["p_above"] * state["mean_overflow"] * rate
            
            state["unit_margin"] = margin
            state["unit_penalty"] = penalty
            state["loss_deficit_per_period"] = loss

            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;border-radius:8px;padding:14px;font-size:15px;color:#7f1d1d'>"
                f"<b>DEFICIT OPPORTUNITY LOSS ENGINE:</b><br>"
                f"• P(Overflow) = {state['p_above']*100:.1f}% &nbsp;|&nbsp; Avg Deficit = {state['mean_overflow']:.2f} units<br>"
                f"• Combined Unit Deficit Rate = {_inr(rate)} <i>(Margin: {_inr(margin)} + Penalty: {_inr(penalty)})</i><br><hr style='border-top:1px dashed #fca5a5;margin:8px 0'>"
                f"<b>EXPECTED DEFICIT LOSS PER PERIOD:</b> <b style='font-size:17px;color:#dc2626'>{_inr(loss)}</b>"
                f"</div>"
            ))

    def render3(btn_=None):
        with out3:
            out3.clear_output(wait=True)
            if state["p_below"] is None:
                print("⚠️ Run Tab 1 first (press 'Update Probability Model') — probabilities originate from Tab 1.")
                return

            overhead = _val(ft_overhead) or 0.0
            loss = state["p_below"] * state["mean_idle"] * overhead
            
            state["unit_overhead"] = overhead
            state["loss_idle_per_period"] = loss

            display(HTML(
                f"<div style='background:#eff6ff;border:2px solid #2563EB;border-radius:8px;padding:14px;font-size:15px;color:#1e3a8a'>"
                f"<b>IDLE CAPACITY OVERHEAD ENGINE:</b><br>"
                f"• P(Underutilization) = {state['p_below']*100:.1f}% &nbsp;|&nbsp; Avg Unused = {state['mean_idle']:.2f} units<br>"
                f"• Fixed Overhead Rate = {_inr(overhead)} / idle unit<br><hr style='border-top:1px dashed #93c5fd;margin:8px 0'>"
                f"<b>EXPECTED IDLE OVERHEAD LOSS PER PERIOD:</b> <b style='font-size:17px;color:#2563eb'>{_inr(loss)}</b>"
                f"</div>"
            ))

    def render4(btn_=None):
        with out4:
            out4.clear_output(wait=True)
            if state["p_above"] is None:
                print("⚠️ Run Tab 1 first (press 'Update Probability Model') — probabilities originate from Tab 1.")
                return

            per = _val(ft_per)
            if per is None or per <= 0:
                print("⚠️ Periods per month must be a positive number.")
                return
            
            state["periods_per_month"] = per

            d_period = state["loss_deficit_per_period"]
            i_period = state["loss_idle_per_period"]
            total_period = d_period + i_period

            display(HTML(
                f"<div style='background:#fef3c7;border:2px solid #D97706;border-radius:8px;padding:16px;font-size:15px;color:#78350f'>"
                f"<h4 style='margin:0 0 10px 0;color:#92400E;'><b>EXECUTIVE P&L SYNTHESIS: NET OPERATIONAL EXPOSURE</b></h4>"
                f"<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
                f"<tr style='border-bottom:1px solid #d97706;text-align:left;'>"
                f"  <th style='padding:5px;'>P&L Component</th><th style='padding:5px;'>Per Operating Period</th><th style='padding:5px;'>Monthly ({per:.0f} Periods)</th><th style='padding:5px;'>Annual ({per*12:.0f} Periods)</th>"
                f"</tr>"
                f"<tr>"
                f"  <td style='padding:6px;'><b>1. Deficit Loss (Opportunity Cost + Penalty)</b></td><td>{_inr(d_period)}</td><td>{_inr(d_period*per)}</td><td><b style='color:#dc2626'>{_inr(d_period*per*12)}</b></td>"
                f"</tr>"
                f"<tr>"
                f"  <td style='padding:6px;'><b>2. Idle Overhead Loss (Unabsorbed Fixed Cost)</b></td><td>{_inr(i_period)}</td><td>{_inr(i_period*per)}</td><td><b style='color:#2563eb'>{_inr(i_period*per*12)}</b></td>"
                f"</tr>"
                f"<tr style='border-top:2px solid #d97706;background:#fde68a;font-weight:bold;'>"
                f"  <td style='padding:6px;'>NET EXPECTED OPERATIONAL LOSS</td><td>{_inr(total_period)}</td><td>{_inr(total_period*per)}</td><td><b style='color:#92400e;font-size:16px'>{_inr(total_period*per*12)}</b></td>"
                f"</tr>"
                f"</table><br>"
                f"<span style='font-size:13px;color:#451a03'>Boardroom Strategy Note: Adjust cutoff threshold $K$ in Tab 1 to find the optimal operational sweet spot that minimizes Net Expected Operational Loss.</span>"
                f"</div>"
            ))

    dd_kind.observe(_toggle_shape, names="value")
    btn1.on_click(render1)
    btn2.on_click(render2)
    btn3.on_click(render3)
    btn4.on_click(render4)
    _toggle_shape()

    # Initial load output messages (No pre-rendered charts)
    with out1:
        print("Set parameters above and click 'Update Probability Model' to run Tab 1.")
    with out2:
        print("Run Tab 1 first, then fill in deficit unit rates above and click 'Calculate Deficit Impact'.")
    with out3:
        print("Run Tab 1 first, then fill in idle overhead rate above and click 'Calculate Idle Impact'.")
    with out4:
        print("Run Tab 1 first, then click 'Synthesize Executive P&L' to generate master summary.")

    tab = widgets.Tab()
    tab.children = [
        widgets.VBox([dd_kind,
                      widgets.HBox([ft_mean, ft_sd]),
                      ft_med,
                      widgets.HBox([ft_thr, dd_dir]),
                      btn1, out1]),
        widgets.VBox([widgets.HTML("<b style='color:#003366;'>Deficit Risk Economics:</b><br>"
                                   "<i>Evaluates lost contribution margins and out-of-pocket breach penalties when demand exceeds cutoff.</i><br><br>"),
                      ft_margin, ft_penalty, btn2, out2]),
        widgets.VBox([widgets.HTML("<b style='color:#003366;'>Idle Capacity Economics:</b><br>"
                                   "<i>Evaluates unabsorbed fixed overhead and holding costs when operations run below cutoff.</i><br><br>"),
                      ft_overhead, btn3, out3]),
        widgets.VBox([widgets.HTML("<b style='color:#003366;'>Executive Master Summary:</b><br>"
                                   "<i>Synthesizes Deficit Opportunity Loss and Idle Capacity Overhead into a single P&L statement.</i><br><br>"),
                      ft_per, btn4, out4])
    ]
    tab.set_title(0, "1. 📊 Shape & Tail Risk")
    tab.set_title(1, "2. 💸 Deficit Risk")
    tab.set_title(2, "3. 🛋️ Idle Capacity Risk")
    tab.set_title(3, "4. 📈 Executive P&L Summary")
    
    display(tab)
