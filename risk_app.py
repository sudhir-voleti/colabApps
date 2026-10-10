# ============================================================================
# Session 02 - General Risk Slide Rule & Financial Impact Engine (v2.0)
# Repo : github.com/sudhir-voleti/colabApps
# Launch code in Colab:
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
        return f"₹{x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"


def _make_dist(mean, sigma, kind):
    """Construct parametric distributions (Normal, Log-normal, Student-t)."""
    if kind.startswith("Normal"):
        return stats.norm(loc=mean, scale=sigma), ""
    if kind.startswith("Log-normal"):
        phi = np.sqrt(sigma ** 2 + mean ** 2)
        s_log = np.sqrt(np.log(phi ** 2 / mean ** 2))
        scale_log = (mean ** 2) / phi
        return stats.lognorm(s=s_log, scale=scale_log), \
            "Log-Normal: Right-skewed distribution. Mean vs. Median gap reflects outlier pull."
    df = 3
    scale_t = sigma * np.sqrt((df - 2.0) / df)
    return stats.t(df=df, loc=mean, scale=scale_t), \
        "Student-t (df=3): Fat-tailed distribution. Heavy tails simulate extreme Black Swan events."


def launch_app():
    # --- TAB 1 CONTROLS: DISTRIBUTION & TAIL RISK ---
    ft_mean = widgets.FloatText(value=40.0, description="Baseline Mean (μ):", style={"description_width": "initial"})
    ft_sd = widgets.FloatText(value=15.0, description="Wobble (sigma):", style={"description_width": "initial"})
    ft_thr = widgets.FloatText(value=50.0, description="Threshold Limit:", style={"description_width": "initial"})
    
    dd_dir = widgets.Dropdown(
        options=["Risk of EXCEEDING Threshold (Overflow / Capacity Breach)",
                 "Risk of FALLING BELOW Threshold (Deficit / Stockout)"],
        value="Risk of EXCEEDING Threshold (Overflow / Capacity Breach)",
        description="Risk Event:", style={"description_width": "initial"}, layout=widgets.Layout(width="480px")
    )
    
    dd_kind = widgets.Dropdown(
        options=["Normal (Symmetric / Bell Curve)",
                 "Log-normal (Right-Skewed / High Outliers)",
                 "Student-t (Fat-Tailed / Black Swan Prone)"],
        value="Normal (Symmetric / Bell Curve)",
        description="Curve Shape:", style={"description_width": "initial"}, layout=widgets.Layout(width="480px")
    )

    # --- TAB 2 CONTROLS: FINANCIAL ECONOMICS & PROFIT IMPACT ---
    ft_rev = widgets.FloatText(value=180000.0, description="Revenue per Unit/Event (₹):", style={"description_width": "initial"})
    ft_margin_pct = widgets.FloatText(value=65.0, description="Contribution Margin (%):", style={"description_width": "initial"})
    ft_fixed_penalty = widgets.FloatText(value=0.0, description="Fixed Penalty / Cost per Breach (₹):", style={"description_width": "initial"})

    btn = widgets.Button(description="Update Analysis & Charts", button_style="success", icon="refresh", layout=widgets.Layout(width="280px"))
    
    out_read_t1 = widgets.Output()
    out_plot_t1 = widgets.Output()
    out_read_t2 = widgets.Output()
    out_plot_t2 = widgets.Output()

    print("=" * 72)
    print("  General Risk Slide Rule: Probability, Tail Risk & Financial Impact Engine")
    print("=" * 72)

    def render(btn_=None):
        mean, sigma = ft_mean.value, ft_sd.value
        thr = ft_thr.value
        above = dd_dir.value.startswith("Risk of EXCEEDING")
        dist, note = _make_dist(mean, sigma, dd_kind.value)
        p_tail = dist.sf(thr) if above else dist.cdf(thr)
        pct = p_tail * 100.0

        # --- FINANCIAL CALCULATIONS ---
        rev_per_unit = ft_rev.value
        margin_pct = ft_margin_pct.value / 100.0
        profit_per_unit = rev_per_unit * margin_pct
        fixed_penalty = ft_fixed_penalty.value
        total_loss_per_event = profit_per_unit + fixed_penalty
        
        exp_loss_per_period = p_tail * total_loss_per_event

        # Plotting Setup
        x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
        x_lo = min(x_lo, thr - sigma)
        x_hi = max(x_hi, thr + sigma)
        xs = np.linspace(x_lo, x_hi, 600)

        # ---------------- RENDER TAB 1: TAIL RISK PROBABILITY ----------------
        with out_read_t1:
            out_read_t1.clear_output(wait=True)
            word = "exceeds" if above else "falls below"
            display(HTML(
                f"<div style='background:#f1f5f9;border-left:5px solid #003366;border-radius:4px;padding:14px;font-size:15px;color:#0f172a'>"
                f"<b>TAIL RISK PROBABILITY:</b> P(Metric {word} {thr:.1f}) = <b style='color:#dc2626;font-size:18px'>{pct:.2f}%</b><br>"
                f"<span style='font-size:13px;color:#475569'>Out of 100 operating periods, expect approximately <b>{pct:.1f} breach/overflow events</b>.</span>"
                f"</div>"
            ))
            if note:
                print(f"\nNote: {note}")

        with out_plot_t1:
            out_plot_t1.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            ax.plot(xs, dist.pdf(xs), color="#003366", lw=2.2, label=f"Distribution ({dd_kind.value.split(' ')[0]})")
            mask = xs >= thr if above else xs <= thr
            ax.fill_between(xs[mask], dist.pdf(xs[mask]), color="#DC2626", alpha=0.45, label=f"Tail Risk Area ({pct:.1f}%)")
            ax.axvline(thr, color="#DC2626", ls="--", lw=2, label=f"Threshold = {thr:.1f}")
            
            m, med = dist.mean(), dist.median()
            ax.axvline(m, color="#D97706", ls="-", lw=1.6, label=f"Mean = {m:.1f}")
            if abs(med - m) > 0.1 * sigma:
                ax.axvline(med, color="#059669", ls=":", lw=1.8, label=f"Median = {med:.1f}")
                
            word_title = "Overflow Zone (> Threshold)" if above else "Deficit Zone (< Threshold)"
            ax.set_title(f"Distribution & Risk Boundary Analysis — {word_title}", fontweight="bold", fontsize=11)
            ax.set_xlabel("Metric Units")
            ax.set_ylabel("Probability Density")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

        # ---------------- RENDER TAB 2: FINANCIAL IMPACT ----------------
        with out_read_t2:
            out_read_t2.clear_output(wait=True)
            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;border-radius:8px;padding:16px;color:#7f1d1d'>"
                f"<h4 style='margin:0 0 8px 0;color:#991b1b'><b>EXPECTED FINANCIAL & PROFIT LOSS (E[Impact])</b></h4>"
                f"• <b>Lost Profit per Breach Unit:</b> {_inr(profit_per_unit)} <i>(Revenue: {_inr(rev_per_unit)} × Margin: {margin_pct*100:.0f}%)</i><br>"
                f"• <b>Fixed Penalty / Cost per Breach:</b> {_inr(fixed_penalty)}<br>"
                f"• <b>Total Cost per Bad Event:</b> <b style='font-size:16px'>{_inr(total_loss_per_event)}</b><br><hr style='border-top:1px dashed #fca5a5;margin:10px 0'>"
                f"<b>EXPECTED LOSS PER OPERATING PERIOD:</b> <b style='font-size:18px;color:#dc2626'>{_inr(exp_loss_per_period)}</b><br>"
                f"<span style='font-size:13.5px;color:#450a0a'>"
                f"<b>Monthly Expected Loss (30 days):</b> {_inr(exp_loss_per_period * 30)} &nbsp;|&nbsp; "
                f"<b>Annual Expected Loss (365 days):</b> {_inr(exp_loss_per_period * 365)}"
                f"</span></div>"
            ))

        with out_plot_t2:
            out_plot_t2.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8.5, 3.5))
            periods = ["Per Period / Day", "Monthly (30 Days)", "Annual (365 Days)"]
            losses = [exp_loss_per_period, exp_loss_per_period * 30, exp_loss_per_period * 365]
            
            bars = ax.bar(periods, losses, color=["#f87171", "#ef4444", "#dc2626"], width=0.45)
            ax.set_ylabel("Expected Bottom-Line Loss (₹)")
            ax.set_title("Expected Financial Impact Across Operating Horizons", fontweight="bold", fontsize=11)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + (0.02 * max(losses)), _inr(yval), ha='center', va='bottom', fontweight='bold', fontsize=9.5)
                
            ax.set_ylim(0, max(losses) * 1.2)
            ax.grid(axis='y', ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

    btn.on_click(render)

    # --- ASSEMBLE 2-TAB UI LAYOUT ---
    tab_ui = widgets.Tab()
    
    ui_tab1 = widgets.VBox([
        widgets.HBox([ft_mean, ft_sd]),
        widgets.HBox([ft_thr, dd_dir]),
        dd_kind,
        btn,
        out_read_t1,
        out_plot_t1
    ])
    
    ui_tab2 = widgets.VBox([
        widgets.HTML("<b>Enter Unit Economics & Profitability Numbers:</b>"),
        widgets.HBox([ft_rev, ft_margin_pct]),
        ft_fixed_penalty,
        btn,
        out_read_t2,
        out_plot_t2
    ])

    tab_ui.children = [ui_tab1, ui_tab2]
    tab_ui.set_title(0, "📊 Tail Risk Probability")
    tab_ui.set_title(1, "💰 Financial Impact & Profit Loss")

    display(tab_ui)
    render()
