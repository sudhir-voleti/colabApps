# ============================================================================
# Risk Slide Rule (Distribution Playground, typed-input version)
# Repo : github.com/sudhir-voleti/colabApps
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/risk_app.py").text)
#   launch_app()
#
# Type your metric's average and wobble, draw the danger line, price the
# risk in rupees. Shapes: Normal / Log-normal (right-skew) / Student-t
# (fat tails, variance-matched so "same wobble" stays honest).
# Expected loss = P(crossing the line) x (cost + margin lost) per event.
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


def _make_dist(mean, sigma, kind):
    if kind.startswith("Normal"):
        return stats.norm(loc=mean, scale=sigma), ""
    if kind.startswith("Log-normal"):
        phi = np.sqrt(sigma ** 2 + mean ** 2)
        s_log = np.sqrt(np.log(phi ** 2 / mean ** 2))
        scale_log = (mean ** 2) / phi
        return stats.lognorm(s=s_log, scale=scale_log), \
            "log-normal: mean/median split on the plot is the skew lesson"
    df = 3
    scale_t = sigma * np.sqrt((df - 2.0) / df)
    return stats.t(df=df, loc=mean, scale=scale_t), \
        "t (df=3), wobble-matched: lower df would be fatter still"


def launch_app():
    ft_mean = widgets.FloatText(value=40.0, description="Average (mean):")
    ft_sd = widgets.FloatText(value=18.0, description="Wobble (sigma):")
    ft_thr = widgets.FloatText(value=50.0, description="Danger line:")
    dd_dir = widgets.Dropdown(
        options=["risk of EXCEEDING the line (overflow)",
                 "risk of FALLING BELOW the line (shortfall)"],
        description="Event:", style={"description_width": "initial"})
    dd_kind = widgets.Dropdown(
        options=["Normal (thin, symmetric)",
                 "Log-normal (right-skewed)",
                 "Student-t (fat-tailed)"],
        description="Shape:", style={"description_width": "initial"})
    ft_cost = widgets.FloatText(value=35000.0,
                                description="Cost per bad event (Rs):")
    ft_margin = widgets.FloatText(
        value=15000.0,
        description="Margin lost per event (Rs, 0 if none):")
    btn = widgets.Button(description="Update", button_style="success",
                         icon="refresh")
    out_plot = widgets.Output()
    out_read = widgets.Output()

    print("=" * 68)
    print("  The Risk Slide Rule: average + wobble + shape = risk, in rupees")
    print("=" * 68)
    display(widgets.VBox([
        widgets.HBox([ft_mean, ft_sd]),
        widgets.HBox([ft_thr, dd_dir]),
        dd_kind,
        widgets.HBox([ft_cost, ft_margin]),
        btn]))
    display(out_read)
    display(out_plot)

    def render(btn_=None):
        mean, sigma = ft_mean.value, ft_sd.value
        thr = ft_thr.value
        above = dd_dir.value.startswith("risk of EXCEEDING")
        dist, note = _make_dist(mean, sigma, dd_kind.value)
        p_tail = dist.sf(thr) if above else dist.cdf(thr)
        cost_per = ft_cost.value + ft_margin.value
        exp_loss = p_tail * cost_per

        x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
        x_lo = min(x_lo, thr - sigma)
        x_hi = max(x_hi, thr + sigma)
        xs = np.linspace(x_lo, x_hi, 600)

        with out_plot:
            out_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            ax.plot(xs, dist.pdf(xs), color="#003366", lw=2.2)
            mask = xs >= thr if above else xs <= thr
            ax.fill_between(xs[mask], dist.pdf(xs[mask]), color="#DC2626",
                            alpha=0.45)
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8)
            m, med = dist.mean(), dist.median()
            ax.axvline(m, color="#059669", ls="-", lw=1.6,
                       label=f"mean = {m:.1f}")
            if abs(med - m) > 0.3 * sigma / max(sigma, 1):
                ax.axvline(med, color="#2563EB", ls=":", lw=1.8,
                           label=f"median = {med:.1f}")
            word = "overflow" if above else "shortfall"
            ax.set_title(f"{dd_kind.value.split(' ')[0]} shape | "
                         f"{word} zone shaded", fontweight="bold")
            ax.legend()
            ax.grid(True, ls=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

        pct = p_tail * 100
        with out_read:
            out_read.clear_output(wait=True)
            word = "exceeds" if above else "falls below"
            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;"
                f"border-radius:8px;padding:14px;font-size:16px'>"
                f"P(demand {word} {thr:.0f}) = <b>{pct:.1f}%</b>"
                f" &nbsp;|&nbsp; expected loss per period = "
                f"P x price = <b>{_inr(exp_loss)}</b><br>"
                f"<span style='font-size:14px'>per month (30 periods): "
                f"{_inr(exp_loss*30)} &nbsp;|&nbsp; per year (365): "
                f"{_inr(exp_loss*365)}</span></div>"))
            print(f"Assumption: one bad event per period that crosses the "
                  f"line; price per event = cost {_inr(ft_cost.value)} + "
                  f"margin {_inr(ft_margin.value)}.")
            if note:
                print("Note: " + note)

    btn.on_click(render)
    render()
