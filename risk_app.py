# ============================================================================
# Session 02 - Distribution & Tail Risk Simulator (Interactive Colab App)
# Repo : github.com/sudhir-voleti/colabApps
# Launch code in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/dist_risk_app.py").text)
#   launch_app()
# Fixes in this revision:
#   - Student-t branch is variance-matched (realised wobble = your sigma),
#     so "same average, same wobble" is honest when flipping shapes
#   - continuous_update=False on sliders (no replot spam while dragging)
#   - threshold slider range extended for monster-line demos
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display, clear_output


def launch_app():
    print("=" * 68)
    print("  Session 02: Interactive Distribution & Tail Risk Simulator")
    print("=" * 68)
    print("Simulate operational risk, capacity overflows, and fat-tail "
          "probabilities.\n")

    w_dist = widgets.Dropdown(
        options=[
            ("Normal (Symmetric / Bell Curve)", "normal"),
            ("Log-Normal (Right-Skewed / High Outlier Pull)", "lognormal"),
            ("Student-t (Fat-Tailed / Black Swan Prone)", "student_t"),
        ],
        value="normal", description="Distribution:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"))

    w_mean = widgets.FloatSlider(value=40.0, min=10.0, max=100.0, step=1.0,
                                 description="Mean (baseline):",
                                 style={"description_width": "initial"},
                                 layout=widgets.Layout(width="400px"),
                                 continuous_update=False)
    w_sigma = widgets.FloatSlider(value=8.0, min=1.0, max=25.0, step=0.5,
                                  description="Wobble (std dev):",
                                  style={"description_width": "initial"},
                                  layout=widgets.Layout(width="400px"),
                                  continuous_update=False)
    w_threshold = widgets.FloatSlider(value=50.0, min=10.0, max=150.0,
                                      step=1.0,
                                      description="Capacity / limit cutoff:",
                                      style={"description_width": "initial"},
                                      layout=widgets.Layout(width="400px"),
                                      continuous_update=False)
    w_tail_dir = widgets.Dropdown(
        options=[("Exceeds limit (> threshold)", "upper"),
                 ("Falls below limit (< threshold)", "lower")],
        value="upper", description="Risk event:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"))

    out_plot = widgets.Output()

    def update_sim(change=None):
        with out_plot:
            clear_output(wait=True)
            dist_type = w_dist.value
            mean_val = w_mean.value
            sigma_val = w_sigma.value
            thresh_val = w_threshold.value
            tail_dir = w_tail_dir.value

            if dist_type == "normal":
                x = np.linspace(mean_val - 4.5 * sigma_val,
                                mean_val + 4.5 * sigma_val, 1000)
                pdf = stats.norm.pdf(x, loc=mean_val, scale=sigma_val)
                median_val = mean_val
                dist = stats.norm(loc=mean_val, scale=sigma_val)

            elif dist_type == "lognormal":
                phi = np.sqrt(sigma_val ** 2 + mean_val ** 2)
                s_log = np.sqrt(np.log(phi ** 2 / mean_val ** 2))
                scale_log = (mean_val ** 2) / phi
                x = np.linspace(0.1, mean_val + 5.0 * sigma_val, 1000)
                pdf = stats.lognorm.pdf(x, s=s_log, scale=scale_log)
                median_val = stats.lognorm.median(s=s_log, scale=scale_log)
                dist = stats.lognorm(s=s_log, scale=scale_log)

            else:  # student_t, variance-matched so wobble stays honest
                df_deg = 3
                scale_t = sigma_val * np.sqrt((df_deg - 2.0) / df_deg)
                x = np.linspace(mean_val - 5.0 * sigma_val,
                                mean_val + 5.0 * sigma_val, 1000)
                pdf = stats.t.pdf(x, df=df_deg, loc=mean_val, scale=scale_t)
                median_val = mean_val
                dist = stats.t(df=df_deg, loc=mean_val, scale=scale_t)

            prob_risk = (dist.sf(thresh_val) if tail_dir == "upper"
                         else dist.cdf(thresh_val))

            print("=" * 68)
            print(" EXECUTIVE TAIL RISK ASSESSMENT")
            print("=" * 68)
            print(f"  Distribution baseline (mean)  : {mean_val:.2f}")
            print(f"  Typical day (median)          : {median_val:.2f}")
            print(f"  Operational wobble (std dev)  : {sigma_val:.2f}")
            print(f"  Capacity / limit cutoff       : {thresh_val:.2f}")
            print("-" * 68)
            pct_risk = prob_risk * 100.0
            if tail_dir == "upper":
                print(f"  RISK PROBABILITY: {pct_risk:.2f}% chance demand "
                      f"EXCEEDS {thresh_val:.1f}.")
                print(f"     -> out of 100 nights, roughly "
                      f"{pct_risk:.0f} overflow events.")
            else:
                print(f"  DEFICIT PROBABILITY: {pct_risk:.2f}% chance the "
                      f"metric FALLS BELOW {thresh_val:.1f}.")
                print(f"     -> out of 100 periods, roughly "
                      f"{pct_risk:.0f} shortage events.")
            print("=" * 68 + "\n")

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            ax.plot(x, pdf, color="#003366", lw=2.5,
                    label=f"Distribution ({dist_type.replace('_', '-')})")
            mask = (x >= thresh_val) if tail_dir == "upper" else (x <= thresh_val)
            ax.fill_between(x[mask], pdf[mask], color="#DC2626", alpha=0.5,
                            label=f"Risk zone ({pct_risk:.1f}%)")
            ax.axvline(mean_val, color="#D97706", linestyle="--", linewidth=2,
                       label=f"mean = {mean_val:.1f}")
            ax.axvline(median_val, color="#059669", linestyle=":", linewidth=2,
                       label=f"median = {median_val:.1f}")
            ax.axvline(thresh_val, color="#1E293B", linestyle="-",
                       linewidth=2.5, label=f"cutoff = {thresh_val:.1f}")
            ax.set_title(f"Distribution & Risk Boundary ({dist_type.upper()})",
                         fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Metric scale (beds demanded, billing Rs, DSO days)")
            ax.set_ylabel("Probability density")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

    for w in (w_dist, w_mean, w_sigma, w_threshold, w_tail_dir):
        w.observe(update_sim, names="value")

    display(widgets.VBox([w_dist,
                          widgets.HBox([w_mean, w_sigma]),
                          widgets.HBox([w_threshold, w_tail_dir])]))
    display(out_plot)
    update_sim()
