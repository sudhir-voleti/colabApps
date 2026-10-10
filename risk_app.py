# ============================================================================
# Risk Slide Rule (Distribution Playground) -- STANDARD TEMPLATE
# Repo : github.com/sudhir-voleti/colabApps
# Launch:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/risk_app.py").text)
#   launch_app()
#
# No data needed - this app turns PARAMETERS into a picture and a probability.
# Enter the average and wobble of any business metric, draw your danger
# line, and read off the chance of crossing it. Flip the distribution and
# watch the risk change while the average stays put.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ipywidgets as widgets
from IPython.display import display, HTML


def _make_dist(mean, sigma, kind, skew, df):
    if kind.startswith("Normal"):
        return stats.norm(loc=mean, scale=sigma), ""
    if kind.startswith("Skewed"):
        return stats.skewnorm(a=skew, loc=mean, scale=sigma), \
            "with skew on, the realised mean drifts from your average - " \
            "watch the two lines split (that gap is yesterday's lesson)"
    scale = sigma * np.sqrt((df - 2.0) / df)   # keep realised sd = sigma
    return stats.t(df=df, loc=mean, scale=scale), \
        f"t with {df:.0f} degrees of freedom - lower df, fatter tails"


def launch_app():
    ft_mean = widgets.FloatText(value=40.0, description="Average (mean):")
    ft_sd = widgets.FloatText(value=18.0, description="Wobble (sigma):")
    ft_thr = widgets.FloatText(value=50.0, description="Danger line:")
    dd_dir = widgets.Dropdown(
        options=["above the line (overflow risk)",
                 "below the line (shortfall risk)"],
        description="Risk of:")
    dd_kind = widgets.Dropdown(
        options=["Normal (thin, symmetric)",
                 "Skewed (drag the skew)",
                 "Fat-tailed (drag the tails)"],
        description="Shape:")
    sl_skew = widgets.FloatSlider(value=0.0, min=-4.0, max=4.0, step=0.5,
                                  description="Skew:",
                                  continuous_update=False)
    sl_df = widgets.IntSlider(value=5, min=3, max=30,
                              description="Tail heaviness (df, low = fat):",
                              style={"description_width": "initial"},
                              continuous_update=False)
    out_plot = widgets.Output()
    out_read = widgets.Output()

    print("=" * 64)
    print("  The Risk Slide Rule: average + wobble + shape = probability")
    print("=" * 64)
    display(widgets.VBox([widgets.HBox([ft_mean, ft_sd]),
                          widgets.HBox([ft_thr, dd_dir]),
                          dd_kind, sl_skew, sl_df]))
    display(out_read)
    display(out_plot)

    def _toggle(*args):
        sl_skew.layout.display = ("flex"
                                  if dd_kind.value.startswith("Skewed")
                                  else "none")
        sl_df.layout.display = ("flex"
                                if dd_kind.value.startswith("Fat")
                                else "none")

    def render(*args):
        mean, sigma = ft_mean.value, ft_sd.value
        thr = ft_thr.value
        above = dd_dir.value.startswith("above")
        dist, note = _make_dist(mean, sigma, dd_kind.value,
                                sl_skew.value, sl_df.value)
        p_tail = dist.sf(thr) if above else dist.cdf(thr)
        x_lo, x_hi = dist.ppf(0.001), dist.ppf(0.999)
        x_lo = min(x_lo, thr - sigma)
        x_hi = max(x_hi, thr + sigma)
        xs = np.linspace(x_lo, x_hi, 600)

        with out_plot:
            out_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            ax.plot(xs, dist.pdf(xs), color="#003366", lw=2)
            mask = xs >= thr if above else xs <= thr
            ax.fill_between(xs[mask], dist.pdf(xs[mask]),
                            color="#DC2626", alpha=0.45)
            ax.axvline(thr, color="#DC2626", ls="--", lw=1.8)
            real_mean = dist.mean()
            ax.axvline(real_mean, color="#059669", ls="-", lw=1.6,
                       label=f"mean = {real_mean:.1f}")
            med = dist.median()
            if abs(med - real_mean) > 0.3:
                ax.axvline(med, color="#2563EB", ls=":", lw=1.8,
                           label=f"median = {med:.1f}")
            ax.set_title("Same average. Same wobble. Different risk.",
                         fontweight="bold")
            ax.legend()
            plt.tight_layout()
            plt.show()

        cover = {}
        m, s = dist.mean(), dist.std()
        for k, nom in [(1, 68), (2, 95), (3, 99.7)]:
            cover[k] = (dist.cdf(m + k * s) - dist.cdf(m - k * s)) * 100
        days = ""
        if 0 < p_tail < 1:
            per = p_tail * 100
            days = (f"About {per:.0f}% - roughly "
                    f"{per/100*30:.0f} nights in a month, "
                    f"{per/100*365:.0f} days in a year.")
        with out_read:
            out_read.clear_output(wait=True)
            word = "exceeds" if above else "falls below"
            display(HTML(
                f"<div style='background:#fee2e2;border:2px solid #DC2626;"
                f"border-radius:8px;padding:14px;font-size:17px'>"
                f"P(demand {word} {thr:.0f}) = <b>{p_tail*100:.1f}%</b>"
                f"<br><span style='font-size:14px'>{days}</span></div>"))
            print(f"Coverage: within +/-1 wobble = {cover[1]:.0f}% "
                  f"(normal promises 68) | +/-2 = {cover[2]:.0f}% "
                  f"(normal promises 95) | +/-3 = {cover[3]:.1f}% "
                  f"(normal promises 99.7)")
            if note:
                print("Note: " + note)

    for w in (ft_mean, ft_sd, ft_thr, dd_dir, dd_kind, sl_skew, sl_df):
        w.observe(render, names="value")
    _toggle()
    render()
