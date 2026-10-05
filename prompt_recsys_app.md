# Prompt: Build the Market-Basket Recommendation App (Colab / ipywidgets)

**What this file is:** a complete, plain-language specification ("prompt") that
an AI coding assistant can follow to recreate `recsys_app.py` from scratch —
a market-basket recommendation app that runs in a Google Colab notebook. It
is written in first person and describes every behavior in terms of what is
visible on screen.

**How to use it:** paste the whole block below into your AI workspace
(section by section if needed), or keep it in the repo as documentation of
what the app does and why.

---

## THE PROMPT (paste everything below)

Build me a single self-contained Python file, `recsys_app.py`, that runs a
**market-basket recommendation application** inside a **Google Colab
notebook** using **ipywidgets** for the interface. The file must define one
function, `launch_app()`, which builds and displays the whole interface. I
will launch it from a Colab cell with:

    import requests
    URL = ("https://raw.githubusercontent.com/<USER>/<REPO>/main/"
           "recsys_app.py?v=1")
    exec(requests.get(URL).text)
    launch_app()

### Who will use it and how it must feel
I will give this tool to people who do not code. Every step must be a visible
button, dropdown, checkbox, or slider with plain-word labels. After every
action I want a one-line plain-English status message. If I press a button
out of order, tell me what to do first ("Load data first."). Use two plain
dictionaries as app state — one for data, one for widgets. Everything the
engine does must be explainable from counts a person could verify by hand:
co-occurrence counts, shares, and lift — no black-box model.

### The engine (compute all of this once, after loading)
From the line-item CSV, build one list of baskets, where each basket is the
list of distinct product codes it contains, **in the order the lines appear
in the file** (the order matters later). Then build a square item × item
table of: **co-occurrence counts** (baskets containing both), **P(B|A)**
(confidence), **support** (share of baskets containing the pair), and
**lift** = P(A&B) / (P(A)·P(B)), which corrects for popularity — the single
most important statistic in the tool.

### Overall layout
A panel with a title, a **tab bar of five tabs**, and a status line at the
bottom. The tabs are:
**1 · Data** → **2 · Single item** → **3 · Complete basket** →
**4 · Evaluate** → **5 · AI copy**.

### Tab 1 · Data
- A file-upload widget accepting `.csv` and a **Load CSV** button. The CSV
  must have columns `basket_id`, `product_code`, `product_name`, `category`;
  if any is missing, say exactly which one.
- On load, print: number of baskets, products, and line items; then show
  (a) the items-per-basket distribution table, (b) the top-10 category
  pairs table (e.g. "Immunity Boosters + Daily Nutrition: 312 baskets"),
  and (c) the first 8 rows. Items (a) and (b) are the material for the
  caselet's "basket anatomy" question — label them as such.

### Tab 2 · Single item
- A product dropdown showing "code — name", a **Top N** slider (5-20,
  default 10), and a **Get recommendations** button.
- Show a table of the N partners with the highest lift: item code, name,
  category, **co-occurrence count**, **P(B|A)**, **support**, **lift** —
  sortable via the `itables` library (pip-install it inside the app if
  missing; fall back to plain display on any failure). Default view sorted
  by lift; I can click the co-occurrence header to compare the two
  rankings — the divergence between them is the lesson ("popularity is not
  personalization").

### Tab 3 · Complete basket
- A multi-select list of products ("in-basket items"), default pre-select
  nothing, and a **Complete my basket** button.
- Score every product NOT in the basket by the **sum of its lift to each
  in-basket item**, considering only products that co-occur with at least
  one of them. Show the top 5: suggestion, name, category, "bought with N
  of your items", summed lift, and minimum support across the pairs. This
  is the real-time "what do I show this shopper" moment.

### Tab 4 · Evaluate
- An **Run evaluation** button that does a leave-one-out test: for every
  basket with at least 2 items, hold out the **last-added** item (last in
  file order — that is why order was preserved), pretend the rest is the
  shopper's cart, produce recommendations the same way as Tab 3, and check
  whether the held-out item appears in the top 1, top 3, and top 5.
- Report a small table: for k = 1, 3, 5 — engine hit-rate and, side by
  side, the **popularity baseline** hit-rate (always recommending the k
  best-selling products). One sentence on the tab: the GAP between engine
  and baseline is the reportable number; the raw hit-rate alone is not.
  (On synthetic, mission-coherent data the engine will look near-perfect —
  that is exactly why the baseline comparison exists.)

### Tab 5 · AI copy
- A large text box and a **Refresh copy block** button that assembles
  everything computed so far — basket anatomy (Q1), the single-item table
  (Q2), the basket completions (Q3), and the evaluation summary (Q4) — into
  one plain-text block with clear Q1-Q4 section markers, ready to select
  all, copy, and paste into an AI prompt. Sections not yet run say so
  instead of appearing empty.

### Conventions and why
- Deterministic everywhere; display rounding only; no randomness.
- No network calls inside the app — the only network access is the initial
  `requests.get` in the Colab cell.
- Display figures/tables with `display()` only — never pair it with
  `plt.show()` or Colab renders everything twice.
- Keep the design decisions as-is: lift (not raw counts) as the primary
  ranking, the last-added item as the hold-out, the popularity baseline as
  the comparison, and counts-based math with no black-box model — the
  explainability IS the pedagogy.
