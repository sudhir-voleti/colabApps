# Prompt: Build the RFM Segmentation App (Colab / ipywidgets)

**What this file is:** a complete, plain-language specification ("prompt") that
an AI coding assistant can follow to recreate `rfm_app.py` from scratch — a
generic RFM (Recency-Frequency-Monetary) analysis app that runs in a Google
Colab notebook. It is written in first person and describes every behavior in
terms of what is visible on screen.

**How to use it:** paste the whole block below into your AI workspace
(section by section if needed), or keep it in the repo as documentation of
what the app does and why.

---

## THE PROMPT (paste everything below)

Build me a single self-contained Python file, `rfm_app.py`, that runs a
generic **RFM segmentation application** inside a **Google Colab notebook**
using **ipywidgets** for the interface and **matplotlib** for plots. The file
must define one function, `launch_app()`, which builds and displays the whole
interface. I will launch it from a Colab cell with:

    import requests
    URL = ("https://raw.githubusercontent.com/<USER>/<REPO>/main/"
           "rfm_app.py?v=1")
    exec(requests.get(URL).text)
    launch_app()

### Who will use it and how it must feel
I will give this tool to people who do not code. Every step must be a visible
button, dropdown, checkbox, slider, or text field with plain-word labels, and
nothing may require keyboard shortcuts. After every action I want a one-line
plain-English status message ("Data prepared."). If I press a button out of
order, tell me what to do first ("Prepare data first (Data tab)."). Use two
plain dictionaries as app state — one for data, one for widgets — so the
pieces can share what they need.

### Overall layout
A panel with a title, a **tab bar of six tabs**, and a status line at the
bottom. The tabs are:
**1 · Data** → **2 · RFM** → **3 · Plots** → **4 · Segments** →
**5 · Dynamic** → **6 · Triggers**.

### Tab 1 · Data
- A file-upload widget accepting `.csv` and a **Load CSV** button.
- Four column-mapping dropdowns — **Customer ID**, **Transaction/Invoice
  ID**, **Transaction date**, **Amount** — plus a toggle "Revenue from:
  Amount column | Quantity × Price" that swaps the Amount dropdown for
  Quantity and Unit-price dropdowns. Pre-select sensible columns by name
  matching when the CSV loads (e.g. names containing customer/cust, invoice/
  txn/order, date, revenue/amount, quantity, price).
- A text field **Date format (optional)**, pre-filled with `%m/%d/%Y %H:%M`;
  if I leave it or it fails to parse, fall back to automatic date inference.
- A checkbox, default ON: "Drop rows with amount <= 0 (cancellations/
  returns)".
- A **Prepare RFM data** button that: parses dates, computes each line's
  amount (selected column, or quantity × unit price), drops non-positive
  rows if the box is ticked, then **aggregates to one row per (customer,
  invoice)** — invoice date = latest line date, invoice amount = sum of its
  lines. Print: number of invoices, number of customers, date range, total
  revenue. Also populate the two snapshot-date dropdowns in Tab 5 with the
  month-end dates found in the data.

### Tab 2 · RFM
- A slider **RFM bins (3-8)**, default 4, and a **Calculate RFM** button.
- Scoring, per customer, as of a snapshot date (default: the latest date in
  the data): **Recency** = days between the snapshot and the customer's last
  purchase; **Frequency** = count of distinct invoices; **Monetary** = total
  spend. Reference date shown explicitly.
- Score each of R, F, M into 1..bins using **equal-frequency binning by
  rank** (rank customers on the value, split into equal-sized groups, break
  ties arbitrarily). This guarantees every score exists even when many
  customers share the same value (e.g. after a threshold filter) — absolute
  quartile edges collapse in that situation, ranks do not. Recency is
  reversed: the most recent customers get the highest score. Concatenate the
  three digits into an `rfm_score` string like "444" or "132".
- Show the per-customer table (customer, recency days, invoice count, spend,
  R, F, M, rfm_score) rendered with the `itables` library — clickable
  sortable headers, search box, pagination (like R's DT::datatable). The app
  must pip-install `itables` itself if missing and silently fall back to
  plain pandas display if anything fails.
- Below the table, print the **absolute quartile cut-points** of each raw
  column as reference, and add a NOTE next to any column whose distinct
  absolute quartiles number fewer than the requested bins — that signals
  tied values (often a filter artifact) and explains why rank-based binning
  was used.
- A **Download RFM scores CSV** button.

### Tab 3 · Plots
A dropdown with five choices and a **Render plot** button:
1. RFM histograms (three panels: recency days, invoice count, spend);
2. Recency vs Monetary scatter;
3. Frequency vs Monetary scatter;
4. R × F heat map of mean Monetary (annotated cells);
5. Top-15 RFM cells by customer count (horizontal bar).

### Tab 4 · Segments
- A segment-sizes table: rfm_score, number of customers, percentage.
- When bins = 4, also show this fixed segment → action mapping, filtered to
  cells that actually occur in the data:

  | rfm_score | Segment | Recommended action |
  | --- | --- | --- |
  | 444 | Best Customers | Loyalty rewards, VIP benefits, early access, referral perks |
  | 241 | Price-Sensitive Regulars | Bundles, cross-sell, personalized promos to lift basket size |
  | 413 | Occasional High-Value | Personalized follow-ups, premium recommendations, loyalty incentives |
  | 132 | At-Risk Customers | Win-back campaign, reminders, limited-time offers before churn |
  | 324 | High-Value Growth | Recommendations + loyalty perks to raise purchase frequency |
  | 111 | Lost / Low-Value | Low-cost reactivation only; suppress expensive campaigns |

  When bins ≠ 4, say the mapping applies at bins = 4.
- A large copy-paste text box with the sizes table and the mapping in plain
  text, labeled for pasting into an AI interpretation prompt.

### Tab 5 · Dynamic
- Two dropdowns, **Snapshot A (from)** and **Snapshot B (to)** (month-ends,
  A must be earlier — guard it), and a **Compute migration** button.
- Compute the full RFM scoring twice: once counting only transactions on or
  before A, once before B (same bins). Join on customer and build a
  **migration matrix**: rows = rfm_score at A, columns = rfm_score at B,
  cells = number of customers who made that move. Everyone present at A is
  also present at B (scores are cumulative), so nobody disappears — quiet
  customers slide into low-recency cells; say so on the tab.
- Print: customers at A, how many are observed at B, and how many stayed in
  the same segment (count and %). Matrix rendered sortable via itables.
- A copy-paste text box with the matrix for an AI exercise.

### Tab 6 · Triggers
- A rule-type toggle: **Segment → segment** (two dropdowns of observed
  rfm_score cells) or **Score drop** (which score: R+F+M total, R only, F
  only, M only; and "dropped ≥ N points", slider 1-9).
- **Run query** → show the count and a sortable table of the matching
  customers (customer, scores at A, scores at B, per-dimension deltas), and
  offer a **Download audience CSV** button — this CSV is the audience file a
  campaign tool would consume.
- A text field **Intervention** (e.g. "send 5% discount coupon") and an
  **Add rule to campaign plan** button that appends the rule to an
  accumulating plan table (trigger | customers | intervention), shown
  sortably, plus a plain-text copy box asking an AI to critique each rule.

### Conventions and why
- Fully deterministic: no randomness anywhere; ranks break ties by first
  occurrence.
- Round for display only, never for computation.
- No network calls inside the app — the only network access is the initial
  `requests.get` in the Colab cell.
- Keep the design decisions above as-is: rank-based equal-frequency binning
  (absolute quartiles collapse under tied values from filtered data), the
  six-cell action mapping shown only at bins = 4, cumulative snapshots
  (nobody leaves the matrix; they slide), and itables with a plain-display
  fallback.
