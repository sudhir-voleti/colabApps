# Prompt: Build the PCPDM Segmentation App (Colab / ipywidgets)

**What this file is:** a complete, plain-language specification ("prompt") that
an AI coding assistant can follow to recreate `segmentation_app.py` from
scratch — the app used in the PCPDM module's VedaFlow caselet. It is written
for a non-specialist reader: every behavior is described in terms of what the
user sees and what the app must do, not in code jargon.

**How to use it:** paste the whole block below into your AI workspace
(preferably in sections if the tool has input limits), or keep it in the
repo as documentation of what the app does and why.

---

## THE PROMPT (paste everything below)

Build a single self-contained Python file, `segmentation_app.py`, that runs a
generic K-Means market-segmentation application inside a **Google Colab
notebook** using **ipywidgets** for the user interface and **matplotlib** for
plots. The file must define one function, `launch_app()`, which builds and
displays the whole interface. Students launch it from a Colab cell with:

    import requests
    URL = ("https://raw.githubusercontent.com/<USER>/<REPO>/main/"
           "segmentation_app.py?v=1")
    exec(requests.get(URL).text)
    launch_app()

### Audience and design philosophy
The users are executive-education students with no coding background. Every
step must be a visible button, dropdown, checkbox, or slider with plain-word
labels. Nothing should require modifier-key tricks. After every action, show
a one-line status message in plain English ("Data prepared. Next: show the
scree plot and pick K."). Guard every action: if a button is pressed out of
order, show a friendly message saying what to do first ("Prepare data first
(Data tab).").

### Overall layout
A panel with a title, a **tab bar of six tabs**, and a status line at the
bottom. The tabs are:
**1 · Data** → **2 · Scree & K** → **3 · Centroids** → **4 · Biplot** →
**5 · Simplified** → **6 · Assignments**.

Use two plain dictionaries as app state, one for data and one for widgets,
so handlers can share what they need. Re-fit nothing silently: every tab
shows exactly what has been computed, and says so if nothing is there yet.

### Tab 1 · Data
- A file-upload widget accepting `.csv` and a **Load CSV** button. On load,
  parse the CSV with pandas and show: row × column counts, the list of
  numeric columns, the list of text columns, and the first 8 rows.
- Split columns into **numeric** and **text/categorical** automatically.
- Show **one checkbox per column** in two tickable lists: "Metric basis
  variables — tick to include" (all ticked by default) and "Categorical
  variables — ticked ones become 0/1 dummies". Under each list, **Select
  all** and **Clear** buttons that tick/untick everything at once.
  (Checkboxes, not multi-select lists: users must never need Ctrl-click to
  deselect.)
- **ID-column guard:** any text column whose number of distinct values is
  one-per-row (like `session_id`) must appear **unticked**, with a printed
  warning naming it and explaining that encoding it would create one dummy
  column per row and break the clustering.
- A checkbox, default ON: "Standardize before clustering (recommended)".
- A **Prepare data** button that: takes the ticked numeric columns as-is,
  one-hot encodes the ticked categoricals (all levels kept as 0/1, none
  dropped), joins them into one design matrix, standardizes it if the box
  is ticked, and prints a summary: rows × features, which variables were
  metric, which levels were encoded, whether standardized.

### Tab 2 · Scree & K
- A **Show scree plot** button that runs K-Means (fixed random seed, several
  restarts for stability) for K = 1 up to 10 (or n−1 if fewer rows) and
  draws ONE figure with two panels side by side: left = inertia
  (within-cluster sum of squares) vs K — "look for the sharpest bend";
  right = silhouette score vs K — "higher = cleaner; its peak often
  corrects the elbow's bias toward K=2".
  Display the figure with `display(fig)` ONLY — do not also call
  `plt.show()`, or Colab renders the plot twice.
- A slider "K (num clusters)" from 2 to 10, and a **Run K-Means** button
  that fits the chosen K on the prepared matrix and stores the segment
  labels. K must be smaller than the number of rows (guard it).

### Tab 3 · Centroids
- A toggle "Centroid units: **Raw units | Z-scores**" (default Raw) above
  the table. Same clusters either way — Raw is the z-scores converted back
  (z × SD + mean), i.e., the plain per-segment averages; Z is the scale
  K-Means actually used. Say this in one line on the tab.
- **Transpose the table: segments are the COLUMNS, basis variables are the
  rows** (a variables-as-columns table extends off the screen). The first
  two rows are `size` and `pct` (share of rows in the segment); then one
  row per basis variable. Numeric cells rounded to 2 decimals.
- Render all tables with the `itables` library (clickable sortable column
  headers, search box, pagination — like R's DT::datatable). The app must
  pip-install `itables` itself on first load if missing, and silently fall
  back to plain pandas display if anything fails.
- Below, a "categorical level shares within segment (%)" table (one block
  per categorical variable: each segment's % in each level).

### Tab 4 · Biplot
- A **Show biplot** button (enabled after Run): a PCA two-dimensional
  projection of the SAME clustering — scatter of all rows colored by
  segment, with an arrow per basis variable whose direction and length show
  its association; axes labeled "PC1 (xx% of variance)". One sentence on
  the tab makes clear this is an eyeball view — K-Means ran in the full
  feature space.

### Tab 5 · Simplified (the LLM-ready table)
- After Run, fill a large text box with a **plain-text** maxima/minima
  summary — NOT an HTML table, which truncates long content. Format:

      K-Means segmentation output | K = 3 | n = 100
      MAXIMA = basis variables where this segment scores HIGHEST across
      segments; MINIMA = where it scores LOWEST (centroids are
      z-standardized).

      SEGMENT 0 (n=42)
        MAXIMA: page_views, product_page_duration, lab_report_clicks, ...
        MINIMA: is_logged_in, past_orders, checkout_success, ...

      (one block per segment)

  Rules: for each basis variable, find which segment has the highest
  z-score (ties allowed within a tiny tolerance) and which has the lowest;
  list every variable accordingly. Segment sizes come from the run.
- The tab text tells the student: click in the box, Ctrl/Cmd-A, copy, paste
  into the AI interpretation prompt from the caselet.

### Tab 6 · Assignments
- After Run: a small "rows per segment" table and a **10-row preview** of
  the original CSV with its assigned segment added as a leading column —
  both sortable via itables.
- A **Download full assignments CSV** button (uses Colab's file download;
  if unavailable, say the file was saved in the Colab file pane).

### Numbers and conventions
- K-Means: fixed seed (42), `n_init=10` so results are reproducible.
- Standardization: z-scores via the column means/SDs of the design matrix.
- Everything rounded for display; never for computation.
- The app never phones home: the only network access is the initial
  `requests.get` in the Colab cell; the app itself makes no network calls.
- If `itables` is unavailable and cannot be installed, plain tables still
  work — nothing else changes.

### Why these choices (so the re-implementation does not "fix" them away)
- Checkboxes instead of multi-select lists, and Select all/Clear buttons:
  multi-select widgets hide their deselect gesture and stranded early
  testers.
- ID-like columns unticked by default: encoding them once created a dummy
  per row and flooded the maxima/minima pools — caught in testing.
- Scree AND silhouette side by side: an elbow-only plot systematically
  favors K=2 (the largest inertia drop is almost always the 1→2 split),
  which misled a real test run on the caselet data.
- Text-format simplified table: the HTML dataframe version cut off
  variables, and the table exists to be copy-pasted into an AI prompt.
- Transposed centroids: with many basis variables, a segments-as-rows
  table overflows the screen width.
