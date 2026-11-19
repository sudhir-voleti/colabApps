# npa_drishti_dtree.py
# SVPNPA 28th MCTP Phase III — Caselet 1: Project Drishti (Decision Tree & Command Triage)
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Colab Launcher:
#   import requests
#   url = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_drishti_dtree.py"
#   response = requests.get(url); response.raise_for_status()
#   exec(compile(response.text, url, "exec"), globals())

import io
import json
import html as html_lib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

APP_STATE = {"df": None, "filename": None}

TABLE_CSS = """
<style>
    .sv-table { border-collapse: collapse; width: 100%; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; margin: 10px 0; }
    .sv-table th { background-color: #002D72; color: #ffffff; padding: 8px 10px; text-align: left; border-bottom: 2px solid #CFB87C; }
    .sv-table td { padding: 7px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
    .sv-table tr:nth-child(even) { background-color: #f8fafc; }
    .sv-card { background: #f4f6f9; border-left: 5px solid #002D72; padding: 14px 18px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; line-height: 1.6; }
    .sv-warn { background: #fffdf5; border-left: 5px solid #CFB87C; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; }
</style>
"""

def df_to_styled_html(df, max_rows=None):
    sub = df.head(max_rows) if max_rows else df
    return TABLE_CSS + sub.to_html(index=False, classes="sv-table", escape=False)

def build_visnetwork_tree_html(clf, feature_names, cat_cols):
    tree = clf.tree_
    total_samples = int(tree.n_node_samples[0])

    coords = {}
    leaf_counter = [0]

    def assign_coords(node, depth):
        if tree.feature[node] == -2:
            coords[node] = (leaf_counter[0] * 290, depth * 175)
            leaf_counter[0] += 1
        else:
            assign_coords(tree.children_left[node], depth + 1)
            assign_coords(tree.children_right[node], depth + 1)
            lx, _ = coords[tree.children_left[node]]
            rx, _ = coords[tree.children_right[node]]
            coords[node] = ((lx + rx) / 2.0, depth * 175)

    assign_coords(0, 0)

    nodes, edges = [], []
    for node in range(tree.node_count):
        n_samp = int(tree.n_node_samples[node])
        pct_traffic = (n_samp / total_samples) * 100.0
        vals = tree.value[node][0]

        total_v = float(vals.sum())
        hit_prob = (float(vals[1]) / total_v) * 100.0 if len(vals) > 1 and total_v > 0 else 0.0
        n_hits = int(round((hit_prob / 100.0) * n_samp))
        n_noise = n_samp - n_hits

        if hit_prob >= 55:
            bg_color, border_color, font_color = "#d1fae5", "#047857", "#064e3b"
            status_lbl = "HIGH YIELD"
        elif hit_prob >= 25:
            bg_color, border_color, font_color = "#fef3c7", "#b45309", "#78350f"
            status_lbl = "SELECTIVE"
        else:
            bg_color, border_color, font_color = "#fee2e2", "#b91c1c", "#7f1d1d"
            status_lbl = "NOISE TRAP"

        is_leaf = bool(tree.feature[node] == -2)
        if not is_leaf:
            feat = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            matched_prefix = next((c for c in cat_cols if feat.startswith(c + "_")), None)
            if matched_prefix and thresh == 0.5:
                cat_val = feat[len(matched_prefix) + 1:]
                split_txt = f"{matched_prefix} = {cat_val}?"
            else:
                split_txt = f"{feat} > {thresh:.1f}?"
            label = f"{split_txt}\nHit: {hit_prob:.1f}% | Vol: {pct_traffic:.1f}%\n(n = {n_samp:,})"
            border_width = 2
        else:
            label = f"LEAF #{node} [{status_lbl}]\nHit Rate: {hit_prob:.1f}%\nTraffic: {pct_traffic:.1f}% (n={n_samp:,})"
            border_width = 3

        tooltip = (
            f"<b>Node #{node} ({'Terminal Leaf' if is_leaf else 'Decision Split'})</b><br>"
            f"Classification: <b>{status_lbl}</b><br>"
            f"Sample Count: {n_samp:,} ({pct_traffic:.1f}% of traffic)<br>"
            f"Positive Class (Y=1): {n_hits:,} ({hit_prob:.1f}%)<br>"
            f"Negative Class (Y=0): {n_noise:,} ({100.0 - hit_prob:.1f}%)"
        )

        nx, ny = coords[node]
        nodes.append({
            "id": int(node), "x": float(nx), "y": float(ny),
            "label": label, "title": tooltip, "shape": "box",
            "color": {"background": bg_color, "border": border_color, "highlight": {"background": "#e0e7ff", "border": "#002D72"}},
            "font": {"color": font_color, "size": 13, "face": "Segoe UI, Arial, sans-serif", "bold": is_leaf},
            "borderWidth": border_width, "margin": 10
        })

        if not is_leaf:
            edges.append({
                "from": int(node), "to": int(tree.children_left[node]), "label": "No",
                "arrows": "to", "color": {"color": "#64748b"},
                "font": {"align": "middle", "size": 12, "color": "#475569", "background": "#ffffff"}
            })
            edges.append({
                "from": int(node), "to": int(tree.children_right[node]), "label": "Yes",
                "arrows": "to", "color": {"color": "#002D72"},
                "font": {"align": "middle", "size": 12, "color": "#002D72", "bold": True, "background": "#ffffff"}
            })

    nodes_json = json.dumps(nodes, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o) if isinstance(o, np.integer) else float(o))
    edges_json = json.dumps(edges)

    iframe_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #ffffff; }}
    #toolbar {{ background: #f8fafc; border-bottom: 1px solid #cbd5e1; padding: 8px 14px; font-size: 12.5px; color: #334155; display: flex; justify-content: space-between; align-items: center; }}
    #mynetwork {{ width: 100%; height: 560px; border: none; }}
    .btn {{ background: #002D72; color: #fff; border: none; padding: 5px 11px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 6px; }}
    .btn:hover {{ background: #1e40af; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <span><b>Interactive Command Tree (visNetwork Style):</b> Drag any node freely in 2D &bull; Scroll to zoom &bull; Hover for exact stats</span>
    <div>
      <button class="btn" onclick="resetTree()">Reset Tree Layout</button>
      <button class="btn" onclick="network.fit({{animation: true}})">Fit to Screen</button>
    </div>
  </div>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    var initialNodes = {nodes_json};
    var nodes = new vis.DataSet(initialNodes);
    var edges = new vis.DataSet({edges_json});
    var container = document.getElementById('mynetwork');
    var options = {{
      layout: {{ hierarchical: {{ enabled: false }} }},
      physics: false,
      interaction: {{ dragNodes: true, dragView: true, hover: true, tooltipDelay: 100, navigationButtons: true, zoomView: true }},
      edges: {{ smooth: {{ type: "cubicBezier", forceDirection: "vertical", roundness: 0.35 }} }}
    }};
    var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
    network.once("afterDrawing", function() {{ network.fit({{ animation: false }}); }});
    function resetTree() {{ nodes.update(initialNodes); network.fit({{ animation: true }}); }}
  </script>
</body>
</html>"""

    escaped_doc = html_lib.escape(iframe_doc, quote=True)
    return f'<iframe srcdoc="{escaped_doc}" style="width:100%; height:610px; border:1px solid #cbd5e1; border-radius:6px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);"></iframe>'

def extract_leaf_profiles(clf, feature_names, cat_cols, X_encoded, y_series, df_orig):
    tree = clf.tree_
    leaf_ids = clf.apply(X_encoded)
    total_n = len(y_series)

    paths = {}
    def recurse(node, rule_parts):
        if tree.feature[node] != -2:
            feat = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            matched_prefix = next((c for c in cat_cols if feat.startswith(c + "_")), None)
            if matched_prefix and thresh == 0.5:
                cat_val = feat[len(matched_prefix) + 1:]
                left_rule = f"{matched_prefix} != {cat_val}"
                right_rule = f"{matched_prefix} == {cat_val}"
            else:
                left_rule = f"{feat} <= {thresh:.2f}"
                right_rule = f"{feat} > {thresh:.2f}"
            recurse(tree.children_left[node], rule_parts + [left_rule])
            recurse(tree.children_right[node], rule_parts + [right_rule])
        else:
            paths[node] = " AND ".join(rule_parts) if rule_parts else "All Records (Root)"

    recurse(0, [])

    rows = []
    has_impact = "est_impact_lakhs" in df_orig.columns
    for leaf_id in np.unique(leaf_ids):
        mask = (leaf_ids == leaf_id)
        cnt = int(mask.sum())
        hits = int(y_series[mask].sum())
        hit_rate = (hits / cnt * 100.0) if cnt > 0 else 0.0
        traffic_pct = (cnt / total_n * 100.0)
        avg_imp = float(df_orig.loc[mask, "est_impact_lakhs"].mean()) if has_impact else 0.0

        if hit_rate >= 55:
            tag = "<b style='color:#0b6e4f;'>HIGH-YIELD PRIORITY</b>"
        elif hit_rate >= 25:
            tag = "<b style='color:#b45309;'>MODERATE / SELECTIVE</b>"
        else:
            tag = "<b style='color:#b91c1c;'>NOISE / FALSE-ALARM TRAP</b>"

        row_dict = {
            "Leaf ID": int(leaf_id),
            "Classification": tag,
            "Volume": cnt,
            "% Traffic": f"{traffic_pct:.1f}%",
            "Positives (Y=1)": hits,
            "Hit Rate (%)": f"{hit_rate:.1f}%",
            "Decision Rule Path": f"<code>{paths.get(leaf_id, '')}</code>"
        }
        if has_impact:
            row_dict["Avg Impact (₹L)"] = f"{avg_imp:.2f}"
        rows.append(row_dict)

    prof_df = pd.DataFrame(rows)
    prof_df["_sort"] = prof_df["Hit Rate (%)"].str.rstrip("%").astype(float)
    return prof_df.sort_values(by="_sort", ascending=False).drop(columns=["_sort"])

# ==============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# ==============================================================================
upload_widget = widgets.FileUpload(accept=".csv", multiple=False, description="Upload CSV", button_style="primary")
upload_status = widgets.HTML(value=TABLE_CSS + "<div class='sv-warn'><b>Step 1:</b> Upload <code>project_drishti_alerts.csv</code> (or any compatible CSV) using the button above to unlock Tabs 2 &amp; 3.</div>")
tab1_preview_out = widgets.Output()

tab1_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>1. Upload Operational Dataset (.csv)</h3>"),
    upload_widget, upload_status, tab1_preview_out
])

# ==============================================================================
# TAB 2: DESCRIPTIVE STATS
# ==============================================================================
tab2_out = widgets.Output()
with tab2_out:
    display(HTML(TABLE_CSS + "<div class='sv-warn'>Upload a CSV file in <b>Tab 1</b> first to view descriptive statistics.</div>"))

tab2_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>2. Dataset Architecture &amp; Descriptive Summary</h3>"),
    tab2_out
])

# ==============================================================================
# TAB 3: INTERACTIVE DECISION TREE & TRIAGE AUDIT
# ==============================================================================
y_dropdown = widgets.Dropdown(options=[], description="Target (Y):", layout=widgets.Layout(width="320px"))
x_select = widgets.SelectMultiple(options=[], description="Predictors (X):", rows=6, layout=widgets.Layout(width="320px"))
depth_slider = widgets.IntSlider(value=3, min=2, max=5, step=1, description="Tree Depth:", layout=widgets.Layout(width="320px"))
leaf_slider = widgets.IntSlider(value=75, min=25, max=250, step=25, description="Min Leaf N:", layout=widgets.Layout(width="320px"))
cap_slider = widgets.IntSlider(value=300, min=100, max=600, step=25, description="Dispatch Cap:", layout=widgets.Layout(width="320px"))
run_tree_btn = widgets.Button(description="Run Decision Tree & Triage Audit", button_style="success", icon="play", layout=widgets.Layout(width="320px"))

tab3_out = widgets.Output()
with tab3_out:
    display(HTML(TABLE_CSS + "<div class='sv-warn'>Upload a CSV in <b>Tab 1</b> to populate Target (Y) and Predictor (X) controls.</div>"))

def on_run_tree(_):
    with tab3_out:
        clear_output(wait=True)
        df = APP_STATE["df"]
        if df is None:
            display(HTML(TABLE_CSS + "<div class='sv-warn'>Please upload a CSV file in <b>Tab 1</b> first.</div>"))
            return

        target_y = y_dropdown.value
        predictors = [c for c in x_select.value if c != target_y]
        if not target_y or not predictors:
            display(HTML(TABLE_CSS + "<div class='sv-warn'>Select a valid <b>Target (Y)</b> and at least one distinct <b>Predictor (X)</b>.</div>"))
            return

        y_raw = df[target_y]
        if y_raw.dtype == object:
            y = (y_raw.astype(str).isin(["1", "True", "true", "Yes", "yes", "Actionable"])).astype(int)
        else:
            y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)

        cat_cols = [c for c in predictors if df[c].dtype == object]
        X_encoded = pd.get_dummies(df[predictors], drop_first=False)
        feature_names = X_encoded.columns.tolist()

        clf = DecisionTreeClassifier(max_depth=depth_slider.value, min_samples_leaf=leaf_slider.value, random_state=42)
        clf.fit(X_encoded, y)

        leaf_df = extract_leaf_profiles(clf, feature_names, cat_cols, X_encoded, y, df)

        total_n, total_hits = len(df), int(y.sum())
        base_rate = total_hits / total_n * 100.0
        cap = cap_slider.value

        vendor_html = ""
        if "ai_risk_score" in df.columns:
            v_mask = df["ai_risk_score"] >= 50
            v_alerts = int(v_mask.sum())
            v_hits = int(y[v_mask].sum())
            v_prec = (v_hits / v_alerts * 100.0) if v_alerts > 0 else 0.0
            vendor_html = f"<li><b>Legacy Vendor Cutoff (<code>ai_risk_score &ge; 50</code>):</b> Fires <b>{v_alerts:,}</b> alerts (exceeds {cap} dispatch cap by {max(0, v_alerts - cap):,}) | Precision: <b>{v_prec:.1f}%</b></li>"

        leaf_ids = clf.apply(X_encoded)
        leaf_rate_map = dict(zip(leaf_df["Leaf ID"], leaf_df["Hit Rate (%)"].str.rstrip("%").astype(float)))
        df_tmp = df.copy()
        df_tmp["_rate"] = [leaf_rate_map[lid] for lid in leaf_ids]
        dispatched = df_tmp.sort_values(by="_rate", ascending=False).head(cap)
        dt_hits = int(y.loc[dispatched.index].sum())
        dt_prec = (dt_hits / len(dispatched) * 100.0) if len(dispatched) > 0 else 0.0
        dt_recall = (dt_hits / total_hits * 100.0) if total_hits > 0 else 0.0

        scorecard_html = f"""{TABLE_CSS}
        <div class='sv-card'>
            <h4 style='margin:0 0 8px 0; color:#002D72;'>Command Triage &amp; Decision Tree Scorecard</h4>
            <ul style='margin:0; padding-left:20px;'>
                <li><b>Incoming Volume:</b> {total_n:,} records | <b>Base Positive Rate:</b> {base_rate:.1f}% ({total_hits:,} cases)</li>
                {vendor_html}
                <li><b>Decision-Tree Prioritized Dispatch (Top {cap} Cap):</b> Catches <b>{dt_hits}</b> positives | Precision: <b>{dt_prec:.1f}%</b> | Recall: <b>{dt_recall:.1f}%</b></li>
            </ul>
        </div>
        """
        display(HTML(scorecard_html))
        display(HTML(build_visnetwork_tree_html(clf, feature_names, cat_cols)))
        display(HTML("<h4 style='color:#002D72; margin-top:16px;'>Leaf-Node Risk &amp; Traffic Profile Table</h4>"))
        display(HTML(df_to_styled_html(leaf_df)))

        ascii_txt = export_text(clf, feature_names=feature_names, decimals=2)
        display(HTML(f"<details style='margin-top:12px;'><summary style='cursor:pointer; color:#002D72; font-weight:bold;'>View ASCII Rule Wireframe</summary><pre style='background:#f8fafc; padding:12px; border:1px solid #ddd;'>{ascii_txt}</pre></details>"))

run_tree_btn.on_click(on_run_tree)

tab3_controls = widgets.HBox([
    widgets.VBox([y_dropdown, x_select]),
    widgets.VBox([depth_slider, leaf_slider, cap_slider, run_tree_btn], layout=widgets.Layout(margin="0 0 0 25px"))
])
tab3_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>3. Caselet 1: Interactive Decision Tree &amp; Triage Lab</h3>"),
    tab3_controls, tab3_out
])

def on_csv_uploaded(change):
    if not upload_widget.value:
        return
    if isinstance(upload_widget.value, dict):
        fname = list(upload_widget.value.keys())[0]
        content = upload_widget.value[fname]["content"]
    else:
        item = upload_widget.value[0]
        fname, content = item["name"], item["content"]

    df = pd.read_csv(io.BytesIO(content))
    APP_STATE["df"], APP_STATE["filename"] = df, fname

    upload_status.value = f"{TABLE_CSS}<div class='sv-card'><b>Loaded:</b> <code>{fname}</code> ({len(df):,} rows &times; {df.shape[1]} columns). Proceed to <b>Tab 2</b> (Descriptive Stats) or <b>Tab 3</b> (Decision Tree Lab).</div>"

    with tab1_preview_out:
        clear_output(wait=True)
        display(HTML("<h4 style='color:#002D72; margin-top:10px;'>First 10 Rows Preview</h4>"))
        display(HTML(df_to_styled_html(df, max_rows=10)))

    with tab2_out:
        clear_output(wait=True)
        num_df = df.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index": "Numeric Variable"})
        for col in ["mean", "std", "min", "50%", "max"]:
            if col in num_df.columns:
                num_df[col] = num_df[col].round(2)

        cat_cols = [c for c in df.columns if c not in df.select_dtypes(include=[np.number]).columns and "id" not in c.lower()]
        cat_rows = []
        for c in cat_cols:
            vc = df[c].value_counts(normalize=True) * 100
            breakdown = " | ".join([f"<b>{k}</b>: {v:.1f}%" for k, v in vc.items()])
            cat_rows.append({"Categorical Field": c, "Unique Levels": df[c].nunique(), "Missing": int(df[c].isna().sum()), "Category Share (%)": breakdown})

        display(HTML("<h4 style='color:#002D72;'>Numeric Variables Summary</h4>"))
        display(HTML(df_to_styled_html(num_df)))
        if cat_rows:
            display(HTML("<h4 style='color:#002D72; margin-top:18px;'>Categorical Variables Breakdown</h4>"))
            display(HTML(df_to_styled_html(pd.DataFrame(cat_rows))))

    valid_cols = [c for c in df.columns if "id" not in c.lower()]
    default_y = "verified_actionable_flag" if "verified_actionable_flag" in valid_cols else valid_cols[0]
    default_x = [c for c in ["source_stream", "zone_type", "ai_risk_score", "prior_linkages", "hour_band"] if c in valid_cols and c != default_y]
    if not default_x:
        default_x = [c for c in valid_cols if c != default_y][:5]

    y_dropdown.options = valid_cols
    y_dropdown.value = default_y
    x_select.options = valid_cols
    x_select.value = tuple(default_x)

    with tab3_out:
        clear_output(wait=True)
        display(HTML(f"{TABLE_CSS}<div class='sv-warn'>Variables mapped for <code>{fname}</code>. Click <b>Run Decision Tree &amp; Triage Audit</b> above.</div>"))

upload_widget.observe(on_csv_uploaded, names="value")

tabs = widgets.Tab(children=[tab1_box, tab2_box, tab3_box])
for i, title in enumerate(["1. Data Upload & Preview", "2. Descriptive Stats", "3. Caselet 1: Decision Tree"]):
    tabs.set_title(i, title)

header_banner = widgets.HTML("""
<div style="background:#002D72; color:#ffffff; padding:16px 22px; border-radius:6px; border-bottom:4px solid #CFB87C; font-family:'Segoe UI', Arial, sans-serif; margin-bottom:12px;">
    <h2 style="margin:0; font-size:19px;">SVPNPA 28th MCTP Phase III &mdash; Caselet 1: Command Decision Tree &amp; Triage</h2>
    <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">Faculty: Prof. Sudhir Voleti | ISB Executive Education | Generic Binary-Outcome Tree Analyzer</div>
</div>
""")

display(widgets.VBox([header_banner, tabs]))
