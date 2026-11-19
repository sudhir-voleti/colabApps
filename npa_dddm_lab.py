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
import base64
import html as html_lib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

APP_STATE = {"df": None, "scored_df": None, "filename": None}

TABLE_CSS = """
<style>
    .sv-table { border-collapse: collapse; width: 100%; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; margin: 10px 0; }
    .sv-table th { background-color: #002D72; color: #ffffff; padding: 8px 10px; text-align: left; border-bottom: 2px solid #CFB87C; }
    .sv-table td { padding: 7px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
    .sv-table tr:nth-child(even) { background-color: #f8fafc; }
    .sv-card { background: #f4f6f9; border-left: 5px solid #002D72; padding: 14px 18px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; line-height: 1.6; }
    .sv-warn { background: #fffdf5; border-left: 5px solid #CFB87C; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; }
    .sv-dl-btn { display: inline-block; background-color: #002D72; color: #ffffff !important; text-decoration: none; padding: 9px 16px; border-radius: 5px; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; font-weight: 600; border-bottom: 3px solid #CFB87C; margin: 8px 0; }
    .sv-dl-btn:hover { background-color: #1e40af; }
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
                right_rule
