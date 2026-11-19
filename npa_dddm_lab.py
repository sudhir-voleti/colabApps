# npa_dddm_lab.py
# SVPNPA 28th MCTP Phase III — Data Driven Decision Making Lab (Outage-Proof ipywidgets + visNetwork Tree)
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Launch via:
#   import requests
#   url = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_dddm_lab.py"
#   response = requests.get(url); response.raise_for_status()
#   exec(compile(response.text, url, "exec"), globals())

import io
import json
import base64
import html as html_lib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from PIL import Image, ImageDraw
import requests
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Global state container for uploaded DataFrame
APP_STATE = {"df": None, "filename": None}

# ==============================================================================
# HELPER FUNCTIONS: STYLING, VISNETWORK TREE BUILDER & LEAF EXTRACTION
# ==============================================================================
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

def build_visnetwork_tree_html(clf, feature_names, target_name, df_orig):
    """
    Generates an interactive Vis.js (visNetwork / visTree style) hierarchical decision tree
    embedded via srcdoc iframe so zoom, pan, drag, and hover tooltips work natively in Colab.
    """
    tree = clf.tree_
    total_samples = int(tree.n_node_samples[0])
    has_impact = "est_impact_lakhs" in df_orig.columns

    nodes = []
    edges = []

    for node in range(tree.node_count):
        n_samp = int(tree.n_node_samples[node])
        pct_traffic = (n_samp / total_samples) * 100.0
        vals = tree.value[node][0]
        n_noise = int(vals[0])
        n_hits = int(vals[1]) if len(vals) > 1 else 0
        hit_prob = (n_hits / n_samp) * 100.0 if n_samp > 0 else 0.0

        # Color scheme matching visTree / executive palette
        if hit_prob >= 55:
            bg_color, border_color, font_color = "#d1fae5", "#047857", "#064e3b"
            status_lbl = "ACTIONABLE (HIGH YIELD)"
        elif hit_prob >= 25:
            bg_color, border_color, font_color = "#fef3c7", "#b45309", "#78350f"
            status_lbl = "SELECTIVE / MODERATE"
        else:
            bg_color, border_color, font_color = "#fee2e2", "#b91c1c", "#7f1d1d"
            status_lbl = "NOISE / FALSE-ALARM TRAP"

        is_leaf = (tree.feature[node] == -2)
        if not is_leaf:
            feat = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            if "_" in feat and thresh == 0.5:
                base_col, cat_val = feat.split("_", 1)
                split_txt = f"{base_col} = {cat_val}?"
            else:
                split_txt = f"{feat} > {thresh:.1f}?"
            label = f"{split_txt}\nHit: {hit_prob:.1f}% | Traffic: {pct_traffic:.1f}%\n(n = {n_samp:,})"
            shape = "box"
            border_width = 2
        else:
            label = f"LEAF #{node}: {status_lbl}\nHit Rate: {hit_prob:.1f}%\nTraffic: {pct_traffic:.1f}% (n = {n_samp:,})"
            shape = "box"
            border_width = 3

        tooltip = (
            f"<b>Node #{node} ({'Terminal Leaf' if is_leaf else 'Decision Split'})</b><br>"
            f"Status: {status_lbl}<br>"
            f"Sample Count: {n_samp:,} ({pct_traffic:.1f}% of traffic)<br>"
            f"True Actionable Hits (Y=1): {n_hits:,} ({hit_prob:.1f}%)<br>"
            f"False Alarms / Noise (Y=0): {n_noise:,} ({100 - hit_prob:.1f}%)"
        )

        nodes.append({
            "id": int(node),
            "label": label,
            "title": tooltip,
            "shape": shape,
            "color": {
                "background": bg_color,
                "border": border_color,
                "highlight": {"background": "#e0e7ff", "border": "#002D72"}
            },
            "font": {"color": font_color, "size": 13, "face": "Segoe UI, Arial, sans-serif", "bold": is_leaf},
            "borderWidth": border_width,
            "margin": 10
        })

        if not is_leaf:
            left_c = int(tree.children_left[node])
            right_c = int(tree.children_right[node])
            edges.append({
                "from": int(node), "to": left_c, "label": "No",
                "arrows": "to", "color": {"color": "#64748b"},
                "font": {"align": "middle", "size": 12, "color": "#475569", "background": "#ffffff"}
            })
            edges.append({
                "from": int(node), "to": right_c, "label": "Yes",
                "arrows": "to", "color": {"color": "#002D72"},
                "font": {"align": "middle", "size": 12, "color": "#002D72", "bold": True, "background": "#ffffff"}
            })

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    iframe_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #ffffff; }}
    #toolbar {{ background: #f8fafc; border-bottom: 1px solid #cbd5e1; padding: 8px 14px; font-size: 12.5px; color: #334155; display: flex; justify-content: space-between; align-items: center; }}
    #mynetwork {{ width: 100%; height: 520px; border: none; }}
    .btn {{ background: #002D72; color: #fff; border: none; padding: 5px 11px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
    .btn:hover {{ background: #1e40af; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <span><b>Interactive Command Tree (visNetwork Style):</b> Scroll to zoom &bull; Drag canvas/nodes &bull; Hover any box for node stats</span>
    <button class="btn" onclick="network.fit({{animation: true}})">Reset Zoom / Fit Tree</button>
  </div>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    var nodes = new vis.DataSet({nodes_json});
    var edges = new vis.DataSet({edges_json});
    var container = document.getElementById('mynetwork');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
      layout: {{
        hierarchical: {{
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 115,
          nodeSpacing: 220,
          treeSpacing: 240
        }}
      }},
      physics: false,
      interaction: {{
        hover: true,
        tooltipDelay: 120,
        navigationButtons: true,
        zoomView: true
      }},
      edges: {{
        smooth: {{
          type: "cubicBezier",
          forceDirection: "vertical",
          roundness: 0.4
        }}
      }}
    }};
    var network = new vis.Network(container, data, options);
  </script>
</body>
</html>"""

    escaped_doc = html_lib.escape(iframe_doc, quote=True)
    return f'<iframe srcdoc="{escaped_doc}" style="width:100%; height:570px; border:1px solid #cbd5e1; border-radius:6px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);"></iframe>'

def extract_leaf_profiles(clf, feature_names, X_encoded, y_series, df_orig):
    tree = clf.tree_
    leaf_ids = clf.apply(X_encoded)
    total_n = len(y_series)

    paths = {}
    def recurse(node, rule_parts):
        if tree.feature[node] != -2:
            feat = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            if "_" in feat and thresh == 0.5:
                base_col, cat_val = feat.split("_", 1)
                left_rule = f"{base_col} != {cat_val}"
                right_rule = f"{base_col} == {cat_val}"
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
            "Command Classification": tag,
            "Alert Volume": cnt,
            "% Traffic": f"{traffic_pct:.1f}%",
            "True Hits (Y=1)": hits,
            "Hit Rate (%)": f"{hit_rate:.1f}%",
            "Decision Rule Path": f"<code>{paths.get(leaf_id, '')}</code>"
        }
        if has_impact:
            row_dict["Avg Impact (₹L)"] = f"{avg_imp:.2f}"
        rows.append(row_dict)

    prof_df = pd.DataFrame(rows)
    prof_df["_sort"] = prof_df["Hit Rate (%)"].str.rstrip("%").astype(float)
    prof_df = prof_df.sort_values(by="_sort", ascending=False).drop(columns=["_sort"])
    return prof_df

def create_demo_cctv_frame():
    img = Image.new("RGB", (640, 360), color=(28, 34, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([35, 55, 605, 325], outline=(110, 130, 150), width=3)
    draw.text((50, 22), "CAM-04 | MALKHANA & ARMOURY PERIMETER | 02:14:33 IST", fill=(230, 238, 245))
    draw.rectangle([425, 230, 495, 285], fill=(185, 50, 40), outline=(255, 205, 75), width=2)
    draw.text((400, 29
