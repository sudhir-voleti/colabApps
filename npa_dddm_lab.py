# npa_dddm_lab.py
# SVPNPA 28th MCTP Phase III — Data Driven Decision Making Lab (Outage-Proof ipywidgets Edition)
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Launch via:
#   import requests
#   url = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_dddm_lab.py"
#   response = requests.get(url); response.raise_for_status()
#   exec(compile(response.text, url, "exec"), globals())

import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from PIL import Image, ImageDraw
import requests
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Global state container for uploaded DataFrame
APP_STATE = {"df": None, "filename": None}

# ==============================================================================
# HELPER FUNCTIONS: STYLING, LEAF EXTRACTION & VLM DEMO
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
    draw.text((400, 294), "[STATIONARY OBJECT > 180s]", fill=(255, 205, 75))
    return img

# ==============================================================================
# TAB 1: DATA UPLOAD & 10-ROW PREVIEW
# ==============================================================================
upload_widget = widgets.FileUpload(
    accept=".csv",
    multiple=False,
    description="Upload CSV",
    button_style="primary"
)
upload_status = widgets.HTML(
    value=TABLE_CSS + "<div class='sv-warn'><b>Step 1:</b> Upload <code>project_drishti_alerts.csv</code> (or any compatible CSV) using the button above to unlock Tabs 2 &amp; 3.</div>"
)
tab1_preview_out = widgets.Output()

tab1_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>1. Upload Operational Dataset (.csv)</h3>"),
    upload_widget,
    upload_status,
    tab1_preview_out
])

# ==============================================================================
# TAB 2: DESCRIPTIVE STATS & VARIABLE SUMMARY
# ==============================================================================
tab2_out = widgets.Output()
with tab2_out:
    display(HTML(TABLE_CSS + "<div class='sv-warn'>Upload a CSV file in <b>Tab 1</b> first to view descriptive statistics.</div>"))

tab2_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>2. Dataset Architecture &amp; Descriptive Summary</h3>"),
    tab2_out
])

# ==============================================================================
# TAB 3: CASELET 1 — DECISION TREE & COMMAND TRIAGE
# ==============================================================================
y_dropdown = widgets.Dropdown(options=[], description="Target (Y):", layout=widgets.Layout(width="320px"))
x_select = widgets.SelectMultiple(options=[], description="Predictors (X):", rows=6, layout=widgets.Layout(width="320px"))
depth_slider = widgets.IntSlider(value=3, min=2, max=5, step=1, description="Tree Depth:", layout=widgets.Layout(width="320px"))
leaf_slider = widgets.IntSlider(value=75, min=25, max=250, step=25, description="Min Leaf N:", layout=widgets.Layout(width="320px"))
cap_slider = widgets.IntSlider(value=300, min=100, max=600, step=25, description="Dispatch Cap:", layout=widgets.Layout(width="320px"))
run_tree_btn = widgets.Button(description="Run Decision Tree & Triage Audit", button_style="success", icon="play", layout=widgets.Layout(width="320px"))

tab3_out = widgets.Output()
with tab3_out:
    display(HTML(TABLE_CSS + "<div class='sv-warn'>Upload <code>project_drishti_alerts.csv</code> in <b>Tab 1</b> to populate Target (Y) and Predictor (X) controls.</div>"))

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

        X_encoded = pd.get_dummies(df[predictors], drop_first=False)
        feature_names = X_encoded.columns.tolist()

        clf = DecisionTreeClassifier(
            max_depth=depth_slider.value,
            min_samples_leaf=leaf_slider.value,
            random_state=42
        )
        clf.fit(X_encoded, y)

        leaf_df = extract_leaf_profiles(clf, feature_names, X_encoded, y, df)

        total_n = len(df)
        total_hits = int(y.sum())
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
                <li><b>Incoming Volume:</b> {total_n:,} alerts | <b>Base Actionable Hit Rate:</b> {base_rate:.1f}% ({total_hits:,} true cases)</li>
                {vendor_html}
                <li><b>Decision-Tree Prioritized Dispatch (Top {cap} Cap):</b> Catches <b>{dt_hits}</b> true hits | Precision: <b>{dt_prec:.1f}%</b> | Recall: <b>{dt_recall:.1f}%</b></li>
            </ul>
        </div>
        """
        display(HTML(scorecard_html))

        # Plot Classical Decision Tree
        fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=["Noise (0)", "Actionable (1)"],
            filled=True,
            rounded=True,
            proportion=True,
            precision=1,
            fontsize=8.5,
            ax=ax
        )
        ax.set_title(f"Command Decision Tree (Target: {target_y} | Max Depth: {depth_slider.value})", fontsize=12, fontweight="bold", color="#002D72")
        plt.tight_layout()
        plt.show()

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
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>3. Caselet 1: Interpretable Decision Tree &amp; Triage Lab</h3>"),
    tab3_controls,
    tab3_out
])

# ==============================================================================
# TAB 4: CASELET 2A — COMMODITY CCTV + VLM SENTINEL
# ==============================================================================
img_upload = widgets.FileUpload(accept="image/*", multiple=False, description="Upload Frame (Opt)")
vlm_prompt = widgets.Textarea(
    value="You are an automated police duty-officer assistant monitoring a sensitive Malkhana/perimeter gate. Flag only genuine operational anomalies (unattended baggage, perimeter breach, fire/smoke, crowd buildup) and suppress routine activity. State confidence and caveats.",
    description="Prompt:",
    layout=widgets.Layout(width="650px", height="85px")
)
api_key_in = widgets.Password(placeholder="Optional OpenRouter API Key (leave blank for offline demo)", description="API Key:", layout=widgets.Layout(width="420px"))
model_in = widgets.Text(value="google/gemini-2.5-flash", description="VLM Model:", layout=widgets.Layout(width="420px"))
vlm_btn = widgets.Button(description="Run VLM Perimeter Analysis", button_style="primary", icon="eye")
tab4_out = widgets.Output()

def on_run_vlm(_):
    with tab4_out:
        clear_output(wait=True)
        if img_upload.value:
            uploaded = list(img_upload.value.values())[0] if isinstance(img_upload.value, dict) else img_upload.value[0]
            pil_img = Image.open(io.BytesIO(uploaded["content"])).convert("RGB")
        else:
            pil_img = create_demo_cctv_frame()

        display(pil_img)
        key = api_key_in.value.strip()
        if key:
            try:
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_in.value.strip() or "google/gemini-2.5-flash",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vlm_prompt.value},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }]
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
                data = resp.json()
                text_out = data["choices"][0]["message"]["content"] if "choices" in data else json.dumps(data)
                display(HTML(f"{TABLE_CSS}<div class='sv-card'><b>Live VLM Assessment:</b><br><pre style='white-space:pre-wrap; font-family:inherit;'>{text_out}</pre></div>"))
                return
            except Exception as e:
                display(HTML(f"{TABLE_CSS}<div class='sv-warn'>API fallback triggered ({e}). Showing offline simulation output:</div>"))

        fallback_html = f"""{TABLE_CSS}
        <div class='sv-card'>
            <h4 style='margin:0 0 8px 0; color:#002D72;'>Automated VLM Perimeter Sentinel Output (Offline Classroom Mode)</h4>
            <ul style='margin:0; padding-left:20px;'>
                <li><b>Alert Status:</b> <span style='color:#b91c1c; font-weight:bold;'>ANOMALY FLAGGED — PRIORITY 2 (Malkhana Perimeter)</span></li>
                <li><b>Visual Reasoning:</b> Stationary rectangular package detected near perimeter gate at <code>02:14 IST</code> with zero human presence for &gt;180 seconds.</li>
                <li><b>Recommended Action:</b> Route snapshot to Station Duty Officer tablet for 30-second visual check before dispatching QRT.</li>
                <li><b>Governance Caveat:</b> <i>Confidence: 81%. Low-light sensor glare or static equipment can mimic unattended objects—Human-in-the-Loop (HITL) sign-off is mandatory.</i></li>
            </ul>
        </div>
        """
        display(HTML(fallback_html))

vlm_btn.on_click(on_run_vlm)

tab4_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>4. Caselet 2A: Commodity CCTV + VLM Sentinel</h3><p>Leave image blank to inspect the built-in Malkhana night-perimeter test frame, or upload any camera snapshot.</p>"),
    img_upload,
    vlm_prompt,
    widgets.HBox([api_key_in, model_in]),
    vlm_btn,
    tab4_out
])

# ==============================================================================
# TAB 5: CASELET 2B — INTERVIEW / WITNESS BEHAVIORAL CUE TRIAGE
# ==============================================================================
scen_radio = widgets.RadioButtons(
    options=[
        "Scenario A: Suspect Alibi Timeline (EOW Shell-Company Inquiry)",
        "Scenario B: Eyewitness Recall Under Trauma (Highway Hit-and-Run)"
    ],
    layout=widgets.Layout(width="550px")
)
cue_btn = widgets.Button(description="Inspect VLM Behavioral Timeline", button_style="primary", icon="search")
tab5_out = widgets.Output()

def on_run_cue(_):
    with tab5_out:
        clear_output(wait=True)
        if "Scenario A" in scen_radio.value:
            html = f"""{TABLE_CSS}
            <div class='sv-card'>
                <h4 style='margin:0 0 8px 0; color:#002D72;'>Multimodal Cue Log — Scenario A (EOW Suspect Alibi)</h4>
                <ul style='margin:0; padding-left:20px;'>
                    <li><b>00:08 – 00:20 (Baseline):</b> Steady gaze and normal blink rate (~14/min) while confirming company incorporation details.</li>
                    <li><b>00:34 – 00:42 (Flagged Cognitive Load Spike):</b> Question asked regarding <i>₹1.8 Cr cash transfer on Nov 14</i>. Blink rate rises to 28/min, 1.9s response latency, subtle asymmetric shoulder shrug.</li>
                    <li><b>01:05 – 01:14 (Vocal Stress Shift):</b> Fundamental vocal pitch elevates +17% when asked about contact with co-accused director.</li>
                </ul>
                <div class='sv-warn' style='margin-top:10px;'><b>Command Governance Rule:</b> Physiological/behavioral cues indicate cognitive load or stress—<b>NOT deception</b>. Use solely to prioritize documentary verification (bank trails, CDR tower dumps), never as standalone evidence.</div>
            </div>"""
        else:
            html = f"""{TABLE_CSS}
            <div class='sv-card'>
                <h4 style='margin:0 0 8px 0; color:#002D72;'>Multimodal Cue Log — Scenario B (Traumatized Eyewitness)</h4>
                <ul style='margin:0; padding-left:20px;'>
                    <li><b>00:05 – 00:24 (Acute Stress Baseline):</b> Rapid breathing, lip compression, and vocal tremor throughout initial narrative.</li>
                    <li><b>00:41 – 00:53 (Memory Retrieval Pattern):</b> Gaze aversion upward-left while recalling partial vehicle plate digits.</li>
                </ul>
                <div class='sv-warn' style='margin-top:10px;'><b>Command Governance Rule:</b> Off-the-shelf emotion/stress classifiers frequently misclassify acute trauma, fatigue, or neurodivergence as 'evasion'. Human investigator context must override raw scores.</div>
            </div>"""
        display(HTML(html))

cue_btn.on_click(on_run_cue)

tab5_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>5. Caselet 2B: Interview Behavioral Cue Triage &amp; HITL Guardrails</h3>"),
    scen_radio,
    cue_btn,
    tab5_out
])

# ==============================================================================
# WIRE FILE UPLOAD EVENT TO POPULATE TABS 1, 2, AND 3
# ==============================================================================
def on_csv_uploaded(change):
    if not upload_widget.value:
        return
    # Handle ipywidgets v7 (dict) and v8 (tuple/list) FileUpload formats
    if isinstance(upload_widget.value, dict):
        fname = list(upload_widget.value.keys())[0]
        content = upload_widget.value[fname]["content"]
    else:
        item = upload_widget.value[0]
        fname = item["name"]
        content = item["content"]

    df = pd.read_csv(io.BytesIO(content))
    APP_STATE["df"] = df
    APP_STATE["filename"] = fname

    upload_status.value = f"{TABLE_CSS}<div class='sv-card'><b>Loaded:</b> <code>{fname}</code> ({len(df):,} rows &times; {df.shape[1]} columns). Proceed to <b>Tab 2</b> (Descriptive Stats) or <b>Tab 3</b> (Decision Tree Lab).</div>"

    # Update Tab 1 Preview
    with tab1_preview_out:
        clear_output(wait=True)
        display(HTML("<h4 style='color:#002D72; margin-top:10px;'>First 10 Rows Preview</h4>"))
        display(HTML(df_to_styled_html(df, max_rows=10)))

    # Update Tab 2 Descriptive Stats
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

    # Update Tab 3 Variable Selectors
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

# ==============================================================================
# RENDER MASTER TAB CONTAINER
# ==============================================================================
tabs = widgets.Tab(children=[tab1_box, tab2_box, tab3_box, tab4_box, tab5_box])
tab_titles = [
    "1. Data Upload & Preview",
    "2. Descriptive Stats",
    "3. Caselet 1: Decision Tree",
    "4. Caselet 2A: CCTV VLM",
    "5. Caselet 2B: Interview Cues"
]
for i, title in enumerate(tab_titles):
    tabs.set_title(i, title)

header_banner = widgets.HTML("""
<div style="background:#002D72; color:#ffffff; padding:16px 22px; border-radius:6px; border-bottom:4px solid #CFB87C; font-family:'Segoe UI', Arial, sans-serif; margin-bottom:12px;">
    <h2 style="margin:0; font-size:19px;">SVPNPA 28th MCTP Phase III &mdash; Data Driven Decision Making Lab</h2>
    <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">Faculty: Prof. Sudhir Voleti | ISB Executive Education | Native Colab Interactive Workspace</div>
</div>
""")

display(widgets.VBox([header_banner, tabs]))
