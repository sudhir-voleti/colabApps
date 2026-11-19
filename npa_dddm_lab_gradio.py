# npa_dddm_lab.py
# SVPNPA 28th MCTP Phase III — Data Driven Decision Making Lab
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Usage in Google Colab:
#   !wget -qO npa_dddm_lab.py https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_dddm_lab.py
#   %run npa_dddm_lab.py

import subprocess
import sys

def ensure_packages():
    required = ["gradio", "scikit-learn", "pandas", "numpy", "matplotlib", "pillow", "requests"]
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_packages()

import os
import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from PIL import Image, ImageDraw
import requests
import gradio as gr

# ==============================================================================
# 1. DATA LOADER & FALLBACK GENERATOR (MATCHES R SCRIPT SCHEMA 1:1)
# ==============================================================================
def generate_fallback_drishti(n=5000, seed=2026):
    rng = np.random.default_rng(seed)
    source_stream = rng.choice(
        ["CCTNS_Pattern", "CCTV_Analytics", "EOW_Financial", "Cyber_UPI_Mule"],
        size=n, p=[0.35, 0.30, 0.20, 0.15]
    )
    zone_type = rng.choice(
        ["Zone1_Central", "Zone2_Industrial", "Zone3_Border", "Zone4_OldCity"],
        size=n, p=[0.28, 0.24, 0.20, 0.28]
    )
    hour_band = rng.choice(
        ["Day_Peak", "Evening_Rush", "Night_Watch"],
        size=n, p=[0.45, 0.35, 0.20]
    )
    prior_linkages = np.minimum(rng.poisson(1.4, size=n), 8)

    true_prob = np.full(n, 0.08)
    idx_fin_high = np.isin(source_stream, ["EOW_Financial", "Cyber_UPI_Mule"]) & (prior_linkages >= 2)
    true_prob[idx_fin_high] = 0.78

    idx_fin_low = np.isin(source_stream, ["EOW_Financial", "Cyber_UPI_Mule"]) & (prior_linkages < 2)
    true_prob[idx_fin_low] = 0.24

    idx_night_patrol = ((source_stream == "CCTNS_Pattern") & (hour_band == "Night_Watch") & (prior_linkages >= 1)) | \
                       ((source_stream == "CCTV_Analytics") & (zone_type == "Zone3_Border") & (hour_band == "Night_Watch"))
    true_prob[idx_night_patrol] = 0.68

    idx_zone4_cctv = (source_stream == "CCTV_Analytics") & (zone_type == "Zone4_OldCity")
    true_prob[idx_zone4_cctv] = 0.06

    verified_actionable_flag = rng.binomial(1, true_prob)

    raw_score = np.where(
        verified_actionable_flag == 1,
        rng.normal(75, 12, size=n),
        rng.normal(40, 16, size=n)
    )
    raw_score[idx_zone4_cctv] = rng.normal(74, 11, size=int(idx_zone4_cctv.sum()))
    ai_risk_score = np.round(np.clip(raw_score, 1, 99), 1)

    est_impact_lakhs = np.where(
        np.isin(source_stream, ["EOW_Financial", "Cyber_UPI_Mule"]),
        np.round(rng.lognormal(mean=2.3, sigma=0.85, size=n), 2),
        np.round(rng.uniform(0.2, 4.5, size=n), 2)
    )

    return pd.DataFrame({
        "alert_id": [f"DRI-{i:05d}" for i in range(1, n + 1)],
        "verified_actionable_flag": verified_actionable_flag,
        "source_stream": source_stream,
        "zone_type": zone_type,
        "ai_risk_score": ai_risk_score,
        "prior_linkages": prior_linkages,
        "hour_band": hour_band,
        "est_impact_lakhs": est_impact_lakhs
    })

def load_initial_df():
    if os.path.exists("project_drishti_alerts.csv"):
        return pd.read_csv("project_drishti_alerts.csv"), "Loaded local `project_drishti_alerts.csv`"
    df = generate_fallback_drishti()
    df.to_csv("project_drishti_alerts.csv", index=False)
    return df, "Initialized built-in `project_drishti_alerts.csv` (5,000 synthetic records) — or upload any compatible CSV above."

DEFAULT_DF, INITIAL_STATUS = load_initial_df()

def build_data_summary(df):
    rows, cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and "id" not in c.lower()]
    summary_rows = []
    for c in df.columns:
        dtype_lbl = "Numeric" if c in num_cols else "Categorical/ID"
        missing = int(df[c].isna().sum())
        if c in num_cols:
            detail = f"Mean: {df[c].mean():.2f} | Min: {df[c].min():.2f} | Max: {df[c].max():.2f}"
        else:
            top_vals = df[c].value_counts().head(3).to_dict()
            detail = f"Unique: {df[c].nunique()} | Top: {top_vals}"
        summary_rows.append({"Field Name": c, "Type": dtype_lbl, "Missing": missing, "Distribution Summary": detail})
    return pd.DataFrame(summary_rows)

def handle_file_upload(file_obj):
    if file_obj is None:
        df = DEFAULT_DF.copy()
        msg = INITIAL_STATUS
    else:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        df = pd.read_csv(path)
        msg = f"Uploaded `{os.path.basename(path)}` successfully ({len(df):,} rows × {df.shape[1]} columns)."

    cols = [c for c in df.columns if "id" not in c.lower()]
    default_y = "verified_actionable_flag" if "verified_actionable_flag" in cols else cols[0]
    default_x = [c for c in ["source_stream", "zone_type", "ai_risk_score", "prior_linkages", "hour_band"] if c in cols and c != default_y]
    if not default_x:
        default_x = [c for c in cols if c != default_y][:5]

    return (
        df,
        f"**Dataset Status:** {msg}",
        df.head(10),
        build_data_summary(df),
        gr.Dropdown(choices=cols, value=default_y),
        gr.CheckboxGroup(choices=cols, value=default_x)
    )

# ==============================================================================
# 2. INTERPRETABLE DECISION TREE & LEAF-NODE TRIAGE ENGINE (CASELET 1)
# ==============================================================================
def extract_leaf_profiles(clf, feature_names, X_encoded, y_series, df_orig):
    tree = clf.tree_
    leaf_ids = clf.apply(X_encoded)
    total_n = len(y_series)

    # Reconstruct human-readable path rules for every leaf
    paths = {}
    def recurse(node, rule_parts):
        if tree.feature[node] != -2:
            feat = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            # Clean formatting for one-hot encoded categorical splits
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
            tag = "HIGH-YIELD PRIORITY"
        elif hit_rate >= 25:
            tag = "MODERATE / SELECTIVE"
        else:
            tag = "NOISE / FALSE-ALARM TRAP"

        row_dict = {
            "Leaf ID": int(leaf_id),
            "Command Classification": tag,
            "Alert Volume": cnt,
            "% of Total Traffic": round(traffic_pct, 1),
            "True Hits (Y=1)": hits,
            "Actionable Hit Rate (%)": round(hit_rate, 1),
            "Decision Rule Path": paths.get(leaf_id, "")
        }
        if has_impact:
            row_dict["Avg Est. Impact (₹L)"] = round(avg_imp, 2)
        rows.append(row_dict)

    prof_df = pd.DataFrame(rows).sort_values(by="Actionable Hit Rate (%)", ascending=False)
    return prof_df

def run_decision_tree_audit(df_state, target_y, predictors_x, max_depth, min_samples_leaf, dispatch_cap):
    if df_state is None or len(df_state) == 0:
        return "Please upload or initialize a dataset first.", None, "", pd.DataFrame()

    df = df_state.copy()
    predictors = [c for c in predictors_x if c != target_y and c in df.columns]
    if not predictors:
        return "Select at least one Predictor (X) distinct from Target (Y).", None, "", pd.DataFrame()

    # Prepare binary target
    y_raw = df[target_y]
    if y_raw.dtype == object:
        y = (y_raw.astype(str).isin(["1", "True", "true", "Yes", "yes", "Actionable"])).astype(int)
    else:
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)

    X_df = df[predictors].copy()
    X_encoded = pd.get_dummies(X_df, drop_first=False)
    feature_names = X_encoded.columns.tolist()

    clf = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    clf.fit(X_encoded, y)

    # Leaf profiles
    leaf_df = extract_leaf_profiles(clf, feature_names, X_encoded, y, df)

    # Compare Legacy Vendor Threshold (ai_risk_score >= 50) vs Top Decision Tree Leaves under Dispatch Cap
    total_records = len(df)
    total_true_hits = int(y.sum())
    base_rate = total_true_hits / total_records * 100.0

    vendor_section = ""
    if "ai_risk_score" in df.columns:
        v_mask = df["ai_risk_score"] >= 50
        v_alerts = int(v_mask.sum())
        v_hits = int(y[v_mask].sum())
        v_prec = (v_hits / v_alerts * 100.0) if v_alerts > 0 else 0.0
        vendor_section = f"* **Legacy Vendor Rule (`ai_risk_score >= 50`):** Fires `{v_alerts:,}` alerts (exceeds `{int(dispatch_cap)}` team capacity by `{max(0, v_alerts - int(dispatch_cap)):,}`) | Precision: `{v_prec:.1f}%`\n"

    # Simulate dispatching top-ranked Decision Tree leaves up to dispatch_cap
    leaf_ids = clf.apply(X_encoded)
    leaf_hit_map = dict(zip(leaf_df["Leaf ID"], leaf_df["Actionable Hit Rate (%)"]))
    df["_leaf_rate"] = [leaf_hit_map[lid] for lid in leaf_ids]
    top_dispatched = df.sort_values(by="_leaf_rate", ascending=False).head(int(dispatch_cap))
    dt_hits = int(y.loc[top_dispatched.index].sum())
    dt_prec = (dt_hits / len(top_dispatched) * 100.0) if len(top_dispatched) > 0 else 0.0
    dt_recall = (dt_hits / total_true_hits * 100.0) if total_true_hits > 0 else 0.0

    scorecard_md = f"""### Command Triage & Decision Tree Scorecard
* **Total Incoming Alerts:** `{total_records:,}` | **Overall Base Hit Rate:** `{base_rate:.1f}%` (`{total_true_hits:,}` true actionable cases)
{vendor_section}* **Decision-Tree Prioritized Dispatch (Top `{int(dispatch_cap)}` Capacity):** Catches **`{dt_hits}`** true actionable hits | **Precision: `{dt_prec:.1f}%`** | **Recall: `{dt_recall:.1f}%`**
* **Actionable Takeaway:** Inspect the **Leaf-Node Profiles Table** below to isolate high-yield rules vs. high-volume false-alarm traps.
"""

    # Plot Classical Decision Tree Chart
    fig, ax = plt.subplots(figsize=(15, 7), dpi=130)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Noise (0)", "Actionable (1)"],
        filled=True,
        rounded=True,
        proportion=True,
        precision=1,
        fontsize=8,
        ax=ax
    )
    ax.set_title(f"Interpretable Command Decision Tree (Target: {target_y} | Max Depth: {int(max_depth)})", fontsize=12, fontweight="bold", color="#002D72")
    plt.tight_layout()

    # ASCII Rule Wireframe
    ascii_rules = export_text(clf, feature_names=feature_names, decimals=2)

    return scorecard_md, fig, ascii_rules, leaf_df

# ==============================================================================
# 3. CASELET 2: MULTIMODAL VLM SENTINEL & BEHAVIORAL CUE DEMOS
# ==============================================================================
def image_to_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def create_demo_cctv_frame():
    img = Image.new("RGB", (640, 360), color=(28, 34, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([35, 55, 605, 325], outline=(110, 130, 150), width=3)
    draw.text((50, 22), "CAM-04 | MALKHANA & ARMOURY PERIMETER | 02:14:33 IST", fill=(230, 238, 245))
    draw.rectangle([425, 230, 495, 285], fill=(185, 50, 40), outline=(255, 205, 75), width=2)
    draw.text((400, 294), "[STATIONARY OBJECT > 180s]", fill=(255, 205, 75))
    return img

def analyze_vlm_frame(image_input, custom_prompt, api_key, model_name):
    if image_input is None:
        pil_img = create_demo_cctv_frame()
    else:
        pil_img = Image.fromarray(image_input) if isinstance(image_input, np.ndarray) else image_input

    if api_key and api_key.strip():
        try:
            b64 = image_to_base64(pil_img)
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name.strip() or "google/gemini-2.5-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }]
            }
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
            data = resp.json()
            if "choices" in data:
                return pil_img, data["choices"][0]["message"]["content"]
            return pil_img, f"**API Notice:** `{json.dumps(data)}`"
        except Exception as e:
            return pil_img, f"**Network/API Error:** `{e}`. Showing offline classroom output below."

    fallback_md = """### Automated VLM Perimeter Sentinel Output (Classroom Simulation Mode)
* **Alert Status:** `ANOMALY FLAGGED — PRIORITY 2 (Malkhana Perimeter)`
* **Visual Reasoning:** Unattended rectangular package detected near perimeter gate at `02:14 IST` with zero authorized personnel visible for >3 minutes.
* **Recommended Action:** Route snapshot to Station Duty Officer tablet for 30-second visual confirmation before dispatching QRT.
* **Governance Caveat:** *Model confidence: 81%. Low-light sensor noise or static equipment can mimic unattended objects—Human-in-the-Loop (HITL) verification is mandatory.*"""
    return pil_img, fallback_md

def analyze_interview_cues(scenario_choice):
    logs = {
        "Scenario A: Suspect Alibi Timeline (EOW Shell-Company Inquiry)": """### Multimodal Behavioral & Vocal Cue Log (90-Sec Clip Analysis)
* **00:08 – 00:20 (Baseline):** Steady gaze, neutral facial muscle tension, normal blink rate (~14/min) while confirming company registration details.
* **00:34 – 00:42 (Flagged Cognitive Load Spike):** Question asked regarding *₹1.8 Cr cash transfer on Nov 14*. Blink rate rises to 28/min, 1.9-second response latency, subtle asymmetric shoulder shrug.
* **01:05 – 01:14 (Vocal Stress Shift):** Fundamental vocal pitch rises +17% when asked about contact with co-accused director.
* **Command Governance Rule:** *Physiological/behavioral cues indicate cognitive load or stress—NOT deception. Use solely to prioritize documentary audit (bank statements, CDR tower dumps), never as standalone proof.*""",
        "Scenario B: Eyewitness Recall Under Trauma (Highway Hit-and-Run)": """### Multimodal Behavioral & Vocal Cue Log (90-Sec Clip Analysis)
* **00:05 – 00:24 (Acute Stress Baseline):** Rapid breathing, lip compression, and elevated vocal tremor throughout narrative.
* **00:41 – 00:53 (Memory Retrieval Pattern):** Gaze aversion upward-left while recalling partial vehicle registration digits.
* **Command Governance Rule:** *Off-the-shelf emotion/stress classifiers frequently misclassify acute trauma, fatigue, or neurodivergence as 'evasion'. Human investigator judgment must override raw scores.*"""
    }
    return logs.get(scenario_choice, "")

# ==============================================================================
# 4. GRADIO MULTI-TAB WORKSPACE
# ==============================================================================
with gr.Blocks(title="SVPNPA Phase III — Data Driven Decision Making Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # SVPNPA 28th MCTP Phase III: Data-Driven Decision Making Lab
    **Faculty:** Prof. Sudhir Voleti | **Caselet 1:** *Project Drishti (Command Decision Tree & Triage)* | **Caselet 2:** *Operation Netra (VLM Sentinels & HITL Guardrails)*
    """)

    df_state = gr.State(DEFAULT_DF)

    with gr.Tabs():
        # TAB 1: UPLOAD & DATA ARCHITECTURE PREVIEW
        with gr.Tab("1. Data Upload & Architecture Preview"):
            gr.Markdown("Upload `project_drishti_alerts.csv` (or any compatible operational CSV). By default, the 5,000-row Project Drishti dataset is preloaded for immediate classroom use.")
            with gr.Row():
                csv_upload = gr.File(label="Upload Operational CSV File (.csv)", file_types=[".csv"])
                status_md = gr.Markdown(f"**Dataset Status:** {INITIAL_STATUS}")
            gr.Markdown("### 10-Row Data Preview")
            preview_df = gr.Dataframe(value=DEFAULT_DF.head(10), interactive=False)
            gr.Markdown("### Variable Architecture & Descriptive Summary")
            summary_df = gr.Dataframe(value=build_data_summary(DEFAULT_DF), interactive=False)

        # TAB 2: CASELET 1 — INTERPRETABLE DECISION TREE & TRIAGE AUDIT
        with gr.Tab("2. Caselet 1: Decision Tree & Command Triage"):
            gr.Markdown("Map your **Command Target (Y)** and **Operational Predictors (X)** below, then click **Run Decision Tree & Triage Audit** to isolate high-yield rules vs. false-alarm traps.")
            with gr.Row():
                with gr.Column(scale=1):
                    y_selector = gr.Dropdown(
                        choices=[c for c in DEFAULT_DF.columns if "id" not in c.lower()],
                        value="verified_actionable_flag",
                        label="Command Target (Y)"
                    )
                    x_selector = gr.CheckboxGroup(
                        choices=[c for c in DEFAULT_DF.columns if "id" not in c.lower()],
                        value=["source_stream", "zone_type", "ai_risk_score", "prior_linkages", "hour_band"],
                        label="Operational Predictors (X1 ... Xn)"
                    )
                    depth_slider = gr.Slider(2, 5, value=3, step=1, label="Tree Depth (Rule Complexity)")
                    leaf_slider = gr.Slider(25, 250, value=75, step=25, label="Min Alerts per Leaf Node")
                    cap_slider = gr.Slider(100, 600, value=300, step=25, label="Max Field Dispatch Capacity (20 Teams × 15 = 300)")
                    run_tree_btn = gr.Button("Run Decision Tree & Triage Audit", variant="primary")

                with gr.Column(scale=2):
                    scorecard_out = gr.Markdown()
                    tree_plot_out = gr.Plot()

            gr.Markdown("### Leaf-Node Risk & Traffic Profile Table")
            leaf_table_out = gr.Dataframe(interactive=False)

            with gr.Accordion("ASCII Decision Rule Wireframe (Text Export)", open=False):
                ascii_out = gr.Code(language="markdown")

        # TAB 3: CASELET 2A — COMMODITY CCTV + VLM SENTINEL
        with gr.Tab("3. Caselet 2A: Smart Chowki VLM Sentinel"):
            gr.Markdown("Demonstrates pairing standard CCTV/webcam frames with an open-ended Vision-Language Model prompt to filter genuine anomalies.")
            with gr.Row():
                with gr.Column(scale=1):
                    cam_in = gr.Image(sources=["upload", "webcam"], type="pil", label="Upload Snapshot or Capture Webcam Frame (Leave blank for Malkhana Demo Frame)")
                    prompt_in = gr.Textbox(
                        value="You are an automated police duty-officer assistant monitoring a sensitive Malkhana/perimeter gate. Flag only genuine operational anomalies (unattended baggage, perimeter breach, fire/smoke, crowd buildup) and suppress routine activity. State confidence and caveats.",
                        lines=4, label="System Prompt to VLM"
                    )
                    with gr.Accordion("Optional Live VLM API Settings (OpenRouter)", open=False):
                        api_key_in = gr.Textbox(type="password", label="OpenRouter API Key (Leave blank for offline simulation mode)")
                        model_in = gr.Textbox(value="google/gemini-2.5-flash", label="VLM Model ID")
                    vlm_run_btn = gr.Button("Analyze Frame with VLM", variant="primary")
                with gr.Column(scale=1):
                    vlm_img_out = gr.Image(label="Inspected Perimeter Frame")
                    vlm_md_out = gr.Markdown()

        # TAB 4: CASELET 2B — INTERVIEW / WITNESS BEHAVIORAL CUE TRIAGE
        with gr.Tab("4. Caselet 2B: Interview Cue Triage & HITL Guardrails"):
            gr.Markdown("Compare multimodal stress/gaze logs across two investigative contexts to establish strict **Human-in-the-Loop (HITL)** rules.")
            with gr.Row():
                with gr.Column(scale=1):
                    scen_in = gr.Radio(
                        [
                            "Scenario A: Suspect Alibi Timeline (EOW Shell-Company Inquiry)",
                            "Scenario B: Eyewitness Recall Under Trauma (Highway Hit-and-Run)"
                        ],
                        value="Scenario A: Suspect Alibi Timeline (EOW Shell-Company Inquiry)",
                        label="Select Pre-Analyzed Video Scenario"
                    )
                    cue_run_btn = gr.Button("Inspect VLM Behavioral Timeline", variant="primary")
                with gr.Column(scale=2):
                    cue_md_out = gr.Markdown()

    # Wire Event Listeners
    csv_upload.change(
        fn=handle_file_upload,
        inputs=[csv_upload],
        outputs=[df_state, status_md, preview_df, summary_df, y_selector, x_selector]
    )

    run_tree_btn.click(
        fn=run_decision_tree_audit,
        inputs=[df_state, y_selector, x_selector, depth_slider, leaf_slider, cap_slider],
        outputs=[scorecard_out, tree_plot_out, ascii_out, leaf_table_out]
    )

    demo.load(
        fn=run_decision_tree_audit,
        inputs=[df_state, y_selector, x_selector, depth_slider, leaf_slider, cap_slider],
        outputs=[scorecard_out, tree_plot_out, ascii_out, leaf_table_out]
    )

    vlm_run_btn.click(
        fn=analyze_vlm_frame,
        inputs=[cam_in, prompt_in, api_key_in, model_in],
        outputs=[vlm_img_out, vlm_md_out]
    )

    cue_run_btn.click(
        fn=analyze_interview_cues,
        inputs=[scen_in],
        outputs=[cue_md_out]
    )

demo.launch(share=True)
