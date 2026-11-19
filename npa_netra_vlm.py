# npa_netra_vlm.py
# SVPNPA 28th MCTP Phase III — Caselet 2: Operation Netra (Commodity VLM Sentinels & HITL Guardrails)
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Colab Launcher:
#   import requests
#   url = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_netra_vlm.py"
#   response = requests.get(url); response.raise_for_status()
#   exec(compile(response.text, url, "exec"), globals())

import io
import json
import base64
from PIL import Image, ImageDraw
import requests
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

TABLE_CSS = """
<style>
    .sv-card { background: #f4f6f9; border-left: 5px solid #002D72; padding: 14px 18px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; line-height: 1.6; }
    .sv-warn { background: #fffdf5; border-left: 5px solid #CFB87C; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; }
</style>
"""

def create_demo_cctv_frame():
    img = Image.new("RGB", (640, 360), color=(28, 34, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([35, 55, 605, 325], outline=(110, 130, 150), width=3)
    draw.text((50, 22), "CAM-04 | MALKHANA & ARMOURY PERIMETER | 02:14:33 IST", fill=(230, 238, 245))
    draw.rectangle([425, 230, 495, 285], fill=(185, 50, 40), outline=(255, 205, 75), width=2)
    draw.text((400, 294), "[STATIONARY OBJECT > 180s]", fill=(255, 205, 75))
    return img

# ==============================================================================
# TAB 1: SMART CHOWKI / MALKHANA VLM SENTINEL
# ==============================================================================
img_upload = widgets.FileUpload(accept="image/*", multiple=False, description="Upload Frame (Opt)")
vlm_prompt = widgets.Textarea(
    value="You are an automated police duty-officer assistant monitoring a sensitive Malkhana/perimeter gate. Flag only genuine operational anomalies (unattended baggage, perimeter breach, fire/smoke, crowd buildup) and suppress routine activity. State confidence and caveats.",
    description="Prompt:",
    layout=widgets.Layout(width="680px", height="85px")
)
api_key_in = widgets.Password(placeholder="Optional OpenRouter API Key (leave blank for offline demo)", description="API Key:", layout=widgets.Layout(width="420px"))
model_in = widgets.Text(value="google/gemini-2.5-flash", description="VLM Model:", layout=widgets.Layout(width="420px"))
vlm_btn = widgets.Button(description="Run VLM Perimeter Analysis", button_style="primary", icon="eye")
tab1_out = widgets.Output()

def on_run_vlm(_):
    with tab1_out:
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

tab1_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>1. Prototype A: Smart Chowki / Malkhana VLM Sentinel</h3><p>Leave image blank to inspect the built-in Malkhana night-perimeter test frame, or upload any camera snapshot.</p>"),
    img_upload, vlm_prompt, widgets.HBox([api_key_in, model_in]), vlm_btn, tab1_out
])

# ==============================================================================
# TAB 2: INTERVIEW & WITNESS BEHAVIORAL CUE TRIAGE
# ==============================================================================
scen_radio = widgets.RadioButtons(
    options=[
        "Scenario A: Suspect Alibi Timeline (EOW Shell-Company Inquiry)",
        "Scenario B: Eyewitness Recall Under Trauma (Highway Hit-and-Run)"
    ],
    layout=widgets.Layout(width="550px")
)
cue_btn = widgets.Button(description="Inspect VLM Behavioral Timeline", button_style="primary", icon="search")
tab2_out = widgets.Output()

def on_run_cue(_):
    with tab2_out:
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

tab2_box = widgets.VBox([
    widgets.HTML("<h3 style='color:#002D72; margin-bottom:4px;'>2. Prototype B: Interview Behavioral Cue Triage &amp; HITL Guardrails</h3>"),
    scen_radio, cue_btn, tab2_out
])

tabs = widgets.Tab(children=[tab1_box, tab2_box])
for i, title in enumerate(["1. Smart Chowki VLM Sentinel", "2. Interview Cue Triage"]):
    tabs.set_title(i, title)

header_banner = widgets.HTML("""
<div style="background:#002D72; color:#ffffff; padding:16px 22px; border-radius:6px; border-bottom:4px solid #CFB87C; font-family:'Segoe UI', Arial, sans-serif; margin-bottom:12px;">
    <h2 style="margin:0; font-size:19px;">SVPNPA 28th MCTP Phase III &mdash; Caselet 2: Operation Netra (Multimodal VLM Lab)</h2>
    <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">Faculty: Prof. Sudhir Voleti | ISB Executive Education | Commodity Prototyping &amp; HITL Governance</div>
</div>
""")

display(widgets.VBox([header_banner, tabs]))
