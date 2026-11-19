# npa_netra_vlm.py
# SVPNPA 28th MCTP Phase III — Caselet 2: Video Frame Sampling & Gemini VLM Sentiment Timeline
# Faculty: Prof. Sudhir Voleti | ISB Executive Education
# Colab Launcher:
#   import requests
#   url = "https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/npa_netra_vlm.py"
#   response = requests.get(url); response.raise_for_status()
#   exec(compile(response.text, url, "exec"), globals())

import os
import io
import json
import base64
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import requests
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

TABLE_CSS = """
<style>
    .sv-table { border-collapse: collapse; width: 100%; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; margin: 10px 0; }
    .sv-table th { background-color: #002D72; color: #ffffff; padding: 8px 10px; text-align: left; border-bottom: 2px solid #CFB87C; }
    .sv-table td { padding: 7px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
    .sv-table tr:nth-child(even) { background-color: #f8fafc; }
    .sv-card { background: #f4f6f9; border-left: 5px solid #002D72; padding: 14px 18px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; line-height: 1.6; }
    .sv-warn { background: #fffdf5; border-left: 5px solid #CFB87C; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13.5px; }
    .sv-dl-btn { display: inline-block; background-color: #002D72; color: #ffffff !important; text-decoration: none; padding: 8px 14px; border-radius: 5px; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; font-weight: 600; border-bottom: 3px solid #CFB87C; margin-top: 8px; }
</style>
"""

def format_ts(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def sample_video_frames(video_bytes, interval_sec=5, max_frames=12):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_frames / fps if fps > 0 else 0

        t = 0.0
        while t <= max(duration_sec, 0.1) and len(frames) < max_frames:
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Resize cleanly for fast VLM upload
            pil_img.thumbnail((512, 512))
            frames.append((format_ts(t), pil_img))
            t += interval_sec
        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return frames

def build_filmstrip_image(frames):
    if not frames:
        return None
    thumb_w, thumb_h = 220, 150
    cols = min(len(frames), 4)
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (thumb_w + 10) + 10, rows * (thumb_h + 28) + 10), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    for idx, (ts, img) in enumerate(frames):
        r, c = divmod(idx, cols)
        x0 = 10 + c * (thumb_w + 10)
        y0 = 10 + r * (thumb_h + 28)
        resized = img.resize((thumb_w, thumb_h))
        canvas.paste(resized, (x0, y0))
        draw.rectangle([x0, y0 + thumb_h, x0 + thumb_w, y0 + thumb_h + 20], fill=(0, 45, 114))
        draw.text((x0 + 8, y0 + thumb_h + 4), f"Timestamp: {ts}", fill=(255, 255, 255))
    return canvas

def call_gemini_vlm_batch(frames, api_key, model_id, custom_instruction):
    """
    Sends all sampled video frames in ONE structured Gemini API request so it completes
    in ~3 seconds and stays well within Free Tier RPM limits.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key.strip()}"

    ts_list = [ts for ts, _ in frames]
    prompt_text = (
        f"{custom_instruction}\n\n"
        f"You are given {len(frames)} sequential video frames sampled at timestamps: {', '.join(ts_list)}.\n"
        "Return ONLY a valid JSON array with one object per frame in exact chronological order, using these exact keys:\n"
        '[{"timestamp": "00:00", "sentiment": "Calm / Neutral", "facial_cues": "Steady eye contact, relaxed brow", "shift_note": "Baseline demeanor", "confidence": "High"}]'
    )

    parts = [{"text": prompt_text}]
    for ts, pil_img in frames:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({"text": f"[Frame at {ts}]"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    resp = requests.post(url, json=payload, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(raw_text)
    return pd.DataFrame(parsed)

def get_offline_demo_df():
    return pd.DataFrame([
        {"timestamp": "00:00", "sentiment": "Calm / Composed", "facial_cues": "Steady eye contact, neutral mouth posture, upright head", "shift_note": "Baseline demeanor during introductory question", "confidence": "High"},
        {"timestamp": "00:05", "sentiment": "Attentive / Neutral", "facial_cues": "Slight nod, relaxed brow, consistent gaze", "shift_note": "No significant affective shift from baseline", "confidence": "High"},
        {"timestamp": "00:10", "sentiment": "Tense / Guarded", "facial_cues": "Brow furrow, lip compression, brief downward gaze shift", "shift_note": "Noticeable tension spike when specific date/transaction raised", "confidence": "Medium"},
        {"timestamp": "00:15", "sentiment": "Anxious / Cognitive Load", "facial_cues": "Upward-left eye movement, tight jaw, raised eyebrows", "shift_note": "High cognitive effort / recall strain (requires documentary check)", "confidence": "Medium"},
        {"timestamp": "00:20", "sentiment": "Defensive / Frustrated", "facial_cues": "Asymmetric smirk/grimace, rapid blink sequence", "shift_note": "Emotional arousal persists during follow-up probe", "confidence": "Medium"},
        {"timestamp": "00:25", "sentiment": "Recovering / Neutral", "facial_cues": "Exhale, shoulders drop, return to steady gaze", "shift_note": "Returning toward baseline composure", "confidence": "High"}
    ])

# ==============================================================================
# WIDGET UI LAYOUT
# ==============================================================================
video_upload = widgets.FileUpload(accept=".mp4,.mov,.avi", multiple=False, description="Upload .mp4 Clip", button_style="primary")
interval_slider = widgets.IntSlider(value=5, min=2, max=15, step=1, description="Sample Every (s):", layout=widgets.Layout(width="340px"))
gemini_key_in = widgets.Password(
    placeholder="Paste Free Gemini API Key (aistudio.google.com/apikey) or leave blank for Demo Mode",
    description="Gemini Key:",
    layout=widgets.Layout(width="540px")
)
model_select = widgets.Dropdown(
    options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
    value="gemini-2.5-flash",
    description="VLM Model:",
    layout=widgets.Layout(width="340px")
)
prompt_box = widgets.Textarea(
    value="Analyze the primary subject's facial expression, visible sentiment, and shifts in demeanor across these sequential interview frames. Remain objective: describe observable facial/gaze cues rather than claiming deception.",
    description="VLM Prompt:",
    layout=widgets.Layout(width="680px", height="75px")
)
run_btn = widgets.Button(description="Sample Frames & Analyze Sentiment Timeline", button_style="success", icon="play", layout=widgets.Layout(width="360px"))
out_area = widgets.Output()

with out_area:
    display(HTML(TABLE_CSS + "<div class='sv-warn'><b>Ready:</b> Upload any short <code>.mp4</code> interview clip (20–40 sec) and paste a free Gemini API key from <code>aistudio.google.com/apikey</code>—or leave both blank and click <b>Run</b> to preview the built-in classroom simulation.</div>"))

def on_analyze_click(_):
    with out_area:
        clear_output(wait=True)
        api_key = gemini_key_in.value.strip()
        interval = interval_slider.value

        # Case 1: Video uploaded by user
        if video_upload.value:
            if isinstance(video_upload.value, dict):
                fname = list(video_upload.value.keys())[0]
                v_bytes = video_upload.value[fname]["content"]
            else:
                item = video_upload.value[0]
                fname, v_bytes = item["name"], item["content"]

            display(HTML(f"{TABLE_CSS}<div class='sv-card'>Sampling frames from <code>{fname}</code> every <b>{interval} seconds</b> via OpenCV...</div>"))
            frames = sample_video_frames(v_bytes, interval_sec=interval)
            if not frames:
                display(HTML(f"{TABLE_CSS}<div class='sv-warn'>Could not extract frames from <code>{fname}</code>. Ensure it is a valid MP4 file.</div>"))
                return

            filmstrip = build_filmstrip_image(frames)
            if filmstrip:
                display(filmstrip)

            if api_key:
                try:
                    df_res = call_gemini_vlm_batch(frames, api_key, model_select.value, prompt_box.value)
                    status_banner = f"<div class='sv-card'><b>Live Gemini VLM Readout ({model_select.value}):</b> Analyzed <b>{len(frames)}</b> sampled frames from <code>{fname}</code> in a single batch call.</div>"
                except Exception as e:
                    df_res = get_offline_demo_df().head(len(frames))
                    df_res["timestamp"] = [ts for ts, _ in frames]
                    status_banner = f"<div class='sv-warn'><b>API Notice ({e}):</b> Displaying simulated sentiment readout aligned to your {len(frames)} extracted timestamps.</div>"
            else:
                df_res = get_offline_demo_df().head(len(frames))
                df_res["timestamp"] = [ts for ts, _ in frames]
                status_banner = f"<div class='sv-warn'><b>No Gemini Key Entered (Classroom Simulation Mode):</b> Extracted <b>{len(frames)}</b> real frames from your video above; paste a free key from <code>aistudio.google.com/apikey</code> to run live VLM inference on them.</div>"
        else:
            # Case 2: No video uploaded -> Classroom simulation mode
            df_res = get_offline_demo_df()
            status_banner = "<div class='sv-warn'><b>Built-In Classroom Simulation Mode:</b> Upload any <code>.mp4</code> clip above to sample live frames via OpenCV. Below is a sample 30-second witness interview sentiment readout (sampled every 5s):</div>"

        csv_b64 = base64.b64encode(df_res.to_csv(index=False).encode("utf-8")).decode("utf-8")
        dl_link = f"<a class='sv-dl-btn' download='vlm_sentiment_timeline.csv' href='data:text/csv;base64,{csv_b64}'>&#11015; Download Timestamped Readout (.csv)</a>"

        governance_box = """
        <div class='sv-card' style='margin-top:14px;'>
            <b>Facilitator &amp; Command Governance Takeaway:</b>
            <ul style='margin:6px 0 0 0; padding-left:20px;'>
                <li><b>Foundation VLM vs. Custom ML:</b> Zero-shot VLMs extract rapid sentiment timelines out of the box; for domain-specific accuracy (e.g., fatigue, acute trauma vs. calm recall), police labs can fine-tune supervised ML models on labeled frame embeddings.</li>
                <li><b>HITL Boundary:</b> Facial sentiment flags <i>emotional/cognitive shifts</i>, NOT deception—use strictly to prioritize documentary verification, never as standalone evidence.</li>
            </ul>
        </div>
        """

        display(HTML(TABLE_CSS + status_banner + df_res.to_html(index=False, classes="sv-table") + dl_link + governance_box))

run_btn.on_click(on_analyze_click)

header_banner = widgets.HTML("""
<div style="background:#002D72; color:#ffffff; padding:16px 22px; border-radius:6px; border-bottom:4px solid #CFB87C; font-family:'Segoe UI', Arial, sans-serif; margin-bottom:12px;">
    <h2 style="margin:0; font-size:19px;">SVPNPA 28th MCTP Phase III &mdash; Caselet 2: Video Frame Sampling &amp; VLM Sentiment Timeline</h2>
    <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">Faculty: Prof. Sudhir Voleti | ISB Executive Education | Unstructured Video-to-Structured Timeline Demo</div>
</div>
""")

controls_box = widgets.VBox([
    widgets.HBox([video_upload, interval_slider]),
    widgets.HBox([gemini_key_in, model_select]),
    prompt_box,
    run_btn
])

display(widgets.VBox([header_banner, controls_box, out_area]))
