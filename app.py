"""VisionVoice - premium assistive dashboard.

    streamlit run app.py

Live camera (webcam / phone IP stream / browser WebRTC) or an uploaded photo is captioned by the
ResNet-50 + LSTM model and spoken aloud by the local TTS engine. The UI thread only paints frames;
captioning and speech run on background threads, so the video never stalls.
"""
from __future__ import annotations

import html
import io
import hashlib 
import json
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
from utils.audio_engine import get_speech_engine
from utils.live_captioner import CaptionEvent, LiveCaptioner
from utils.video_stream import VideoStream, normalise_phone_url, parse_source

try:  # optional: live camera straight from a phone / laptop browser
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    HAS_WEBRTC = True
except Exception:  # noqa: BLE001
    HAS_WEBRTC = False

st.set_page_config(page_title="VisionVoice", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
THEMES = {
    "Midnight Gold": dict(bg="#07080B", surface="#101319", surface2="#171B23", border="#2A303C",
                          text="#F5F0E6", muted="#A9A596", accent="#E2B857", accent2="#58CDB6", on_accent="#0A0A0A"),
    "High Contrast": dict(bg="#000000", surface="#0B0B0B", surface2="#151515", border="#FFFFFF",
                          text="#FFFFFF", muted="#E6E6E6", accent="#FFE600", accent2="#00F0FF", on_accent="#000000"),
    "Emerald Noir": dict(bg="#050A09", surface="#0C1513", surface2="#111E1B", border="#23362F",
                         text="#EAF5EF", muted="#9DB5AB", accent="#5EE0B0", accent2="#E8C872", on_accent="#04120C"),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');
:root{__VARS__}
html{font-size:calc(16px * var(--ui-scale))}
.stApp{background:radial-gradient(1100px 560px at 10% -8%,color-mix(in srgb,var(--accent) 13%,transparent),transparent 62%),
 radial-gradient(900px 520px at 100% 0%,color-mix(in srgb,var(--accent2) 10%,transparent),transparent 58%),var(--bg);
 color:var(--text);font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif}
header[data-testid="stHeader"]{background:color-mix(in srgb,var(--bg) 72%,transparent);backdrop-filter:blur(10px)}
.block-container{padding-top:2rem;max-width:1560px}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:var(--text)}
h1,h2,h3,h4{font-family:'Fraunces',Georgia,serif;letter-spacing:-.01em;color:var(--text)}
p,label,li,span,.stMarkdown{font-size:1rem}
:focus-visible{outline:3px solid var(--accent)!important;outline-offset:3px!important;border-radius:10px}
.stButton>button,.stDownloadButton>button{background:var(--surface2);color:var(--text);border:1px solid var(--border);
 border-radius:14px;min-height:3rem;font-weight:700;font-size:1rem;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{height:3rem;border-radius:12px 12px 0 0;color:var(--muted);font-weight:700}
.stTabs [aria-selected="true"]{color:var(--accent)!important}
[data-testid="stImage"] img{border-radius:22px;border:1px solid var(--border);box-shadow:0 24px 60px -30px #000}
.vv-hero{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:1rem 2rem;margin-bottom:1.4rem}
.vv-hero>div:first-child{flex:1 1 420px}
.vv-pills{justify-content:flex-end}
.vv-title{font-family:'Fraunces',Georgia,serif;font-size:clamp(2.4rem,4.6vw,3.8rem);font-weight:600;line-height:1.05;
 background:linear-gradient(100deg,var(--text) 10%,var(--accent) 90%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent}
.vv-h3{font-family:'Fraunces',Georgia,serif;font-size:1.6rem;font-weight:600;color:var(--text);margin:.2rem 0 .5rem}
.vv-sub{color:var(--muted);margin:.5rem 0 0;font-size:1.05rem}
.vv-pills{display:flex;flex-wrap:wrap;gap:.5rem}
.vv-pill{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .85rem;border:1px solid var(--border);border-radius:999px;
 background:color-mix(in srgb,var(--surface2) 80%,transparent);font-size:.85rem;font-weight:600;color:var(--muted)}
.vv-pill b{color:var(--text)}
.vv-dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;background:var(--accent2);box-shadow:0 0 0 0 var(--accent2)}
.vv-dot.live{background:#FF5A5F;animation:vvpulse 1.6s infinite}
.vv-dot.off{background:var(--muted);animation:none}
@keyframes vvpulse{0%{box-shadow:0 0 0 0 rgba(255,90,95,.55)}100%{box-shadow:0 0 0 .7rem rgba(255,90,95,0)}}
.vv-card{background:linear-gradient(180deg,var(--surface2),var(--surface));border:1px solid var(--border);border-radius:24px;
 padding:1.5rem 1.7rem;box-shadow:0 30px 70px -40px #000}
.vv-eyebrow{text-transform:uppercase;letter-spacing:.18em;font-size:.72rem;font-weight:700;color:var(--accent);margin-bottom:.7rem;
 display:flex;justify-content:space-between;align-items:center}
.vv-caption{font-family:'Fraunces',Georgia,serif;font-size:var(--caption-size);line-height:1.18;font-weight:500;margin:.2rem 0 1.1rem;
 min-height:3.6em;word-break:break-word}
.vv-caption.idle{color:var(--muted);font-weight:400}
.vv-wave{display:flex;align-items:center;gap:4px;height:3.4rem;margin:.4rem 0 1rem}
.vv-wave span{flex:1;height:100%;border-radius:999px;background:linear-gradient(var(--accent),var(--accent2));
 transform:scaleY(.14);opacity:.45;transform-origin:center}
.vv-wave.on span{opacity:1;animation:vvwave .9s ease-in-out calc(var(--i) * -.08s) var(--iters,infinite)}
@keyframes vvwave{0%,100%{transform:scaleY(.14)}50%{transform:scaleY(var(--h,1))}}
.vv-wave.on.calm span{animation:none;transform:scaleY(var(--h,1))}
@media (prefers-reduced-motion:reduce){.vv-wave.on span{animation:none;transform:scaleY(var(--h,1))}.vv-dot.live{animation:none}}
.vv-chips{display:flex;flex-wrap:wrap;gap:.45rem}
.vv-chip{padding:.28rem .7rem;border-radius:999px;border:1px solid var(--border);font-size:.8rem;color:var(--muted);font-weight:600}
.vv-chip.hot{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 55%,var(--border))}
.vv-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:1rem}
.vv-kpi{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:.9rem 1rem}
.vv-kpi small{display:block;color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700}
.vv-kpi strong{font-family:'Fraunces',Georgia,serif;font-size:1.7rem;font-weight:600}
.vv-kpi em{font-style:normal;color:var(--muted);font-size:.85rem;margin-left:.25rem}
.vv-stage{aspect-ratio:16/9;display:grid;place-items:center;text-align:center;border:1px dashed var(--border);border-radius:24px;
 color:var(--muted);background:repeating-linear-gradient(135deg,transparent 0 22px,color-mix(in srgb,var(--border) 35%,transparent) 22px 23px)}
.vv-status{display:flex;gap:1.2rem;flex-wrap:wrap;color:var(--muted);font-size:.85rem;margin:.6rem .2rem 1rem;font-weight:600}
.vv-status b{color:var(--text)}
.vv-flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem}
.vv-node{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:.9rem 1rem}
.vv-node small{color:var(--accent);text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-size:.68rem}
.vv-node b{display:block;margin:.25rem 0;font-size:1.02rem}
.vv-node span{color:var(--muted);font-size:.85rem}
.vv-steps code{background:var(--surface);border:1px solid var(--border);padding:.15rem .45rem;border-radius:8px}
"""

WAVE_HEIGHTS = [0.35 + 0.65 * abs(np.sin(i * 0.9) * np.cos(i * 0.37)) for i in range(32)]


def inject_css(theme: dict, caption_px: int, ui_scale: float, calm: bool) -> None:
    variables = (f"--bg:{theme['bg']};--surface:{theme['surface']};--surface2:{theme['surface2']};--border:{theme['border']};"
                 f"--text:{theme['text']};--muted:{theme['muted']};--accent:{theme['accent']};--accent2:{theme['accent2']};"
                 f"--on-accent:{theme['on_accent']};--caption-size:{caption_px}px;--ui-scale:{ui_scale:.2f}")
    st.markdown(f"<style>{CSS.replace('__VARS__', variables)}</style>", unsafe_allow_html=True)


def block(markup: str) -> str:
    """Collapse whitespace so Markdown never mistakes indented HTML for a code block."""
    return re.sub(r">\s+<", "><", re.sub(r"\s*\n\s*", " ", markup)).strip()


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
def artifacts_ready() -> bool:
    return config.MODEL_PATH.exists() and config.VOCAB_PATH.exists()


@st.cache_resource(show_spinner="Loading ResNet-50 encoder and LSTM decoder ...")
def load_services() -> SimpleNamespace:
    from model import CaptionGenerator

    speech = get_speech_engine()
    generator = CaptionGenerator(load_cnn=True)
    generator.warmup()
    captioner = LiveCaptioner(generator, speech)
    captioner.start()
    return SimpleNamespace(speech=speech, captioner=captioner, generator=generator, stream=None, stream_is_new=False)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json(path: Path) -> Optional[dict]:
    return _read_json(str(path), path.stat().st_mtime) if path.exists() else None


def ensure_stream(sv: SimpleNamespace, source) -> VideoStream:
    """Reuse the running camera across Streamlit reruns; reopen only when the source changed."""
    current = sv.stream
    if current is not None and current.source == source and current.alive:
        sv.stream_is_new = False
        return current
    release_stream(sv)
    sv.stream = VideoStream(source).start()
    sv.stream_is_new = True
    return sv.stream


def release_stream(sv: SimpleNamespace) -> None:
    if sv.stream is not None:
        sv.stream.stop()
        sv.stream = None


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #
def hero_html(ready: bool, voice: str, source: str, live: bool) -> str:
    dot = "live" if live else ("" if ready else "off")
    state = "Live" if live else ("Ready" if ready else "Setup needed")
    return block(f"""
    <div class="vv-hero"><div>
      <div class="vv-title" role="heading" aria-level="1">VisionVoice</div>
      <p class="vv-sub">See the world, spoken aloud. Real-time scene description for people who cannot see it.</p></div>
      <div class="vv-pills" role="status">
        <span class="vv-pill"><i class="vv-dot {dot}"></i><b>{state}</b></span>
        <span class="vv-pill">Voice <b>{html.escape(voice)}</b></span>
        <span class="vv-pill">Source <b>{html.escape(source)}</b></span>
        <span class="vv-pill">ResNet-50 <b>+</b> LSTM</span></div></div>""")


def stage_html(title: str, detail: str) -> str:
    return block(f'<div class="vv-stage"><div><div class="vv-h3" role="heading" aria-level="3">{html.escape(title)}</div><p>{html.escape(detail)}</p></div></div>')


def caption_card_html(event: Optional[CaptionEvent], speaking: bool, voice_on: bool, calm: bool,
                      busy: bool, iterations: str = "infinite") -> str:
    bars = "".join(f'<span style="--i:{i};--h:{h:.2f}"></span>' for i, h in enumerate(WAVE_HEIGHTS))
    if event is None:
        text, css = "Waiting for the first scene ...", " idle"
        chips = ""
    else:
        text, css = html.escape(event.caption), ""
        chips = (f'<span class="vv-chip hot">{event.latency_ms:.0f} ms</span>'
                 f'<span class="vv-chip">beam {event.beam_width}</span>'
                 f'<span class="vv-chip">{html.escape(event.source)}</span>'
                 f'<span class="vv-chip">{datetime.fromtimestamp(event.timestamp):%H:%M:%S}</span>')
    voice = "Speaking" if speaking else ("Voice on" if voice_on else "Voice muted")
    wave = f'vv-wave{" on" if speaking else ""}{" calm" if calm else ""}'
    thinking = '<span class="vv-chip hot">thinking ...</span>' if busy else ""
    return block(f"""
    <div class="vv-card">
      <div class="vv-eyebrow"><span>Scene description</span><span>{voice}</span></div>
      <div class="vv-caption{css}" role="status" aria-live="polite" aria-atomic="true">{text}</div>
      <div class="{wave}" style="--iters:{iterations}" aria-hidden="true">{bars}</div>
      <div class="vv-chips">{chips}{thinking}</div>
    </div>""")


def kpi_html(history) -> str:
    totals = [e.latency_ms for e in history]
    last = f"{totals[-1]:.0f}" if totals else "-"
    median = f"{np.median(totals):.0f}" if totals else "-"
    return block(f"""
    <div class="vv-kpis">
      <div class="vv-kpi"><small>Last</small><strong>{last}</strong><em>ms</em></div>
      <div class="vv-kpi"><small>Median</small><strong>{median}</strong><em>ms</em></div>
      <div class="vv-kpi"><small>Captions</small><strong>{len(totals)}</strong></div>
    </div>""")


def onboarding_html() -> str:
    return block(f"""
    <div class="vv-card vv-steps"><div class="vv-eyebrow"><span>One-time setup</span><span>3 steps</span></div>
    <div class="vv-h3" role="heading" aria-level="2">The model has not been trained yet</div>
    <p>VisionVoice needs the trained decoder (<code>{config.MODEL_PATH.name}</code>) and vocabulary
    (<code>{config.VOCAB_PATH.name}</code>) in <code>outputs/models</code>.</p>
    <ol>
      <li>Download Flickr8k and extract the images and <code>Flickr8k.token.txt</code> (or <code>captions.txt</code>) into <code>data/flickr8k/</code>.</li>
      <li>Run <code>python train.py</code> (cached ResNet-50 features make this fast on a laptop GPU, patient on CPU).</li>
      <li>Reload this page.</li>
    </ol></div>""")


def architecture_html() -> str:
    nodes = [
        ("Input", "Camera frame", "RGB, resized to 224x224"),
        ("Encoder", "ResNet-50", "ImageNet weights, frozen, global-avg-pool to 2048-d"),
        ("Image branch", "Dropout + Dense 256", "projects features into the caption space"),
        ("Text branch", "Embedding + LSTM 256", "reads the words generated so far"),
        ("Merge", "Add + Dense 256", "fuses vision and language"),
        ("Output", "Softmax + beam search", "next word until endseq, then speech"),
    ]
    parts = []
    for i, (tag, title, desc) in enumerate(nodes):
        parts.append(f'<div class="vv-node"><small>{i + 1:02d} &middot; {tag}</small><b>{title}</b><span>{desc}</span></div>')
    return block(f'<div class="vv-flow">{"".join(parts)}</div>')


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def style_figure(fig: go.Figure, theme: dict, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=24, r=16, t=76, b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, system-ui, sans-serif", color=theme["text"], size=14),
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=20), x=0.0, y=0.97, yanchor="top"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=theme["surface2"], bordercolor=theme["accent"], font_color=theme["text"]),
    )
    fig.update_xaxes(gridcolor=theme["border"], zerolinecolor=theme["border"], linecolor=theme["border"], automargin=True)
    fig.update_yaxes(gridcolor=theme["border"], zerolinecolor=theme["border"], linecolor=theme["border"], automargin=True)
    return fig


def bleu_figure(metrics: dict, theme: dict) -> go.Figure:
    bleu = metrics["bleu"]
    labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    values = [bleu[f"bleu{i}"] for i in range(1, 5)]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=theme["accent"], text=[f"{v:.3f}" for v in values],
                           textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(range=[0, min(1.0, max(values) * 1.25 + 0.05)], title="score (higher is better)")
    return style_figure(fig.update_layout(title=f"Caption quality on {metrics.get('test_images', '?')} Flickr8k test images"), theme)


def training_figure(history: dict, theme: dict) -> go.Figure:
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_scatter(x=epochs, y=history["loss"], name="train loss", mode="lines+markers", line=dict(color=theme["accent"], width=3))
    if "val_loss" in history:
        fig.add_scatter(x=epochs, y=history["val_loss"], name="validation loss", mode="lines+markers",
                        line=dict(color=theme["accent2"], width=3))
        best = int(np.argmin(history["val_loss"]))
        fig.add_scatter(x=[epochs[best]], y=[history["val_loss"][best]], name="best epoch", mode="markers",
                        marker=dict(size=14, symbol="star", color=theme["text"]))
    fig.update_xaxes(title="epoch", dtick=1 if len(epochs) < 16 else None)
    fig.update_yaxes(title="cross-entropy")
    return style_figure(fig.update_layout(title="Training curve"), theme)


def latency_figure(history, theme: dict) -> go.Figure:
    events = list(history)
    fig = go.Figure()
    if events:
        xs = [e.event_id for e in events]
        fig.add_bar(x=xs, y=[e.feature_ms for e in events], name="ResNet-50 features", marker_color=theme["accent2"])
        fig.add_bar(x=xs, y=[e.decode_ms for e in events], name="LSTM decoding", marker_color=theme["accent"])
        median = float(np.median([e.latency_ms for e in events]))
        fig.add_hline(y=median, line_dash="dot", line_color=theme["text"], annotation_text=f"median {median:.0f} ms",
                      annotation_font_color=theme["text"])
    fig.update_layout(barmode="stack", title="Inference latency per caption")
    fig.update_xaxes(title="caption #")
    fig.update_yaxes(title="milliseconds")
    return style_figure(fig, theme)


def benchmark_figure(rows, theme: dict) -> go.Figure:
    df = pd.DataFrame(rows, columns=["beam", "feature_ms", "decode_ms"]).groupby("beam").mean().reset_index()
    fig = go.Figure()
    fig.add_bar(x=df["beam"].astype(str), y=df["feature_ms"], name="ResNet-50 features", marker_color=theme["accent2"])
    fig.add_bar(x=df["beam"].astype(str), y=df["decode_ms"], name="LSTM decoding", marker_color=theme["accent"])
    fig.update_layout(barmode="stack", title="Beam width vs latency (mean of runs)")
    fig.update_xaxes(title="beam width")
    fig.update_yaxes(title="milliseconds")
    return style_figure(fig, theme)


def show_chart(slot, fig: go.Figure) -> None:
    slot.plotly_chart(fig, theme=None, config={"displaylogo": False, "displayModeBar": False})


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def jpeg_bytes(rgb: np.ndarray, quality: int = config.LIVE_JPEG_QUALITY) -> bytes:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else b""


def history_frame(history) -> pd.DataFrame:
    return pd.DataFrame([{
        "time": datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"), "caption": e.caption, "source": e.source,
        "feature_ms": round(e.feature_ms, 1), "decode_ms": round(e.decode_ms, 1), "total_ms": round(e.latency_ms, 1),
        "beam": e.beam_width} for e in history])


# =========================================================================== #
#                                    PAGE
# =========================================================================== #
st.session_state.setdefault("snapshots", [])
st.session_state.setdefault("bench_rows", None)
st.session_state.setdefault("last_upload", None)

ready = artifacts_ready()

# ---- sidebar: accessibility, voice, captioning, camera ---------------------- #
with st.sidebar:
    st.markdown("### ◉ VisionVoice")
    st.caption("Assistive scene narration")

    st.markdown("#### Accessibility")
    theme_name = st.selectbox("Colour theme", list(THEMES), help="High Contrast is yellow-on-black with white borders.")
    caption_px = st.slider("Caption text size", 20, 96, 44, 2, format="%d px")
    ui_scale = st.slider("Interface scale", 85, 150, 100, 5, format="%d%%") / 100
    calm = st.toggle("Reduce motion", value=False, help="Static waveform instead of animated bars.")

    st.markdown("#### Voice")
    auto_speak = st.toggle("Speak new captions automatically", value=True)
    rate = st.slider("Speech rate", 100, 260, config.TTS_RATE, 5, format="%d wpm")
    volume = st.slider("Volume", 0, 100, int(config.TTS_VOLUME * 100), 5, format="%d%%")

    st.markdown("#### Captioning")
    beam = st.slider("Beam width", 1, 5, config.DEFAULT_BEAM_WIDTH, help="1 = fastest (greedy). Higher = better sentences, slower.")
    interval = st.slider("Minimum seconds between captions", 1.0, 15.0, float(config.LIVE_CAPTION_INTERVAL_S), 0.5)
    threshold = st.slider("Scene-change threshold", 1.0, 30.0, float(config.LIVE_SCENE_CHANGE_THRESHOLD), 0.5,
                          help="Lower = describes smaller changes. Higher = stays quiet unless the scene really changes.")

    st.markdown("#### Camera")
    source_options = ["Webcam", "Phone / IP camera", "Browser camera (WebRTC)", "Upload image"]
    source_kind = st.selectbox("Input source", source_options)
    webcam_index = phone_url = None
    if source_kind == "Webcam":
        webcam_index = st.number_input("Camera index", 0, 9, 0)
    elif source_kind == "Phone / IP camera":
        phone_url = st.text_input("Stream address", placeholder="192.168.1.8:8080",
                                  help="Android 'IP Webcam' / DroidCam / any MJPEG or RTSP URL on your Wi-Fi.")
    elif source_kind == "Browser camera (WebRTC)" and not HAS_WEBRTC:
        st.warning("Install `streamlit-webrtc` to enable this source.")

theme = THEMES[theme_name]
inject_css(theme, caption_px, ui_scale, calm)

sv: Optional[SimpleNamespace] = load_services() if ready else None
speech = sv.speech if sv else get_speech_engine()
speech.configure(rate=rate, volume=volume / 100)
if sv:
    sv.captioner.configure(interval_s=interval, scene_threshold=threshold, beam_width=beam, auto_speak=auto_speak)

LIVE_SOURCES = ("Webcam", "Phone / IP camera")
live_on = bool(st.session_state.get("live_on", False)) and source_kind in LIVE_SOURCES
hero_slot = st.empty()
hero_slot.markdown(hero_html(ready, speech.backend_name if speech.available else "unavailable", source_kind, live_on),
                   unsafe_allow_html=True)

if not ready:
    st.markdown(onboarding_html(), unsafe_allow_html=True)
    st.stop()
assert sv is not None
if not speech.available:
    st.warning(f"No text-to-speech engine could be started ({speech.error}). Captions will still be shown. "
               "On Linux install eSpeak: `sudo apt install espeak-ng`.")

# ---- main stage -------------------------------------------------------------- #
col_video, col_caption = st.columns([1.8, 1], gap="large")

with col_video:
    video_slot = st.empty()
    status_slot = st.empty()
    c_live, c_snap, c_mute = st.columns([1.1, 1, 1])
    if source_kind not in LIVE_SOURCES:
        c_live.markdown("&nbsp;")
    else:
        c_live.toggle("Start live stream", key="live_on", help="Opens the selected camera and describes what it sees.")
    snap_clicked = c_snap.button("Capture & describe", icon=":material/photo_camera:", width="stretch", type="primary")
    mute_clicked = c_mute.button("Stop voice", icon=":material/volume_off:", width="stretch")

with col_caption:
    caption_slot = st.empty()
    kpi_slot = st.empty()
    replay_clicked = st.button("Repeat caption", icon=":material/replay:", width="stretch")

# upload path -------------------------------------------------------------------- #
# upload path --------------------------------------------------------------------
uploaded_rgb = None

if source_kind == "Upload image":
    with col_video:
        upload = st.file_uploader(
            "Choose a photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    if upload is not None:
        uploaded_bytes = upload.getvalue()
        uploaded_rgb = np.array(
            Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        )
        video_slot.image(uploaded_bytes, width="stretch")

        image_hash = hashlib.sha256(uploaded_bytes).hexdigest()

        if st.session_state.last_upload != image_hash:
            st.session_state.last_upload = image_hash
            sv.captioner.submit(uploaded_rgb)

            event = sv.captioner.caption_now(
                uploaded_rgb,
                "upload",
                speak=False,
            )

            if event is None:
                st.warning("DEBUG: No caption event returned.")
            else:
                st.write("DEBUG — Caption:", event.caption)

                if auto_speak:
                    speech_queued = speech.speak(event.caption)
                    time.sleep(0.5)
                    st.write("DEBUG — Speaking:", speech.is_speaking)
                    st.write("DEBUG — Error after queue:", speech.error)
                    st.write("DEBUG — Speech queued:", speech_queued)
                    st.write("DEBUG — Audio available:", speech.available)
                    st.write("DEBUG — Audio backend:", speech.backend_name)
                    st.write("DEBUG — Audio error:", speech.error)
                else:
                    st.warning("DEBUG: Automatic speech is disabled.")

    else:
        video_slot.markdown(
            stage_html(
                "Drop a photo to describe it",
                "JPEG, PNG or WebP up to 20 MB",
            ),
            unsafe_allow_html=True,
        )

elif not live_on and source_kind != "Browser camera (WebRTC)":
    video_slot.markdown(
        stage_html(
            "Camera is off",
            "Turn on 'Start live stream' to begin narrating what the camera sees.",
        ),
        unsafe_allow_html=True,
    )


# WebRTC path ---------------------------------------------------------------------- #
webrtc_ctx = None
if source_kind == "Browser camera (WebRTC)" and HAS_WEBRTC:
    last_push = [0.0]

    def on_frame(frame):
        now = time.time()
        if now - last_push[0] >= 0.25:                       # convert at most 4x per second
            last_push[0] = now
            sv.captioner.submit(frame.to_ndarray(format="rgb24"))
        return frame

    with video_slot.container():
        webrtc_ctx = webrtc_streamer(
            key="visionvoice-webrtc", mode=WebRtcMode.SENDRECV, video_frame_callback=on_frame,
            media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True)
elif source_kind == "Browser camera (WebRTC)":
    video_slot.markdown(stage_html("WebRTC not installed", "pip install streamlit-webrtc"), unsafe_allow_html=True)

# button actions ----------------------------------------------------------------------- #
if mute_clicked:
    speech.stop()

if snap_clicked:
    frame = uploaded_rgb if uploaded_rgb is not None else sv.captioner.latest_frame
    if frame is None:
        st.toast("No frame yet - start the stream (or upload a photo) first.", icon=":material/info:")
    else:
        with st.spinner("Describing ..."):
            event = sv.captioner.caption_now(frame, "snapshot", speak=True)
        st.session_state.snapshots.insert(0, dict(jpg=jpeg_bytes(frame, 85), caption=event.caption,
                                                  ms=event.latency_ms, ts=event.timestamp))
        del st.session_state.snapshots[12:]
        st.toast(event.caption, icon=":material/graphic_eq:")

if replay_clicked and sv.captioner.latest_event:
    speech.speak(sv.captioner.latest_event.caption)


def paint_caption(live: bool) -> None:
    event = sv.captioner.latest_event
    remaining = max(0.0, speech.speaking_until - time.time())
    iterations = "infinite" if live else f"{remaining / 0.9:.2f}"
    caption_slot.markdown(caption_card_html(event, speech.is_speaking, auto_speak or speech.is_speaking, calm,
                                            sv.captioner.busy, iterations), unsafe_allow_html=True)
    kpi_slot.markdown(kpi_html(sv.captioner.history), unsafe_allow_html=True)


paint_caption(live=False)

# ---- analytics & history ------------------------------------------------------------------- #
tab_perf, tab_snap, tab_log, tab_arch = st.tabs(["Performance", "Snapshots", "Caption log", "Architecture"])

with tab_perf:
    metrics, history_json = read_json(config.METRICS_PATH), read_json(config.HISTORY_PATH)
    left, right = st.columns(2, gap="large")
    with left:
        if metrics and "bleu" in metrics:
            show_chart(st, bleu_figure(metrics, theme))
        else:
            st.info("Run `python train.py` to compute BLEU scores for this chart.")
    with right:
        if history_json and "loss" in history_json:
            show_chart(st, training_figure(history_json, theme))
        else:
            st.info("Training history appears here after `python train.py`.")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Parameters (decoder)", f"{metrics.get('parameters', 0) / 1e6:.2f} M")
        m2.metric("Vocabulary", f"{metrics.get('vocab_size', 0):,} words")
        m3.metric("Best val. loss", f"{metrics.get('best_val_loss', float('nan')):.3f}")
        m4.metric("Benchmark latency", f"{metrics.get('latency', {}).get('total_ms_mean', float('nan')):.0f} ms")

    st.markdown("#### Live inference latency")
    latency_slot = st.empty()
    show_chart(latency_slot, latency_figure(sv.captioner.history, theme))

    bcol1, bcol2 = st.columns([1, 3])
    if bcol1.button("Run beam-width benchmark", icon=":material/speed:"):
        frame = sv.captioner.latest_frame
        if frame is None:
            frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        with st.spinner("Timing greedy vs beam search ..."):
            st.session_state.bench_rows = sv.captioner.benchmark(frame, beams=(1, 3, 5), runs=6)
    bcol2.caption("Uses the newest camera frame (or a synthetic image) and the compiled inference graphs.")
    if st.session_state.bench_rows:
        show_chart(st, benchmark_figure(st.session_state.bench_rows, theme))

with tab_snap:
    if not st.session_state.snapshots:
        st.info("Press 'Capture & describe' to save a frame together with its spoken caption.")
    cols = st.columns(3, gap="large")
    for i, snap in enumerate(st.session_state.snapshots):
        with cols[i % 3]:
            st.image(snap["jpg"], width="stretch")
            st.markdown(f"**{snap['caption']}**")
            st.caption(f"{datetime.fromtimestamp(snap['ts']):%H:%M:%S} - {snap['ms']:.0f} ms")

with tab_log:
    log = history_frame(sv.captioner.history)
    if log.empty:
        st.info("Captions generated in this session are listed here.")
    else:
        st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)
        st.download_button("Download CSV", log.to_csv(index=False).encode("utf-8"), "visionvoice_captions.csv", "text/csv")

with tab_arch:
    st.markdown(architecture_html(), unsafe_allow_html=True)
    st.markdown("")
    st.caption("The CNN is frozen and its features are cached, so training only updates the light decoder. "
               "At inference both stages run as compiled graphs with fixed signatures - no retracing, no per-frame warm-up.")

# ---- live loops (must stay last: they block this script run until the user interacts) ---- #
def refresh_loop_ui(state: dict, force: bool = False) -> None:
    event = sv.captioner.latest_event
    key = (event.event_id if event else 0, speech.is_speaking, sv.captioner.busy)
    if force or key != state["key"]:
        state["key"] = key
        paint_caption(live=True)
        if len(sv.captioner.history) != state["count"] and time.time() - state["chart_at"] > 3:
            state["count"], state["chart_at"] = len(sv.captioner.history), time.time()
            show_chart(latency_slot, latency_figure(sv.captioner.history, theme))


if live_on:
    if source_kind == "Webcam":
        source = parse_source(int(webcam_index or 0))
    else:
        source = normalise_phone_url(phone_url or "")
    if source == "":
        video_slot.markdown(stage_html("Enter the phone's stream address", "Example: 192.168.1.8:8080"), unsafe_allow_html=True)
    else:
        with st.spinner("Opening camera ..."):
            stream = ensure_stream(sv, source)
        if sv.stream_is_new:
            sv.captioner.reset_scene()
        if stream.error:
            video_slot.markdown(stage_html("Camera unavailable", stream.error), unsafe_allow_html=True)
            hero_slot.markdown(hero_html(ready, speech.backend_name, source_kind, False), unsafe_allow_html=True)
            release_stream(sv)
        else:
            state = dict(key=None, count=len(sv.captioner.history), chart_at=0.0)
            last_id, last_status, frame_period = -1, 0.0, 1.0 / config.LIVE_DISPLAY_FPS
            while True:
                tick = time.time()
                frame, frame_id = stream.read()
                if not stream.alive and frame is None:
                    video_slot.markdown(stage_html("Camera stream ended", stream.error or "Toggle the stream to reconnect."),
                                        unsafe_allow_html=True)
                    break
                if frame is not None and frame_id != last_id:
                    last_id = frame_id
                    sv.captioner.submit(frame)
                    video_slot.image(jpeg_bytes(frame), width="stretch")
                refresh_loop_ui(state)
                if tick - last_status > 1.0:
                    last_status = tick
                    last = sv.captioner.latest_event
                    status_slot.markdown(block(f"""<div class="vv-status">
                        <span><i class="vv-dot live"></i> <b>LIVE</b></span><span>camera <b>{stream.fps:.0f} fps</b></span>
                        <span>last caption <b>{(f'{last.latency_ms:.0f} ms') if last else '-'}</b></span>
                        <span>model <b>{'thinking' if sv.captioner.busy else 'idle'}</b></span></div>"""), unsafe_allow_html=True)
                time.sleep(max(0.0, frame_period - (time.time() - tick)))
elif source_kind in LIVE_SOURCES:
    release_stream(sv)                                         # toggle off -> camera light off

elif webrtc_ctx is not None:
    state = dict(key=None, count=len(sv.captioner.history), chart_at=0.0)
    while webrtc_ctx.state.playing:
        refresh_loop_ui(state)
        time.sleep(0.25)
