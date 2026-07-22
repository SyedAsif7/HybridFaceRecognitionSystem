"""
Streamlit UI — Hybrid Face Recognition
Made by Syed Asif
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.features.extractors import extract_research_fusion
from src.preprocessing.processor import detect_and_align_face

ROOT = Path(__file__).resolve().parent
FACES_DIR = ROOT / "Faces"
FACES_INDEX = FACES_DIR / "index.json"
HISTORY_FILE = ROOT / "data" / "recognition_history.json"
MATCH_THRESHOLD = 65.0

st.set_page_config(
    page_title="Hybrid Face Recognition",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root {
  --bg: #e8eef1;
  --card: #ffffff;
  --ink: #12202a;
  --mute: #5a6b75;
  --line: #cfd9df;
  --teal: #0d6b5c;
  --teal-deep: #0a4f44;
  --navy: #1c3d4f;
  --ok: #0d6b5c;
  --warn: #9a6b12;
  --bad: #a33a3a;
  --soft: #e7f2ef;
}

html, body, [data-testid="stAppViewContainer"], .stApp,
p, span, label, div, input, button, textarea {
  font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 420px at 0% 0%, #d5e5e8 0%, transparent 55%),
    radial-gradient(700px 360px at 100% 0%, #d9e4de 0%, transparent 50%),
    var(--bg) !important;
  color: var(--ink) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
#MainMenu, footer { display: none !important; }

.block-container {
  max-width: 1040px !important;
  padding: 1.5rem 1.25rem 3rem !important;
}

/* Type scale */
.ui-title {
  margin: 0;
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-size: 34px !important;
  font-weight: 700 !important;
  line-height: 1.15 !important;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.ui-byline {
  margin: 10px 0 0;
  font-size: 15px !important;
  line-height: 1.5 !important;
  color: var(--mute);
}
.ui-byline a {
  color: var(--navy);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid rgba(28,61,79,0.35);
}
.ui-byline a:hover { border-bottom-color: var(--navy); }
.ui-lead {
  margin: 14px 0 0;
  font-size: 16px !important;
  line-height: 1.55 !important;
  color: var(--mute);
  max-width: 46rem;
}
.ui-count {
  text-align: right;
}
.ui-count b {
  display: block;
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  line-height: 1 !important;
  color: var(--ink);
}
.ui-count span {
  display: block;
  margin-top: 6px;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--mute);
}
.ui-h2 {
  margin: 0 0 6px;
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-size: 24px !important;
  font-weight: 700 !important;
  line-height: 1.25 !important;
  color: var(--ink);
}
.ui-help {
  margin: 0 0 18px;
  font-size: 15px !important;
  line-height: 1.5 !important;
  color: var(--mute);
}
.ui-label {
  margin: 0 0 8px;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--navy);
}
.ui-panel {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 16px;
  min-height: 120px;
}
.ui-name {
  margin: 0 0 12px;
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  line-height: 1.15 !important;
}
.ui-kv {
  display: grid;
  grid-template-columns: 110px 1fr;
  gap: 6px 12px;
  margin-bottom: 12px;
  font-size: 15px !important;
}
.ui-kv i { color: var(--mute); font-style: normal; }
.ui-kv b { font-weight: 650; color: var(--ink); }
.ok { color: var(--ok) !important; }
.warn { color: var(--warn) !important; }
.bad { color: var(--bad) !important; }
.ui-bar {
  height: 8px;
  background: #edf2f4;
  border: 1px solid var(--line);
  margin: 0 0 12px;
  overflow: hidden;
}
.ui-bar > span {
  display: block;
  height: 100%;
  background: var(--teal);
}
.ui-soft {
  margin: 0;
  font-size: 14px !important;
  line-height: 1.5 !important;
  color: var(--mute);
}
.ui-person {
  margin: 10px 0 2px;
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}
.ui-person-meta {
  margin: 0;
  font-size: 13px !important;
  color: var(--mute);
}
.ui-note {
  margin-top: 20px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  font-size: 13px !important;
  line-height: 1.5 !important;
  color: var(--mute);
}
.ui-header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
  background: var(--card);
  border: 1px solid var(--line);
  padding: 22px 24px;
  margin-bottom: 18px;
}
@media (max-width: 720px) {
  .ui-header { flex-direction: column; }
  .ui-count { text-align: left; }
  .ui-title { font-size: 28px !important; }
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: var(--card);
  border: 1px solid var(--line);
  padding: 0 !important;
  margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
  flex: 1;
  justify-content: center;
  height: auto !important;
  padding: 14px 10px !important;
  font-size: 15px !important;
  font-weight: 650 !important;
  color: var(--mute) !important;
  border-right: 1px solid var(--line) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:last-child { border-right: none !important; }
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  background: var(--soft) !important;
  box-shadow: inset 0 -3px 0 var(--teal);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Widgets */
div[data-testid="stRadio"] p,
div[data-testid="stRadio"] label,
.stMarkdown, .stCaption, .stTextInput label,
[data-testid="stFileUploader"] label,
[data-testid="stWidgetLabel"] p {
  font-size: 14px !important;
  line-height: 1.4 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 10px !important;
  margin-bottom: 12px !important;
}
div[data-testid="stRadio"] label {
  background: #fff !important;
  border: 1px solid var(--line) !important;
  padding: 8px 14px !important;
  margin: 0 !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: var(--soft) !important;
  border-color: var(--teal) !important;
}
.stButton > button {
  border-radius: 4px !important;
  min-height: 46px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"] {
  background: var(--teal-deep) !important;
  border-color: var(--teal-deep) !important;
  color: #fff !important;
}
.stTextInput input {
  font-size: 15px !important;
  min-height: 44px !important;
  border-radius: 4px !important;
}
div[data-testid="stImage"] img {
  border: 1px solid var(--line);
  border-radius: 2px;
}
div[data-testid="stFileUploader"] section {
  background: #fff !important;
  border: 1px dashed #9eb0ba !important;
  border-radius: 4px !important;
  padding: 18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def confidence_level(pct: float) -> str:
    if pct >= 80:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"


def load_face_index() -> dict:
    if not FACES_INDEX.exists():
        return {"registered_faces": {}, "next_id": 1}
    with open(FACES_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def save_face_index(index: dict) -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACES_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def recognize(image_bgr: np.ndarray) -> dict:
    face = detect_and_align_face(image_bgr)
    faces = load_face_index().get("registered_faces", {})
    if not faces:
        return {
            "label": "No faces registered",
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": face,
            "detail": "Register at least one person first.",
        }

    query = extract_research_fusion(face)
    best_name, best_id, best_score = None, None, -1e9
    for _fid, data in faces.items():
        stored = np.asarray(data["avg_encoding"], dtype=np.float64)
        denom = np.linalg.norm(query) * np.linalg.norm(stored)
        score = float(np.dot(query, stored) / denom) if denom > 0 else -1.0
        if score > best_score:
            best_score = score
            best_name = data["name"]
            best_id = data["id"]

    confidence = (best_score + 1) / 2 * 100
    if confidence < MATCH_THRESHOLD:
        return {
            "label": "Not Found",
            "confidence": float(confidence),
            "level": confidence_level(confidence),
            "found": False,
            "aligned": face,
            "detail": f"Best score below {MATCH_THRESHOLD:.0f}% match threshold.",
            "score": float(best_score),
        }

    return {
        "label": best_name,
        "id": best_id,
        "confidence": float(confidence),
        "level": confidence_level(confidence),
        "found": True,
        "aligned": face,
        "detail": "Matched with research fusion (SIFT + HOG + Gabor).",
        "score": float(best_score),
    }


def register_person(name: str, images_bgr: list[np.ndarray]) -> tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Enter a person name."
    if not images_bgr:
        return False, "Add at least one image."

    index = load_face_index()
    person_dir = FACES_DIR / name.replace(" ", "_")
    person_dir.mkdir(parents=True, exist_ok=True)

    encodings = []
    for idx, img in enumerate(images_bgr):
        aligned = detect_and_align_face(img)
        if aligned is None:
            continue
        cv2.imwrite(str(person_dir / f"{idx + 1}.jpg"), (aligned * 255).astype(np.uint8))
        encodings.append(extract_research_fusion(aligned))

    if not encodings:
        return False, "No valid faces detected in the uploaded images."

    existing_id = None
    for fid, data in index["registered_faces"].items():
        if data["name"].lower() == name.lower():
            existing_id = fid
            break

    if existing_id is None:
        person_id = index.get("next_id", 1)
        index["next_id"] = person_id + 1
        key = str(person_id)
    else:
        person_id = index["registered_faces"][existing_id]["id"]
        key = existing_id

    index["registered_faces"][key] = {
        "id": person_id,
        "name": name,
        "num_images": len(encodings),
        "directory": str(person_dir.relative_to(ROOT)).replace("\\", "/"),
        "avg_encoding": np.mean(encodings, axis=0).tolist(),
    }
    save_face_index(index)
    return True, f"Registered {name} with {len(encodings)} image(s)."


def append_history(record: dict) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.insert(0, record)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[:200], f, indent=2)
    except Exception:
        pass


def tone(found: bool, level: str) -> str:
    if not found:
        return "bad"
    return {"High": "ok", "Medium": "warn", "Low": "bad"}.get(level, "bad")


faces_map = load_face_index().get("registered_faces", {})
n_faces = len(faces_map)

st.markdown(
    f"""
    <div class="ui-header">
      <div>
        <h1 class="ui-title">Hybrid Face Recognition</h1>
        <p class="ui-byline">
          Made by <strong>Syed Asif</strong> ·
          <a href="https://www.linkedin.com/in/the-syed-asif" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          ·
          <a href="https://github.com/SyedAsif7" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
        <p class="ui-lead">
          Align a face, fuse SIFT + HOG + Gabor features, and match it to your gallery
          with a confidence score.
        </p>
      </div>
      <div class="ui-count">
        <b>{n_faces}</b>
        <span>Enrolled faces</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_rec, tab_reg, tab_gal = st.tabs(["Recognize", "Register", "Gallery"])

with tab_rec:
    st.markdown(
        '<p class="ui-h2">Recognize a face</p>'
        '<p class="ui-help">Upload or capture one clear frontal photo, then run matching.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.08, 1], gap="large")

    with left:
        st.markdown('<p class="ui-label">Input</p>', unsafe_allow_html=True)
        source = st.radio(
            "Source",
            ["Upload image", "Camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="src",
        )
        image_bgr = None
        if source == "Upload image":
            up = st.file_uploader(
                "Choose a face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="up",
            )
            if up is not None:
                image_bgr = bytes_to_bgr(up.getvalue())
                st.image(up.getvalue(), use_container_width=True)
        else:
            cam = st.camera_input("Take a photo", key="cam")
            if cam is not None:
                image_bgr = bytes_to_bgr(cam.getvalue())

        go = st.button(
            "Run recognition",
            type="primary",
            use_container_width=True,
            disabled=image_bgr is None,
            key="go",
        )

    with right:
        st.markdown('<p class="ui-label">Result</p>', unsafe_allow_html=True)
        if go and image_bgr is not None:
            with st.spinner("Matching face…"):
                result = recognize(image_bgr)
            st.session_state["result"] = result
            append_history(
                {
                    "label": result["label"],
                    "confidence": f"{result['confidence']:.2f}%",
                    "confidence_level": result["level"],
                    "found": result["found"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "streamlit",
                    "model_used": "Custom Face Database (Research Fusion)",
                }
            )

        result = st.session_state.get("result")
        if not result:
            st.markdown(
                '<div class="ui-panel"><p class="ui-soft">'
                "Your match result will appear here after recognition."
                "</p></div>",
                unsafe_allow_html=True,
            )
        else:
            aligned = (result["aligned"] * 255).astype(np.uint8)
            cls = tone(result["found"], result["level"])
            width = max(0.0, min(100.0, float(result["confidence"])))
            st.image(aligned, caption="Aligned face", use_container_width=True, clamp=True)
            st.markdown(
                f"""
                <div class="ui-panel" style="margin-top:12px;">
                  <p class="ui-name">{html.escape(str(result['label']))}</p>
                  <div class="ui-kv">
                    <i>Confidence</i>
                    <b class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</b>
                    <i>Method</i>
                    <b>SIFT + HOG + Gabor</b>
                  </div>
                  <div class="ui-bar"><span style="width:{width:.1f}%;"></span></div>
                  <p class="ui-soft">{html.escape(str(result.get('detail', '')))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_reg:
    st.markdown(
        '<p class="ui-h2">Register a person</p>'
        '<p class="ui-help">Add a name and one or more clear face images.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="ui-label">Details</p>', unsafe_allow_html=True)
        name = st.text_input("Full name", placeholder="e.g. Alex Lacamoire", key="name")
        files = st.file_uploader(
            "Face images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="files",
        )
        shot = st.camera_input("Optional camera photo", key="regcam")

    images: list[np.ndarray] = []
    if files:
        for f in files:
            img = bytes_to_bgr(f.getvalue())
            if img is not None:
                images.append(img)
    if shot is not None:
        img = bytes_to_bgr(shot.getvalue())
        if img is not None:
            images.append(img)

    with c2:
        st.markdown('<p class="ui-label">Preview</p>', unsafe_allow_html=True)
        if not images:
            st.markdown(
                '<div class="ui-panel"><p class="ui-soft">Previews appear here after upload.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{len(images)} image(s) ready")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images[:6]):
                cols[i % len(cols)].image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

    if st.button("Save to gallery", type="primary", use_container_width=True, key="save"):
        with st.spinner("Saving…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.session_state.pop("result", None)
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        '<p class="ui-note">On Streamlit Cloud, new enrollments may reset when the app sleeps. '
        "Commit updated <code>Faces/</code> files to GitHub to keep them.</p>",
        unsafe_allow_html=True,
    )

with tab_gal:
    st.markdown(
        '<p class="ui-h2">Enrolled gallery</p>'
        '<p class="ui-help">People currently available for recognition.</p>',
        unsafe_allow_html=True,
    )
    if not faces_map:
        st.markdown(
            '<div class="ui-panel"><p class="ui-soft">No faces yet. Use Register to add someone.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        people = [faces_map[k] for k in sorted(faces_map.keys(), key=lambda x: int(x))]
        for start in range(0, len(people), 3):
            cols = st.columns(3, gap="medium")
            for col, person in zip(cols, people[start : start + 3]):
                folder = ROOT / person.get("directory", "")
                photos = sorted(folder.glob("*.jpg")) if folder.exists() else []
                with col:
                    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
                    if photos:
                        st.image(str(photos[0]), use_container_width=True)
                    st.markdown(
                        f'<p class="ui-person">{html.escape(str(person["name"]))}</p>'
                        f'<p class="ui-person-meta">ID {html.escape(str(person["id"]))} · '
                        f'{int(person["num_images"])} image(s)</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
