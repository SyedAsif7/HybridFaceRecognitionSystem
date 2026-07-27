"""
Streamlit UI — Hybrid Face Recognition
"""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.db.face_store import (
    image_url,
    is_supabase_enabled,
    recognize_face,
    register_person,
    safe_load_faces,
)

ROOT = Path(__file__).resolve().parent
HISTORY_FILE = ROOT / "data" / "recognition_history.json"

st.set_page_config(
    page_title="Hybrid Face Recognition",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Nunito+Sans:opsz,wght@6..12,400;6..12,500;6..12,600;6..12,700&display=swap');

:root {
  --ink: #14242c;
  --mute: #5d6e77;
  --line: #d0dbe1;
  --bg: #edf2f4;
  --panel: #ffffff;
  --teal: #0b5f52;
  --teal-soft: #e6f3ef;
  --ok: #0b5f52;
  --warn: #8f6a14;
  --bad: #9b3535;
}

html, body, .stApp {
  font-family: 'Nunito Sans', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(800px 380px at 0% 0%, #d7e6e8 0%, transparent 55%),
    radial-gradient(700px 340px at 100% 0%, #dce8e2 0%, transparent 50%),
    var(--bg);
}

section[data-testid="stSidebar"] { display: none !important; }

div.block-container {
  max-width: 1000px !important;
  padding-top: 1.75rem !important;
  padding-bottom: 3rem !important;
}

/* Only style our markup — leave Streamlit widgets alone */
.hero {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1.5rem 1.6rem 1.35rem;
  margin-bottom: 1.25rem;
}
.hero-title {
  margin: 0;
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(1.75rem, 3vw, 2.15rem);
  font-weight: 700;
  line-height: 1.2;
  color: var(--ink);
}
.hero-desc {
  margin: 0.75rem 0 0;
  max-width: 40rem;
  font-size: 1rem;
  line-height: 1.55;
  color: var(--mute);
}
.hero-row {
  display: none;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 1.25rem;
  padding-top: 1.15rem;
  border-top: 1px solid var(--line);
}
.stat-card {
  background: linear-gradient(180deg, #f8fbfb 0%, #f3f8f6 100%);
  border: 1px solid #c5ddd6;
  border-radius: 10px;
  padding: 0.85rem 0.95rem;
  min-height: 5.5rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.stat-label {
  margin: 0;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mute);
  line-height: 1.3;
}
.stat-value {
  margin: 0.35rem 0 0.15rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.45rem;
  font-weight: 700;
  line-height: 1.15;
  color: var(--ink);
}
.stat-hint {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.35;
  color: var(--teal);
  font-weight: 600;
}
.stat-card.feature .stat-value,
.stat-card.storage .stat-value {
  font-size: 1.05rem;
}
@media (max-width: 720px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
.chip {
  display: none;
}

.sec-title {
  margin: 0.25rem 0 0.35rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.4rem;
  font-weight: 700;
  line-height: 1.25;
}
.sec-help {
  margin: 0 0 1.1rem;
  font-size: 0.95rem;
  line-height: 1.5;
  color: var(--mute);
}
.panel-title {
  margin: 0 0 0.65rem;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--teal);
  line-height: 1.3;
}
.result-box {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem 1.05rem;
  margin-top: 0.75rem;
}
.result-name {
  margin: 0 0 0.65rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.2;
}
.result-name.not-found { color: var(--bad); }
.result-line {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  line-height: 1.45;
}
.result-line .k { color: var(--mute); margin-right: 0.35rem; }
.ok { color: var(--ok); font-weight: 800; }
.warn { color: var(--warn); font-weight: 800; }
.bad { color: var(--bad); font-weight: 800; }
.person-name {
  margin: 0.55rem 0 0.15rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.3;
}
.person-meta {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.4;
  color: var(--mute);
}
.foot {
  margin-top: 1.25rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--line);
  font-size: 0.85rem;
  line-height: 1.5;
  color: var(--mute);
}

div[data-testid="stImage"] img {
  border: 1px solid var(--line);
  border-radius: 8px;
}
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  min-height: 2.75rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)




def bytes_to_bgr(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bytes_fingerprint(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def apply_recognition(image_bgr: np.ndarray, *, history_source: str) -> dict | None:
    try:
        result = recognize_face(image_bgr)
    except Exception as exc:
        st.error(f"Recognition failed: {exc}")
        return None

    st.session_state["result"] = result
    append_history(
        {
            "label": result["label"],
            "confidence": f"{result['confidence']:.2f}%",
            "confidence_level": result["level"],
            "found": result["found"],
            "timestamp": datetime.now().isoformat(),
            "source": history_source,
            "model_used": "Custom Face Database (Research Fusion)",
        }
    )
    return result


def render_recognition_result(result: dict) -> None:
    aligned = (result["aligned"] * 255).astype(np.uint8)
    cls = tone(result["found"], result["level"])
    if not result["found"]:
        st.warning(
            f"**{html.escape(str(result['label']))}** — "
            f"{html.escape(str(result.get('detail', 'This person is not enrolled in the gallery.')))}"
        )
    st.image(aligned, caption="Aligned face crop", use_container_width=True, clamp=True)
    if result["found"]:
        st.progress(min(1.0, max(0.0, float(result["confidence"]) / 100.0)))
    name_cls = "" if result["found"] else " not-found"
    st.markdown(
        f"""
        <div class="result-box">
          <p class="result-name{name_cls}">{html.escape(str(result['label']))}</p>
          <p class="result-line">
            <span class="k">Status</span>
            <span class="{cls}">{'Identified' if result['found'] else 'Not identified'}</span>
          </p>
          <p class="result-line">
            <span class="k">Confidence</span>
            <span class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</span>
          </p>
          <p class="result-line"><span class="k">Method</span>SIFT keypoint matching</p>
          <p class="result-line" style="color:var(--mute);margin-top:0.55rem;">
            {html.escape(str(result.get('detail', '')))}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_gallery_images(person: dict) -> str:
    n = int(person.get("num_images") or 0)
    if n <= 0:
        return "No images"
    if n == 1:
        return "1 image"
    nums = ", ".join(str(i) for i in range(1, n + 1))
    return f"{n} images · {nums}"


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


faces_index, db_error = safe_load_faces()
faces_map = faces_index.get("registered_faces", {})
n_faces = len(faces_map)
storage_label = "Supabase" if is_supabase_enabled() else "Local"

if db_error:
    st.error(db_error)

# ── Header ────────────────────────────────────────────────
enrolled_label = "person" if n_faces == 1 else "people"
storage_hint = "Cloud database" if is_supabase_enabled() else "Saved on this device"

st.markdown(
    f"""
    <div class="hero">
      <h1 class="hero-title">Hybrid Face Recognition</h1>
      <p class="hero-desc">
        Enroll faces, then recognize them with hybrid feature fusion against your secure gallery.
      </p>
      <div class="stats-grid">
        <div class="stat-card">
          <p class="stat-label">Gallery</p>
          <p class="stat-value">{n_faces}</p>
          <p class="stat-hint">{enrolled_label} enrolled</p>
        </div>
        <div class="stat-card feature">
          <p class="stat-label">Features</p>
          <p class="stat-value">SIFT · HOG · Gabor</p>
          <p class="stat-hint">Hybrid fusion method</p>
        </div>
        <div class="stat-card storage">
          <p class="stat-label">Storage</p>
          <p class="stat-value">{html.escape(storage_label)}</p>
          <p class="stat-hint">{html.escape(storage_hint)}</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Prefer segmented control when available (Streamlit ≥ 1.44)
try:
    page = st.segmented_control(
        "Choose a section",
        options=["Recognize", "Register", "Gallery"],
        default="Recognize",
        label_visibility="collapsed",
        key="nav",
    )
    if page is None:
        page = "Recognize"
except Exception:
    page = st.radio(
        "Choose a section",
        ["Recognize", "Register", "Gallery"],
        horizontal=True,
        key="nav_radio",
    )

st.write("")  # breathing room

# ── Recognize ─────────────────────────────────────────────
if page == "Recognize":
    st.markdown(
        '<p class="sec-title">Recognize</p>'
        '<p class="sec-help">Upload a photo or use the camera — turn the camera on/off with the button, then recognition runs automatically.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="panel-title">1 · Add photo</p>', unsafe_allow_html=True)
        source = st.radio(
            "How do you want to add a photo?",
            ["Upload image", "Use camera"],
            horizontal=True,
            key="src",
            on_change=lambda: st.session_state.update({"camera_on": False}),
        )

        image_bgr = None
        go = False
        camera_auto = False

        if source == "Upload image":
            up = st.file_uploader(
                "Choose a clear face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="up",
                help="JPG, PNG, BMP, or WEBP — one face works best.",
            )
            if up is not None:
                image_bgr = bytes_to_bgr(up.getvalue())
                if image_bgr is None:
                    st.error("Image could not be read. Please upload a valid JPG, PNG, or WEBP file.")
                else:
                    st.image(up.getvalue(), use_container_width=True)

            go = st.button(
                "Run recognition",
                type="primary",
                use_container_width=True,
                disabled=image_bgr is None,
                key="go",
            )
            if image_bgr is None:
                st.caption("Add a photo to enable recognition.")
        else:
            if "camera_on" not in st.session_state:
                st.session_state["camera_on"] = False

            if st.session_state["camera_on"]:
                if st.button(
                    "Turn Camera Off",
                    use_container_width=True,
                    key="cam_off",
                ):
                    st.session_state["camera_on"] = False
                    st.session_state.pop("last_cam_id", None)
                    st.rerun()

                cam = st.camera_input(
                    "Live camera",
                    key="cam",
                    help="Recognition runs automatically when a face photo appears.",
                )
                if cam is not None:
                    cam_bytes = cam.getvalue()
                    image_bgr = bytes_to_bgr(cam_bytes)
                    if image_bgr is None:
                        st.error("Camera image could not be read. Please try again.")
                    else:
                        cam_id = bytes_fingerprint(cam_bytes)
                        if st.session_state.get("last_cam_id") != cam_id:
                            st.session_state["last_cam_id"] = cam_id
                            camera_auto = True
                st.caption("Camera is on. Recognition runs when a photo appears.")
            else:
                st.info("Camera is off.")
                if st.button(
                    "Turn Camera On",
                    type="primary",
                    use_container_width=True,
                    key="cam_on",
                ):
                    st.session_state["camera_on"] = True
                    st.rerun()
                st.caption("Click **Turn Camera On** to start the camera.")

    with right:
        st.markdown('<p class="panel-title">2 · Match result</p>', unsafe_allow_html=True)

        if camera_auto and image_bgr is not None:
            with st.spinner("Recognizing live camera image…"):
                apply_recognition(image_bgr, history_source="streamlit-camera")
        elif go and image_bgr is not None:
            with st.spinner("Aligning face and matching…"):
                apply_recognition(image_bgr, history_source="streamlit")

        result = st.session_state.get("result")
        if not result:
            if source == "Use camera" and not st.session_state.get("camera_on"):
                st.info("Turn the camera on to start recognition.")
            elif source == "Use camera":
                st.info("Look at the camera — the match result will appear here automatically.")
            else:
                st.info("Results show here after you run recognition.")
        else:
            render_recognition_result(result)

# ── Register ──────────────────────────────────────────────
elif page == "Register":
    st.markdown(
        '<p class="sec-title">Register</p>'
        '<p class="sec-help">Enroll a new person so they can be recognized later.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="panel-title">Person details</p>', unsafe_allow_html=True)
        name = st.text_input(
            "Full name",
            placeholder="e.g. Syed Asif",
            key="name",
            help="This name appears in recognition results.",
        )
        files = st.file_uploader(
            "Face images (one or more)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="files",
        )
        shot = st.camera_input("Or capture one photo now", key="regcam")

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
        st.markdown('<p class="panel-title">Preview</p>', unsafe_allow_html=True)
        if not images:
            st.info("Upload or capture photos to preview them here.")
        else:
            st.success(f"{len(images)} image(s) ready to save")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images[:6]):
                cols[i % len(cols)].image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

    saved = st.button("Save to gallery", type="primary", use_container_width=True, key="save")
    if saved:
        if not name.strip():
            st.error("Enter a person name before saving.")
        elif not images:
            st.error("Add at least one photo before saving.")
        else:
            with st.spinner("Building face encoding…"):
                ok, msg = register_person(name, images)
            if ok:
                st.success(msg)
                st.session_state.pop("result", None)
                st.rerun()
            else:
                st.error(msg)

    st.markdown(
        '<p class="foot">'
        + (
            "Faces are saved permanently in Supabase."
            if is_supabase_enabled()
            else "Tip: add Supabase secrets to keep enrollments on Streamlit Cloud."
        )
        + "</p>",
        unsafe_allow_html=True,
    )

# ── Gallery ───────────────────────────────────────────────
else:
    st.markdown(
        '<p class="sec-title">Gallery</p>'
        '<p class="sec-help">Everyone currently enrolled for recognition.</p>',
        unsafe_allow_html=True,
    )

    if not faces_map:
        st.warning("No faces enrolled yet. Go to Register to add the first person.")
    else:
        people = sorted(faces_map.values(), key=lambda p: str(p.get("name", "")).lower())
        for start in range(0, len(people), 3):
            cols = st.columns(3, gap="medium")
            for col, person in zip(cols, people[start : start + 3]):
                with col:
                    with st.container(border=True):
                        thumb = image_url(person)
                        if thumb:
                            st.image(thumb, use_container_width=True)
                        st.markdown(
                            f'<p class="person-name">{html.escape(str(person["name"]))}</p>'
                            f'<p class="person-meta">{html.escape(format_gallery_images(person))}</p>',
                            unsafe_allow_html=True,
                        )
