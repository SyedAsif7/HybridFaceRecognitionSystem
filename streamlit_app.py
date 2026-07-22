"""
Streamlit UI — Hybrid Face Recognition System
Research fusion (SIFT + HOG + Gabor) + custom face database.
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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink: #10242c;
  --muted: #5d727b;
  --line: #c5d3d9;
  --line-strong: #9eb0b9;
  --paper: #f3f7f8;
  --panel: #ffffff;
  --teal: #145f6d;
  --teal-soft: #d7e8ec;
  --ok: #1a6848;
  --warn: #8a6410;
  --bad: #8f2f2f;
  --track: #e2eaed;
}

html, body, [data-testid="stAppViewContainer"], .stApp, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--ink);
}

[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0)),
    radial-gradient(900px 420px at 0% 0%, #cfe0e5 0%, transparent 60%),
    radial-gradient(700px 380px at 100% 10%, #d5e3dc 0%, transparent 55%),
    var(--paper) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer { display: none !important; }

[data-testid="stSidebar"] { display: none !important; }

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 980px !important;
}

/* ── Brand ── */
.app-hero {
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 1.5rem;
  align-items: end;
  padding-bottom: 1.35rem;
  margin-bottom: 1.1rem;
  border-bottom: 1px solid var(--line);
}
@media (max-width: 800px) {
  .app-hero { grid-template-columns: 1fr; }
}
.kicker {
  margin: 0 0 0.4rem;
  font-size: 0.9rem;
  font-weight: 500;
  letter-spacing: 0.01em;
  text-transform: none;
  color: var(--muted);
  line-height: 1.45;
}
.kicker a {
  color: var(--teal);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid transparent;
}
.kicker a:hover {
  border-bottom-color: var(--teal);
}
.hero-title {
  margin: 0;
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(2rem, 4.5vw, 2.75rem);
  font-weight: 650;
  letter-spacing: -0.03em;
  line-height: 1.02;
  color: var(--ink);
}
.hero-copy {
  margin: 0.7rem 0 0;
  max-width: 34rem;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.5;
}
.stat-board {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.55rem;
}
.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 0.75rem 0.8rem;
}
.stat b {
  display: block;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 650;
  line-height: 1;
  color: var(--ink);
}
.stat span {
  display: block;
  margin-top: 0.35rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}

/* ── Section chrome ── */
.sec-head { margin: 0.35rem 0 1.15rem; }
.sec-title {
  margin: 0;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.55rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.sec-desc {
  margin: 0.35rem 0 0;
  color: var(--muted);
  font-size: 0.96rem;
  line-height: 1.45;
  max-width: 40rem;
}
.label {
  margin: 0 0 0.55rem;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--teal);
}
.frame {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 1rem 1.05rem 1.1rem;
}
.frame + .frame { margin-top: 0.85rem; }
.muted {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.45;
}
.divider {
  height: 1px;
  background: var(--line);
  margin: 1.25rem 0;
}

/* Result */
.identity {
  margin: 0 0 0.85rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.05;
}
.kv { display: grid; grid-template-columns: 7rem 1fr; gap: 0.35rem 0.75rem; margin-bottom: 0.85rem; }
.kv i { color: var(--muted); font-style: normal; font-size: 0.9rem; }
.kv b { font-weight: 600; font-size: 0.95rem; }
.ok { color: var(--ok); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.bar {
  width: 100%;
  height: 8px;
  background: var(--track);
  border: 1px solid var(--line);
  overflow: hidden;
  margin: 0.35rem 0 0.9rem;
}
.bar > i {
  display: block;
  height: 100%;
  background: var(--teal);
  font-style: normal;
}
.foot-note {
  margin-top: 1.4rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.45;
}

/* Gallery tile */
.tile-name {
  margin: 0.65rem 0 0.15rem;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.15rem;
  font-weight: 650;
}
.tile-meta { margin: 0; color: var(--muted); font-size: 0.84rem; }

/* Controls */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex !important;
  gap: 0 !important;
  border: 1px solid var(--line-strong);
  background: var(--panel);
  padding: 0 !important;
  margin-bottom: 1.1rem !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
  flex: 1 !important;
  justify-content: center !important;
  margin: 0 !important;
  padding: 0.7rem 0.5rem !important;
  border-right: 1px solid var(--line) !important;
  background: transparent !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child {
  border-right: none !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--teal-soft) !important;
  box-shadow: inset 0 -2px 0 var(--teal);
}
div[data-testid="stRadio"] p {
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  text-align: center;
}

.stButton > button {
  border-radius: 2px !important;
  min-height: 2.75rem !important;
  font-weight: 650 !important;
  letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  border: 1px solid var(--teal) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.08);
}
div[data-testid="stImage"] img {
  border: 1px solid var(--line);
  width: 100%;
}
div[data-testid="stFileUploader"] section {
  border: 1px dashed var(--line-strong) !important;
  background: #fafcfd !important;
}
.stTextInput input {
  border-radius: 2px !important;
  border-color: var(--line-strong) !important;
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
    index = load_face_index()
    faces = index.get("registered_faces", {})

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

    for _face_id, data in faces.items():
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
        out = person_dir / f"{idx + 1}.jpg"
        cv2.imwrite(str(out), (aligned * 255).astype(np.uint8))
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


def level_class(found: bool, level: str) -> str:
    if not found:
        return "bad"
    return {"High": "ok", "Medium": "warn", "Low": "bad"}.get(level, "bad")


# ── Load data ─────────────────────────────────────────────
index_data = load_face_index()
faces_map = index_data.get("registered_faces", {})
n_faces = len(faces_map)

st.markdown(
    f"""
    <div class="app-hero">
      <div>
        <p class="kicker">
          Project made by <strong>Syed Asif</strong> ·
          <a href="https://www.linkedin.com/in/the-syed-asif" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          ·
          <a href="https://github.com/SyedAsif7" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
        <h1 class="hero-title">Hybrid Face Recognition</h1>
        <p class="hero-copy">
          Align faces, fuse SIFT + HOG + Gabor features, and match against
          your enrolled gallery with a clear confidence score.
        </p>
      </div>
      <div class="stat-board">
        <div class="stat"><b>{n_faces}</b><span>Enrolled</span></div>
        <div class="stat"><b>{MATCH_THRESHOLD:.0f}%</b><span>Threshold</span></div>
        <div class="stat"><b>3</b><span>Features</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["Recognize", "Register", "Gallery"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)


# ── Recognize ─────────────────────────────────────────────
if page == "Recognize":
    st.markdown(
        """
        <div class="sec-head">
          <h2 class="sec-title">Recognize</h2>
          <p class="sec-desc">Upload or capture one clear frontal photo, then run matching against the enrolled gallery.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="label">1 · Image source</p>', unsafe_allow_html=True)
        source = st.radio(
            "Source",
            ["Upload image", "Use camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="recognize_source",
        )

        image_bgr = None
        st.markdown('<p class="label">2 · Provide face</p>', unsafe_allow_html=True)
        if source == "Upload image":
            upload = st.file_uploader(
                "Face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="recognize_upload",
            )
            if upload is not None:
                image_bgr = bytes_to_bgr(upload.getvalue())
                st.image(upload.getvalue(), use_container_width=True)
        else:
            shot = st.camera_input("Camera", key="recognize_cam")
            if shot is not None:
                image_bgr = bytes_to_bgr(shot.getvalue())

        run = st.button(
            "Run recognition",
            type="primary",
            use_container_width=True,
            disabled=image_bgr is None,
            key="btn_recognize",
        )

    with right:
        st.markdown('<p class="label">3 · Result</p>', unsafe_allow_html=True)

        if run and image_bgr is not None:
            with st.spinner("Aligning face and matching…"):
                result = recognize(image_bgr)
            st.session_state["last_result"] = result
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

        result = st.session_state.get("last_result")
        if not result:
            st.markdown(
                """
                <div class="frame">
                  <p class="muted">No result yet. Add a face on the left and press <b>Run recognition</b>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            aligned = (result["aligned"] * 255).astype(np.uint8)
            cls = level_class(result["found"], result["level"])
            width = max(0.0, min(100.0, float(result["confidence"])))
            name = html.escape(str(result["label"]))
            detail = html.escape(str(result.get("detail", "")))
            st.image(aligned, caption="Aligned face", use_container_width=True, clamp=True)
            st.markdown(
                f"""
                <div class="frame" style="margin-top:0.85rem;">
                  <p class="identity">{name}</p>
                  <div class="kv">
                    <i>Confidence</i><b class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</b>
                    <i>Method</i><b>SIFT + HOG + Gabor</b>
                  </div>
                  <div class="bar"><i style="width:{width:.1f}%;"></i></div>
                  <p class="muted">{detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Register ──────────────────────────────────────────────
elif page == "Register":
    st.markdown(
        """
        <div class="sec-head">
          <h2 class="sec-title">Register</h2>
          <p class="sec-desc">Enroll a person with a name and one or more clear face images. Extra angles improve stability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.05], gap="large")
    with c1:
        st.markdown('<p class="label">Person details</p>', unsafe_allow_html=True)
        name = st.text_input("Full name", placeholder="e.g. Alex Lacamoire", key="register_name")
        st.markdown('<p class="label">Face images</p>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="register_uploads",
        )
        cam = st.camera_input("Optional camera photo", key="register_cam")

    images: list[np.ndarray] = []
    if uploads:
        for f in uploads:
            img = bytes_to_bgr(f.getvalue())
            if img is not None:
                images.append(img)
    if cam is not None:
        img = bytes_to_bgr(cam.getvalue())
        if img is not None:
            images.append(img)

    with c2:
        st.markdown('<p class="label">Preview</p>', unsafe_allow_html=True)
        if not images:
            st.markdown(
                """
                <div class="frame">
                  <p class="muted">Previews appear here after you upload or capture photos.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{len(images)} ready")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images[:6]):
                cols[i % len(cols)].image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("Save to gallery", type="primary", use_container_width=True, key="btn_register"):
        with st.spinner("Building face encoding…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.session_state.pop("last_result", None)
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        """
        <p class="foot-note">
          Streamlit Cloud storage is temporary. To keep new enrollments permanently,
          commit the updated <code>Faces/</code> folder to GitHub.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ── Gallery ───────────────────────────────────────────────
else:
    st.markdown(
        """
        <div class="sec-head">
          <h2 class="sec-title">Gallery</h2>
          <p class="sec-desc">People currently enrolled and available for recognition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not faces_map:
        st.markdown(
            """
            <div class="frame">
              <p class="muted">No faces enrolled yet. Open <b>Register</b> to add the first person.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        people = [faces_map[fid] for fid in sorted(faces_map.keys(), key=lambda x: int(x))]
        for row_start in range(0, len(people), 3):
            cols = st.columns(3, gap="medium")
            for col, data in zip(cols, people[row_start : row_start + 3]):
                sample = ROOT / data.get("directory", "")
                jpgs = sorted(sample.glob("*.jpg")) if sample.exists() else []
                with col:
                    st.markdown('<div class="frame">', unsafe_allow_html=True)
                    if jpgs:
                        st.image(str(jpgs[0]), use_container_width=True)
                    st.markdown(
                        f"""
                        <p class="tile-name">{html.escape(str(data['name']))}</p>
                        <p class="tile-meta">ID {html.escape(str(data['id']))} · {int(data['num_images'])} image(s)</p>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
