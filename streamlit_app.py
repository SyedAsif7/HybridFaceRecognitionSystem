"""
Streamlit UI — Hybrid Face Recognition System
==============================================
Research fusion (SIFT + HOG + Gabor) + custom face database.
"""

from __future__ import annotations

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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --ink: #0c1f26;
  --muted: #5a6f78;
  --line: #c9d6dc;
  --panel: rgba(255,255,255,0.72);
  --teal: #176b7a;
  --teal-deep: #0e4a56;
  --ok: #1b6b4a;
  --warn: #8a6a12;
  --bad: #9b2c2c;
  --mist: #dce8ec;
}

html, body, [class*="css"], .stApp {
  font-family: 'Source Sans 3', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 8% -10%, #c5dde4 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #d5e4df 0%, transparent 50%),
    linear-gradient(180deg, #eef4f6 0%, #e3ecef 45%, #dfe8eb 100%);
}

.block-container {
  padding-top: 1.75rem !important;
  padding-bottom: 3rem !important;
  max-width: 1080px;
}

h1, h2, h3, .brand-title {
  font-family: 'Fraunces', Georgia, serif !important;
  font-weight: 550 !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
}

/* Hide default chrome noise */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

div[data-testid="stToolbar"] { display: none; }

/* Brand */
.brand-wrap {
  margin-bottom: 1.75rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--line);
}
.brand-kicker {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal-deep);
  margin: 0 0 0.45rem 0;
}
.brand-title {
  font-size: clamp(2.1rem, 4vw, 2.85rem);
  line-height: 1.05;
  margin: 0 0 0.55rem 0;
}
.brand-sub {
  margin: 0;
  max-width: 36rem;
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.45;
}
.brand-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.25rem;
  margin-top: 1rem;
  color: var(--muted);
  font-size: 0.9rem;
}
.brand-meta strong { color: var(--ink); font-weight: 600; }

/* Panels */
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1.15rem 1.25rem 1.25rem;
  backdrop-filter: blur(8px);
}
.panel-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--teal-deep);
  margin: 0 0 0.75rem 0;
}
.section-title {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.45rem;
  margin: 0 0 0.35rem 0;
  color: var(--ink);
}
.section-help {
  color: var(--muted);
  font-size: 0.95rem;
  margin: 0 0 1.1rem 0;
  line-height: 1.4;
}

/* Result */
.result-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1.25rem;
  min-height: 220px;
}
.result-name {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2rem;
  line-height: 1.1;
  margin: 0 0 0.65rem 0;
  color: var(--ink);
}
.result-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
  font-size: 0.95rem;
}
.result-key { color: var(--muted); min-width: 6.5rem; }
.result-val { font-weight: 600; color: var(--ink); }
.level-high { color: var(--ok); }
.level-medium { color: var(--warn); }
.level-low, .level-none { color: var(--bad); }
.empty-state {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
  padding: 1.5rem 0.25rem;
}

/* Gallery */
.person-tile {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.85rem;
  height: 100%;
}
.person-name {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.15rem;
  margin: 0.55rem 0 0.15rem 0;
}
.person-meta { color: var(--muted); font-size: 0.85rem; margin: 0; }

/* Streamlit controls */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.25rem;
}
.stTabs [data-baseweb="tab"] {
  height: auto;
  padding: 0.65rem 1rem;
  background: transparent;
  border: none !important;
  color: var(--muted);
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--teal) !important;
}
.stButton > button[kind="primary"] {
  background: var(--teal-deep) !important;
  border: 1px solid var(--teal-deep) !important;
  color: #fff !important;
  border-radius: 3px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  min-height: 2.7rem;
}
.stButton > button[kind="primary"]:hover {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
}
.stTextInput input, .stFileUploader, div[data-testid="stFileUploader"] {
  border-radius: 3px !important;
}
div[data-testid="stImage"] img {
  border-radius: 3px;
  border: 1px solid var(--line);
}
.note-muted {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.4;
  margin-top: 1.5rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--line);
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


def render_brand(n_faces: int) -> None:
    st.markdown(
        f"""
        <div class="brand-wrap">
          <p class="brand-kicker">SSIEMS Parbhani · M.Tech</p>
          <h1 class="brand-title">Hybrid Face Recognition</h1>
          <p class="brand-sub">
            Eye-landmark alignment and research feature fusion
            (SIFT + HOG + Gabor) for custom identity matching.
          </p>
          <div class="brand-meta">
            <span><strong>{n_faces}</strong> enrolled</span>
            <span>Threshold <strong>{MATCH_THRESHOLD:.0f}%</strong></span>
            <span>Fusion <strong>SIFT · HOG · Gabor</strong></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict) -> None:
    aligned = (result["aligned"] * 255).astype(np.uint8)
    level = result["level"] if result["found"] else "none"
    level_cls = f"level-{level.lower()}" if result["found"] else "level-none"

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown('<p class="panel-label">Aligned face</p>', unsafe_allow_html=True)
        st.image(aligned, use_container_width=True, clamp=True)
    with right:
        st.markdown(
            f"""
            <div class="result-shell">
              <p class="panel-label">Match result</p>
              <p class="result-name">{result['label']}</p>
              <div class="result-row">
                <span class="result-key">Confidence</span>
                <span class="result-val {level_cls}">{result['confidence']:.1f}% · {result['level']}</span>
              </div>
              <div class="result-row">
                <span class="result-key">Method</span>
                <span class="result-val">Research fusion</span>
              </div>
              <p class="section-help" style="margin-top:0.9rem;margin-bottom:0;">{result.get('detail', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_result() -> None:
    st.markdown(
        """
        <div class="result-shell">
          <p class="panel-label">Match result</p>
          <p class="empty-state">
            Upload or capture a face, then run recognition.
            The aligned crop and identity match will appear here.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data ──────────────────────────────────────────────────
index_preview = load_face_index()
faces_map = index_preview.get("registered_faces", {})
n_faces = len(faces_map)

render_brand(n_faces)

tab_recognize, tab_register, tab_gallery = st.tabs(
    ["Recognize", "Register", "Gallery"]
)


# ── Recognize ─────────────────────────────────────────────
with tab_recognize:
    st.markdown('<p class="section-title">Identify a face</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-help">Provide one clear frontal photo. The system aligns the face, extracts fused features, and compares against the enrolled gallery.</p>',
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1.05, 1], gap="large")

    with col_in:
        st.markdown('<p class="panel-label">Input</p>', unsafe_allow_html=True)
        source = st.radio(
            "Source",
            ["Upload", "Camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="recognize_source",
        )

        image_bgr = None
        if source == "Upload":
            upload = st.file_uploader(
                "Face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="recognize_upload",
                label_visibility="collapsed",
            )
            if upload is not None:
                image_bgr = bytes_to_bgr(upload.getvalue())
                st.image(upload.getvalue(), use_container_width=True)
        else:
            shot = st.camera_input("Capture", label_visibility="collapsed", key="recognize_cam")
            if shot is not None:
                image_bgr = bytes_to_bgr(shot.getvalue())

        run = st.button(
            "Run recognition",
            type="primary",
            use_container_width=True,
            disabled=image_bgr is None,
            key="btn_recognize",
        )

    with col_out:
        if run and image_bgr is not None:
            with st.spinner("Aligning face and matching identities…"):
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

        if "last_result" in st.session_state:
            render_result(st.session_state["last_result"])
        else:
            render_empty_result()


# ── Register ──────────────────────────────────────────────
with tab_register:
    st.markdown('<p class="section-title">Enroll a person</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-help">Add a name and one or more clear face images. Multiple angles improve match stability.</p>',
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns([1, 1.1], gap="large")
    with r1:
        st.markdown('<p class="panel-label">Details</p>', unsafe_allow_html=True)
        name = st.text_input(
            "Full name",
            placeholder="e.g. Alex Lacamoire",
            key="register_name",
        )
        uploads = st.file_uploader(
            "Face images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="register_uploads",
        )
        cam = st.camera_input("Optional camera capture", key="register_cam")

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

    with r2:
        st.markdown('<p class="panel-label">Preview</p>', unsafe_allow_html=True)
        if images:
            st.caption(f"{len(images)} image(s) ready")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images[:6]):
                cols[i % len(cols)].image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div class="result-shell">
                  <p class="empty-state">Image previews will show here after you upload or capture photos.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button(
        "Save to gallery",
        type="primary",
        use_container_width=True,
        key="btn_register",
    ):
        with st.spinner("Building face encoding…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.cache_data.clear()
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        """
        <p class="note-muted">
          On Streamlit Cloud, enrollments written at runtime may reset when the app sleeps.
          Commit an updated <code>Faces/</code> folder to GitHub to keep identities permanently.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ── Gallery ───────────────────────────────────────────────
with tab_gallery:
    st.markdown('<p class="section-title">Enrolled gallery</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-help">People currently available for recognition.</p>',
        unsafe_allow_html=True,
    )

    if not faces_map:
        st.markdown(
            """
            <div class="result-shell">
              <p class="empty-state">No faces enrolled yet. Open the Register tab to add the first person.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        people = [
            faces_map[fid]
            for fid in sorted(faces_map.keys(), key=lambda x: int(x))
        ]
        rows = (len(people) + 2) // 3
        idx = 0
        for _ in range(rows):
            cols = st.columns(3, gap="medium")
            for c in cols:
                if idx >= len(people):
                    break
                data = people[idx]
                idx += 1
                sample = ROOT / data.get("directory", "")
                jpgs = sorted(sample.glob("*.jpg")) if sample.exists() else []
                with c:
                    if jpgs:
                        st.image(str(jpgs[0]), use_container_width=True)
                    st.markdown(
                        f"""
                        <div class="person-tile">
                          <p class="person-name">{data['name']}</p>
                          <p class="person-meta">ID {data['id']} · {data['num_images']} image(s)</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
