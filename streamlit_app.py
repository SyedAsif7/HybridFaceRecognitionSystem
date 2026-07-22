"""
Streamlit UI — Hybrid Face Recognition System
==============================================
Research fusion (SIFT + HOG + Gabor) + custom face database.
No TensorFlow required — suitable for Streamlit Community Cloud.
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
MATCH_THRESHOLD = 65.0  # percent

st.set_page_config(
    page_title="Hybrid Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Instrument+Serif:ital@0;1&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .block-container { padding-top: 1.5rem; max-width: 1100px; }
      h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important; }
      .hero-sub { color: #5c6b73; font-size: 1.05rem; margin-top: -0.4rem; margin-bottom: 1.5rem; }
      .result-card {
        border: 1px solid #d5dde3; border-radius: 12px; padding: 1.1rem 1.25rem;
        background: linear-gradient(160deg, #f7fafb 0%, #eef3f6 100%);
      }
      .name-xl { font-size: 1.85rem; font-family: 'Instrument Serif', Georgia, serif; margin: 0; color: #14323f; }
      .meta { color: #5c6b73; font-size: 0.95rem; }
      .badge-high { color: #0f6b4c; font-weight: 700; }
      .badge-med  { color: #9a6700; font-weight: 700; }
      .badge-low  { color: #a32020; font-weight: 700; }
      .badge-none { color: #6b7280; font-weight: 700; }
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


def level_class(level: str) -> str:
    return {
        "High": "badge-high",
        "Medium": "badge-med",
        "Low": "badge-low",
    }.get(level, "badge-none")


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

    for face_id, data in faces.items():
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
            "detail": f"Best score below {MATCH_THRESHOLD:.0f}% threshold.",
            "score": float(best_score),
        }

    return {
        "label": best_name,
        "id": best_id,
        "confidence": float(confidence),
        "level": confidence_level(confidence),
        "found": True,
        "aligned": face,
        "detail": "Matched via research fusion (SIFT + HOG + Gabor).",
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

    # Replace existing person with same name if present
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


def show_result(result: dict) -> None:
    aligned = (result["aligned"] * 255).astype(np.uint8)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.image(aligned, caption="Aligned face", use_container_width=True, clamp=True)
    with c2:
        cls = level_class(result["level"]) if result["found"] else "badge-none"
        st.markdown(
            f"""
            <div class="result-card">
              <p class="name-xl">{result['label']}</p>
              <p class="meta">Confidence:
                <span class="{cls}">{result['confidence']:.1f}% ({result['level']})</span>
              </p>
              <p class="meta">{result.get('detail', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Hybrid Face Recognition")
    st.caption("SSIEMS Parbhani · M.Tech research demo")
    page = st.radio(
        "Navigate",
        ["Recognize", "Register", "Database"],
        label_visibility="collapsed",
    )
    index_preview = load_face_index()
    n_faces = len(index_preview.get("registered_faces", {}))
    st.markdown(f"**{n_faces}** registered face(s)")
    st.markdown(
        f"Match threshold: **{MATCH_THRESHOLD:.0f}%**  \n"
        "Fusion: SIFT + HOG + Gabor"
    )
    st.info(
        "On Streamlit Cloud, new registrations may reset when the app sleeps. "
        "Commit updated `Faces/` to GitHub to keep them."
    )


st.title("Hybrid Face Recognition")
st.markdown(
    '<p class="hero-sub">Eye-landmark alignment · research feature fusion · custom identity match</p>',
    unsafe_allow_html=True,
)


# ── Recognize ─────────────────────────────────────────────
if page == "Recognize":
    st.subheader("Identify a face")
    source = st.radio("Input", ["Upload image", "Camera"], horizontal=True)

    image_bgr = None
    if source == "Upload image":
        upload = st.file_uploader("Face image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload is not None:
            image_bgr = bytes_to_bgr(upload.getvalue())
            st.image(upload.getvalue(), caption="Uploaded", use_container_width=True)
    else:
        shot = st.camera_input("Capture face")
        if shot is not None:
            image_bgr = bytes_to_bgr(shot.getvalue())

    if image_bgr is not None and st.button("Recognize", type="primary", use_container_width=True):
        with st.spinner("Extracting features and matching…"):
            result = recognize(image_bgr)
        show_result(result)
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


# ── Register ──────────────────────────────────────────────
elif page == "Register":
    st.subheader("Register a person")
    name = st.text_input("Person name", placeholder="e.g. Salman Khan")
    uploads = st.file_uploader(
        "One or more face images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    cam = st.camera_input("Or capture one photo")

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

    if images:
        st.caption(f"{len(images)} image(s) ready")
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images[:4]):
            cols[i].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Register face", type="primary", use_container_width=True):
        with st.spinner("Aligning faces and building encoding…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.cache_data.clear()
        else:
            st.error(msg)


# ── Database ──────────────────────────────────────────────
else:
    st.subheader("Registered faces")
    index = load_face_index()
    faces = index.get("registered_faces", {})
    if not faces:
        st.warning("No faces in the database yet. Use Register to add someone.")
    else:
        for fid in sorted(faces.keys(), key=lambda x: int(x)):
            data = faces[fid]
            with st.container(border=True):
                left, right = st.columns([2, 1])
                left.markdown(f"**{data['name']}**  \nID `{data['id']}` · {data['num_images']} image(s)")
                sample = ROOT / data.get("directory", "")
                jpgs = sorted(sample.glob("*.jpg")) if sample.exists() else []
                if jpgs:
                    right.image(str(jpgs[0]), use_container_width=True)

