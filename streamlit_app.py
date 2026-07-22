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

# Keep CSS minimal — heavy BaseWeb overrides cause text overlap on Streamlit Cloud
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

:root {
  --bg: #e8eef1;
  --card: #ffffff;
  --ink: #12202a;
  --mute: #5a6b75;
  --line: #cfd9df;
  --teal: #0a4f44;
  --soft: #e7f2ef;
  --ok: #0d6b5c;
  --warn: #9a6b12;
  --bad: #a33a3a;
}

.stApp {
  background: var(--bg);
  font-family: 'DM Sans', sans-serif;
  color: var(--ink);
}

section[data-testid="stSidebar"] { display: none !important; }

.block-container {
  max-width: 980px !important;
  padding-top: 2.5rem !important;
  padding-bottom: 3rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

/* Custom content only — never restyle Streamlit tab/radio internals */
.ui-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 1.35rem 1.4rem;
  margin: 0 0 1.15rem 0;
  position: relative;
  z-index: 1;
  overflow: visible;
}
.ui-title {
  margin: 0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.25;
  color: var(--ink);
}
.ui-byline {
  margin: 0.65rem 0 0 0;
  font-size: 0.95rem;
  line-height: 1.55;
  color: var(--mute);
}
.ui-byline a {
  color: #1c3d4f;
  font-weight: 600;
  text-decoration: underline;
  text-underline-offset: 2px;
}
.ui-lead {
  margin: 0.85rem 0 0 0;
  font-size: 1rem;
  line-height: 1.55;
  color: var(--mute);
}
.ui-stats {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
}
.ui-stat {
  min-width: 5.5rem;
}
.ui-stat b {
  display: block;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
  color: var(--ink);
}
.ui-stat span {
  display: block;
  margin-top: 0.2rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--mute);
  line-height: 1.3;
}
.ui-h {
  margin: 0 0 0.4rem 0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.3;
  color: var(--ink);
}
.ui-p {
  margin: 0 0 1rem 0;
  font-size: 0.95rem;
  line-height: 1.5;
  color: var(--mute);
}
.ui-label {
  display: block;
  margin: 0 0 0.5rem 0;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #1c3d4f;
  line-height: 1.3;
}
.ui-box {
  background: #f7fafb;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 1rem;
  margin-top: 0.75rem;
}
.ui-name {
  margin: 0 0 0.75rem 0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.25;
}
.ui-line {
  margin: 0 0 0.4rem 0;
  font-size: 0.95rem;
  line-height: 1.45;
}
.ui-line span { color: var(--mute); margin-right: 0.4rem; }
.ok { color: var(--ok); font-weight: 700; }
.warn { color: var(--warn); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
.ui-soft {
  margin: 0.5rem 0 0 0;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--mute);
}
.ui-person {
  margin: 0.6rem 0 0.15rem 0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.3;
}
.ui-meta {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.4;
  color: var(--mute);
}
.ui-note {
  margin: 1.25rem 0 0 0;
  padding-top: 0.9rem;
  border-top: 1px solid var(--line);
  font-size: 0.85rem;
  line-height: 1.5;
  color: var(--mute);
}

div[data-testid="stImage"] img {
  border: 1px solid var(--line);
  border-radius: 4px;
}
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
  color: #fff !important;
  font-weight: 650 !important;
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
    <div class="ui-card">
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
      <div class="ui-stats">
        <div class="ui-stat"><b>{n_faces}</b><span>Enrolled</span></div>
        <div class="ui-stat"><b>{MATCH_THRESHOLD:.0f}%</b><span>Threshold</span></div>
        <div class="ui-stat"><b>3</b><span>Features</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# selectbox avoids Streamlit tab CSS overlap bugs
page = st.selectbox(
    "Section",
    ["Recognize", "Register", "Gallery"],
    index=0,
)

if page == "Recognize":
    st.markdown(
        '<p class="ui-h">Recognize a face</p>'
        '<p class="ui-p">Upload or capture one clear frontal photo, then run matching.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<span class="ui-label">Input</span>', unsafe_allow_html=True)
        source = st.radio(
            "Image source",
            ["Upload image", "Camera"],
            horizontal=True,
            key="src",
        )
        image_bgr = None
        if source == "Upload image":
            up = st.file_uploader(
                "Face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="up",
            )
            if up is not None:
                image_bgr = bytes_to_bgr(up.getvalue())
                st.image(up.getvalue(), use_container_width=True)
        else:
            cam = st.camera_input("Camera photo", key="cam")
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
        st.markdown('<span class="ui-label">Result</span>', unsafe_allow_html=True)
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
            st.info("Your match result will appear here after recognition.")
        else:
            aligned = (result["aligned"] * 255).astype(np.uint8)
            cls = tone(result["found"], result["level"])
            st.image(aligned, caption="Aligned face", use_container_width=True, clamp=True)
            st.markdown(
                f"""
                <div class="ui-box">
                  <p class="ui-name">{html.escape(str(result['label']))}</p>
                  <p class="ui-line">
                    <span>Confidence</span>
                    <b class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</b>
                  </p>
                  <p class="ui-line"><span>Method</span><b>SIFT + HOG + Gabor</b></p>
                  <p class="ui-soft">{html.escape(str(result.get('detail', '')))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Register":
    st.markdown(
        '<p class="ui-h">Register a person</p>'
        '<p class="ui-p">Add a name and one or more clear face images.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<span class="ui-label">Details</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="ui-label">Preview</span>', unsafe_allow_html=True)
        if not images:
            st.info("Previews appear here after upload.")
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

else:
    st.markdown(
        '<p class="ui-h">Enrolled gallery</p>'
        '<p class="ui-p">People currently available for recognition.</p>',
        unsafe_allow_html=True,
    )
    if not faces_map:
        st.info("No faces yet. Open Register to add someone.")
    else:
        people = [faces_map[k] for k in sorted(faces_map.keys(), key=lambda x: int(x))]
        for start in range(0, len(people), 3):
            cols = st.columns(3, gap="medium")
            for col, person in zip(cols, people[start : start + 3]):
                folder = ROOT / person.get("directory", "")
                photos = sorted(folder.glob("*.jpg")) if folder.exists() else []
                with col:
                    if photos:
                        st.image(str(photos[0]), use_container_width=True)
                    st.markdown(
                        f'<p class="ui-person">{html.escape(str(person["name"]))}</p>'
                        f'<p class="ui-meta">ID {html.escape(str(person["id"]))} · '
                        f'{int(person["num_images"])} image(s)</p>',
                        unsafe_allow_html=True,
                    )
