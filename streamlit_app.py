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
.hero-by {
  margin: 0.55rem 0 0;
  font-size: 0.95rem;
  line-height: 1.5;
  color: var(--mute);
}
.hero-by a {
  color: var(--teal);
  font-weight: 700;
  text-decoration: none;
}
.hero-by a:hover { text-decoration: underline; }
.hero-desc {
  margin: 0.75rem 0 0;
  max-width: 40rem;
  font-size: 1rem;
  line-height: 1.55;
  color: var(--mute);
}
.hero-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 1.1rem;
}
.chip {
  background: var(--teal-soft);
  border: 1px solid #c5ddd6;
  border-radius: 999px;
  padding: 0.4rem 0.8rem;
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--teal);
  line-height: 1.2;
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

# ── Header ────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1 class="hero-title">Hybrid Face Recognition</h1>
      <p class="hero-by">
        Made by <strong>Syed Asif</strong> ·
        <a href="https://www.linkedin.com/in/the-syed-asif" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        ·
        <a href="https://github.com/SyedAsif7" target="_blank" rel="noopener noreferrer">GitHub</a>
      </p>
      <p class="hero-desc">
        Upload a photo, align the face, fuse SIFT + HOG + Gabor features,
        and match against your enrolled gallery.
      </p>
      <div class="hero-row">
        <div class="chip">{n_faces} enrolled</div>
        <div class="chip">{MATCH_THRESHOLD:.0f}% threshold</div>
        <div class="chip">SIFT · HOG · Gabor</div>
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
        '<p class="sec-help">Step 1: add a face photo. Step 2: run recognition. Step 3: see the match.</p>',
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
        )

        image_bgr = None
        if source == "Upload image":
            up = st.file_uploader(
                "Choose a clear face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="up",
                help="JPG, PNG, BMP, or WEBP — one face works best.",
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
        if image_bgr is None:
            st.caption("Add a photo to enable recognition.")

    with right:
        st.markdown('<p class="panel-title">2 · Match result</p>', unsafe_allow_html=True)

        if go and image_bgr is not None:
            with st.spinner("Aligning face and matching…"):
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
            st.info("Results show here after you run recognition.")
        else:
            aligned = (result["aligned"] * 255).astype(np.uint8)
            cls = tone(result["found"], result["level"])
            st.image(aligned, caption="Aligned face crop", use_container_width=True, clamp=True)
            st.progress(min(1.0, max(0.0, float(result["confidence"]) / 100.0)))
            st.markdown(
                f"""
                <div class="result-box">
                  <p class="result-name">{html.escape(str(result['label']))}</p>
                  <p class="result-line">
                    <span class="k">Confidence</span>
                    <span class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</span>
                  </p>
                  <p class="result-line"><span class="k">Method</span>SIFT + HOG + Gabor</p>
                  <p class="result-line" style="color:var(--mute);margin-top:0.55rem;">
                    {html.escape(str(result.get('detail', '')))}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            placeholder="e.g. Alex Lacamoire",
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
        with st.spinner("Building face encoding…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.session_state.pop("result", None)
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        '<p class="foot">Tip: on Streamlit Cloud, enrollments may reset when the app sleeps. '
        "Commit updated <code>Faces/</code> files to GitHub to keep them permanently.</p>",
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
        people = [faces_map[k] for k in sorted(faces_map.keys(), key=lambda x: int(x))]
        for start in range(0, len(people), 3):
            cols = st.columns(3, gap="medium")
            for col, person in zip(cols, people[start : start + 3]):
                folder = ROOT / person.get("directory", "")
                photos = sorted(folder.glob("*.jpg")) if folder.exists() else []
                with col:
                    with st.container(border=True):
                        if photos:
                            st.image(str(photos[0]), use_container_width=True)
                        st.markdown(
                            f'<p class="person-name">{html.escape(str(person["name"]))}</p>'
                            f'<p class="person-meta">ID {html.escape(str(person["id"]))} · '
                            f'{int(person["num_images"])} image(s)</p>',
                            unsafe_allow_html=True,
                        )
