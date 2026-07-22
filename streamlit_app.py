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
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --bg: #ebe7df;
  --sheet: #f7f5f0;
  --ink: #1a1a1a;
  --mute: #5e5a54;
  --line: #d2cdc3;
  --accent: #0f5c4c;
  --accent-2: #163a4a;
  --ok: #0f5c4c;
  --warn: #8a6b1f;
  --bad: #8a3030;
  --soft: #e4efe9;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  font-family: 'Manrope', sans-serif !important;
  color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 8%, #dfece6 0%, transparent 42%),
    radial-gradient(circle at 88% 0%, #d9e3ea 0%, transparent 40%),
    linear-gradient(180deg, #f0ece4 0%, var(--bg) 100%) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
#MainMenu, footer { display: none !important; }

.block-container {
  max-width: 880px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
}

/* Shell */
.shell {
  background: var(--sheet);
  border: 1px solid var(--line);
  box-shadow: 0 18px 50px rgba(26, 26, 26, 0.06);
  padding: 1.5rem 1.6rem 1.7rem;
}

.top {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
  padding-bottom: 1.15rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.1rem;
}
.brand {
  margin: 0;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: clamp(1.55rem, 3.5vw, 2rem);
  line-height: 1.15;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.byline {
  margin: 0.55rem 0 0;
  color: var(--mute);
  font-size: 0.9rem;
  line-height: 1.5;
}
.byline a {
  color: var(--accent-2);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid rgba(22, 58, 74, 0.35);
}
.byline a:hover { border-bottom-color: var(--accent-2); }
.meta {
  text-align: right;
  min-width: 9rem;
}
.meta strong {
  display: block;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.4rem;
  line-height: 1;
}
.meta span {
  display: block;
  margin-top: 0.25rem;
  color: var(--mute);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.lead {
  margin: 0 0 1.15rem;
  color: var(--mute);
  font-size: 0.98rem;
  line-height: 1.5;
}

.h {
  margin: 0 0 0.25rem;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.28rem;
}
.p {
  margin: 0 0 1rem;
  color: var(--mute);
  font-size: 0.92rem;
  line-height: 1.45;
}
.lbl {
  margin: 0 0 0.45rem;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--accent-2);
}
.box {
  background: #fff;
  border: 1px solid var(--line);
  padding: 0.95rem 1rem;
}
.name {
  margin: 0 0 0.7rem;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.7rem;
  line-height: 1.1;
}
.row {
  display: grid;
  grid-template-columns: 6.5rem 1fr;
  gap: 0.3rem 0.7rem;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}
.row i { color: var(--mute); font-style: normal; }
.row b { font-weight: 650; }
.ok { color: var(--ok); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.bar {
  height: 7px;
  background: #ece8e0;
  border: 1px solid var(--line);
  margin: 0.2rem 0 0.75rem;
  overflow: hidden;
}
.bar > i {
  display: block;
  height: 100%;
  background: var(--accent);
  font-style: normal;
}
.soft {
  color: var(--mute);
  font-size: 0.88rem;
  line-height: 1.45;
  margin: 0;
}
.person {
  margin: 0.55rem 0 0.1rem;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 1.05rem;
}
.person-sub { margin: 0; color: var(--mute); font-size: 0.8rem; }
.note {
  margin-top: 1.2rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--line);
  color: var(--mute);
  font-size: 0.78rem;
  line-height: 1.45;
}

/* Nav radio */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 0 !important;
  border: 1px solid var(--line) !important;
  background: #fff !important;
  margin: 0 0 1.2rem 0 !important;
}
div[data-testid="stRadio"] label {
  flex: 1 !important;
  justify-content: center !important;
  padding: 0.72rem 0.4rem !important;
  margin: 0 !important;
  border-right: 1px solid var(--line) !important;
  background: transparent !important;
}
div[data-testid="stRadio"] label:last-child { border-right: none !important; }
div[data-testid="stRadio"] label:has(input:checked) {
  background: var(--soft) !important;
  box-shadow: inset 0 -2px 0 var(--accent);
}
div[data-testid="stRadio"] p {
  font-size: 0.9rem !important;
  font-weight: 650 !important;
  text-align: center !important;
}

.stButton > button {
  border-radius: 2px !important;
  min-height: 2.7rem !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}
div[data-testid="stImage"] img {
  border: 1px solid var(--line);
}
div[data-testid="stFileUploader"] section {
  background: #fff !important;
  border: 1px dashed #b9b3a8 !important;
}
.stTextInput input {
  border-radius: 2px !important;
  background: #fff !important;
}

@media (max-width: 700px) {
  .top { flex-direction: column; }
  .meta { text-align: left; }
  .shell { padding: 1.15rem; }
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

st.markdown('<div class="shell">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="top">
      <div>
        <h1 class="brand">Hybrid Face Recognition</h1>
        <p class="byline">
          Made by <strong>Syed Asif</strong> ·
          <a href="https://www.linkedin.com/in/the-syed-asif" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          ·
          <a href="https://github.com/SyedAsif7" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
      </div>
      <div class="meta">
        <strong>{n_faces}</strong>
        <span>Enrolled faces</span>
      </div>
    </div>
    <p class="lead">
      Align a face, fuse SIFT + HOG + Gabor features, and match it to your gallery
      with a confidence score.
    </p>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Section",
    ["Recognize", "Register", "Gallery"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav",
)

# ── Recognize ─────────────────────────────────────────────
if page == "Recognize":
    st.markdown(
        '<p class="h">Recognize</p><p class="p">Add one clear frontal photo, then run matching.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="lbl">Input</p>', unsafe_allow_html=True)
        source = st.radio(
            "Source",
            ["Upload", "Camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="src",
        )
        image_bgr = None
        if source == "Upload":
            up = st.file_uploader(
                "Face image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="up",
            )
            if up is not None:
                image_bgr = bytes_to_bgr(up.getvalue())
                st.image(up.getvalue(), use_container_width=True)
        else:
            cam = st.camera_input("Camera", key="cam")
            if cam is not None:
                image_bgr = bytes_to_bgr(cam.getvalue())

        go = st.button(
            "Run recognition",
            type="primary",
            use_container_width=True,
            disabled=image_bgr is None,
        )

    with right:
        st.markdown('<p class="lbl">Result</p>', unsafe_allow_html=True)
        if go and image_bgr is not None:
            with st.spinner("Matching…"):
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
                '<div class="box"><p class="soft">Results appear here after you run recognition.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            aligned = (result["aligned"] * 255).astype(np.uint8)
            cls = tone(result["found"], result["level"])
            width = max(0.0, min(100.0, float(result["confidence"])))
            st.image(aligned, use_container_width=True, clamp=True)
            st.markdown(
                f"""
                <div class="box" style="margin-top:0.75rem;">
                  <p class="name">{html.escape(str(result['label']))}</p>
                  <div class="row">
                    <i>Confidence</i>
                    <b class="{cls}">{result['confidence']:.1f}% · {html.escape(result['level'])}</b>
                    <i>Method</i>
                    <b>SIFT + HOG + Gabor</b>
                  </div>
                  <div class="bar"><i style="width:{width:.1f}%;"></i></div>
                  <p class="soft">{html.escape(str(result.get('detail', '')))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Register ──────────────────────────────────────────────
elif page == "Register":
    st.markdown(
        '<p class="h">Register</p><p class="p">Enroll a person with a name and one or more face images.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="lbl">Details</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="lbl">Preview</p>', unsafe_allow_html=True)
        if not images:
            st.markdown(
                '<div class="box"><p class="soft">Image previews show here.</p></div>',
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

    if st.button("Save to gallery", type="primary", use_container_width=True):
        with st.spinner("Saving…"):
            ok, msg = register_person(name, images)
        if ok:
            st.success(msg)
            st.session_state.pop("result", None)
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        """
        <p class="note">
          On Streamlit Cloud, new enrollments may reset when the app sleeps.
          Commit updated <code>Faces/</code> files to GitHub to keep them.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ── Gallery ───────────────────────────────────────────────
else:
    st.markdown(
        '<p class="h">Gallery</p><p class="p">People enrolled for recognition.</p>',
        unsafe_allow_html=True,
    )
    if not faces_map:
        st.markdown(
            '<div class="box"><p class="soft">No faces yet. Use Register to add someone.</p></div>',
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
                    if photos:
                        st.image(str(photos[0]), use_container_width=True)
                    st.markdown(
                        f"""
                        <p class="person">{html.escape(str(person['name']))}</p>
                        <p class="person-sub">ID {html.escape(str(person['id']))} · {int(person['num_images'])} image(s)</p>
                        """,
                        unsafe_allow_html=True,
                    )

st.markdown("</div>", unsafe_allow_html=True)
