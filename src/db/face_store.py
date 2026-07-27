"""
Face storage — Supabase (cloud) or local JSON fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.features.extractors import extract_research_fusion
from src.features.sift_matcher import (
    MIN_MATCH_MARGIN,
    MIN_SIFT_MATCHES,
    extract_sift_descriptors,
    matches_to_confidence,
    pick_best_match,
)
from src.preprocessing.processor import detect_and_align_face, face_detected

ROOT = Path(__file__).resolve().parents[2]
FACES_DIR = ROOT / "Faces"
FACES_INDEX = FACES_DIR / "index.json"
BUCKET = "faces"
TABLE = "registered_faces"


def _slug(name: str) -> str:
    return name.strip().replace(" ", "_")


def get_credentials() -> tuple[str | None, str | None]:
    """Read Supabase URL + key from env or Streamlit secrets."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if not url and "SUPABASE_URL" in st.secrets:
                url = st.secrets["SUPABASE_URL"]
            if not key:
                if "SUPABASE_KEY" in st.secrets:
                    key = st.secrets["SUPABASE_KEY"]
                elif "SUPABASE_ANON_KEY" in st.secrets:
                    key = st.secrets["SUPABASE_ANON_KEY"]
                elif "supabase" in st.secrets:
                    url = url or st.secrets.supabase.get("url")
                    key = key or st.secrets.supabase.get("key")
    except Exception:
        pass

    return url, key


def is_supabase_enabled() -> bool:
    url, key = get_credentials()
    return bool(url and key)


def _get_client():
    from supabase import create_client

    url, key = get_credentials()
    if not url or not key:
        raise RuntimeError("Supabase credentials not configured.")
    return create_client(url, key)


# ── Local fallback ────────────────────────────────────────

def _load_local() -> dict:
    if not FACES_INDEX.exists():
        return {"registered_faces": {}, "next_id": 1}
    with open(FACES_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_local(index: dict) -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACES_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _load_supabase() -> dict[str, Any]:
    client = _get_client()
    rows = client.table(TABLE).select("*").order("created_at").execute().data or []
    faces: dict[str, Any] = {}
    for row in rows:
        faces[str(row["id"])] = {
            "id": row["id"],
            "name": row["name"],
            "num_images": row["num_images"],
            "directory": row.get("image_folder") or "",
            "avg_encoding": row["avg_encoding"],
            "sift_descriptors": row.get("sift_descriptors") or [],
            "storage": True,
        }
    return {"registered_faces": faces, "next_id": len(faces) + 1}


MATCH_THRESHOLD = 65.0  # legacy display only; recognition uses SIFT matching
MIN_SIFT_MATCHES_REQUIRED = MIN_SIFT_MATCHES
NOT_IDENTIFIED_LABEL = "Not Identified"
NO_GALLERY_LABEL = "Not Found"


def confidence_level(pct: float) -> str:
    if pct >= 80:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"


def safe_load_faces() -> tuple[dict[str, Any], str | None]:
    """Load faces; return (index, error_message). error_message is set only on failure."""
    if is_supabase_enabled():
        try:
            return _load_supabase(), None
        except Exception as exc:
            return {"registered_faces": {}, "next_id": 1}, f"Supabase read failed: {exc}"
    return _load_local(), None


def load_faces() -> dict[str, Any]:
    index, err = safe_load_faces()
    if err:
        raise RuntimeError(err)
    return index


def _build_sift_descriptor_sets(aligned_images: list[np.ndarray]) -> list[list[list[float]]]:
    sets: list[list[list[float]]] = []
    for aligned in aligned_images:
        desc = extract_sift_descriptors(aligned)
        if desc:
            sets.append(desc)
    return sets


def _sift_sets_from_local_images(directory: str) -> list[list[list[float]]]:
    folder = ROOT / directory.replace("\\", "/")
    if not folder.exists():
        return []

    sets: list[list[list[float]]] = []
    for jpg in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        aligned = detect_and_align_face(img)
        desc = extract_sift_descriptors(aligned)
        if desc:
            sets.append(desc)
    return sets


def _sift_sets_from_supabase_images(folder: str, num_images: int) -> list[list[list[float]]]:
    if not folder or not is_supabase_enabled():
        return []

    client = _get_client()
    sets: list[list[list[float]]] = []
    for idx in range(1, max(num_images, 1) + 1):
        path = f"{folder}/{idx}.jpg"
        try:
            raw = client.storage.from_(BUCKET).download(path)
        except Exception:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        aligned = detect_and_align_face(img)
        desc = extract_sift_descriptors(aligned)
        if desc:
            sets.append(desc)
    return sets


def _descriptor_sets_for_person(data: dict[str, Any]) -> list[list[list[float]]]:
    stored = data.get("sift_descriptors") or []
    if stored:
        return stored

    if data.get("storage"):
        rebuilt = _sift_sets_from_supabase_images(
            data.get("directory", ""),
            int(data.get("num_images") or 1),
        )
    else:
        rebuilt = _sift_sets_from_local_images(data.get("directory", ""))

    if rebuilt:
        data["sift_descriptors"] = rebuilt
    return rebuilt


def recognize_face(image_bgr: np.ndarray, threshold: float = MATCH_THRESHOLD) -> dict:
    """Align face, match against enrolled gallery, return result dict."""
    if image_bgr is None or not hasattr(image_bgr, "size") or image_bgr.size == 0:
        blank = np.zeros((128, 128), dtype=np.float64)
        return {
            "label": NOT_IDENTIFIED_LABEL,
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": blank,
            "detail": "Image could not be read. Upload a valid JPG, PNG, or WEBP file.",
        }

    if not face_detected(image_bgr):
        return {
            "label": NOT_IDENTIFIED_LABEL,
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": detect_and_align_face(image_bgr),
            "detail": "No face detected in the image. Use a clear front-facing photo.",
        }

    face = detect_and_align_face(image_bgr)
    faces = load_faces().get("registered_faces", {})

    if not faces:
        return {
            "label": NO_GALLERY_LABEL,
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": face,
            "detail": "No faces are enrolled yet. Register a person first.",
        }

    query_desc = extract_sift_descriptors(face)
    if not query_desc:
        return {
            "label": NOT_IDENTIFIED_LABEL,
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": face,
            "detail": "Could not extract face features. Use a clearer, front-facing photo.",
        }

    candidates: list[tuple[str, Any, list[list[list[float]]]]] = []
    for data in faces.values():
        desc_sets = _descriptor_sets_for_person(data)
        if desc_sets:
            candidates.append((data["name"], data["id"], desc_sets))

    if not candidates:
        return {
            "label": NO_GALLERY_LABEL,
            "confidence": 0.0,
            "level": "Low",
            "found": False,
            "aligned": face,
            "detail": "No usable face data in the gallery. Re-register enrolled people.",
        }

    best_name, best_id, best_count, margin, ranked = pick_best_match(query_desc, candidates)
    confidence = matches_to_confidence(best_count)

    if best_name is None:
        top_name, top_count = ranked[0] if ranked else ("—", 0)
        if best_count < MIN_SIFT_MATCHES_REQUIRED:
            detail = (
                f"No strong match found (best: {top_name} with {best_count} keypoint matches, "
                f"need {MIN_SIFT_MATCHES_REQUIRED}+)."
            )
        else:
            detail = (
                f"Match too close to call (best: {top_count} matches, margin {margin}, "
                f"need {MIN_MATCH_MARGIN}+ gap)."
            )
        return {
            "label": NOT_IDENTIFIED_LABEL,
            "confidence": float(confidence),
            "level": confidence_level(confidence),
            "found": False,
            "aligned": face,
            "detail": detail,
            "match_count": int(best_count),
            "ranked": ranked,
        }

    return {
        "label": best_name,
        "id": best_id,
        "confidence": float(confidence),
        "level": confidence_level(confidence),
        "found": True,
        "aligned": face,
        "detail": f"Matched with SIFT keypoints ({best_count} good matches, margin {margin}).",
        "match_count": int(best_count),
        "ranked": ranked,
    }


def _find_existing_name(client, name: str) -> dict | None:
    rows = client.table(TABLE).select("id,name").execute().data or []
    name_l = name.lower()
    for row in rows:
        if str(row.get("name", "")).lower() == name_l:
            return row
    return None


def image_url(person: dict) -> str | None:
    """Public URL for gallery thumbnail."""
    if person.get("storage") and is_supabase_enabled():
        folder = person.get("directory", "")
        if not folder:
            return None
        url, _ = get_credentials()
        return f"{url}/storage/v1/object/public/{BUCKET}/{folder}/1.jpg"
    folder = ROOT / person.get("directory", "")
    photos = sorted(folder.glob("*.jpg")) if folder.exists() else []
    return str(photos[0]) if photos else None


def register_person(name: str, images_bgr: list[np.ndarray]) -> tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Enter a person name."
    if not images_bgr:
        return False, "Add at least one image."

    valid_images = [img for img in images_bgr if img is not None and getattr(img, "size", 0) > 0]
    if not valid_images:
        return False, "Image could not be read. Upload a valid JPG, PNG, or WEBP file."

    encodings: list[np.ndarray] = []
    aligned_images: list[np.ndarray] = []

    for img in valid_images:
        if not face_detected(img):
            continue
        aligned = detect_and_align_face(img)
        aligned_images.append(aligned)
        encodings.append(extract_research_fusion(aligned))

    if not encodings:
        return False, "Face not detected. Image could not be stored — use a clear front-facing photo."

    avg_encoding = np.mean(encodings, axis=0).tolist()
    sift_sets = _build_sift_descriptor_sets(aligned_images)
    if not sift_sets:
        return False, "Could not extract face features from the images. Try clearer photos."

    slug = _slug(name)
    folder = slug

    if is_supabase_enabled():
        return _register_supabase(name, slug, folder, aligned_images, avg_encoding, sift_sets, len(encodings))
    return _register_local(name, slug, aligned_images, avg_encoding, sift_sets, len(encodings))


def _register_local(
    name: str,
    slug: str,
    aligned_images: list[np.ndarray],
    avg_encoding: list[float],
    sift_sets: list[list[list[float]]],
    n: int,
) -> tuple[bool, str]:
    index = _load_local()
    person_dir = FACES_DIR / slug
    person_dir.mkdir(parents=True, exist_ok=True)

    for idx, aligned in enumerate(aligned_images):
        cv2.imwrite(str(person_dir / f"{idx + 1}.jpg"), (aligned * 255).astype(np.uint8))

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
        "num_images": n,
        "directory": str(person_dir.relative_to(ROOT)).replace("\\", "/"),
        "avg_encoding": avg_encoding,
        "sift_descriptors": sift_sets,
    }
    _save_local(index)
    return True, f"Registered {name} with {n} image(s)."


def _register_supabase(
    name: str,
    slug: str,
    folder: str,
    aligned_images: list[np.ndarray],
    avg_encoding: list[float],
    sift_sets: list[list[list[float]]],
    n: int,
) -> tuple[bool, str]:
    try:
        client = _get_client()

        for idx, aligned in enumerate(aligned_images):
            _, buf = cv2.imencode(".jpg", (aligned * 255).astype(np.uint8))
            path = f"{folder}/{idx + 1}.jpg"
            client.storage.from_(BUCKET).upload(
                path,
                buf.tobytes(),
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

        existing = _find_existing_name(client, name)

        row = {
            "name": name,
            "num_images": n,
            "avg_encoding": avg_encoding,
            "image_folder": folder,
            "sift_descriptors": sift_sets,
        }

        if existing:
            try:
                client.table(TABLE).update(row).eq("id", existing["id"]).execute()
            except Exception:
                row.pop("sift_descriptors", None)
                client.table(TABLE).update(row).eq("id", existing["id"]).execute()
        else:
            try:
                client.table(TABLE).insert(row).execute()
            except Exception:
                row.pop("sift_descriptors", None)
                client.table(TABLE).insert(row).execute()

        return True, f"Registered {name} with {n} image(s) — saved to Supabase."
    except Exception as exc:
        return False, f"Supabase save failed: {exc}"
