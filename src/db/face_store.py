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
from src.preprocessing.processor import detect_and_align_face

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
            "storage": True,
        }
    return {"registered_faces": faces, "next_id": len(faces) + 1}


def load_faces() -> dict[str, Any]:
    if is_supabase_enabled():
        try:
            return _load_supabase()
        except Exception as exc:
            raise RuntimeError(f"Supabase read failed: {exc}") from exc
    return _load_local()


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

    encodings: list[np.ndarray] = []
    aligned_images: list[np.ndarray] = []

    for img in images_bgr:
        aligned = detect_and_align_face(img)
        if aligned is None:
            continue
        aligned_images.append(aligned)
        encodings.append(extract_research_fusion(aligned))

    if not encodings:
        return False, "No valid faces detected in the uploaded images."

    avg_encoding = np.mean(encodings, axis=0).tolist()
    slug = _slug(name)
    folder = slug

    if is_supabase_enabled():
        return _register_supabase(name, slug, folder, aligned_images, avg_encoding, len(encodings))
    return _register_local(name, slug, aligned_images, avg_encoding, len(encodings))


def _register_local(
    name: str,
    slug: str,
    aligned_images: list[np.ndarray],
    avg_encoding: list[float],
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
    }
    _save_local(index)
    return True, f"Registered {name} with {n} image(s)."


def _register_supabase(
    name: str,
    slug: str,
    folder: str,
    aligned_images: list[np.ndarray],
    avg_encoding: list[float],
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

        existing = (
            client.table(TABLE).select("id").eq("name", name).limit(1).execute().data
        )

        row = {
            "name": name,
            "num_images": n,
            "avg_encoding": avg_encoding,
            "image_folder": folder,
        }

        if existing:
            client.table(TABLE).update(row).eq("id", existing[0]["id"]).execute()
        else:
            client.table(TABLE).insert(row).execute()

        return True, f"Registered {name} with {n} image(s) — saved to Supabase."
    except Exception as exc:
        return False, f"Supabase save failed: {exc}"
