"""
One-time migration: upload local Faces/ data to Supabase.
Usage:
  set SUPABASE_URL=...
  set SUPABASE_KEY=...
  python scripts/migrate_faces_to_supabase.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.face_store import _get_client, is_supabase_enabled, BUCKET, TABLE  # noqa: E402

INDEX = ROOT / "Faces" / "index.json"


def _name_exists(client, name: str) -> bool:
    name_l = name.lower()
    for row in client.table(TABLE).select("name").execute().data or []:
        if row.get("name", "").lower() == name_l:
            return True
    return False


def main() -> None:
    if not is_supabase_enabled():
        print("Set SUPABASE_URL and SUPABASE_KEY first.")
        sys.exit(1)

    if not INDEX.exists():
        print("No Faces/index.json found.")
        sys.exit(1)

    with open(INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)

    client = _get_client()
    count = 0

    for _fid, data in index.get("registered_faces", {}).items():
        name = data["name"]
        if _name_exists(client, name):
            print(f"  Skipped (already exists): {name}")
            continue

        folder = data.get("directory", "").replace("\\", "/")
        local_dir = ROOT / folder
        if local_dir.exists():
            for jpg in sorted(local_dir.glob("*.jpg")):
                rel = f"{Path(folder).name}/{jpg.name}"
                with open(jpg, "rb") as f:
                    client.storage.from_(BUCKET).upload(
                        rel,
                        f.read(),
                        file_options={"content-type": "image/jpeg", "upsert": "true"},
                    )

        row = {
            "name": data["name"],
            "num_images": data["num_images"],
            "avg_encoding": data["avg_encoding"],
            "image_folder": Path(folder).name if folder else None,
        }
        client.table(TABLE).insert(row).execute()
        count += 1
        print(f"  Migrated: {data['name']}")

    print(f"Done — {count} face(s) uploaded to Supabase.")


if __name__ == "__main__":
    main()
