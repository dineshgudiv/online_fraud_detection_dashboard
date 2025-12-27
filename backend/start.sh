#!/bin/sh
set -e

cd /app

echo "Running migrations with /app/alembic.ini ..."
python - <<'PY'
import os
from pathlib import Path

from sqlalchemy import create_engine, text

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("DATABASE_URL not set; skipping alembic version check.")
else:
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT to_regclass('public.alembic_version') IS NOT NULL"
                )
            ).scalar()
            if exists:
                version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                if version:
                    print(f"Current alembic_version: {version}")
                    if len(version) > 32:
                        print("WARNING: alembic_version exceeds 32 chars; migrations may fail.")
            else:
                print("alembic_version table not found yet.")
    except Exception as exc:
        print(f"Unable to read alembic_version: {exc}")

versions_dir = Path("/app/alembic/versions")
if versions_dir.exists():
    for path in versions_dir.glob("*.py"):
        text_content = path.read_text(encoding="utf-8")
        for line in text_content.splitlines():
            if line.strip().startswith("revision ="):
                revision = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(revision) > 32:
                    print(f"WARNING: revision id in {path.name} exceeds 32 chars: {revision}")
PY
alembic -c /app/alembic.ini upgrade head

echo "Seeding demo data (optional) ..."
python /app/seed.py || true

echo "Starting API ..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
