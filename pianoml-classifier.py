"""
pianoml-classifier.py — Batch-classify ungraded scores on PianoML.

Searches for scores with no grade, downloads their MIDI, runs the
piano-syllabus-classifier inference, and updates each score's grade.

Usage:
    python pianoml-classifier.py [--model_dir ./ps_model] [--limit 100]
"""

import argparse
import getpass
import sys
import tempfile
from pathlib import Path

import requests
import torch

from inference import load_model, predict_grade

API_BASE_URL = "https://api.pianoml.org"
LOGIN_EMAIL = "emmanuel.florent@gmail.com"


def login(session: requests.Session, password: str) -> str:
    """Authenticate and set the Bearer token on the session."""
    resp = session.post(
        f"{API_BASE_URL}/account/login",
        json={"email": LOGIN_EMAIL, "password": password},
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        print("Erreur: pas de token reçu")
        sys.exit(1)
    session.headers.update({"Authorization": f"Bearer {token}"})
    print(f"✓ Connecté ({data.get('username')})")
    return token


def fetch_ungraded_batch(session: requests.Session, limit: int, offset: int = 0) -> list[dict]:
    """Fetch a single batch of scores where grade is NONE."""
    resp = session.get(
        f"{API_BASE_URL}/score/search",
        #params={"offset": offset, "limit": limit, "gradeStart": "NONE", "gradeEnd": "NONE"},
        params={"gradeStart": "NONE", "gradeEnd": "NONE"},
    )
    resp.raise_for_status()
    return resp.json()


def download_midi(session: requests.Session, score: dict) -> bytes | None:
    """Download the MIDI binary for a score. Returns bytes or None."""
    owner_id = score.get("owner_id")
    score_id = score.get("id")
    version = score.get("version", 1)
    if not owner_id or not score_id:
        return None
    url = f"{API_BASE_URL}/score/{owner_id}/{score_id}/midi/{version}/1"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.content
    print(f"  ⚠ Téléchargement MIDI échoué ({resp.status_code}): {score_id}")
    return None


def update_score_grade(session: requests.Session, score: dict, grade: int) -> bool:
    """PUT the score with the new grade."""
    score_id = score["id"]
    payload = dict(score)
    payload["grade"] = grade
    resp = session.put(
        f"{API_BASE_URL}/score/{score_id}/info",
        json=payload,
    )
    if resp.status_code == 200:
        return True
    print(f"  ⚠ Mise à jour échouée ({resp.status_code}): {resp.text[:200]}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Classify ungraded PianoML scores")
    parser.add_argument("--model_dir", default="./ps_model", help="Model directory")
    parser.add_argument("--limit", type=int, default=10, help="Batch size per request")
    args = parser.parse_args()

    password = getpass.getpass("PianoML password: ")

    # Login
    session = requests.Session()
    login(session, password)

    # Load ML model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble, normalizer, cfg = load_model(args.model_dir)
    print(f"✓ Modèle chargé depuis {args.model_dir}")

    # Process ungraded scores batch by batch
    updated = 0
    errors = 0
    total = 0
    skipped_ids: set[str] = set()  # scores that failed and still have grade=NONE

    while True:
        # offset past scores we already tried but couldn't update
        batch = fetch_ungraded_batch(session, args.limit, offset=len(skipped_ids))
        if not batch:
            break

        # If every score in the batch was already seen, we're done
        new_scores = [s for s in batch if s["id"] not in skipped_ids]
        if not new_scores:
            print(f"\n→ Plus aucune nouvelle partition à traiter (skip {len(skipped_ids)})")
            break

        print(f"\n→ Batch de {len(new_scores)} partitions récupéré")

        for score in new_scores:
            total += 1
            title = score.get("title", "?")
            author = score.get("author", "?")
            score_id = score["id"]
            print(f"[{total}] {author} — {title} ({score_id})")
            

            midi_data = download_midi(session, score)
            if midi_data is None:
                errors += 1
                skipped_ids.add(score_id)
                continue

            # Write MIDI to a temp file and run inference
            try:
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                    tmp.write(midi_data)
                    tmp_path = Path(tmp.name)

                result = predict_grade(tmp_path, ensemble, normalizer, device=device)
                grade = round(result["predicted_value"])
                print(f"  → grade={grade} (valeur={result['predicted_value']}, label={result['predicted_grade']})")

                if update_score_grade(session, score, grade):
                    updated += 1
                    print(f"  ✓ Mis à jour")
                else:
                    errors += 1
                    skipped_ids.add(score_id)
            except Exception as e:
                print(f"  ✗ Erreur: {e}")
                errors += 1
                skipped_ids.add(score_id)
            finally:
                tmp_path.unlink(missing_ok=True)

        if len(batch) < args.limit:
            break
        

    print(f"\nTerminé: {updated} mis à jour, {errors} erreurs sur {total} partitions.")


if __name__ == "__main__":
    main()
