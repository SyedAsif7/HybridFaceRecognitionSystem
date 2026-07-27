"""
SIFT descriptor matching for face identity.
More reliable than mean fused-vector cosine similarity on small aligned crops.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

MIN_SIFT_MATCHES = 12
MIN_MATCH_MARGIN = 6
MATCHES_FOR_FULL_CONF = 75
LOWE_RATIO = 0.75


def extract_sift_descriptors(aligned_face: np.ndarray) -> list[list[float]] | None:
    """Extract SIFT descriptors from one aligned grayscale face [0, 1]."""
    if aligned_face is None or not hasattr(aligned_face, "size") or aligned_face.size == 0:
        return None

    img = (aligned_face * 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None or len(descriptors) == 0:
        return None
    return descriptors.tolist()


def count_good_matches(
    query_desc: np.ndarray | list[list[float]] | None,
    stored_descs: list[list[list[float]]] | list[np.ndarray],
) -> int:
    """Count Lowe-ratio SIFT matches between query and stored descriptor sets."""
    if query_desc is None or not stored_descs:
        return 0

    query = np.asarray(query_desc, dtype=np.float32)
    if query.ndim != 2 or len(query) < 2:
        return 0

    bf = cv2.BFMatcher()
    total = 0
    for stored in stored_descs:
        ref = np.asarray(stored, dtype=np.float32)
        if ref.ndim != 2 or len(ref) < 2:
            continue
        pairs = bf.knnMatch(query, ref, k=2)
        for pair in pairs:
            if len(pair) != 2:
                continue
            match, nxt = pair
            if match.distance < LOWE_RATIO * nxt.distance:
                total += 1
    return total


def matches_to_confidence(match_count: int) -> float:
    """Map raw SIFT match count to a 0–100 confidence score."""
    return float(min(100.0, (match_count / MATCHES_FOR_FULL_CONF) * 100.0))


def pick_best_match(
    query_desc: list[list[float]] | None,
    candidates: list[tuple[str, Any, list[list[list[float]]]]],
) -> tuple[str | None, Any | None, int, int, list[tuple[str, int]]]:
    """
    Return (best_name, best_id, best_count, margin, ranked_scores).
    candidates: list of (name, id, sift_descriptor_sets)
    """
    scores: list[tuple[str, Any, int]] = []
    for name, person_id, desc_sets in candidates:
        count = count_good_matches(query_desc, desc_sets)
        scores.append((name, person_id, count))

    scores.sort(key=lambda item: item[2], reverse=True)
    if not scores or scores[0][2] < MIN_SIFT_MATCHES:
        best_count = scores[0][2] if scores else 0
        return None, None, best_count, 0, [(n, c) for n, _, c in scores]

    best_name, best_id, best_count = scores[0]
    second_count = scores[1][2] if len(scores) > 1 else 0
    margin = best_count - second_count
    if margin < MIN_MATCH_MARGIN:
        return None, None, best_count, margin, [(n, c) for n, _, c in scores]

    return best_name, best_id, best_count, margin, [(n, c) for n, _, c in scores]
