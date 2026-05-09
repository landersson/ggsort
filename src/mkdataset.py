#!/usr/bin/env python3
"""
Export script for GGSort
Exports images from database to organized output directory structure
"""

import os
import sqlite3
import argparse
import shutil
import sys
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# A bounding box in normalized [0, 1] top-left coords: (x, y, width, height).
Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Diversity-aware sampling helpers
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    """Tuning knobs for diversity-aware sampling. enabled=False keeps legacy stride behavior."""
    enabled: bool = False
    iou_threshold: float = 0.5
    max_gap_seconds: float = 60.0
    budget_exponent: float = 0.5
    seed: int = 0


@dataclass
class ImageRecord:
    """A single image considered for sampling. Replaces the bare sqlite3.Row tuples we used before."""
    id: int
    file_path: str
    datetime_original: str | None = None
    boxes: tuple[Box, ...] = ()
    sequence_id: int | None = None
    sequence_length: int | None = None
    novelty_score: float | None = None
    selected_for_classes: list[int] = field(default_factory=list)


def _iou(a: Box, b: Box) -> float:
    """Intersection-over-union for two (x, y, w, h) boxes in any consistent unit space."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _max_pairwise_iou(boxes_a: tuple[Box, ...], boxes_b: tuple[Box, ...]) -> float:
    """Max IoU across the cross-product of two box lists. 0 if either list is empty."""
    best = 0.0
    for a in boxes_a:
        for b in boxes_b:
            v = _iou(a, b)
            if v > best:
                best = v
    return best


def _parse_exif_datetime(s: str | None) -> datetime | None:
    """Parse an EXIF DateTimeOriginal string ('YYYY:MM:DD HH:MM:SS'). Returns None on failure."""
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y:%m:%d %H:%M:%S')
    except (ValueError, TypeError):
        return None


def _detect_sequences(records: list[ImageRecord], config: DiversityConfig) -> None:
    """Walk records (already chronologically sorted), populate sequence_id/length/novelty in place.

    Adjacent records link into the same sequence iff:
      - both have a parseable datetime_original, AND
      - the time gap is <= max_gap_seconds, AND
      - max pairwise IoU between their box sets >= iou_threshold.

    NULL/unparseable timestamps force a sequence break — those frames become their own
    length-1 sequence with novelty 1.0. This is deliberately conservative: if we don't know
    when a frame was captured, we don't link it to anything.
    """
    if not records:
        return

    seq_id = 0
    records[0].sequence_id = seq_id
    records[0].novelty_score = 1.0
    prev_dt = _parse_exif_datetime(records[0].datetime_original)

    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        curr_dt = _parse_exif_datetime(curr.datetime_original)

        link = False
        if prev_dt is not None and curr_dt is not None:
            gap = (curr_dt - prev_dt).total_seconds()
            if 0 <= gap <= config.max_gap_seconds:
                box_sim = _max_pairwise_iou(prev.boxes, curr.boxes)
                if box_sim >= config.iou_threshold:
                    link = True
                    curr.novelty_score = max(0.0, 1.0 - box_sim)

        if link:
            curr.sequence_id = prev.sequence_id
        else:
            seq_id += 1
            curr.sequence_id = seq_id
            curr.novelty_score = 1.0

        prev_dt = curr_dt

    # Second pass to populate sequence_length on every record.
    counts: dict[int, int] = defaultdict(int)
    for r in records:
        counts[r.sequence_id] += 1
    for r in records:
        r.sequence_length = counts[r.sequence_id]


def _allocate_sequence_budgets(seq_lengths: list[int], target: int, exponent: float) -> list[int]:
    """Distribute `target` slots across sequences, weighted by len**exponent, capped at len.

    Largest-remainder method, then a redistribute pass to soak up rounding leftover into
    sequences that still have headroom. Returns one int per sequence summing to
    min(target, sum(seq_lengths)).
    """
    n = len(seq_lengths)
    if n == 0 or target <= 0:
        return [0] * n

    total_available = sum(seq_lengths)
    if target >= total_available:
        return list(seq_lengths)

    weights = [max(0.0, float(L)) ** exponent if L > 0 else 0.0 for L in seq_lengths]
    total_weight = sum(weights)
    if total_weight <= 0:
        return [0] * n

    raw = [target * w / total_weight for w in weights]
    floors = [min(int(math.floor(r)), seq_lengths[i]) for i, r in enumerate(raw)]
    remainders = [(raw[i] - math.floor(raw[i]), i) for i in range(n)]

    # Largest-remainder pass — only give an extra slot if the sequence still has headroom.
    remainders.sort(key=lambda t: t[0], reverse=True)
    leftover = target - sum(floors)
    alloc = list(floors)
    for _, i in remainders:
        if leftover <= 0:
            break
        if alloc[i] < seq_lengths[i]:
            alloc[i] += 1
            leftover -= 1

    # If caps blocked some assignments, round-robin the remainder onto whichever sequences
    # still have capacity, prioritizing the longest first (most "natural" home for extras).
    if leftover > 0:
        order = sorted(range(n), key=lambda i: seq_lengths[i] - alloc[i], reverse=True)
        while leftover > 0:
            distributed = 0
            for i in order:
                if leftover <= 0:
                    break
                if alloc[i] < seq_lengths[i]:
                    alloc[i] += 1
                    leftover -= 1
                    distributed += 1
            if distributed == 0:
                break

    return alloc


def _sample_within_sequence(
    records: list[ImageRecord], allocation: int, rng: random.Random
) -> list[ImageRecord]:
    """Pick `allocation` records from one sequence: anchor first+last, fill remainder by novelty."""
    L = len(records)
    if allocation <= 0 or L == 0:
        return []
    if allocation >= L:
        return list(records)
    if allocation == 1:
        return [records[0]]
    if allocation == 2:
        return [records[0], records[-1]]

    # Anchor first and last; fill the middle weighted by novelty.
    chosen_idx = {0, L - 1}
    interior = [i for i in range(1, L - 1)]
    weights = [max(1e-6, records[i].novelty_score or 0.0) for i in interior]

    needed = allocation - 2
    # Weighted sampling without replacement via rejection. needed is small (<= L-2)
    # and L is sequence-length-bounded, so the loop is trivially fast.
    attempts = 0
    max_attempts = max(1000, needed * 50)
    while needed > 0 and attempts < max_attempts and interior:
        pick = rng.choices(interior, weights=weights, k=1)[0]
        if pick not in chosen_idx:
            chosen_idx.add(pick)
            needed -= 1
        attempts += 1

    # Safety net: if rejection sampling stalled, fill from highest-novelty remaining.
    if needed > 0:
        remaining = sorted(
            (i for i in interior if i not in chosen_idx),
            key=lambda i: records[i].novelty_score or 0.0,
            reverse=True,
        )
        for i in remaining[:needed]:
            chosen_idx.add(i)

    return [records[i] for i in sorted(chosen_idx)]


# ---------------------------------------------------------------------------
# Per-location image fetch
# ---------------------------------------------------------------------------

def _fetch_location_records_legacy(cursor, category: int, location: str) -> list[ImageRecord]:
    """Legacy stride path: minimal fields, alphabetical file_path order. No bbox/datetime data."""
    cursor.execute(
        """
        SELECT DISTINCT i.id, i.file_path, i.datetime_original
        FROM images i
        JOIN detections d ON i.id = d.image_id
        WHERE d.category = ? AND d.deleted = 0
          AND (d.hard = 0 OR d.hard IS NULL)
          AND d.confidence >= 0.20
          AND i.file_path LIKE ?
        ORDER BY i.file_path
        """,
        (category, f"{location}%"),
    )
    return [
        ImageRecord(id=row['id'], file_path=row['file_path'], datetime_original=row['datetime_original'])
        for row in cursor.fetchall()
    ]


def _fetch_location_records_diversity(cursor, category: int, location: str) -> list[ImageRecord]:
    """Diversity path: join with detections to bring back per-image bbox lists, sort chronologically."""
    cursor.execute(
        """
        SELECT i.id, i.file_path, i.datetime_original,
               d.x, d.y, d.width, d.height
        FROM images i
        JOIN detections d ON i.id = d.image_id
        WHERE d.category = ? AND d.deleted = 0
          AND (d.hard = 0 OR d.hard IS NULL)
          AND d.confidence >= 0.20
          AND i.file_path LIKE ?
        ORDER BY i.file_path
        """,
        (category, f"{location}%"),
    )

    # Group rows by image. dict preserves insertion order (Python 3.7+) → matches the SQL ordering.
    grouped: dict[int, dict] = {}
    for row in cursor.fetchall():
        img_id = row['id']
        if img_id not in grouped:
            grouped[img_id] = {
                'id': img_id,
                'file_path': row['file_path'],
                'datetime_original': row['datetime_original'],
                'boxes': [],
            }
        grouped[img_id]['boxes'].append((row['x'], row['y'], row['width'], row['height']))

    records = [
        ImageRecord(
            id=g['id'],
            file_path=g['file_path'],
            datetime_original=g['datetime_original'],
            boxes=tuple(g['boxes']),
        )
        for g in grouped.values()
    ]

    # Sort chronologically; NULL/unparseable timestamps go to the end (sentinel = +inf timestamp)
    # then by file_path as a deterministic tiebreaker.
    SENTINEL = datetime.max
    records.sort(key=lambda r: (_parse_exif_datetime(r.datetime_original) or SENTINEL, r.file_path))
    return records


# ---------------------------------------------------------------------------
# Sampling driver
# ---------------------------------------------------------------------------

def sample_images_by_category(
    cursor,
    location_stats,
    category: int,
    target_total: int,
    category_name: str,
    diversity_config: DiversityConfig | None = None,
) -> list[ImageRecord]:
    """
    Sample images with detections of a specific category, distributed as evenly as possible across locations.

    When diversity_config.enabled is True, within each location detections are clustered into
    sequences by (timestamp gap + bbox IoU); long sequences receive a sublinear-in-length budget
    and within-sequence picks anchor first+last and weight the rest by per-frame novelty.

    Returns:
        List of ImageRecord, each tagged with selected_for_classes=[category].
    """
    if diversity_config is None:
        diversity_config = DiversityConfig()

    print(f"\n--- {category_name} Sampling ---")
    print(f"Target {category_name} images: {target_total}")
    if diversity_config.enabled:
        print(
            f"Diversity sampling ON  (iou>={diversity_config.iou_threshold}, "
            f"gap<={diversity_config.max_gap_seconds}s, exponent={diversity_config.budget_exponent}, "
            f"seed={diversity_config.seed})"
        )

    # First pass: get available counts and per-location record lists for this category.
    location_availability: dict[str, dict] = {}
    total_available = 0

    for location in sorted(location_stats.keys()):
        if diversity_config.enabled:
            records = _fetch_location_records_diversity(cursor, category, location)
            _detect_sequences(records, diversity_config)
        else:
            records = _fetch_location_records_legacy(cursor, category, location)

        location_availability[location] = {
            'available': len(records),
            'images': records,
        }
        total_available += len(records)

    if total_available == 0:
        print(f"No {category_name} images available")
        return []

    if total_available <= target_total:
        print(f"Taking all {total_available} available images (less than target)")
        all_images: list[ImageRecord] = []
        for location_data in location_availability.values():
            all_images.extend(location_data['images'])
        for rec in all_images:
            rec.selected_for_classes = [category]
        return all_images

    # Even distribution strategy with scaling to reach target — only count locations that
    # actually have images for this category.
    locations_with_images = {loc: data for loc, data in location_availability.items() if data['available'] > 0}
    total_locations_with_images = len(locations_with_images)

    if total_locations_with_images == 0:
        print(f"No locations have {category_name} images")
        return []

    # Add small buffer to ensure we reach target after redistribution.
    buffer_target = target_total + min(10, total_locations_with_images)
    base_per_location = buffer_target // total_locations_with_images

    print(f"Locations with {category_name} images: {total_locations_with_images}")
    print(f"Initial base per location: {base_per_location} (with buffer)")

    # First pass: allocate base amount, collect shortfalls.
    location_allocations: dict[str, int] = {loc: 0 for loc in location_availability.keys()}
    total_shortfall = 0

    for location, data in locations_with_images.items():
        available_count = data['available']
        if available_count <= base_per_location:
            location_allocations[location] = available_count
            total_shortfall += base_per_location - available_count
        else:
            location_allocations[location] = base_per_location

    print(f"Total shortfall from small locations: {total_shortfall}")

    # Second pass: redistribute shortfall to locations with more availability.
    if total_shortfall > 0:
        print(f"Redistributing {total_shortfall} images to locations with capacity...")

        remaining_shortfall = total_shortfall
        while remaining_shortfall > 0:
            locations_with_capacity = []
            for location, data in location_availability.items():
                available = data['available']
                current_allocation = location_allocations[location]
                if available > current_allocation:
                    capacity = available - current_allocation
                    locations_with_capacity.append((location, capacity))

            if not locations_with_capacity:
                print(f"Warning: Cannot redistribute remaining {remaining_shortfall} images - no locations have capacity")
                break

            locations_with_capacity.sort(key=lambda x: x[1], reverse=True)

            distributed_this_round = 0
            for location, capacity in locations_with_capacity:
                if remaining_shortfall <= 0:
                    break
                if capacity > 0:
                    location_allocations[location] += 1
                    remaining_shortfall -= 1
                    distributed_this_round += 1

            if distributed_this_round == 0:
                print(f"Warning: Could not distribute any more images - stopping with {remaining_shortfall} remaining")
                break

        print(f"Successfully redistributed {total_shortfall - remaining_shortfall} images")

    # Now sample images according to final allocations. Per-category seed isolation prevents
    # changing one category's target from perturbing another category's sample.
    rng = random.Random(diversity_config.seed + category) if diversity_config.enabled else None

    selected_images: list[ImageRecord] = []
    actual_total = 0

    for location, target_count in sorted(location_allocations.items()):
        data = location_availability[location]
        available_count = data['available']
        records: list[ImageRecord] = data['images']

        if target_count <= 0:
            print(f"{location}: {available_count} available, 0 selected")
            continue

        if target_count >= available_count:
            selected_location = list(records)
        elif diversity_config.enabled:
            selected_location = _sample_location_diversity(
                records, target_count, rng, diversity_config.budget_exponent
            )
        else:
            # Legacy: evenly-spaced stride sampling.
            step = available_count / target_count
            selected_location = [records[int(j * step)] for j in range(target_count)]

        if diversity_config.enabled:
            seq_lens = sorted({r.sequence_length for r in records if r.sequence_length is not None})
            n_seqs = len({r.sequence_id for r in records})
            longest = max((r.sequence_length or 0) for r in records) if records else 0
            print(
                f"{location}: {available_count} available in {n_seqs} sequences "
                f"(max len {longest}), {len(selected_location)} selected"
            )
        else:
            print(f"{location}: {available_count} available, {len(selected_location)} selected")

        selected_images.extend(selected_location)
        actual_total += len(selected_location)

    print(f"Total selected {category_name} images: {actual_total}")

    # Final step: trim to exact target if we oversampled. In diversity mode, prefer to drop
    # from the longest sequences (preserving anchors and isolates); the legacy stride path keeps
    # its original behavior of dropping the chronological tail.
    if len(selected_images) > target_total:
        print(f"Trimming from {len(selected_images)} to exactly {target_total} images")
        if diversity_config.enabled:
            selected_images = _trim_diversity_aware(selected_images, target_total)
        else:
            selected_images = selected_images[:target_total]

    for rec in selected_images:
        rec.selected_for_classes = [category]

    return selected_images


def _trim_diversity_aware(records: list[ImageRecord], target: int) -> list[ImageRecord]:
    """Drop excess records from the longest sequences first, preserving anchors and isolates.

    Score each record by `anchor_bonus + 1/sequence_length + novelty_score`. Anchors
    (first and last record of each contiguous same-sequence-id run in the input list) get a
    large bonus so they survive. Isolates (length-1 sequences) score high via 1/L=1. Long-
    sequence interior records score low via 1/L≈0 and novelty≈0. Drop the lowest-scored.
    """
    n_to_drop = len(records) - target
    if n_to_drop <= 0:
        return list(records)

    # Mark first and last record of each contiguous same-sequence-id run as anchors.
    # _sample_location_diversity returns records grouped by sequence_id within each
    # (location, category) call, so any block of same-sequence-id records in the input is
    # contiguous — even when sequence_ids happen to collide across calls (different blocks).
    is_anchor = [False] * len(records)
    i = 0
    while i < len(records):
        sid = records[i].sequence_id
        j = i
        while j + 1 < len(records) and records[j + 1].sequence_id == sid:
            j += 1
        is_anchor[i] = True
        is_anchor[j] = True
        i = j + 1

    def score(idx: int) -> float:
        r = records[idx]
        L = r.sequence_length or 1
        n = r.novelty_score if r.novelty_score is not None else 1.0
        anchor_bonus = 10.0 if is_anchor[idx] else 0.0
        return anchor_bonus + (1.0 / L) + n

    # Stable sort: ties broken by index order so the result is deterministic.
    drop_order = sorted(range(len(records)), key=lambda idx: (score(idx), idx))
    drop_set = set(drop_order[:n_to_drop])
    return [r for idx, r in enumerate(records) if idx not in drop_set]


def _sample_location_diversity(
    records: list[ImageRecord],
    target_count: int,
    rng: random.Random,
    budget_exponent: float,
) -> list[ImageRecord]:
    """Group records by sequence_id, allocate budgets sublinearly in length, pick within each."""
    seq_buckets: dict[int, list[ImageRecord]] = defaultdict(list)
    seq_order: list[int] = []
    for r in records:
        if r.sequence_id not in seq_buckets:
            seq_order.append(r.sequence_id)
        seq_buckets[r.sequence_id].append(r)

    seq_records = [seq_buckets[sid] for sid in seq_order]
    seq_lengths = [len(s) for s in seq_records]

    allocations = _allocate_sequence_budgets(seq_lengths, target_count, budget_exponent)

    selected: list[ImageRecord] = []
    for seq, alloc in zip(seq_records, allocations):
        selected.extend(_sample_within_sequence(seq, alloc, rng))

    return selected


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def export_images(
    db_file: str,
    images_dir: str,
    output_dir: str,
    max_images: int | None = None,
    target_gang_gang: int | None = None,
    target_possum: int | None = None,
    target_other: int | None = None,
    include_locations: str | None = None,
    exclude_locations: str | None = None,
    diversity_config: DiversityConfig | None = None,
    skip_export: bool = False,
):
    """Export images from database to output directory with subdirectory organization"""

    if diversity_config is None:
        diversity_config = DiversityConfig()

    if not os.path.exists(db_file):
        print(f"Error: Database file not found: {db_file}")
        return 1

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Get all images from database
        cursor.execute("SELECT id, file_path FROM images ORDER BY file_path")
        images = cursor.fetchall()

        total_images = len(images)
        print(f"Found {total_images} images in database")

        # Collect statistics by location
        location_stats: dict[str, int] = {}
        for image in images:
            file_path = image['file_path']
            path_parts = file_path.split(os.sep)
            location = path_parts[0] if path_parts else file_path
            location_stats[location] = location_stats.get(location, 0) + 1

        # Get detection counts by category and location
        def get_detection_stats_by_category(category):
            cursor.execute(
                """
                SELECT i.file_path, COUNT(*) as detection_count
                FROM images i
                JOIN detections d ON i.id = d.image_id
                WHERE d.category = ? AND d.deleted = 0
                  AND (d.hard = 0 OR d.hard IS NULL)
                  AND d.confidence >= 0.20
                GROUP BY i.id, i.file_path
                """,
                (category,),
            )
            results = cursor.fetchall()

            stats: dict[str, int] = {}
            for result in results:
                file_path = result['file_path']
                path_parts = file_path.split(os.sep)
                location = path_parts[0] if path_parts else file_path
                stats[location] = stats.get(location, 0) + result['detection_count']
            return stats

        gang_gang_stats = get_detection_stats_by_category(1)
        possum_stats = get_detection_stats_by_category(4)
        other_stats = get_detection_stats_by_category(5)

        # Print location statistics
        print("\nImage count by location:")
        print("-" * 60)
        for location, count in sorted(location_stats.items()):
            gang_gang_count = gang_gang_stats.get(location, 0)
            possum_count = possum_stats.get(location, 0)
            other_count = other_stats.get(location, 0)
            print(
                f"{location}: {count} images, {gang_gang_count} Gang Gang detections, "
                f"{possum_count} Possum detections, {other_count} Other detections"
            )

        print(f"\nTotal locations: {len(location_stats)}")
        total_gang_gangs = sum(gang_gang_stats.values())
        total_possums = sum(possum_stats.values())
        total_others = sum(other_stats.values())
        print(f"Total Gang Gang detections: {total_gang_gangs}")
        print(f"Total Possum detections: {total_possums}")
        print(f"Total Other detections: {total_others}")

        # Apply location filtering
        filtered_location_stats = location_stats.copy()

        if include_locations:
            include_list = [loc.strip() for loc in include_locations.split(',')]
            filtered_location_stats = {
                loc: count for loc, count in location_stats.items() if loc in include_list
            }
            print(f"\nIncluding only locations: {', '.join(include_list)}")
            print(f"Filtered to {len(filtered_location_stats)} locations")

        if exclude_locations:
            exclude_list = [loc.strip() for loc in exclude_locations.split(',')]
            filtered_location_stats = {
                loc: count for loc, count in filtered_location_stats.items() if loc not in exclude_list
            }
            print(f"\nExcluding locations: {', '.join(exclude_list)}")
            print(f"Filtered to {len(filtered_location_stats)} locations")

        if filtered_location_stats != location_stats:
            print(f"\nFiltered location statistics:")
            for location in sorted(filtered_location_stats.keys()):
                image_count = filtered_location_stats[location]
                gang_gang_count = gang_gang_stats.get(location, 0)
                possum_count = possum_stats.get(location, 0)
                other_count = other_stats.get(location, 0)
                print(
                    f"{location}: {image_count} images, {gang_gang_count} Gang Gang detections, "
                    f"{possum_count} Possum detections, {other_count} Other detections"
                )

        # Use filtered locations for sampling
        location_stats = filtered_location_stats

        # Sample images by category
        all_selected_images: list[ImageRecord] = []

        if target_gang_gang:
            all_selected_images.extend(
                sample_images_by_category(cursor, location_stats, 1, target_gang_gang, "Gang Gang", diversity_config)
            )

        if target_possum:
            all_selected_images.extend(
                sample_images_by_category(cursor, location_stats, 4, target_possum, "Possum", diversity_config)
            )

        if target_other:
            all_selected_images.extend(
                sample_images_by_category(cursor, location_stats, 5, target_other, "Other", diversity_config)
            )

        # Remove duplicates based on file_path (some images may be selected for multiple categories).
        # Keep first occurrence's diagnostics, but accumulate selected_for_classes so downstream
        # tooling sees the full set of categories that targeted this image.
        if all_selected_images:
            seen_paths: dict[str, ImageRecord] = {}
            unique_images: list[ImageRecord] = []
            duplicates_removed = 0

            for image in all_selected_images:
                if image.file_path not in seen_paths:
                    seen_paths[image.file_path] = image
                    unique_images.append(image)
                else:
                    kept = seen_paths[image.file_path]
                    for cat in image.selected_for_classes:
                        if cat not in kept.selected_for_classes:
                            kept.selected_for_classes.append(cat)
                    duplicates_removed += 1

            all_selected_images = unique_images
            if duplicates_removed > 0:
                print(f"\nRemoved {duplicates_removed} duplicate images (images selected by multiple categories)")
                print(f"Final unique image count: {len(all_selected_images)}")

        # Export all selected images if output directory is specified
        if output_dir and all_selected_images:
            os.makedirs(output_dir, exist_ok=True)

            if skip_export:
                print(f"\n--skip-export: writing metadata.json only for {len(all_selected_images)} images, "
                      f"not copying files")
            else:
                print(f"\nExporting {len(all_selected_images)} total images to {output_dir}...")

            metadata = []
            copied_count = 0

            for idx, image in enumerate(all_selected_images):
                image_id = image.id
                src_file_path = image.file_path

                # Extract location from first directory component of original path.
                path_parts = src_file_path.split(os.sep)
                location = path_parts[0] if path_parts else src_file_path

                # Get detections for this image (all categories, not just the one that targeted it).
                cursor.execute(
                    """
                    SELECT category, x, y, width, height, hard
                    FROM detections
                    WHERE image_id = ? AND deleted = 0
                      AND (hard = 0 OR hard IS NULL)
                      AND confidence >= 0.20
                    ORDER BY confidence DESC
                    """,
                    (image_id,),
                )
                detections_rows = cursor.fetchall()

                detections = []
                for det in detections_rows:
                    detections.append({
                        'category': det['category'],
                        'x': det['x'],
                        'y': det['y'],
                        'width': det['width'],
                        'height': det['height'],
                        'hard': bool(det['hard']) if det['hard'] is not None else False,
                    })

                if not detections:
                    print(f"Warning: No detections found for {src_file_path}, skipping")
                    continue

                # Construct full input path
                if os.path.isabs(src_file_path):
                    src_path = src_file_path
                else:
                    src_path = os.path.join(images_dir, src_file_path)

                if not os.path.exists(src_path):
                    print(f"Warning: Source file not found: {src_path}")
                    continue

                # Generate sequential filename: image_000001.jpg, image_000002.jpg, etc.
                dst_filename = f"image_{copied_count + 1:06d}.jpg"
                dst_path = os.path.join(output_dir, dst_filename)

                try:
                    if not skip_export:
                        shutil.copy2(src_path, dst_path)

                    entry = {
                        'file_path': dst_filename,
                        'location': location,
                        'datetime_original': image.datetime_original,
                        'detections': detections,
                    }
                    if diversity_config.enabled:
                        entry['sequence_id'] = image.sequence_id
                        entry['sequence_length'] = image.sequence_length
                        entry['novelty_score'] = image.novelty_score
                        entry['selected_for_classes'] = list(image.selected_for_classes)
                    metadata.append(entry)

                    copied_count += 1

                    if copied_count % 100 == 0:
                        action = "Recorded" if skip_export else "Exported"
                        print(f"{action} {copied_count}/{len(all_selected_images)} images...")

                except Exception as e:
                    print(f"Error copying {src_path} to {dst_path}: {e}")
                    continue

            # Write metadata to JSON file
            metadata_path = os.path.join(output_dir, 'metadata.json')
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Metadata saved to: {metadata_path}")
            except Exception as e:
                print(f"Error writing metadata file: {e}")

            verb = "recorded" if skip_export else "exported"
            print(f"Successfully {verb} {copied_count} images")
            print(f"Total detections {verb}: {sum(len(img['detections']) for img in metadata)}")

        return 0

    except Exception as e:
        print(f"Database error: {e}")
        return 1

    finally:
        conn.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Export images from GGSort database')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--images-dir', required=True,
                        help='Base directory containing input image files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for exported images')
    parser.add_argument('--max-images', type=int,
                        help='Maximum number of images to export')
    parser.add_argument('--target-gang-gang', type=int,
                        help='Target number of Gang Gang images to export, distributed evenly across locations')
    parser.add_argument('--target-possum', type=int,
                        help='Target number of Possum images to export, distributed evenly across locations')
    parser.add_argument('--target-other', type=int,
                        help='Target number of Other images to export, distributed evenly across locations')
    parser.add_argument('--include-locations', type=str,
                        help='Comma-separated list of locations to include (only these locations will be considered)')
    parser.add_argument('--exclude-locations', type=str,
                        help='Comma-separated list of locations to exclude from the dataset')
    parser.add_argument('--diversity-sampling', action='store_true',
                        help='Enable diversity-aware sampling: cluster bursts of similar-bbox '
                             'frames into sequences and downsample long sequences')
    parser.add_argument('--diversity-iou-threshold', type=float, default=0.5,
                        help='IoU threshold to link adjacent frames into the same sequence (default 0.5)')
    parser.add_argument('--diversity-max-gap-seconds', type=float, default=60.0,
                        help='Max time gap (seconds) between adjacent frames to consider linking (default 60)')
    parser.add_argument('--diversity-budget-exponent', type=float, default=0.5,
                        help='Per-sequence budget weight exponent: 0=equal per sequence, '
                             '0.5=sqrt dampening (default), 1=no dampening')
    parser.add_argument('--diversity-seed', type=int, default=0,
                        help='RNG seed for diversity sampling (default 0)')
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip copying image files; write metadata.json only')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.db_file):
        print(f"Error: Database file not found: {args.db_file}")
        return 1

    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1

    if args.max_images and args.max_images <= 0:
        print(f"Error: --max-images must be a positive integer")
        return 1

    diversity_config = DiversityConfig(
        enabled=args.diversity_sampling,
        iou_threshold=args.diversity_iou_threshold,
        max_gap_seconds=args.diversity_max_gap_seconds,
        budget_exponent=args.diversity_budget_exponent,
        seed=args.diversity_seed,
    )

    print(f"Exporting from database: {args.db_file}")
    print(f"Input images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_images:
        print(f"Maximum images: {args.max_images}")
    if args.skip_export:
        print("Skip-export mode: will not copy image files")

    return export_images(
        args.db_file,
        args.images_dir,
        args.output_dir,
        args.max_images,
        args.target_gang_gang,
        args.target_possum,
        args.target_other,
        args.include_locations,
        args.exclude_locations,
        diversity_config=diversity_config,
        skip_export=args.skip_export,
    )


if __name__ == "__main__":
    sys.exit(main())
