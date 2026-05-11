#!/usr/bin/env python3
"""
Round-trip merger for GGSort QA split databases.

Combines several ``*_pN.db`` files produced by ``split_db.py`` (each
independently QA'd) back into a single database that is data-equivalent
to having QA'd the original database in one go.

Why a separate script from ``merge_dbs.py``?
``merge_dbs.py`` is designed for the ML-export pipeline and intentionally
*drops* detections marked ``deleted=1`` or ``hard=1`` (and any image whose
detections are all so marked). That filtering would silently destroy
manual QA decisions on the round-trip. This script preserves every
column of every row.

Safety guarantees:
* No row is filtered: ``deleted=1``, ``hard=1``, ``subcategory``, custom
  bbox tweaks, ``manually_processed`` all survive.
* ``app_state`` and ``autodelete_templates`` tables are recreated.
* By default, a ``file_path`` appearing in more than one input is a hard
  error (splits should be disjoint). Use ``--allow-duplicate-paths`` to
  fall back to consolidation if you know what you're doing.
* The whole merge runs inside a single transaction; on any error the
  output file is removed so the workspace stays clean.
* Optional ``--verify-against ORIGINAL.db`` performs end-to-end diff
  checks against the pre-split database.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path


SKIP_TABLES = {"sqlite_sequence"}

REQUIRED_TABLES = {"images", "detections", "app_state", "autodelete_templates"}


def open_readonly(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def read_schema(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE sql IS NOT NULL
          AND name NOT LIKE 'sqlite_%'
        ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,
                 name
        """
    )
    return [(t, n, s) for (t, n, s) in cursor.fetchall() if n not in SKIP_TABLES]


def replay_schema(target_conn, schema):
    cursor = target_conn.cursor()
    for _type, _name, sql in schema:
        cursor.execute(sql)
    target_conn.commit()


def table_columns(conn, table):
    """Return the ordered list of column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def validate_input_schema(conn, path):
    """Raise if the input doesn't look like a GGSort split DB."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    present = {row[0] for row in cursor.fetchall()}
    missing = REQUIRED_TABLES - present
    if missing:
        raise ValueError(
            f"{path}: missing required tables {sorted(missing)} - is this a GGSort database?"
        )

    image_cols = set(table_columns(conn, "images"))
    expected_image_cols = {
        "id", "file_path", "width", "height", "datetime_original", "manually_processed",
    }
    if not expected_image_cols.issubset(image_cols):
        raise ValueError(
            f"{path}: images table is missing columns {sorted(expected_image_cols - image_cols)}"
        )

    det_cols = set(table_columns(conn, "detections"))
    expected_det_cols = {
        "id", "image_id", "category", "confidence",
        "x", "y", "width", "height", "deleted", "subcategory", "hard",
    }
    if not expected_det_cols.issubset(det_cols):
        raise ValueError(
            f"{path}: detections table is missing columns {sorted(expected_det_cols - det_cols)}"
        )


def merge_one_split(merged_conn, split_path, next_image_id, next_detection_id,
                    file_path_origin, allow_duplicate_paths):
    """Merge a single split DB into ``merged_conn``.

    Returns the next available image_id and detection_id, plus the per-split
    counts (images_inserted, detections_inserted, duplicate_paths).
    """
    src = open_readonly(split_path)
    try:
        validate_input_schema(src, split_path)

        src_cur = src.cursor()
        merged_cur = merged_conn.cursor()

        # Iterate images in stable order so re-numbering is deterministic.
        src_cur.execute(
            """
            SELECT id, file_path, width, height, datetime_original, manually_processed
            FROM images
            ORDER BY id
            """
        )

        image_id_map = {}            # old_id -> new_id (within this split)
        duplicate_paths = []         # list of (file_path, other_split)
        images_inserted = 0

        for old_id, file_path, width, height, datetime_original, manually_processed in src_cur.fetchall():
            existing_split = file_path_origin.get(file_path)
            if existing_split is not None:
                # file_path already came in from an earlier split.
                duplicate_paths.append((file_path, existing_split))
                if not allow_duplicate_paths:
                    # Defer raising until we've gathered all duplicates in this
                    # split for a more useful error message.
                    image_id_map[old_id] = None
                    continue
                # Consolidation mode: reuse the existing image row.
                merged_cur.execute(
                    "SELECT id FROM images WHERE file_path = ?", (file_path,)
                )
                image_id_map[old_id] = merged_cur.fetchone()[0]
                continue

            merged_cur.execute(
                """
                INSERT INTO images (id, file_path, width, height, datetime_original, manually_processed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (next_image_id, file_path, width, height, datetime_original, manually_processed),
            )
            image_id_map[old_id] = next_image_id
            file_path_origin[file_path] = split_path
            next_image_id += 1
            images_inserted += 1

        if duplicate_paths and not allow_duplicate_paths:
            # Build a useful error message before bailing.
            msg_lines = [
                f"Duplicate file_path(s) found in {split_path}:",
                "  (a file_path should appear in exactly one split - this likely indicates a bug)",
            ]
            for fp, other in duplicate_paths[:10]:
                msg_lines.append(f"    {fp}  (also in {other})")
            if len(duplicate_paths) > 10:
                msg_lines.append(f"    ... and {len(duplicate_paths) - 10} more")
            msg_lines.append("  Pass --allow-duplicate-paths to merge anyway (detections will be consolidated).")
            raise ValueError("\n".join(msg_lines))

        # Now copy detections. We re-number detection IDs because two splits
        # may have independently inserted new detections with the same IDs
        # (ggsort supports adding detections during QA).
        src_cur.execute(
            """
            SELECT id, image_id, category, confidence, x, y, width, height,
                   deleted, subcategory, hard
            FROM detections
            ORDER BY id
            """
        )

        detections_inserted = 0
        for row in src_cur.fetchall():
            (_old_det_id, old_image_id, category, confidence,
             x, y, w, h, deleted, subcategory, hard) = row

            new_image_id = image_id_map.get(old_image_id)
            if new_image_id is None:
                # The parent image was a duplicate skipped in strict mode -
                # we already errored out above, so this branch is unreachable.
                continue

            merged_cur.execute(
                """
                INSERT INTO detections
                    (id, image_id, category, confidence, x, y, width, height,
                     deleted, subcategory, hard)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (next_detection_id, new_image_id, category, confidence,
                 x, y, w, h, deleted, subcategory, hard),
            )
            next_detection_id += 1
            detections_inserted += 1

        return next_image_id, next_detection_id, images_inserted, detections_inserted, len(duplicate_paths)
    finally:
        src.close()


def copy_autodelete_templates(merged_conn, first_split_path):
    """Copy autodelete_templates from the first split.

    split_db.py duplicates these into every split (they're global UI state),
    so taking them from any one split is sufficient. We pick the first for
    determinism.
    """
    src = open_readonly(first_split_path)
    try:
        src_cur = src.cursor()
        src_cur.execute("SELECT COUNT(*) FROM autodelete_templates")
        n = src_cur.fetchone()[0]
        if n == 0:
            return 0

        # Pull rows by named columns so we don't depend on column order.
        cols = table_columns(src, "autodelete_templates")
        col_list = ",".join(cols)
        placeholders = ",".join(["?"] * len(cols))

        src_cur.execute(f"SELECT {col_list} FROM autodelete_templates")
        rows = src_cur.fetchall()

        merged_cur = merged_conn.cursor()
        merged_cur.executemany(
            f"INSERT INTO autodelete_templates ({col_list}) VALUES ({placeholders})",
            rows,
        )
        return len(rows)
    finally:
        src.close()


def hash_file_paths(conn):
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM images ORDER BY file_path")
    h = hashlib.sha256()
    for (fp,) in cur:
        h.update(fp.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def summarise(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM images")
    images = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM detections")
    detections = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM detections WHERE deleted = 1")
    deleted = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM detections WHERE hard = 1")
    hard = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM images WHERE manually_processed = 1")
    mp = cur.fetchone()[0]
    return {
        "images": images,
        "detections": detections,
        "deleted": deleted,
        "hard": hard,
        "manually_processed": mp,
        "file_path_hash": hash_file_paths(conn),
    }


def verify_against_original(merged_conn, original_path):
    """Diff the merged DB against the pre-split original."""
    merged = summarise(merged_conn)
    orig_conn = open_readonly(original_path)
    try:
        original = summarise(orig_conn)
    finally:
        orig_conn.close()

    problems = []
    for key in ("images", "detections", "deleted", "hard", "manually_processed"):
        if merged[key] != original[key]:
            problems.append(f"{key}: merged={merged[key]} original={original[key]}")
    if merged["file_path_hash"] != original["file_path_hash"]:
        problems.append("file_path set differs between merged and original")

    return problems, merged, original


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge GGSort QA split databases back into a single database, "
            "losslessly preserving every QA flag."
        )
    )
    parser.add_argument("output_db", help="Path for the merged output .db file.")
    parser.add_argument("split_dbs", nargs="+", help="Two or more split .db files to merge.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it exists.")
    parser.add_argument(
        "--allow-duplicate-paths",
        action="store_true",
        help=(
            "If a file_path appears in more than one split, silently consolidate "
            "(reusing the first split's image row). By default this is an error."
        ),
    )
    parser.add_argument(
        "--verify-against",
        metavar="ORIGINAL_DB",
        default=None,
        help=(
            "Optional path to the pre-split original database. If supplied, "
            "the merged DB will be diffed against it after the merge completes."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output_db)
    split_paths = [Path(p) for p in args.split_dbs]

    if len(split_paths) < 1:
        print("Error: at least one split DB must be supplied.", file=sys.stderr)
        sys.exit(1)

    missing = [p for p in split_paths if not p.exists()]
    if missing:
        print("Error: the following input files were not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    # Detect the same physical file listed twice (e.g. when a glob overlaps an
    # explicit path: ``merge_split_dbs.py out.db splits/p1.db splits/*``).
    # Without this check we'd later fail with a confusing "duplicate file_path"
    # error referring to rows inside the DB rather than the duplicated input.
    seen = {}
    for p in split_paths:
        real = p.resolve()
        if real in seen:
            print(
                f"Error: input file listed more than once: {p}  "
                f"(same file as {seen[real]})",
                file=sys.stderr,
            )
            print(
                "  Likely cause: a glob (e.g. splits/*) overlaps an explicit "
                "path you also passed.",
                file=sys.stderr,
            )
            sys.exit(1)
        seen[real] = p

    # Don't let the output overwrite one of the inputs - we read from each input
    # multiple times during the merge, and clobbering one mid-run would corrupt
    # the result silently.
    out_real = output_path.resolve()
    if out_real in seen:
        print(
            f"Error: output path '{output_path}' is also listed as an input "
            f"({seen[out_real]}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path.exists():
        if not args.force:
            print(
                f"Error: output file '{output_path}' exists (pass --force to overwrite).",
                file=sys.stderr,
            )
            sys.exit(1)
        output_path.unlink()

    # Read the schema from the first split. All splits should have identical
    # schemas; we don't bother checking byte-equality because validate_input_schema
    # already ensures the required tables/columns are present in every split.
    first = open_readonly(split_paths[0])
    try:
        validate_input_schema(first, split_paths[0])
        schema = read_schema(first)
    finally:
        first.close()

    print(f"Creating merged database: {output_path}")
    merged_conn = sqlite3.connect(str(output_path))
    # Manual transaction control - see split_db.py for the same rationale.
    merged_conn.isolation_level = None
    try:
        replay_schema(merged_conn, schema)

        merged_conn.execute("BEGIN")

        next_image_id = 1
        next_detection_id = 1
        file_path_origin = {}  # file_path -> first split that contributed it
        total_dupes = 0

        for i, split_path in enumerate(split_paths, start=1):
            print(f"  [{i}/{len(split_paths)}] Merging {split_path}...")
            try:
                (next_image_id, next_detection_id,
                 n_img, n_det, n_dup) = merge_one_split(
                    merged_conn,
                    split_path,
                    next_image_id,
                    next_detection_id,
                    file_path_origin,
                    args.allow_duplicate_paths,
                )
            except ValueError as exc:
                # Schema or duplicate-path errors - actionable user-facing.
                merged_conn.rollback()
                merged_conn.close()
                if output_path.exists():
                    output_path.unlink()
                print(f"\nError: {exc}", file=sys.stderr)
                sys.exit(1)

            total_dupes += n_dup
            print(
                f"      inserted {n_img} images, {n_det} detections"
                + (f" (skipped {n_dup} duplicate paths)" if n_dup else "")
            )

        # autodelete_templates: take from the first split only (they're global
        # and split_db.py duplicates them to every split, so the splits agree).
        n_templates = copy_autodelete_templates(merged_conn, split_paths[0])
        if n_templates:
            print(f"  Copied {n_templates} autodelete template(s) from {split_paths[0].name}.")

        # Fresh starting point for whoever opens the merged DB.
        merged_conn.execute(
            "INSERT INTO app_state (key, value) VALUES ('last_image_index', '0')"
        )

        merged_conn.execute("COMMIT")
    except Exception as exc:
        merged_conn.rollback()
        merged_conn.close()
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        print(f"\nError during merge: {exc}", file=sys.stderr)
        sys.exit(1)

    # Post-merge invariants. We re-open the merged DB read-only to keep this
    # check honest (no in-flight transaction state).
    merged_conn.close()
    check_conn = open_readonly(output_path)
    try:
        stats = summarise(check_conn)

        # Orphan check - every detection must reference a real image.
        cur = check_conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM detections d
            LEFT JOIN images i ON i.id = d.image_id
            WHERE i.id IS NULL
            """
        )
        orphans = cur.fetchone()[0]
    finally:
        check_conn.close()

    print()
    print("Merged database summary:")
    print(f"  images:             {stats['images']}")
    print(f"  detections:         {stats['detections']}")
    print(f"  deleted=1:          {stats['deleted']}")
    print(f"  hard=1:             {stats['hard']}")
    print(f"  manually_processed: {stats['manually_processed']}")

    if orphans:
        print(
            f"\nERROR: merged DB contains {orphans} orphan detections "
            "(image_id with no matching image row).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verify_against:
        print()
        print(f"Verifying against original: {args.verify_against}")
        check_conn = open_readonly(output_path)
        try:
            problems, merged_stats, original_stats = verify_against_original(
                check_conn, args.verify_against
            )
        finally:
            check_conn.close()

        if problems:
            print("Verification FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            sys.exit(1)

        print("  Verification passed - merged DB matches original on:")
        print(f"    images={original_stats['images']}, detections={original_stats['detections']}, "
              f"deleted={original_stats['deleted']}, hard={original_stats['hard']}, "
              f"manually_processed={original_stats['manually_processed']}")
        print("    and on the exact set of file_paths.")

    print(f"\nDone. Merged {len(split_paths)} splits into {output_path}.")


if __name__ == "__main__":
    main()
