#!/usr/bin/env python3
"""
Database splitter for the GGSort QA workflow.

Splits a single GGSort SQLite database into multiple smaller databases that
can be QA'd independently and later recombined with ``merge_split_dbs.py``.

The split is lossless: every column of every row of every table (images,
detections, app_state, autodelete_templates) is preserved, including
``deleted=1`` and ``hard=1`` detections that encode manual QA decisions.
The only intentional change is that each split has its own
``last_image_index = 0`` in ``app_state`` so QA workers start at the
beginning of their chunk.

Images are grouped into chunks in the same order ggsort iterates them
(``ORDER BY file_path``) so each chunk corresponds to a contiguous slice
of the natural QA order.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path


# Tables we intentionally do not replay or copy. ``sqlite_sequence`` is managed
# by SQLite itself; we don't use AUTOINCREMENT on the user-visible tables we
# care about (only autodelete_templates uses AUTOINCREMENT, and it will be
# recreated by SQLite as needed when that table is created).
SKIP_TABLES = {"sqlite_sequence"}


def open_readonly(path):
    """Open a SQLite database in read-only mode (URI form)."""
    # ``file:`` URI with mode=ro guarantees we never accidentally write.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def read_schema(conn):
    """Return a list of (type, name, sql) for every user object in the DB.

    Order is preserved so tables are created before their indexes.
    """
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
    return [
        (t, n, s) for (t, n, s) in cursor.fetchall() if n not in SKIP_TABLES
    ]


def replay_schema(target_conn, schema):
    """Replay schema statements verbatim into a fresh target connection."""
    cursor = target_conn.cursor()
    for _type, _name, sql in schema:
        cursor.execute(sql)
    target_conn.commit()


def fetch_image_order(conn):
    """Return [image_id, ...] in ggsort iteration order (ORDER BY file_path)."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM images ORDER BY file_path")
    return [row[0] for row in cursor.fetchall()]


def plan_chunks(image_ids, chunk_size):
    """Group images into chunks of at most ``chunk_size`` images each.

    Splits the file_path-sorted list into contiguous slices so each QA worker
    gets a natural-order range. The last chunk holds the remainder (so it
    may be smaller, but never larger, than ``chunk_size``).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return [
        image_ids[i:i + chunk_size]
        for i in range(0, len(image_ids), chunk_size)
    ]


def write_chunk(source_path, target_path, image_ids, schema, force):
    """Write a single split DB containing exactly ``image_ids`` and their detections."""
    if target_path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {target_path}")
        target_path.unlink()

    conn = sqlite3.connect(str(target_path))
    # Take manual control of transactions - Python's sqlite3 default auto-begins
    # a transaction on DML, which conflicts with our explicit BEGIN below.
    conn.isolation_level = None
    try:
        # Replay the canonical schema (CREATE TABLE/INDEX text copied from the
        # source's sqlite_master) so the split is byte-equivalent to what
        # ggsort and the other tools expect.
        replay_schema(conn, schema)

        # ATTACH the source so we can stream rows server-side without
        # round-tripping through Python.
        cursor = conn.cursor()
        cursor.execute("ATTACH DATABASE ? AS src", (f"file:{source_path}?mode=ro",))

        # Stage image IDs in a temp table - cleaner than building a giant
        # IN (...) clause for tens of thousands of IDs.
        cursor.execute("CREATE TEMP TABLE chunk_ids (id INTEGER PRIMARY KEY)")
        cursor.executemany(
            "INSERT INTO chunk_ids (id) VALUES (?)", [(i,) for i in image_ids]
        )

        cursor.execute("BEGIN")

        # Copy images verbatim (preserving original primary keys).
        cursor.execute(
            """
            INSERT INTO images
            SELECT * FROM src.images
            WHERE id IN (SELECT id FROM chunk_ids)
            """
        )

        # Copy *every* detection for those images - including deleted=1 and
        # hard=1. This is the key QA-safety guarantee vs merge_dbs.py.
        cursor.execute(
            """
            INSERT INTO detections
            SELECT * FROM src.detections
            WHERE image_id IN (SELECT id FROM chunk_ids)
            """
        )

        # Global UI state: copy autodelete templates to every split so workers
        # see the same auto-delete rules.
        cursor.execute("INSERT INTO autodelete_templates SELECT * FROM src.autodelete_templates")

        # Each split starts at frame 0 - the worker is opening a fresh chunk.
        cursor.execute(
            "INSERT INTO app_state (key, value) VALUES ('last_image_index', '0')"
        )

        cursor.execute("COMMIT")

        cursor.execute("DETACH DATABASE src")

        # Self-stats for the post-write summary.
        cursor.execute("SELECT COUNT(*) FROM images")
        n_images = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM detections")
        n_detections = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM detections WHERE deleted = 1")
        n_deleted = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM detections WHERE hard = 1")
        n_hard = cursor.fetchone()[0]
    except Exception:
        conn.close()
        # Best-effort cleanup so partial files don't confuse the user.
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise
    finally:
        conn.close()

    return n_images, n_detections, n_deleted, n_hard


def hash_file_paths(conn):
    """Return a sha256 of every file_path in the DB (sorted), for cheap set comparison."""
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM images ORDER BY file_path")
    h = hashlib.sha256()
    for (fp,) in cursor:
        h.update(fp.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def verify_splits(source_path, split_paths):
    """Assert the union of splits matches the source on every metric we care about."""
    src = open_readonly(source_path)
    try:
        src_cur = src.cursor()
        src_cur.execute("SELECT COUNT(*) FROM images")
        src_images = src_cur.fetchone()[0]
        src_cur.execute("SELECT COUNT(*) FROM detections")
        src_detections = src_cur.fetchone()[0]
        src_cur.execute("SELECT COUNT(*) FROM detections WHERE deleted = 1")
        src_deleted = src_cur.fetchone()[0]
        src_cur.execute("SELECT COUNT(*) FROM detections WHERE hard = 1")
        src_hard = src_cur.fetchone()[0]
        src_hash = hash_file_paths(src)
    finally:
        src.close()

    total_images = 0
    total_detections = 0
    total_deleted = 0
    total_hard = 0
    combined_hash = hashlib.sha256()
    combined_paths = []

    for split_path in split_paths:
        conn = open_readonly(split_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM images")
            total_images += cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM detections")
            total_detections += cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM detections WHERE deleted = 1")
            total_deleted += cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM detections WHERE hard = 1")
            total_hard += cur.fetchone()[0]
            cur.execute("SELECT file_path FROM images")
            for (fp,) in cur:
                combined_paths.append(fp)
        finally:
            conn.close()

    combined_paths.sort()
    for fp in combined_paths:
        combined_hash.update(fp.encode("utf-8"))
        combined_hash.update(b"\n")
    combined_hash = combined_hash.hexdigest()

    problems = []
    if total_images != src_images:
        problems.append(f"image count mismatch: source={src_images} splits={total_images}")
    if total_detections != src_detections:
        problems.append(f"detection count mismatch: source={src_detections} splits={total_detections}")
    if total_deleted != src_deleted:
        problems.append(f"deleted=1 count mismatch: source={src_deleted} splits={total_deleted}")
    if total_hard != src_hard:
        problems.append(f"hard=1 count mismatch: source={src_hard} splits={total_hard}")
    if combined_hash != src_hash:
        problems.append("file_path set mismatch between source and union of splits")

    return problems, {
        "images": (src_images, total_images),
        "detections": (src_detections, total_detections),
        "deleted": (src_deleted, total_deleted),
        "hard": (src_hard, total_hard),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split a GGSort database into independently-editable chunks. "
            "Use merge_split_dbs.py to recombine them after QA."
        )
    )
    parser.add_argument("input_db", help="Path to the input .db file (read-only).")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Maximum number of images per output chunk (default: 10000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write split files into (default: alongside the input).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing _pN.db output files.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_db)
    if not input_path.exists():
        print(f"Error: input database not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plan the chunks before we write anything - cheaper to bail early on bad input.
    src = open_readonly(input_path)
    try:
        schema = read_schema(src)
        expected = {"images", "detections", "app_state", "autodelete_templates"}
        present = {n for (_t, n, _s) in schema if _t == "table"}
        missing = expected - present
        if missing:
            print(
                f"Error: input database is missing expected tables: {sorted(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

        image_ids = fetch_image_order(src)
    finally:
        src.close()

    if not image_ids:
        print("Error: input database has no images.", file=sys.stderr)
        sys.exit(1)

    chunks = plan_chunks(image_ids, args.chunk_size)

    stem = input_path.stem
    split_paths = [
        output_dir / f"{stem}_p{i + 1}.db" for i in range(len(chunks))
    ]

    # Pre-flight: bail before writing anything if any target exists and --force wasn't set.
    if not args.force:
        existing = [p for p in split_paths if p.exists()]
        if existing:
            print(
                "Error: the following output files already exist (pass --force to overwrite):",
                file=sys.stderr,
            )
            for p in existing:
                print(f"  {p}", file=sys.stderr)
            sys.exit(1)

    total_images = sum(len(c) for c in chunks)
    print(
        f"Splitting {input_path} ({total_images} images, max {args.chunk_size} images/chunk) "
        f"into {len(chunks)} files..."
    )

    written = []
    try:
        for i, image_ids in enumerate(chunks, start=1):
            target = split_paths[i - 1]
            n_img, n_det, n_del, n_hard = write_chunk(
                input_path, target, image_ids, schema, args.force
            )
            written.append(target)
            print(
                f"  [{i}/{len(chunks)}] {target.name}: "
                f"{n_img} images, {n_det} detections "
                f"(deleted={n_del}, hard={n_hard})"
            )
    except Exception as exc:
        # Roll back: remove any already-written splits so we don't leave the
        # workspace in a half-split state.
        print(f"\nError while writing splits: {exc}", file=sys.stderr)
        for p in written:
            try:
                p.unlink()
            except OSError:
                pass
        sys.exit(1)

    print("\nVerifying splits against source...")
    problems, stats = verify_splits(input_path, written)
    if problems:
        print("Verification FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "The split files have been kept for inspection; do NOT distribute them.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"  images:     source={stats['images'][0]:>7}   splits={stats['images'][1]:>7}   OK"
    )
    print(
        f"  detections: source={stats['detections'][0]:>7}   splits={stats['detections'][1]:>7}   OK"
    )
    print(
        f"  deleted=1:  source={stats['deleted'][0]:>7}   splits={stats['deleted'][1]:>7}   OK"
    )
    print(
        f"  hard=1:     source={stats['hard'][0]:>7}   splits={stats['hard'][1]:>7}   OK"
    )
    print("  file_path set matches source exactly.")
    print(f"\nDone. {len(written)} split files written to {output_dir}.")


if __name__ == "__main__":
    main()
