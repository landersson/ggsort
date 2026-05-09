#!/usr/bin/env python3
"""Tests for src/mkdataset.py — diversity sampling helpers and end-to-end behavior."""

import os
import random
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mkdataset
from mkdataset import (
    DiversityConfig,
    ImageRecord,
    _allocate_sequence_budgets,
    _detect_sequences,
    _iou,
    _max_pairwise_iou,
    _parse_exif_datetime,
    _sample_within_sequence,
    sample_images_by_category,
)


def _ts(t):
    """Format a datetime as the EXIF DateTimeOriginal string."""
    return t.strftime('%Y:%m:%d %H:%M:%S')


class TestIoU(unittest.TestCase):
    """_iou must agree with hand-computed values across the standard cases."""

    def test_identical_boxes(self):
        self.assertAlmostEqual(_iou((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(_iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)), 0.0)

    def test_half_overlap(self):
        # Two unit squares overlapping by half on the x-axis: inter=0.5, union=1.5 → 1/3.
        self.assertAlmostEqual(_iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.0, 1.0)), 1.0 / 3.0)

    def test_zero_area_box(self):
        # No divide-by-zero, result is 0 (no overlap possible with a degenerate box).
        self.assertEqual(_iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)), 0.0)

    def test_negative_dimensions(self):
        # Defensive: negative dims should give 0, not nonsense.
        self.assertEqual(_iou((0.0, 0.0, -1.0, 1.0), (0.0, 0.0, 1.0, 1.0)), 0.0)

    def test_max_pairwise_iou(self):
        boxes_a = ((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.4, 0.4))
        boxes_b = ((0.5, 0.5, 0.4, 0.4), (0.9, 0.9, 0.05, 0.05))  # second pair identical
        self.assertAlmostEqual(_max_pairwise_iou(boxes_a, boxes_b), 1.0)

    def test_max_pairwise_iou_empty(self):
        self.assertEqual(_max_pairwise_iou((), ((0.0, 0.0, 1.0, 1.0),)), 0.0)


class TestParseExifDatetime(unittest.TestCase):
    def test_parses_valid(self):
        self.assertEqual(_parse_exif_datetime('2024:03:15 13:45:30'),
                         datetime(2024, 3, 15, 13, 45, 30))

    def test_none_input(self):
        self.assertIsNone(_parse_exif_datetime(None))

    def test_empty_string(self):
        self.assertIsNone(_parse_exif_datetime(''))

    def test_malformed_zero_date(self):
        # Some broken cameras emit this; we don't want to raise.
        self.assertIsNone(_parse_exif_datetime('0000:00:00 00:00:00'))

    def test_garbage_string(self):
        self.assertIsNone(_parse_exif_datetime('not a date'))


class TestDetectSequences(unittest.TestCase):
    """The sequence detector links adjacent frames by time gap + IoU only — not by index."""

    def _record(self, t, boxes, file_path='x'):
        return ImageRecord(
            id=hash((t, boxes, file_path)) & 0xffff,
            file_path=file_path,
            datetime_original=_ts(t) if t is not None else None,
            boxes=boxes,
        )

    def test_three_frame_burst_links(self):
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        box = (0.4, 0.4, 0.2, 0.2)
        records = [self._record(t0 + timedelta(seconds=i * 5), (box,)) for i in range(3)]
        _detect_sequences(records, DiversityConfig(enabled=True))
        self.assertEqual({r.sequence_id for r in records}, {0})  # all linked
        self.assertEqual({r.sequence_length for r in records}, {3})
        self.assertEqual(records[0].novelty_score, 1.0)  # first is always 1.0
        # Identical boxes → near-zero novelty for linked frames.
        self.assertAlmostEqual(records[1].novelty_score, 0.0)
        self.assertAlmostEqual(records[2].novelty_score, 0.0)

    def test_time_gap_breaks_sequence(self):
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        box = (0.4, 0.4, 0.2, 0.2)
        records = [
            self._record(t0, (box,)),
            self._record(t0 + timedelta(seconds=5), (box,)),
            self._record(t0 + timedelta(seconds=200), (box,)),  # > 60s default → break
        ]
        _detect_sequences(records, DiversityConfig(enabled=True))
        self.assertEqual(records[0].sequence_id, 0)
        self.assertEqual(records[1].sequence_id, 0)
        self.assertEqual(records[2].sequence_id, 1)

    def test_low_iou_breaks_sequence(self):
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        records = [
            self._record(t0, ((0.0, 0.0, 0.2, 0.2),)),
            self._record(t0 + timedelta(seconds=5), ((0.8, 0.8, 0.2, 0.2),)),  # disjoint
        ]
        _detect_sequences(records, DiversityConfig(enabled=True, iou_threshold=0.5))
        self.assertEqual(records[0].sequence_id, 0)
        self.assertEqual(records[1].sequence_id, 1)

    def test_null_datetime_forces_isolation(self):
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        box = (0.4, 0.4, 0.2, 0.2)
        records = [
            self._record(t0, (box,)),
            self._record(None, (box,)),         # no timestamp → break
            self._record(t0 + timedelta(seconds=5), (box,)),
        ]
        _detect_sequences(records, DiversityConfig(enabled=True))
        # Three distinct sequences: prev->null breaks, null->next breaks (no datetime on null side).
        self.assertEqual({r.sequence_id for r in records}, {0, 1, 2})
        for r in records:
            self.assertEqual(r.sequence_length, 1)
            self.assertEqual(r.novelty_score, 1.0)

    def test_multi_box_link_by_any_pair(self):
        """If any pair across the two frames' box sets clears IoU threshold, link."""
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        prev_boxes = ((0.0, 0.0, 0.2, 0.2), (0.7, 0.7, 0.2, 0.2))  # two boxes
        # curr's first box matches prev's first; second box is wildly different.
        curr_boxes = ((0.0, 0.0, 0.2, 0.2), (0.4, 0.1, 0.1, 0.1))
        records = [
            ImageRecord(id=1, file_path='a', datetime_original=_ts(t0), boxes=prev_boxes),
            ImageRecord(id=2, file_path='b', datetime_original=_ts(t0 + timedelta(seconds=5)),
                        boxes=curr_boxes),
        ]
        _detect_sequences(records, DiversityConfig(enabled=True, iou_threshold=0.5))
        self.assertEqual(records[0].sequence_id, records[1].sequence_id)

    def test_handles_missing_frames_in_burst(self):
        """Manually-deleted intermediate frames must not break linking when survivors are still
        within max_gap_seconds — the algorithm depends on time + IoU, never on frame indices."""
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        box = (0.4, 0.4, 0.2, 0.2)
        # Frames at 0s, 5s, 10s, 15s, 20s — but 5s and 10s "deleted" so we only see the rest.
        records = [
            ImageRecord(id=i, file_path=f'f{i}', datetime_original=_ts(t0 + timedelta(seconds=t)),
                        boxes=(box,))
            for i, t in enumerate([0, 15, 20])
        ]
        _detect_sequences(records, DiversityConfig(enabled=True))
        # 15s gap < 60s default, so 0s↔15s links; 5s gap < 60s, 15s↔20s links → one sequence.
        self.assertEqual({r.sequence_id for r in records}, {0})
        self.assertEqual({r.sequence_length for r in records}, {3})

    def test_empty_records(self):
        _detect_sequences([], DiversityConfig(enabled=True))  # must not raise


class TestAllocateSequenceBudgets(unittest.TestCase):
    def test_largest_remainder_caps_at_length(self):
        # Lengths 10/5/1, target 8, exponent 0.5 → weights ~3.16/2.24/1.0.
        # Budget split should sum to 8 and never exceed each cap.
        alloc = _allocate_sequence_budgets([10, 5, 1], target=8, exponent=0.5)
        self.assertEqual(sum(alloc), 8)
        self.assertEqual(len(alloc), 3)
        for a, L in zip(alloc, [10, 5, 1]):
            self.assertLessEqual(a, L)

    def test_target_exceeds_total_takes_all(self):
        self.assertEqual(_allocate_sequence_budgets([3, 2, 1], target=99, exponent=0.5), [3, 2, 1])

    def test_zero_target(self):
        self.assertEqual(_allocate_sequence_budgets([5, 5], target=0, exponent=0.5), [0, 0])

    def test_empty(self):
        self.assertEqual(_allocate_sequence_budgets([], target=10, exponent=0.5), [])

    def test_exponent_zero_equal_per_sequence(self):
        # Exponent 0 → weight 1 per nonzero-length sequence → equal slots, capped.
        alloc = _allocate_sequence_budgets([100, 100, 100], target=6, exponent=0.0)
        self.assertEqual(alloc, [2, 2, 2])

    def test_exponent_one_proportional(self):
        # Exponent 1 → weights are the lengths themselves → roughly proportional.
        alloc = _allocate_sequence_budgets([8, 4, 4], target=8, exponent=1.0)
        self.assertEqual(sum(alloc), 8)
        # Largest sequence should get the largest allocation.
        self.assertEqual(max(alloc), alloc[0])

    def test_redistribute_when_caps_bind(self):
        # Tiny first sequence saturates immediately; remainder must flow to the other one.
        alloc = _allocate_sequence_budgets([1, 100], target=20, exponent=0.5)
        self.assertEqual(sum(alloc), 20)
        self.assertEqual(alloc[0], 1)  # saturated at length
        self.assertEqual(alloc[1], 19)


class TestSampleWithinSequence(unittest.TestCase):
    def _seq(self, n, novelty_pattern=None):
        out = []
        for i in range(n):
            r = ImageRecord(id=i, file_path=f'p{i:04d}', sequence_id=0, sequence_length=n)
            r.novelty_score = novelty_pattern[i] if novelty_pattern else (1.0 if i == 0 else 0.5)
            out.append(r)
        return out

    def test_anchors_first_and_last(self):
        seq = self._seq(20)
        rng = random.Random(0)
        out = _sample_within_sequence(seq, allocation=5, rng=rng)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0].id, seq[0].id)
        self.assertEqual(out[-1].id, seq[-1].id)

    def test_take_all_when_allocation_exceeds_length(self):
        seq = self._seq(3)
        rng = random.Random(0)
        out = _sample_within_sequence(seq, allocation=10, rng=rng)
        self.assertEqual(len(out), 3)

    def test_zero_allocation_returns_empty(self):
        self.assertEqual(_sample_within_sequence(self._seq(5), 0, random.Random(0)), [])

    def test_single_pick_returns_first(self):
        seq = self._seq(5)
        out = _sample_within_sequence(seq, allocation=1, rng=random.Random(0))
        self.assertEqual([r.id for r in out], [seq[0].id])

    def test_two_picks_returns_first_and_last(self):
        seq = self._seq(5)
        out = _sample_within_sequence(seq, allocation=2, rng=random.Random(0))
        self.assertEqual([r.id for r in out], [seq[0].id, seq[-1].id])

    def test_deterministic_with_seed(self):
        seq = self._seq(50)
        a = _sample_within_sequence(seq, 10, random.Random(42))
        b = _sample_within_sequence(seq, 10, random.Random(42))
        self.assertEqual([r.id for r in a], [r.id for r in b])

    def test_different_seeds_differ(self):
        seq = self._seq(50)
        a = _sample_within_sequence(seq, 10, random.Random(1))
        b = _sample_within_sequence(seq, 10, random.Random(2))
        # Anchors are the same, but interior picks should diverge for at least one position.
        self.assertNotEqual([r.id for r in a], [r.id for r in b])


# ---------------------------------------------------------------------------
# End-to-end tests against an in-memory SQLite DB matching the production schema
# ---------------------------------------------------------------------------

def _make_db():
    """Build an in-memory sqlite DB matching src/make_db.py's schema."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER, height INTEGER,
            datetime_original TEXT,
            manually_processed BOOLEAN DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            category INTEGER, confidence REAL,
            x REAL, y REAL, width REAL, height REAL,
            deleted BOOLEAN DEFAULT 0, subcategory INTEGER, hard BOOLEAN DEFAULT 0,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
    """)
    return conn


def _insert_image(conn, img_id, file_path, dt_str, boxes, category=1, confidence=0.9):
    """Insert one image plus a detection row per box. Boxes are (x, y, w, h)."""
    c = conn.cursor()
    c.execute("INSERT INTO images (id, file_path, datetime_original, width, height) VALUES (?, ?, ?, ?, ?)",
              (img_id, file_path, dt_str, 1920, 1080))
    for box in boxes:
        c.execute(
            "INSERT INTO detections (image_id, category, confidence, x, y, width, height, deleted, hard) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
            (img_id, category, confidence, box[0], box[1], box[2], box[3]),
        )


class TestEndToEnd(unittest.TestCase):
    """Hand-build a DB with bursts + isolated frames, run sampling both ways, assert behavior."""

    def setUp(self):
        self.conn = _make_db()
        # Burst location: 50 frames, near-identical boxes, 5s apart.
        t0 = datetime(2024, 1, 1, 8, 0, 0)
        burst_box = (0.4, 0.4, 0.2, 0.2)
        for i in range(50):
            _insert_image(
                self.conn,
                img_id=1000 + i,
                file_path=f"BurstLoc/frame_{i:04d}.jpg",
                dt_str=_ts(t0 + timedelta(seconds=i * 5)),
                boxes=[burst_box],
            )
        # 5 isolated frames at the same location, hours apart, with drifting boxes.
        for i in range(5):
            _insert_image(
                self.conn,
                img_id=2000 + i,
                file_path=f"BurstLoc/iso_{i:04d}.jpg",
                dt_str=_ts(t0 + timedelta(hours=2 + i * 3)),
                boxes=[(0.1 + i * 0.15, 0.1 + i * 0.05, 0.15, 0.15)],
            )
        # 3 NULL-datetime frames: same location, similar boxes, but unparseable EXIF.
        for i in range(3):
            _insert_image(
                self.conn,
                img_id=3000 + i,
                file_path=f"BurstLoc/null_{i:04d}.jpg",
                dt_str=None,
                boxes=[burst_box],
            )

    def tearDown(self):
        self.conn.close()

    def _sample(self, target, **cfg_overrides):
        cursor = self.conn.cursor()
        location_stats = {"BurstLoc": 58}
        config = DiversityConfig(enabled=True, **cfg_overrides)
        return sample_images_by_category(cursor, location_stats, 1, target, "Gang Gang", config)

    def test_legacy_path_unchanged(self):
        """No diversity flag → stride sampling over alphabetical file_path order — same as before."""
        cursor = self.conn.cursor()
        location_stats = {"BurstLoc": 58}
        result = sample_images_by_category(cursor, location_stats, 1, 10, "Gang Gang", None)
        self.assertEqual(len(result), 10)
        # Each record should be tagged with the targeted class.
        for rec in result:
            self.assertEqual(rec.selected_for_classes, [1])
        # Stride is over file_path order: BurstLoc/frame_*, then iso_*, then null_*.
        # With 58 available and 10 target, step=5.8, indices = 0,5,11,17,23,29,34,40,46,52.
        # We don't pin exact paths (alphabetical sort across mixed names is fragile), but we
        # should have sampled mostly from frame_* (the bulk).
        frame_count = sum(1 for r in result if 'frame_' in r.file_path)
        self.assertGreater(frame_count, 5)

    def test_diversity_downsamples_burst(self):
        """The 50-frame burst should contribute much less than its proportional share."""
        result = self._sample(target=20)
        self.assertLessEqual(len(result), 20)
        # All 5 isolates should be present (each is its own length-1 sequence with novelty 1.0,
        # and we never have more sequences than the target so each gets at least one slot).
        iso_paths = {r.file_path for r in result if 'iso_' in r.file_path}
        self.assertEqual(len(iso_paths), 5, "all isolated frames should be retained")
        # All 3 NULL-datetime frames are also forced isolates → each its own sequence.
        null_paths = {r.file_path for r in result if 'null_' in r.file_path}
        self.assertEqual(len(null_paths), 3, "all NULL-datetime frames should be retained")
        # The remaining slots come from the burst, which had 50 frames in one sequence.
        burst_picks = sum(1 for r in result if 'frame_' in r.file_path)
        self.assertEqual(burst_picks, 20 - 5 - 3)
        # First and last frames of the burst should be anchored.
        burst_paths = sorted(r.file_path for r in result if 'frame_' in r.file_path)
        self.assertEqual(burst_paths[0], "BurstLoc/frame_0000.jpg")
        self.assertEqual(burst_paths[-1], "BurstLoc/frame_0049.jpg")

    def test_diversity_diagnostic_fields_set(self):
        result = self._sample(target=20)
        for rec in result:
            self.assertIsNotNone(rec.sequence_id)
            self.assertIsNotNone(rec.sequence_length)
            self.assertIsNotNone(rec.novelty_score)
            self.assertEqual(rec.selected_for_classes, [1])

    def test_diversity_deterministic_with_seed(self):
        a = self._sample(target=20, seed=7)
        b = self._sample(target=20, seed=7)
        self.assertEqual(
            sorted(r.file_path for r in a),
            sorted(r.file_path for r in b),
        )

    def test_diversity_seed_changes_picks(self):
        a = self._sample(target=20, seed=1)
        b = self._sample(target=20, seed=2)
        # Same totals but different burst-interior picks (anchors will match).
        self.assertEqual(len(a), len(b))
        a_paths = {r.file_path for r in a}
        b_paths = {r.file_path for r in b}
        # We only require at least one differing pick somewhere.
        self.assertNotEqual(a_paths, b_paths)


# ---------------------------------------------------------------------------
# --save-as-db: write the sample to a fresh SQLite DB matching make_db.py schema
# ---------------------------------------------------------------------------

from mkdataset import _write_output_db


class TestSaveAsDb(unittest.TestCase):
    """Verify the new DB output mode produces a valid, ggsort-compatible database."""

    def setUp(self):
        self.conn = _make_db()
        # A small mix: one short burst + a couple of isolated frames + one image with
        # both gang_gang AND possum detections (exercises multi-class dedup) + one image
        # with a hard detection that should be filtered out of the output.
        t0 = datetime(2024, 5, 1, 9, 0, 0)
        for i in range(6):
            _insert_image(self.conn, img_id=100 + i,
                          file_path=f"locA/burst_{i:02d}.jpg",
                          dt_str=_ts(t0 + timedelta(seconds=i * 5)),
                          boxes=[(0.4, 0.4, 0.2, 0.2)])
        _insert_image(self.conn, img_id=200,
                      file_path="locA/iso_00.jpg",
                      dt_str=_ts(t0 + timedelta(hours=4)),
                      boxes=[(0.1, 0.1, 0.15, 0.15)])
        _insert_image(self.conn, img_id=201,
                      file_path="locA/iso_01.jpg",
                      dt_str=_ts(t0 + timedelta(hours=8)),
                      boxes=[(0.7, 0.7, 0.1, 0.1)])
        # Image with a hard detection — should never make it into the output DB.
        c = self.conn.cursor()
        c.execute("INSERT INTO images (id, file_path, datetime_original, width, height) "
                  "VALUES (300, 'locA/hard_only.jpg', ?, 1920, 1080)", (_ts(t0 + timedelta(hours=12)),))
        c.execute("INSERT INTO detections (image_id, category, confidence, x, y, width, height, deleted, hard) "
                  "VALUES (300, 1, 0.9, 0.3, 0.3, 0.1, 0.1, 0, 1)")
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmpfile.close()
        self.out_path = self.tmpfile.name
        os.unlink(self.out_path)  # _write_output_db will create it fresh

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.out_path):
            os.unlink(self.out_path)

    def _sample_and_write(self, target=10):
        cursor = self.conn.cursor()
        location_stats = {"locA": 9}
        records = sample_images_by_category(cursor, location_stats, 1, target, "Gang Gang", None)
        _write_output_db(cursor, records, self.out_path,
                         images_dir="/some/base", confidence_threshold=0.20)
        return records

    def test_writes_valid_schema(self):
        self._sample_and_write(target=10)
        self.assertTrue(os.path.exists(self.out_path))

        out = sqlite3.connect(self.out_path)
        out.row_factory = sqlite3.Row
        c = out.cursor()

        # Check the four expected sqlite_master entries are present.
        c.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'index') ORDER BY name")
        names = {row['name'] for row in c.fetchall()}
        self.assertIn('images', names)
        self.assertIn('detections', names)
        self.assertIn('idx_image_id', names)
        self.assertIn('idx_file_path', names)

        # Column lists match the canonical schema (order doesn't matter).
        c.execute("PRAGMA table_info(images)")
        img_cols = {row['name'] for row in c.fetchall()}
        self.assertEqual(img_cols, {'id', 'file_path', 'width', 'height',
                                    'datetime_original', 'manually_processed'})
        c.execute("PRAGMA table_info(detections)")
        det_cols = {row['name'] for row in c.fetchall()}
        self.assertEqual(det_cols, {'id', 'image_id', 'category', 'confidence',
                                    'x', 'y', 'width', 'height', 'deleted',
                                    'subcategory', 'hard'})
        out.close()

    def test_invariants_on_written_rows(self):
        self._sample_and_write(target=10)
        out = sqlite3.connect(self.out_path)
        out.row_factory = sqlite3.Row
        c = out.cursor()

        # Every detection row has hard=0, deleted=0, subcategory=NULL — matches the canonical
        # filter mkdataset applies. The hard-only image (id=300) must be absent from the
        # output entirely (it had no surviving detections, so it shouldn't have been picked).
        c.execute("SELECT COUNT(*) FROM detections WHERE hard != 0 OR deleted != 0 OR subcategory IS NOT NULL")
        self.assertEqual(c.fetchone()[0], 0)

        c.execute("SELECT COUNT(*) FROM images WHERE file_path = 'locA/hard_only.jpg'")
        self.assertEqual(c.fetchone()[0], 0,
                         "image with only hard detections should not appear in output")

        # All file_paths are relative (no absolute paths leaking through).
        c.execute("SELECT file_path FROM images")
        for row in c.fetchall():
            self.assertFalse(os.path.isabs(row['file_path']),
                             f"unexpected absolute path: {row['file_path']!r}")

        # datetime_original carries through.
        c.execute("SELECT datetime_original FROM images WHERE file_path = 'locA/burst_00.jpg'")
        row = c.fetchone()
        self.assertIsNotNone(row)
        self.assertTrue(row['datetime_original'].startswith('2024:05:01'))

        # FK integrity: every detections.image_id must exist in images.id.
        c.execute("""
            SELECT COUNT(*) FROM detections d
            LEFT JOIN images i ON i.id = d.image_id
            WHERE i.id IS NULL
        """)
        self.assertEqual(c.fetchone()[0], 0)

        # confidence preserved (we used 0.9 in the fixture).
        c.execute("SELECT MIN(confidence), MAX(confidence) FROM detections")
        lo, hi = c.fetchone()
        self.assertAlmostEqual(lo, 0.9)
        self.assertAlmostEqual(hi, 0.9)

        out.close()

    def test_no_duplicate_image_rows_for_multi_class(self):
        """Mark one image as also a possum detection; sampling both classes should yield
        one row in the output's images table, not two (UNIQUE(file_path) holds)."""
        # Add a possum detection to burst_00.
        c = self.conn.cursor()
        c.execute("INSERT INTO detections (image_id, category, confidence, x, y, width, height, deleted, hard) "
                  "VALUES (100, 4, 0.9, 0.5, 0.5, 0.1, 0.1, 0, 0)")

        cursor = self.conn.cursor()
        location_stats = {"locA": 9}
        gg = sample_images_by_category(cursor, location_stats, 1, 5, "Gang Gang", None)
        possum = sample_images_by_category(cursor, location_stats, 4, 5, "Possum", None)
        combined = list(gg) + list(possum)

        # Mimic the dedup logic in export_images.
        seen: dict[str, ImageRecord] = {}
        unique = []
        for r in combined:
            if r.file_path not in seen:
                seen[r.file_path] = r
                unique.append(r)
        # burst_00 was selected for both classes — must still appear exactly once in output.
        self.assertLessEqual(len(unique), len(combined))

        _write_output_db(cursor, unique, self.out_path,
                         images_dir="/some/base", confidence_threshold=0.20)

        out = sqlite3.connect(self.out_path)
        c = out.cursor()
        c.execute("SELECT COUNT(*) FROM images WHERE file_path = 'locA/burst_00.jpg'")
        self.assertEqual(c.fetchone()[0], 1)
        # The possum detection should be present alongside the gang_gang one.
        c.execute("SELECT category FROM detections d JOIN images i ON i.id = d.image_id "
                  "WHERE i.file_path = 'locA/burst_00.jpg' ORDER BY category")
        cats = [row[0] for row in c.fetchall()]
        self.assertEqual(cats, [1, 4])
        out.close()

    def test_relative_path_passthrough(self):
        self._sample_and_write(target=3)
        out = sqlite3.connect(self.out_path)
        c = out.cursor()
        c.execute("SELECT file_path FROM images")
        paths = sorted(row[0] for row in c.fetchall())
        # Source paths were already relative, so they should survive verbatim — no
        # accidental joining with images_dir.
        for p in paths:
            self.assertTrue(p.startswith("locA/"), f"unexpected path: {p!r}")
        out.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
