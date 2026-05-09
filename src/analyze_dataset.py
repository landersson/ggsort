#!/usr/bin/env python3
"""
Inspect a metadata.json produced by mkdataset.py.

Prints a textual summary and ASCII histograms of the exported sample. When the metadata
includes diversity-mode diagnostic fields (sequence_id, sequence_length, novelty_score),
also prints sequence-length distribution, picks-per-sequence stats, and a per-(location, class)
ASCII timeline. Pass two metadata files to compare them side by side.

  python src/analyze_dataset.py /tmp/test-dataset/metadata.json
  python src/analyze_dataset.py /tmp/legacy/metadata.json /tmp/diversity/metadata.json
  python src/analyze_dataset.py /tmp/test-dataset/metadata.json --plot /tmp/dataset.png
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

CATEGORY_NAMES = {
    1: "Gang Gang",
    2: "Person",
    3: "Vehicle",
    4: "Possum",
    5: "Other",
}


def _parse_exif_datetime(s):
    """Same parser as mkdataset.py — duplicated to keep this script standalone."""
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y:%m:%d %H:%M:%S')
    except (ValueError, TypeError):
        return None


def _load(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of metadata entries")
    return data


def _has_diversity_fields(entries):
    return any('sequence_id' in e for e in entries)


def _ascii_bar(value, max_value, width=40, fill='█', empty='░'):
    if max_value <= 0:
        return empty * width
    n = int(round(width * value / max_value))
    n = max(0, min(width, n))
    return fill * n + empty * (width - n)


def _length_bucket(L):
    if L <= 1:
        return "1"
    if L <= 4:
        return "2-4"
    if L <= 10:
        return "5-10"
    if L <= 25:
        return "11-25"
    if L <= 50:
        return "26-50"
    return "51+"


_BUCKET_ORDER = ["1", "2-4", "5-10", "11-25", "26-50", "51+"]


def _summarize_basic(entries):
    """Counts and time range info common to legacy and diversity outputs."""
    by_location = Counter()
    by_class = Counter()
    by_loc_class = Counter()
    times = []

    for e in entries:
        loc = e.get('location', '?')
        by_location[loc] += 1
        cats = {d['category'] for d in e.get('detections', [])}
        for cat in cats:
            by_class[cat] += 1
            by_loc_class[(loc, cat)] += 1
        dt = _parse_exif_datetime(e.get('datetime_original'))
        if dt is not None:
            times.append(dt)

    return {
        'total': len(entries),
        'by_location': by_location,
        'by_class': by_class,
        'by_loc_class': by_loc_class,
        'time_min': min(times) if times else None,
        'time_max': max(times) if times else None,
        'times': times,
    }


def _summarize_diversity(entries):
    """Sequence-aware stats. Returns None if no diversity fields are present."""
    if not _has_diversity_fields(entries):
        return None

    # Group by (location, class, sequence_id). The same sequence_id may be reused across
    # different (location, class) pairs since each sample_images_by_category call assigns
    # its own sequence ids — so the bucket key includes location and class to keep them apart.
    buckets = defaultdict(list)
    for e in entries:
        sid = e.get('sequence_id')
        if sid is None:
            continue
        loc = e.get('location', '?')
        # Use the first selected_for_classes entry as the "primary" class for grouping;
        # fall back to the first detection's category.
        sel = e.get('selected_for_classes') or []
        primary = sel[0] if sel else (e['detections'][0]['category'] if e.get('detections') else None)
        buckets[(loc, primary, sid)].append(e)

    seq_lengths = []
    picks_per_seq = []
    pick_to_length = []
    novelty_values = []

    for key, group in buckets.items():
        # All entries in a sequence carry the same sequence_length (set by _detect_sequences).
        L = group[0].get('sequence_length') or len(group)
        picks = len(group)
        seq_lengths.append(L)
        picks_per_seq.append(picks)
        pick_to_length.append((L, picks))
        for e in group:
            n = e.get('novelty_score')
            if n is not None:
                novelty_values.append(n)

    return {
        'n_sequences': len(buckets),
        'seq_lengths': seq_lengths,
        'picks_per_seq': picks_per_seq,
        'pick_to_length': pick_to_length,
        'novelty_values': novelty_values,
    }


def _print_basic(label, summary):
    print(f"\n=== {label} ===")
    print(f"Total images: {summary['total']}")
    if summary['time_min'] and summary['time_max']:
        print(f"Time range: {summary['time_min']} → {summary['time_max']}")
    else:
        print("Time range: (no datetime_original on any entry)")

    print("\nBy location:")
    for loc, n in sorted(summary['by_location'].items(), key=lambda kv: -kv[1]):
        print(f"  {loc:<28} {n:>5}")

    print("\nBy class (images containing >=1 detection of class):")
    for cat, n in sorted(summary['by_class'].items(), key=lambda kv: -kv[1]):
        name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
        print(f"  {name:<14} {n:>5}")


def _print_time_histogram(label, summary, n_buckets=24, width=40):
    """Per-(location, class) horizontal histogram across the full export time range."""
    if not summary['times']:
        return

    t_min = summary['time_min']
    t_max = summary['time_max']
    span = (t_max - t_min).total_seconds()
    if span <= 0:
        span = 1.0  # all selections at the same instant — degenerate but not a divide-by-zero.

    # Build per-(location, class) bucket counts.
    per_lc = defaultdict(lambda: [0] * n_buckets)
    for entry, dt in _iter_with_time(summary):
        loc = entry.get('location', '?')
        cats = {d['category'] for d in entry.get('detections', [])} or {None}
        # Use first selected_for_classes if available so diversity entries align with which
        # category targeted them; otherwise charge each represented detection class.
        sel = entry.get('selected_for_classes') or []
        if sel:
            cats = set(sel)
        b = int(((dt - t_min).total_seconds() / span) * n_buckets)
        if b == n_buckets:
            b -= 1
        for cat in cats:
            per_lc[(loc, cat)][b] += 1

    print(f"\nTime histogram ({n_buckets} buckets across {t_min} → {t_max}):")
    for (loc, cat), counts in sorted(per_lc.items()):
        name = CATEGORY_NAMES.get(cat, f"cat-{cat}" if cat is not None else "?")
        max_c = max(counts) if counts else 0
        bar = ''.join(
            '█' if c == max_c and c > 0 else
            '▓' if c > 0 and c >= max_c * 0.66 else
            '▒' if c > 0 and c >= max_c * 0.33 else
            '░' if c > 0 else
            ' '
            for c in counts
        )
        print(f"  {loc} / {name:<10} |{bar}| max={max_c}")


def _iter_with_time(summary):
    """Yield (entry, datetime) pairs by re-walking the entries list — but we don't have it
    on the summary. Caller supplies it via summary['_entries']."""
    for e in summary['_entries']:
        dt = _parse_exif_datetime(e.get('datetime_original'))
        if dt is not None:
            yield e, dt


def _print_diversity(label, div):
    if div is None:
        return
    print(f"\n--- Diversity diagnostics for {label} ---")
    print(f"Total sequences: {div['n_sequences']}")
    if div['seq_lengths']:
        avg_len = sum(div['seq_lengths']) / len(div['seq_lengths'])
        print(f"Mean sequence length: {avg_len:.1f}, max: {max(div['seq_lengths'])}")

    # Sequence-length histogram by bucket.
    bucket_counts = Counter(_length_bucket(L) for L in div['seq_lengths'])
    max_b = max(bucket_counts.values()) if bucket_counts else 0
    print("\nSequence-length distribution (number of sequences in each length bucket):")
    for b in _BUCKET_ORDER:
        n = bucket_counts.get(b, 0)
        bar = _ascii_bar(n, max_b, width=40)
        print(f"  {b:<6} |{bar}| {n}")

    # Picks per sequence as a function of length — the sublinearity verification.
    pick_by_bucket = defaultdict(list)
    for L, picks in div['pick_to_length']:
        pick_by_bucket[_length_bucket(L)].append(picks)
    print("\nMean picks per sequence by length bucket (verifies sublinear allocation):")
    for b in _BUCKET_ORDER:
        ps = pick_by_bucket.get(b, [])
        if ps:
            mean_p = sum(ps) / len(ps)
            print(f"  {b:<6}  mean picks = {mean_p:>5.2f}   (n={len(ps)} sequences)")

    # Novelty histogram.
    if div['novelty_values']:
        edges = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["[0.0,0.1)", "[0.1,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
        bins = [0] * (len(edges) - 1)
        for v in div['novelty_values']:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    bins[i] += 1
                    break
        max_b = max(bins) if bins else 0
        print("\nNovelty score distribution (selected frames):")
        for label_, n in zip(labels, bins):
            bar = _ascii_bar(n, max_b, width=40)
            print(f"  {label_:<10} |{bar}| {n}")


def _maybe_plot(out_path, summaries):
    """Render side-by-side matplotlib figures for one or two summaries. Optional dep."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Error: --plot requires matplotlib. Install it with: pip install matplotlib", file=sys.stderr)
        return 1

    n = len(summaries)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 12), squeeze=False)

    for col, (label, basic, div) in enumerate(summaries):
        # Row 0: per-location image counts.
        locs = sorted(basic['by_location'].items(), key=lambda kv: kv[1])
        if locs:
            names = [k for k, _ in locs]
            vals = [v for _, v in locs]
            axes[0][col].barh(names, vals)
        axes[0][col].set_title(f"{label}\nImages per location")
        axes[0][col].set_xlabel('count')

        # Row 1: time scatter.
        if basic['times']:
            axes[1][col].hist(basic['times'], bins=48)
        axes[1][col].set_title('Selections over time (48 bins)')
        axes[1][col].set_xlabel('time')
        axes[1][col].set_ylabel('count')

        # Row 2: sequence length vs picks (diversity only).
        if div is not None and div['pick_to_length']:
            xs = [L for L, _ in div['pick_to_length']]
            ys = [p for _, p in div['pick_to_length']]
            axes[2][col].scatter(xs, ys, alpha=0.5)
            # Reference sqrt curve.
            if xs:
                import math
                xmax = max(xs)
                ref_xs = list(range(1, xmax + 1))
                # Scale sqrt so it matches mean picks at the largest x.
                scale = (max(ys) / math.sqrt(xmax)) if xmax > 0 else 1.0
                ref_ys = [scale * math.sqrt(x) for x in ref_xs]
                axes[2][col].plot(ref_xs, ref_ys, 'r--', label='sqrt reference', alpha=0.6)
                axes[2][col].legend()
            axes[2][col].set_xscale('log')
            axes[2][col].set_yscale('log')
            axes[2][col].set_xlabel('sequence length')
            axes[2][col].set_ylabel('picks from sequence')
            axes[2][col].set_title('Sublinearity check')
        else:
            axes[2][col].text(0.5, 0.5, '(legacy mode — no sequence info)',
                              ha='center', va='center', transform=axes[2][col].transAxes)
            axes[2][col].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved to: {out_path}")
    return 0


def analyze(paths, plot_path=None):
    summaries = []  # (label, basic_summary, diversity_summary_or_None)

    for path in paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            return 1
        entries = _load(path)
        basic = _summarize_basic(entries)
        basic['_entries'] = entries  # used by time-histogram printer
        div = _summarize_diversity(entries)
        label = os.path.basename(os.path.dirname(path)) or path
        summaries.append((label, basic, div))

        mode = "DIVERSITY" if div is not None else "LEGACY"
        _print_basic(f"{label}  ({mode})", basic)
        _print_time_histogram(label, basic)
        _print_diversity(label, div)

    if len(summaries) == 2:
        _print_comparison(summaries)

    if plot_path:
        return _maybe_plot(plot_path, summaries)

    return 0


def _print_comparison(summaries):
    """Side-by-side comparison stats for exactly two summaries."""
    (la, ba, da), (lb, bb, db) = summaries
    print("\n=== Comparison ===")
    print(f"{la:<30} vs {lb}")
    print(f"  Total:        {ba['total']:>6}   {bb['total']:>6}")

    locs = sorted(set(ba['by_location']) | set(bb['by_location']))
    print("  Per location:")
    for loc in locs:
        a = ba['by_location'].get(loc, 0)
        b = bb['by_location'].get(loc, 0)
        delta = b - a
        sign = '+' if delta >= 0 else ''
        print(f"    {loc:<28} {a:>5}   {b:>5}   ({sign}{delta})")

    if da and db:
        print("  Sequences detected:")
        print(f"    {da['n_sequences']:>5}   {db['n_sequences']:>5}")
        if da['seq_lengths'] and db['seq_lengths']:
            print(f"    max len: {max(da['seq_lengths']):>3}   {max(db['seq_lengths']):>3}")
            print(f"    mean len: {sum(da['seq_lengths'])/len(da['seq_lengths']):.1f}   "
                  f"{sum(db['seq_lengths'])/len(db['seq_lengths']):.1f}")
    elif da or db:
        print("  (one side is legacy, one is diversity — sequence stats only available for diversity)")


def main():
    parser = argparse.ArgumentParser(description="Analyze a mkdataset.py metadata.json")
    parser.add_argument('metadata_files', nargs='+',
                        help='One or two metadata.json paths. Two paths triggers comparison mode.')
    parser.add_argument('--plot', type=str, default=None,
                        help='Optional output path for a matplotlib PNG visualization.')
    args = parser.parse_args()

    if len(args.metadata_files) > 2:
        print("Error: at most two metadata files for side-by-side comparison.", file=sys.stderr)
        return 1

    return analyze(args.metadata_files, plot_path=args.plot)


if __name__ == '__main__':
    sys.exit(main())
