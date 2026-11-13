#!/usr/bin/env python3
"""Quality gate: BD-rate of every GPU mode (and radius variant) against mode 0.

Reads the CSV produced by bench-quality.ps1.  For every (clip, config) it
compares each GPU mode's rate/quality curve against mode 0 (which is
bit-exact with stock x264) over the overlapping quality range, separately
for PSNR, SSIM and VMAF.  Rows are grouped by their GPU refinement radius
(`--gpu-me-radius`, default 2), and every radius variant is compared against
the same unpruned mode-0 reference, so the gate covers candidate-quality
experiments as well.

Gate (override with QG_PER_CLIP_MAX / QG_MEAN_MAX / QG_STRICT_MAX, in percent):
  - per (clip, config, mode, radius): BD-rate <= +2.0 %
  - mean over all comparisons:        BD-rate <= +1.0 %
  - every encode must have exited with code 0
  - "strict" count reports comparisons above +0.5 % (the recommended mode-2 level)

The band is set around the known content-dependent worst case of the
shipped modes (mode 3/4 on one short 1080p clip, +1.5..+1.9 % PSNR/SSIM,
see the README quality-gate section), so the gate acts as a regression
detector for further work instead of flagging the recorded baseline.
Radius-variant rows (experiments) are skipped unless QG_INCLUDE_VARIANTS=1.

For radius != 2 an informational line also compares the variant against the
same mode at the default radius (isolated effect, PSNR).

Usage:
    py -3 tools/quality-gate.py [bench-results/quality/quality_summary.csv]
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdrate import bd_rate  # noqa: E402

PER_CLIP_MAX = float(os.environ.get("QG_PER_CLIP_MAX", "2.0")) / 100.0
MEAN_MAX = float(os.environ.get("QG_MEAN_MAX", "1.0")) / 100.0
STRICT_MAX = float(os.environ.get("QG_STRICT_MAX", "0.5")) / 100.0
DEFAULT_RADIUS = 2
INCLUDE_VARIANTS = os.environ.get("QG_INCLUDE_VARIANTS") == "1"

METRICS = (("psnr_y", "PSNR"), ("ssim_y", "SSIM"), ("vmaf", "VMAF"))


def radius_of(row):
    try:
        return int(row.get("radius") or DEFAULT_RADIUS)
    except (TypeError, ValueError):
        return DEFAULT_RADIUS


def curves(rows, metric):
    return [(float(r["kbps"]), float(r[metric])) for r in rows if r[metric]]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "bench-results", "quality", "quality_summary.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows in %s" % path)
        return 1

    groups = {}
    for r in rows:
        groups.setdefault((r["clip"], r["config"], radius_of(r)), []).append(r)

    all_bd = {name: [] for _, name in METRICS}
    failures = []
    info = []

    for (clip, config, radius) in sorted(groups):
        if radius != DEFAULT_RADIUS and not INCLUDE_VARIANTS:
            continue
        # the reference is mode 0 of the same (clip, config) family, whatever
        # radius/prune tag it carries (mode-0 rows are identical encodes)
        base = []
        for key, grp in groups.items():
            if key[0] == clip and key[1] == config:
                base += [r for r in grp if int(r["mode"]) == 0]
        if len(base) < 2:
            print("== %s / %s / r%d: no usable mode-0 baseline, skipped"
                  % (clip, config, radius))
            continue
        modes = sorted(set(int(r["mode"]) for r in groups[(clip, config, radius)]))
        for mode in modes:
            if mode == 0:
                continue
            cur = [r for r in groups[(clip, config, radius)] if int(r["mode"]) == mode]
            parts = []
            for key, name in METRICS:
                cpts, bpts = curves(cur, key), curves(base, key)
                if len(cpts) < 2 or len(bpts) < 2:
                    continue
                d, _cover, _span = bd_rate(cpts, bpts)
                all_bd[name].append(d)
                if d > PER_CLIP_MAX:
                    failures.append((clip, config, mode, radius, name, d))
                parts.append("%s %+6.2f%% [%s]" % (
                    name, 100 * d, "OK " if d <= PER_CLIP_MAX else "FAIL"))
            if [r for r in cur if int(r["exit"]) != 0]:
                failures.append((clip, config, mode, radius, "exit", None))
            print("== %-11s %-8s m%-2d r%-2d %s" % (clip, config, mode, radius, "  ".join(parts)))

            # isolated radius effect (informational, PSNR vs the same mode at r2)
            if radius != DEFAULT_RADIUS:
                ref = [r for r in groups.get((clip, config, DEFAULT_RADIUS), [])
                       if int(r["mode"]) == mode]
                cpts, rpts = curves(cur, "psnr_y"), curves(ref, "psnr_y")
                if len(cpts) >= 2 and len(rpts) >= 2:
                    d, _c, _s = bd_rate(cpts, rpts)
                    info.append("radius effect %s/%s m%d: PSNR %+0.2f%% (r%d vs r%d)"
                                % (clip, config, mode, 100 * d, radius, DEFAULT_RADIUS))

    for line in info:
        print(line)

    print()
    for _, name in METRICS:
        vals = all_bd[name]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        worst = max(vals)
        ok = mean <= MEAN_MAX and worst <= PER_CLIP_MAX
        print("aggregate %s: mean %+0.2f%%  worst %+0.2f%%  (%d comparisons)  [%s]"
              % (name, 100 * mean, 100 * worst, len(vals), "PASS" if ok else "FAIL"))
    strict = [v for vals in all_bd.values() for v in vals if v > STRICT_MAX]
    print("strict check (> +%.2f%%): %d comparison(s)" % (100 * STRICT_MAX, len(strict)))

    print()
    if failures:
        print("GATE: FAIL (%d violation(s))" % len(failures))
        for clip, config, mode, radius, name, d in failures:
            extra = "exit!=0" if d is None else "%+0.2f%%" % (100 * d)
            print("   %s / %s / m%d / r%d / %s: %s" % (clip, config, mode, radius, name, extra))
        return 1
    print("GATE: PASS (per-comparison <= +%.2f%%, mean <= +%.2f%%)"
          % (100 * PER_CLIP_MAX, 100 * MEAN_MAX))
    return 0


if __name__ == "__main__":
    sys.exit(main())
