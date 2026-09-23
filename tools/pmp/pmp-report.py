#!/usr/bin/env python3
"""Report on a pmp.c profile: flat (self) and inclusive time by function.

Usage: pmp-report.py <pmp.out.PID> [top N] [t0:t1,t2:t3,...]

The optional windows are wall-clock ranges (seconds since the epoch, as time.time() prints) that
restrict the report to those samples, e.g. only the generation phase of a server run.
Inclusive counts only see the DEPTH frames pmp.c records, so deep call chains undercount
their outer callers.
"""
import bisect, collections, subprocess, sys

lines = open(sys.argv[1]).read().split('\n')
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
windows = [tuple(map(float, w.split(':'))) for w in sys.argv[3].split(',')] if len(sys.argv) > 3 else None

n_all = int(lines[0].split()[1])
samples = []
for l in lines[1:1 + n_all]:
    f = l.split()
    if windows is None or any(a <= float(f[0]) <= b for a, b in windows):
        samples.append([int(x, 16) for x in f[1:]])

maps = []
for l in lines[lines.index('MAPS') + 1:]:
    p = l.split()
    if len(p) >= 6 and 'x' in p[1]:
        a, b = (int(x, 16) for x in p[0].split('-'))
        maps.append((a, b, int(p[2], 16), p[5]))
maps.sort()
starts = [m[0] for m in maps]


def locate(a):
    i = bisect.bisect_right(starts, a) - 1
    if i < 0 or a >= maps[i][1]:
        return None
    return maps[i][3], a - maps[i][0] + maps[i][2]


where = {}
by_file = collections.defaultdict(set)
for a in {a for s in samples for a in s if a}:
    r = locate(a)
    if r:
        where[a] = r
        by_file[r[0]].add(r[1])

names = {}
for path, offs in by_file.items():
    offs = sorted(offs)
    out = subprocess.run(['addr2line', '-f', '-C', '-e', path] + [hex(o) for o in offs],
                         capture_output=True, text=True).stdout.split('\n')
    for i, o in enumerate(offs):
        if 2 * i < len(out):
            names[(path, o)] = out[2 * i]


def name(a):
    r = where.get(a)
    if not r:
        return '?'
    if '/libcuda.so' in r[0]:  # stripped: addr2line would name the nearest exported symbol
        return 'libcuda.so (driver internals)'
    s = names.get(r, '??')
    if s == '??':  # stripped library: file and offset
        s = r[0].split('/')[-1] + '+' + hex(r[1])
    return s[:100]


flat, incl = collections.Counter(), collections.Counter()
for s in samples:
    flat[name(s[0])] += 1
    for nm in {name(a) for a in s if a}:
        incl[nm] += 1

n = max(1, len(samples))
print(f"{len(samples)} samples")
print("== flat (self) ==")
for k, v in flat.most_common(top):
    print(f"{v / n * 100:6.2f}% {k}")
print("== inclusive ==")
for k, v in incl.most_common(top):
    print(f"{v / n * 100:6.2f}% {k}")
