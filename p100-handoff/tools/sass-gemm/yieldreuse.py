#!/usr/bin/env python3
"""post-pass over a bankfixed .cuasm: in the HFMA2 range, drop yield flags and set .reuse wherever the
next FMA reads the same register in the same operand slot. usage: tune.py in out [noyield=1]"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
noy = int(sys.argv[3]) if len(sys.argv) > 3 else 1
L = open(src).read().splitlines()
INS = re.compile(r'^(\s*\[)([^\]]*)(\]\s*/\*([0-9a-f]+)\*/\s*)(\{?\s*)(.*)$')
idx = [i for i, l in enumerate(L) if INS.match(l)]
fma = [i for i in idx if re.match(r'(HFMA2|HMUL2)\b', INS.match(L[i]).group(6))]
lo, hi = fma[0], fma[-1]
def ops(t):
    op, rest = t.split(None, 1)
    rest = rest.rstrip()
    tail = ''
    m = re.search(r'\s*;\s*\}?\s*$', rest); tail = rest[m.start():]; rest = rest[:m.start()]
    return op, [o.strip() for o in rest.split(',')], tail
def reg(o):
    m = re.match(r'^(-?)R(\d+)(\.reuse)?((?:\.\w+)*)$', o)
    return int(m.group(2)) if m else None
seq = [i for i in idx if lo <= i <= hi]
ny = nr = 0
for n, i in enumerate(seq):
    m = INS.match(L[i]); ctl = m.group(2); t = m.group(6)
    if noy and ctl[14] == 'Y' and re.match(r'(HFMA2|HMUL2)\b', t):
        ctl = ctl[:14] + '-' + ctl[15:]; ny += 1
    if re.match(r'(HFMA2|HMUL2)\b', t) and n + 1 < len(seq):
        j = seq[n + 1]
        if j == i + 1 or all(not L[k].strip().startswith('.') for k in range(i + 1, j)):
            t2 = INS.match(L[j]).group(6)
            if re.match(r'(HFMA2|HMUL2)\b', t2):
                op, a, tail = ops(t); _, b, _ = ops(t2)
                for k in range(1, min(len(a), len(b))):
                    ra, rb = reg(a[k]), reg(b[k])
                    if ra is not None and ra == rb and '.reuse' not in a[k]:
                        a[k] = re.sub(r'R(\d+)', r'R\1.reuse', a[k], count=1); nr += 1
                t = op + ' ' + ', '.join(a) + tail
    L[i] = m.group(1) + ctl + m.group(3) + m.group(5) + t
open(dst, 'w').write('\n'.join(L) + '\n')
print(f'yield dropped {ny}, reuse added {nr}', file=sys.stderr)
