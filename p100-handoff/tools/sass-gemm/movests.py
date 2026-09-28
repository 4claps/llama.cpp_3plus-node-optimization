#!/usr/bin/env python3
"""Move the fold GEMM's next-tile shared-memory stores (16 STS + their 3 DEPBAR waits) out of the
store phase after the HFMA2 block and interleave them into the second half of that block.

The ablation (GOAL-PREFILL300.md) showed the store + barrier phase costs ~14% of the kernel: every
warp finishes its FMAs, then stores, then waits. The buffer being stored (buf ^ 1) is not read
during the current tile, so the stores may issue any time after their global loads have landed.
The store addresses are recomputed at the loop top into free registers (R200-R204); the original
address code stays where it was (its outputs are then dead). Exact: same stores, same values, same
addresses; only the issue order changes.

usage: movests.py fk.cuasm out.cuasm
Written against the sm_60 SASS of fold_kernel.cuh; it asserts on every anchor it relies on.
"""
import re, sys

src, dst = sys.argv[1], sys.argv[2]
L = open(src).read().split('\n')

def find(pat, start=0, end=None):
    end = len(L) if end is None else end
    for i in range(start, end):
        if re.search(pat, L[i]):
            return i
    raise SystemExit(f"anchor not found: {pat}")

top    = find(r'^\s*\.L_x_11:')
lds4   = find(r'LDS\.U\.128 R68, \[R190\+0x4100\] ;', top)
blk    = find(r'^\s*\.L_x_8:', top)
foldbr = find(r'@P5 BRA CC\.NEU', blk)
st0    = find(r'DEPBAR\.LE SB5, 0x3 ;', foldbr)
st1    = find(r'STS \[R42\+0x4600\], R59 ;', st0)
store_lines = L[st0:st1 + 1]
assert sum('STS' in l for l in store_lines) == 16 and sum('DEPBAR' in l for l in store_lines) == 3

def ctrl(line):
    return re.match(r'^(\s*\[[^\]]*\])', line).group(1)

def mk(c, text):
    return f"      {c}         /*0000*/                   {text}"

# 1. addresses at the loop top: A (was R40) -> R203, B (was R42) -> R202
addr = [
    "SHL R201, R102, 0xb ;",
    "SHL R202, R111, 0x3 ;",
    "LOP32I.AND R200, ~R103, 0x2000 ;",
    "SHR.U32 R203, R102, 0x2 ;",
    "LOP32I.AND R201, R201, 0x1800 ;",
    "LOP32I.AND R204, R202, 0x18 ;",
    "SHR.U32 R202, R111, 0x2 ;",
    "IADD R201, R200, R201 ;",
    "LOP.XOR R203, R204, R203 ;",
    "LOP.XOR R202, R204, R202 ;",
    "ISCADD R203, R203, R201, 0x2 ;",
    "ISCADD R202, R202, R201, 0x2 ;",
]
addr_lines = [mk("[B------:R-:W-:-:S06]", a) for a in addr]

# 2. the store groups, rewritten to the new address registers and predicated on !P4 (next tile exists)
groups, cur = [], []
for l in store_lines:
    if 'DEPBAR' in l and cur:
        groups.append(cur); cur = []
    if 'STS' in l:
        c = ctrl(l)
        body = l.split('*/', 1)[1].strip()
        body = body.replace('[R40', '[R203').replace('[R42', '[R202')
        l = mk(c, '@!P4 ' + body.lstrip())
    cur.append(l)
groups.append(cur)
assert len(groups) == 3

# 3. insertion points: single-issue HFMA2 lines (not inside a { } dual-issue pair) in the second half
fma = [i for i in range(blk, foldbr) if re.search(r'\bHFMA2\b', L[i]) and '{' not in L[i] and '}' not in L[i]
       and '{' not in L[i - 1]]
picks = [fma[int(len(fma)*f)] for f in (0.55, 0.70, 0.85)]

out = []
for i, l in enumerate(L):
    if st0 <= i <= st1:
        continue                      # moved
    out.append(l)
    if i == lds4:
        out.extend(addr_lines)
    if i in picks:
        out.extend(groups[picks.index(i)])
txt = '\n'.join(out)
txt = re.sub(r'SHI_REGISTERS=(\d+)', lambda m: f"SHI_REGISTERS={max(int(m.group(1)), 205)}", txt)
txt = re.sub(r'(EIATTR_REGCOUNT[\s\S]*?index@\(\w+\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\s+)0x([0-9a-f]+)',
             lambda m: m.group(1) + f"0x{max(int(m.group(2), 16), 205):08x}", txt, count=1)
open(dst, 'w').write(txt)
print(f"moved {len(store_lines)} lines into the HFMA2 block at lines {picks}; addresses at the loop top")
