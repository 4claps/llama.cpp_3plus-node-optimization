#!/usr/bin/env python3
"""Register-bank conflict removal for Pascal (sm_60) SASS, by register renaming only.

Maxwell/Pascal read registers from 4 banks (register index mod 4). When an FMA-type instruction
reads two source operands from the same bank (and neither comes from the operand reuse cache), it
stalls a cycle. ptxas does not optimise for this in the fold GEMM's HFMA2 loop (~42% of the FMAs
conflict). A global permutation of register names changes no arithmetic: the output is
bit-identical by construction. This pass only moves *scalar* registers (never part of a 64/128-bit
operand anywhere in the kernel) and prefers free registers, so vector alignment is untouched.

usage: bankfix.py in.cuasm out.cuasm <loop start hex> <loop end hex> [max registers, default 255]
(the cap keeps a kernel's occupancy: e.g. 128 for two 256-thread CTAs per SM)
"""
import re, sys
from collections import defaultdict

INS_RE = re.compile(r'^(\s*\[[^\]]*\]\s*/\*([0-9a-f]+)\*/\s*)(.*)$')
REG_RE = re.compile(r'(?<![\w.])R(\d+)((?:\.reuse)?)')
FMA_OPS = ('HFMA2', 'HMUL2', 'HADD2', 'FFMA', 'FADD', 'FMUL')


def parse(lines):
    ins = []
    for li, l in enumerate(lines):
        m = INS_RE.match(l)
        if m and not m.group(3).startswith('.'):
            ins.append((li, int(m.group(2), 16), m.group(3)))
    return ins


def opcode(text):
    t = re.sub(r'^@!?P\w+\s+', '', text.strip())
    return t.split()[0] if t else ''


def vector_regs(ins):
    """registers that are part of a multi-register operand somewhere: never moved"""
    vec = set()
    for _, _, t in ins:
        op = opcode(t)
        width = 4 if '.128' in op else 2 if ('.64' in op or op.startswith('D')) else 1
        regs = [int(r) for r, _ in REG_RE.findall(t)]
        if width > 1 and regs:
            # destination / data register group of the wide access
            for r in regs[:1] if not op.startswith('ST') else regs[-1:]:
                vec.update(range(r, r + width))
        # 64-bit addresses: [Rn] on .E global/generic accesses
        if re.match(r'(LDG|STG|LD|ST|ATOM|RED)\b', op) and '.E' in op:
            for a in re.findall(r'\[R(\d+)', t):
                vec.update((int(a), int(a) + 1))
        if op.startswith('ST') and width > 1:
            vec.update(range(regs[-1], regs[-1] + width))
    return vec


def sources(t):
    """(dest, [(src reg, reuse flag, slot)]) for a simple 'OP Rd, Ra, Rb[, Rc]' instruction"""
    body = re.sub(r'^@!?P\w+\s+', '', t.strip()).rstrip(' ;}')
    parts = body.split(None, 1)
    if len(parts) < 2:
        return None, []
    ops = [o.strip() for o in parts[1].split(',')]
    dest = None
    m = re.match(r'-?\|?R(\d+)', ops[0])
    if m:
        dest = int(m.group(1))
    src = []
    for slot, o in enumerate(ops[1:]):
        m = re.match(r'-?\|?R(\d+)(\.reuse)?', o)
        if m:
            src.append((int(m.group(1)), bool(m.group(2)), slot))
    return dest, src


def conflicts(hot, bank):
    n = 0
    prev = {}
    for t in hot:
        d, src = sources(t)
        banks = defaultdict(int)
        seen = set()
        for r, reuse, slot in src:
            cached = prev.get(slot) == r
            if not cached and r != 255 and r not in seen:
                banks[bank(r)] += 1
                seen.add(r)
        n += sum(v - 1 for v in banks.values() if v > 1)
        prev = {slot: r for r, reuse, slot in src if reuse}
    return n


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    lines = open(src_path).read().split('\n')
    ins = parse(lines)
    # hot loop address range (from the disassembly): argv[3], argv[4] as hex
    best = (int(sys.argv[3], 16), int(sys.argv[4], 16))
    lo, hi = best
    hot_ins = [(li, a, t) for li, a, t in ins if lo <= a <= hi and opcode(t).split('.')[0] in FMA_OPS]
    hot = [t for _, _, t in hot_ins]
    used = set()
    for _, _, t in ins:
        used.update(int(r) for r, _ in REG_RE.findall(t))
    used.discard(255)
    vec = vector_regs(ins)
    maxreg = max(used)
    print(f"hot loop 0x{lo:x}-0x{hi:x}: {len(hot)} FMA-type instructions; regs used {len(used)} (max R{maxreg}); "
          f"vector-bound {len(vec & used)}")
    ident = lambda r: r % 4
    print(f"bank conflicts before: {conflicts(hot, ident)}")

    # candidates to move: scalar registers that are destinations of hot HFMA2/HMUL2 (the accumulators)
    acc = set()
    for t in hot:
        d, s = sources(t)
        if opcode(t).startswith(('HFMA2', 'HMUL2')) and d is not None:
            acc.add(d)
            for r, _, slot in s:
                if slot == 2:
                    acc.add(r)
    acc -= vec
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 255
    free = [r for r in range(0, cap) if r not in used]
    pool = sorted(acc | set(free))
    print(f"accumulators to place: {len(acc)}; pool {len(pool)} (free {len(free)})")

    # greedy: place accumulators one at a time into the bank that adds the fewest conflicts
    mapping = {}
    avail = defaultdict(list)
    for r in pool:
        avail[r % 4].append(r)
    cur = lambda r: mapping.get(r, r)
    order = sorted(acc)
    # start from all accumulators unplaced -> evaluate with a provisional identity, then refine
    for it in range(3):
        for r in order:
            if r in mapping:
                avail[mapping[r] % 4].append(mapping.pop(r))
            best_b, best_c = None, None
            for b in range(4):
                if not avail[b]:
                    continue
                mapping[r] = avail[b][0]
                c = conflicts(hot, lambda x: cur(x) % 4)
                del mapping[r]
                if best_c is None or c < best_c:
                    best_b, best_c = b, c
            mapping[r] = avail[best_b].pop(0)
        print(f"pass {it}: conflicts {conflicts(hot, lambda x: cur(x) % 4)}")

    # apply as a permutation over the pool: accumulators -> chosen regs; displaced pool regs -> vacated regs
    perm = dict(mapping)
    targets = set(perm.values())
    vacated = [r for r in sorted(acc) if r not in targets]
    displaced = [r for r in sorted(targets) if r not in acc and r in used]
    assert len(displaced) <= len(vacated)
    for a, b in zip(displaced, vacated):
        perm[a] = b
    assert len(set(perm.values())) == len(perm), "not a permutation"
    def ren(t):
        return REG_RE.sub(lambda m: f"R{perm.get(int(m.group(1)), int(m.group(1)))}{m.group(2)}", t)
    out = list(lines)
    for li, a, t in ins:
        m = INS_RE.match(lines[li])
        out[li] = m.group(1) + ren(m.group(3))
    new_hot = [ren(t) for t in hot]
    print(f"bank conflicts after: {conflicts(new_hot, ident)}; max reg R{max(perm.get(r, r) for r in used)}")
    txt = '\n'.join(out)
    # register count: EIATTR_REGCOUNT word and the section's SHI_REGISTERS
    nreg = max(perm.get(r, r) for r in used) + 1
    txt = re.sub(r'(EIATTR_REGCOUNT[\s\S]*?index@\(\w+\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\s+)0x([0-9a-f]+)',
                 lambda m: m.group(1) + f"0x{max(int(m.group(2), 16), nreg):08x}", txt, count=1)
    txt = re.sub(r'SHI_REGISTERS=(\d+)', lambda m: f"SHI_REGISTERS={max(int(m.group(1)), nreg)}", txt)
    print(f"register count {nreg}")
    open(dst_path, 'w').write(txt)


if __name__ == '__main__':
    main()
