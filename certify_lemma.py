"""Certified covering of [f_{2,9}^tree >= b] by <u boxes.

Proves that f_{2,9}^tree <= u everywhere and that f_{2,9}^K(w) > u
for some witness w.

  S    centred form on f_{2,9}^tree:            sup f_{2,9}^tree <= u

Rigour model
------------
* every arithmetic step is an arb/acb ball operation: FLINT rounds the
  midpoint and inflates the radius so the true value is always enclosed
* transcendental constants (pi) come from `arb.pi()` at working precision
* box endpoints are exact binary64
* radii are read with `.rad()`
* every float extracted from a ball is bumped outward with `math.nextafter`
* sup/inf extracted via fup/flo (outward-rounded)

Usage:
  python3 certify_lemma.py [certificate.jsonl.gz]
"""

import gzip
import json
import math
import sys
import time

from flint import arb

from certify_theorem import (
    B, D, PREC, fup, _ball, _set_prec,
    tree_acb, tree_dual, kdd_ball, Dual,
)

U = 0.6391276594250365      # 0x1.473bbd96662c7p-1
assert U >= B
WITNESS_HEX = ["0x1.0189541bf092fp-1", "0x1.752b5bc9d2407p+1",
               "-0x1.e0c856d2ed88ap-2", "0x1.6f478b15e56c7p+1"]

# --------------------------------------------------------------------------
# verify upper bound
# --------------------------------------------------------------------------


def verify_u(path):
    """Verify u bounds sup_D f_{2,9{^tree by checking it on every S leaf."""
    _set_prec(PREC)
    fh = gzip.open(path, "rt")
    header = json.loads(fh.readline())
    dom = [(float.fromhex(x), float.fromhex(y)) for x, y in header["dom_hex"]]
    stack = [tuple(dom)]
    counts = {"N": 0, "P n": 0, "P c": 0, "S": 0}
    worst = -math.inf
    t0 = time.time()
    nline = 1
    for line in fh:
        nline += 1
        if nline % 500_000 == 0:
            print(f"  replay: {nline} records, worst_bound={worst!r}, "
                  f"t={time.time()-t0:.0f}s", flush=True)
        tok = line.split()
        if not stack:
            raise SystemExit(f"line {nline}: tree exhausted early")
        box = stack.pop()
        head = tok[0]
        if head == "N":
            k = int(tok[1])
            lo_k, hi_k = box[k]
            mid = 0.5 * (lo_k + hi_k)
            if not lo_k < mid < hi_k:
                raise SystemExit(f"line {nline}: "
                                 f"degenerate split at coord {k}")
            child_lo, child_hi = list(box), list(box)
            child_lo[k] = (lo_k, mid)
            child_hi[k] = (mid, hi_k)
            stack.append(tuple(child_hi))
            stack.append(tuple(child_lo))
            counts["N"] += 1
            continue
        if head == "P":
            counts["P " + tok[1]] += 1
            continue  # already independently proven sup f_tree < b <= U
        if head != "S":
            raise SystemExit(f"line {nline}: bad record {tok!r} "
                             "(an ABORT line means the run never finished)")
        counts["S"] += 1
        mr = [_ball(x, y) for x, y in box]
        tree_c = tree_acb([arb(m) for m, _ in mr])
        _, g = tree_dual([Dual.var(arb(m, r), i)
                         for i, (m, r) in enumerate(mr)])
        bound = tree_c
        for i in range(4):
            bound = bound + g[i] * arb(0.0, mr[i][1])
        ctr = fup(bound)
        if ctr > U:
            raise SystemExit(f"line {nline}: S leaf bound {ctr!r} > "
                             f"U={U!r}, box={box}")
        worst = max(worst, ctr)
    fh.close()
    if stack:
        raise SystemExit(f"tree incomplete: {len(stack)} boxes never "
                         "discharged")
    print(f"replay done: {nline-1} records, {counts}, "
          f"t={time.time()-t0:.0f}s")
    return worst, counts

# --------------------------------------------------------------------------
# verify witness
# --------------------------------------------------------------------------


def verify_witness(hexes, prec=256):
    """Rigorous lower bound on f_{2,9}^K at the witness point."""
    _set_prec(prec)
    x = [arb(float.fromhex(h)) for h in hexes]
    v = kdd_ball(*x)
    return math.nextafter(float(v.lower()), -math.inf)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "certificate.jsonl.gz"
    worst_S, cnts = verify_u(path)
    print(f"\nu = {U!r} verified: bound <= u on all {cnts['S']} S leaves "
          f"(tightest observed: {worst_S!r})")

    l = verify_witness(WITNESS_HEX)
    print(f"\nl (rigorous lower bound on f_{{2,9}}^K(w)) = {l!r}")
    print(f"witness w (hex): {WITNESS_HEX}")

    ok = l > U
    margin = l - U
    print(f"\nl - u = {margin!r}  ({'PASS' if ok else 'FAIL'}: "
          f"f_{{2,9}}^K(w) {'>' if ok
          else '<='} u >= f_{{2,9}}^{{tree,*}})")
    assert ok, "witness does not certifiably beat the tree upper bound"

    result = dict(ok=ok, u=U, l=l, margin=margin, worst_S_bound=worst_S,
                  witness_hex=WITNESS_HEX, b_hex=float.hex(B), counts=cnts)
    with open("witness_result.json", "w") as f:
        json.dump(result, f, indent=1)
    print("\nwrote witness_result.json")


if __name__ == "__main__":
    main()

