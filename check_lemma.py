"""Independent checker for the global upper bound certificate (mpmath.iv).

  * arithmetic is mpmath's inf-sup interval type (`mpmath.iv`)
  * stdlib + mpmath only.

What is verified
----------------
  1. f_{2,9}^tree < u on all S leaves of imported covering certificate
  2. u < f_{2,9}^K(w) for the witness w provided by the certifier 

Usage:
  python3 check_lemma.py [certificate.jsonl.gz] [witness_result.json]
"""

import gzip
import json
import math
import sys
import time

from mpmath import iv, mp

from check_theorem import (
    D, set_dps, expected_domain, enclosing_ball, check_prune,
    IVC, kdd,
)

DPS = 25
WIT_DPS = 40

# --------------------------------------------------------------------------
# verify upper bound
# --------------------------------------------------------------------------


def verify_u(path, u, b):
    """Verify u bounds sup_D f_{2,9}^tree by checking it on every S leaf."""
    set_dps(DPS)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
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
            continue
        if head != "S":
            raise SystemExit(f"line {nline}: bad record {tok!r}")
        counts["S"] += 1
        mr = [enclosing_ball(x, y) for x, y in box]
        _, _, val = check_prune(mr, b, I)
        if val > u:
            raise SystemExit(f"line {nline}: S leaf bound "
                             f"{val!r} > u={u!r}, box={box}")
        worst = max(worst, val)
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

def verify_witness(hexes):
    set_dps(WIT_DPS)
    mp.dps = max(mp.dps, WIT_DPS + 10)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    x = [IVC.const(iv.mpf([float.fromhex(h), float.fromhex(h)]))
         for h in hexes]
    v = kdd(D, x, I)
    return math.nextafter(float(mp.mpf(v.a)), -math.inf)


def main():
    cert_path = sys.argv[1] if len(sys.argv) > 1 else "certificate.jsonl.gz"
    wit_path = sys.argv[2] if len(sys.argv) > 2 else "witness_result.json"
    wit = json.load(open(wit_path))
    u, b = wit["u"], float.fromhex(wit["b_hex"])

    worst_S, counts = verify_u(cert_path, u, b)
    print(f"\nu = {u!r} verified: bound <= u on all {counts['S']} S leaves "
          f"(tightest observed: {worst_S!r})")

    l = verify_witness(wit["witness_hex"])
    print(f"\nl (rigorous lower bound on f_{{2,9}}^K(w)) = {l!r}")
    if l > wit["l"] + 1e-9 or l < wit["l"] - 1e-2:
        raise SystemExit(f"l wildly disagrees with certifier: "
                         f"mpmath.iv={l!r} vs arb={wit['l']!r}")

    ok = l > u
    margin = l - u
    print(f"\nl - u = {margin!r}  ({'PASS' if ok else 'FAIL'})")
    if not ok:
        raise SystemExit("witness does NOT certifiably beat upper bound")

    print(f"\nPASS (mpmath.iv only, no python-flint): f_{{2,9}}^K(w) >= "
          f"{l!r} > u = {u!r} >= f_{{2,9}}^{{tree,*}} for "
          f"w = {wit['witness_hex']}")


if __name__ == "__main__":
    main()

