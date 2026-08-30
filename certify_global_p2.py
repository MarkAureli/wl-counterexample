"""Producer: certified covering of {theta : f_2(d;theta) >= L} by delta<0 boxes.

Proves (counterexample/README.md):  every theta in the reduced domain
with f_2(d;theta) >= L satisfies  delta(theta) = <C_e>(K_dd;theta) - f_2(d;theta) < 0,
by a two-verdict interval branch and bound:

  P n  natural ball evaluation:                 sup f_2(B) < L
  P c  centred form (midpoint + AD gradient):   sup f_2(B) < L
  S    centred form on delta:                   sup delta(B) < 0

Boxes achieving no verdict are bisected at the exact float midpoint of one
coordinate, so the leaves tile the domain EXACTLY (children share the computed
midpoint; no midpoint/radius rounding gaps).  There is deliberately NO
monotonicity face-collapse: collapsing is sound for pruning but unsound as a
covering step (interior points with f_2 >= L would escape the S test).

Certificate: gzipped text, one JSON header line, then a preorder serialization
of the box tree, one node per line:
  N <k>              internal node, split coordinate k; low child subtree
                     follows, then high child subtree
  P n | P c          prune leaf (verdict re-derivable by a checker)
  S <hex>            safe leaf; <hex> = certified upper bound on delta there
All angle constants in the header are exact binary64 hex (E3/E7 traps).

Rigour: pure arb ops at fixed precision; sup/inf extracted via f2_ub.fup/flo
(outward-rounded); enclosing balls for [lo,hi] use radius nextafter'd outward.
Run from counterexample/.  Usage:
  python3 certify_global_p2.py run9        # the real run (d=9, real L)
"""

import gzip
import json
import math
import sys
import time
from typing import TypedDict

from flint import arb, acb  # noqa: E402

import f2_ub as M  # noqa: E402
import kdd_ball  # noqa: E402

PREC = 96
GAP_SKIP = 0.03      # natural bound this far above L: split, skip gradient work
W_SAFE = 1.5e-4      # only attempt the (15 ms) delta test below this width
MIN_W = 1e-8         # width floor: reaching it means the theorem is in doubt
LOG_EVERY = 250_000

class _Const(TypedDict):
    L: float
    U: float
    witness: list[str]


# certified lower bounds on f_2^tree(d) and witness points (arb point
# enclosures, radius ~8.7e-77); re-certified at run time by verify_L, so
# nothing here is trusted (README.md §2)
CONSTANTS: dict[int, _Const] = {
    9: {
        "L": float.fromhex("0x1.473bb99c3c5eap-1"),   # 0.6391275408952286
        "U": float.fromhex("0x1.473bb9f2229a6p-1"),   # certified upper bound
        "witness": ["-0x1.0b184cde2ef9ap-2", "0x1.56f3f40791132p+1",
                    "-0x1.04f3d4e04111ap-1", "-0x1.6f1f7f7dd666ep+1"],
    },
}


def verify_L(d, L, witness_hex):
    """Certify L <= f_2(d; witness) (hence L <= f_2^tree(d))."""
    M._set_prec(256)
    x = [acb(arb(float.fromhex(h))) for h in witness_hex]
    v = M.f2_acb(d - 1, x)
    lo = M.flo(v)
    assert lo >= L, f"L verification failed: {lo} < {L}"
    return lo


def _ball(lo, hi):
    m = 0.5 * (lo + hi)
    r = max(m - lo, hi - m)
    r = math.nextafter(r, math.inf) if r > 0.0 else 0.0
    return m, r


def cover(d, L, out_path, max_seconds=36000.0,
          max_boxes=200_000_000) -> dict[str, object]:
    """Run the covering search; returns stats dict. Writes certificate."""
    M._set_prec(PREC)
    E = d - 1
    dom = M.domain(d)
    t0 = time.time()
    f = gzip.open(out_path, "wt")
    header = dict(kind="wl-global-p2-cover", version=1, d=d, p=2, prec=PREC,
                  L_hex=float.hex(L), domain_hex=[[float.hex(a), float.hex(b)]
                                                  for a, b in dom],
                  witness_hex=(CONSTANTS[d]["witness"] if d in CONSTANTS
                               else None),
                  gap_skip=GAP_SKIP, w_safe=W_SAFE, min_w=MIN_W,
                  order="g1,g2,b1,b2",
                  tree="preorder; N k -> low subtree then high subtree")
    f.write(json.dumps(header) + "\n")

    stack = [tuple((a, b) for a, b in dom)]
    emitted = []          # replay buffer is the file itself; stream directly
    n = dict(boxes=0, pn=0, pc=0, safe=0, evals=0, duals=0, dduals=0)
    worst_safe = -math.inf
    safe_boxes = []

    while stack:
        if n["boxes"] > max_boxes or (n["boxes"] % 4096 == 0
                                      and time.time() - t0 > max_seconds):
            f.write("ABORT budget\n")
            f.close()
            return dict(ok=False, reason="budget", stats=n,
                        seconds=time.time() - t0)
        box = stack.pop()
        n["boxes"] += 1
        if n["boxes"] % LOG_EVERY == 0:
            print(f"  boxes={n['boxes']} stack={len(stack)} "
                  f"pn={n['pn']} pc={n['pc']} S={n['safe']} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        lo = [a for a, _ in box]
        hi = [b for _, b in box]
        mr = [_ball(a, b) for a, b in box]
        maxw = max(b - a for a, b in box)

        # 1. natural prune
        n["evals"] += 1
        val = M.f2_acb(E, [acb(arb(m, r)) for m, r in mr])
        gap = M.fup(val) - L
        if gap < 0.0:
            f.write("P n\n")
            n["pn"] += 1
            continue

        contrib = [b - a for a, b in box]
        if gap <= GAP_SKIP:
            # 2. centred prune
            n["duals"] += 1
            _, g = M.f2_dual(E, [M.Dual.var(arb(m, r), i)
                                 for i, (m, r) in enumerate(mr)])
            gmag = [math.nextafter(max(abs(float(gi.upper())),
                                       abs(float(gi.lower()))), math.inf)
                    for gi in g]
            n["evals"] += 1
            fc = M.f2_acb(E, [acb(arb(m)) for m, _ in mr])
            # mean-value form; the radius term g_i * [-r_i, r_i] is formed IN
            # arb, never as a float product (float rounds to nearest and can
            # shave the radius inward — referee defect E-G1). contrib stays
            # float: it only steers the split heuristic, which needs no rigour.
            bound = fc
            contrib = [gmag[i] * mr[i][1] for i in range(4)]
            for i in range(4):
                bound = bound + g[i] * arb(0.0, mr[i][1])
            if M.fup(bound) < L:
                f.write("P c\n")
                n["pc"] += 1
                continue

            # 3. safe test (only near-optimal boxes get this far)
            if maxw <= W_SAFE:
                n["dduals"] += 1
                g1, g2, b1, b2 = (arb(m) for m, _ in mr)
                ce_c = kdd_ball.ce_ball(d, g1, g2, b1, b2, PREC)
                _, gd = kdd_ball.ce_dual(
                    d, [M.Dual.var(arb(m, r), i)
                        for i, (m, r) in enumerate(mr)], PREC)
                # gradient of delta = ce - f2 as an arb enclosure difference;
                # radius term in ball arithmetic (E-G1), and tighter than the
                # |ce'|+|f2'| triangle bound.
                dbound = ce_c - fc
                for i in range(4):
                    dbound = dbound + (gd[i] - g[i]) * arb(0.0, mr[i][1])
                sup_d = M.fup(dbound)
                if sup_d < 0.0:
                    f.write(f"S {float.hex(sup_d)}\n")
                    n["safe"] += 1
                    worst_safe = max(worst_safe, sup_d)
                    safe_boxes.append([list(map(float.hex, lo)),
                                       list(map(float.hex, hi))])
                    continue
                if maxw < MIN_W:
                    f.write("ABORT floor\n")
                    f.close()
                    diag: dict[str, object] = {
                        "ok": False,
                        "reason": "width floor: cannot discharge",
                        "box_lo": list(map(float.hex, lo)),
                        "box_hi": list(map(float.hex, hi)),
                        "delta_center": M.fup(ce_c - fc),
                        "stats": n, "seconds": time.time() - t0}
                    return diag

        # 4. split
        k = max(range(4), key=lambda i: contrib[i])
        if hi[k] <= lo[k]:
            k = max(range(4), key=lambda i: hi[i] - lo[i])
        midk = 0.5 * (lo[k] + hi[k])
        if not (lo[k] < midk < hi[k]):
            f.write("ABORT split-exhausted\n")
            f.close()
            return dict(ok=False, reason="split exhausted", stats=n,
                        seconds=time.time() - t0)
        f.write(f"N {k}\n")
        child_hi = list(box)
        child_lo = list(box)
        child_lo[k] = (lo[k], midk)
        child_hi[k] = (midk, hi[k])
        stack.append(tuple(child_hi))
        stack.append(tuple(child_lo))

    f.close()
    del emitted
    return dict(ok=True, stats=n, seconds=time.time() - t0,
                c=(-worst_safe if n["safe"] else None),
                worst_safe_sup_delta=worst_safe if n["safe"] else None,
                safe_boxes=safe_boxes)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run9"
    if mode != "run9":
        raise SystemExit(f"unknown mode {mode}")
    d, L = 9, CONSTANTS[9]["L"]
    out = "cert_global_p2_d9.jsonl.gz"
    wlo = verify_L(d, L, CONSTANTS[d]["witness"])
    print(f"L verified: {L!r} <= certified point value {wlo!r}")
    print(f"mode={mode} d={d} L={L!r} prec={PREC}")
    res = cover(d, L, out, max_seconds=36000.0)
    res["mode"] = mode
    res["L_hex"] = float.hex(L)
    # The S-leaf boxes are deliberately NOT part of the run record: they are
    # reconstructed by replaying the bisection tree, so storing them would ship
    # a coordinate a checker could be tempted to trust.  Keep the dumped record
    # to counts and constants (README section 4).
    res.pop("safe_boxes", None)
    res["safe_boxes_note"] = (
        "not stored: S-leaf boxes are reconstructed by replaying the bisection "
        f"tree in {out}, which is what verify/check_global_p2_d9.py does.")
    json.dump(res, open(out.replace(".jsonl.gz", "_result.json"), "w"),
              indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
