"""Certified covering of [f_{2,9}^tree >= b] by delta<0 boxes.

Proves that very theta with f_{2,9}^tree(theta) >= b satisfies
delta(theta) = f_{2,9}^K(theta) - f_{2,9}^tree(theta) < 0
via interval branch and bound:

  P n  natural ball evaluation:                 sup f_{2,d}^tree(B) < b
  P c  centred form (midpoint + AD gradient):   sup f_{2,d}^tree(B) < b
  S    centred form on delta:                   sup delta(B) < 0

Boxes achieving no verdict are bisected at the exact float midpoint of one
coordinate, so the leaves tile the domain EXACTLY.

Certificate: gzipped text, one JSON header line, then preorder serialization
of the box tree, one node per line:
  N <k>              internal node, split coordinate k; low child subtree
                     follows, then high child subtree
  P n | P c          prune leaf (verdict re-derivable by a checker)
  S <hex>            safe leaf; <hex> = certified upper bound on delta there
All angle constants in the header are exact binary64 hex.

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
  python3 certify_theorem.py
"""

import gzip
import json
import math
import sys
import time
from math import comb
from typing import TypedDict

from flint import ctx, arb, acb
from fractions import Fraction as F

B = float.fromhex("0x1.473bb99c3c5eap-1")   # 0.6391275408952286
D = 9
PREC = 96
WITNESS_HEX = ["-0x1.04f3d4e04111ap-1", "-0x1.0b184cde2ef9ap-2",
               "-0x1.6f1f7f7dd666ep+1", "0x1.56f3f40791132p+1"]
GAP_SKIP = 0.03      # natural bound this far above b: split, skip gradient
W_SAFE = 1.5e-4      # only attempt the (15 ms) delta test below this width
MIN_W = 1e-8         # width floor
LOG_EVERY = 250_000

# --------------------------------------------------------------------------
# outward-rounded float extraction
# --------------------------------------------------------------------------


def fup(a):
    """A float >= sup(a) for an arb ball a."""
    return math.nextafter(float(a.upper()), math.inf)


def flo(a):
    """A float <= inf(a) for an arb ball a."""
    return math.nextafter(float(a.lower()), -math.inf)


# --------------------------------------------------------------------------
# forward-mode AD over acb balls: value + 4 partial derivatives
# --------------------------------------------------------------------------

_Z = None  # acb zero, set per precision


class Dual:
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v = v
        self.g = g

    # -- coercion ---------------------------------------------------------
    @staticmethod
    def const(v):
        return Dual(acb(v), (_Z, _Z, _Z, _Z))

    @staticmethod
    def var(v, i):
        g = [_Z, _Z, _Z, _Z]
        g[i] = acb(1)
        return Dual(acb(v), tuple(g))

    # -- real/imag parts, needed by formulas that split real/imag mid-way -
    @property
    def real(self):
        return Dual(self.v.real, tuple(a.real for a in self.g))

    @property
    def imag(self):
        return Dual(self.v.imag, tuple(a.imag for a in self.g))

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v,
                        tuple(a + b for a, b in zip(self.g, o.g)))
        return Dual(self.v + o, self.g)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, tuple(-a for a in self.g))

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v - o.v,
                        tuple(a - b for a, b in zip(self.g, o.g)))
        return Dual(self.v - o, self.g)

    def __rsub__(self, o):
        return Dual(o - self.v, tuple(-a for a in self.g))

    def __mul__(self, o):
        if isinstance(o, Dual):
            u, w = self.v, o.v
            return Dual(u * w, tuple(a * w + u * b
                                     for a, b in zip(self.g, o.g)))
        return Dual(self.v * o, tuple(a * o for a in self.g))

    __rmul__ = __mul__

    def __truediv__(self, o):  # only by scalars here
        return Dual(self.v / o, tuple(a / o for a in self.g))

    def __pow__(self, n):
        n = int(n)
        if n == 0:
            return Dual.const(1)
        if n == 1:
            return self
        pm1 = self.v ** (n - 1)
        v = pm1 * self.v
        c = n * pm1
        return Dual(v, tuple(c * a for a in self.g))

    def cos(self):
        s, c = acb_sin_cos(self.v)
        return Dual(c, tuple(-s * a for a in self.g))

    def sin(self):
        s, c = acb_sin_cos(self.v)
        return Dual(s, tuple(c * a for a in self.g))


def acb_sin_cos(x):
    return x.sin(), x.cos()


def _set_prec(prec):
    global _Z
    ctx.prec = prec
    _Z = acb(0)


# --------------------------------------------------------------------------
# reduced search domain
# --------------------------------------------------------------------------


def domain():
    """Reduced fundamental domain."""
    pi = math.nextafter(math.pi, math.inf)
    qpi = math.nextafter(math.pi / 4, math.inf)
    return [(-qpi, qpi), (0.0, pi), (-qpi, qpi), (-pi, pi)]


# --------------------------------------------------------------------------
# Marwaha's closed form, generic over acb and Dual
# --------------------------------------------------------------------------
#
#   f_{2,d}^tree = 1/2 + c^2 r t y^E z - (c s/2) alpha - (s^2 t z/4) kappa
#   c = cos(2 beta2), m = cos(gamma2), r = cos(2 beta1), y = cos(gamma1)
#   s = sin(2 beta2), n = sin(gamma2), t = sin(2 beta1), z = sin(gamma1),
#   E = d-1
#   A = (m y - n r z)^E,  B = (m y + n r z)^E
#   P = (m + i n t y^E z)^E,  Q = (m - i n t y^E z)^E
#   kappa = ((1+r) A - (1-r) B) (P + Q)
#   alpha = (1+r)(-m r z - n y) A + (1-r)(m r z - n y) B
#           + t ((m t y^E z + i n) P + (m t y^E z - i n) Q)


def tree_form(b1, g1, b2, g2, I):
    """Generic; b1,g1,b2,g2 are acb balls or Duals."""
    tb1 = 2 * b1
    tb2 = 2 * b2
    c = tb2.cos()
    s = tb2.sin()
    r = tb1.cos()
    t = tb1.sin()
    m = g2.cos()
    n = g2.sin()
    y = g1.cos()
    z = g1.sin()
    E = D - 1

    yE = y ** E
    nrz = n * r * z
    my = m * y
    A = (my - nrz) ** E
    B = (my + nrz) ** E

    ntyEz = n * t * yE * z
    P = (m + I * ntyEz) ** E
    Q = (m - I * ntyEz) ** E

    kappa = ((1 + r) * A - (1 - r) * B) * (P + Q)

    mrz = m * r * z
    ny = n * y
    mtyEz = m * t * yE * z
    alpha = ((1 + r) * (-mrz - ny) * A
             + (1 - r) * (mrz - ny) * B
             + t * ((mtyEz + I * n) * P + (mtyEz - I * n) * Q))

    return (0.5 + c * c * r * t * yE * z
            - (c * s) * alpha / 2
            - (s * s * t * z) * kappa / 4)


def tree_acb(x):
    """Natural acb ball evaluation. Returns arb (real part)."""
    return tree_form(x[0], x[1], x[2], x[3], acb(0, 1)).real


def tree_dual(x):
    """Value + gradient over a box.  Returns (arb value, 4 arb partials)."""
    res = tree_form(x[0], x[1], x[2], x[3], acb(0, 1))
    return res.v.real, tuple(a.real for a in res.g)


# --------------------------------------------------------------------------
# K_{d,d} per-edge engine, generic over acb and Dual
# --------------------------------------------------------------------------
#
# State lives in the (d+1)x(d+1) two-sided symmetric (Dicke) subspace;
# psi[p][q] stays symmetric throughout (Cm and the mixer Mx are applied
# identically to both indices), so only the upper triangle p<=q is stored;
# the `mid` contraction still needs the full DxD psi, but `new` and the
# final readout sum only touch the upper triangle.
#
# Natural ball evaluation amplifies input radii too much on wide boxes, so
# kdd_dual supplies a gradient enclosure for the centred form
#       sup delta(B) <= delta(mid) + sum_i sup|d delta/d x_i (B)| * rad_i.


def kdd_form(b1, g1, b2, g2, I):
    """Generic; b1,g1,b2,g2 are acb/arb balls or Duals."""
    binom = [arb(comb(D, p)) for p in range(D + 1)]
    sq = [x.sqrt() for x in binom]
    Cm = [[D * (p + q) - 2 * p * q for q in range(D + 1)] for p in range(D + 1)]
    two_pow = arb(2) ** arb(D)

    psi = [[None] * (D + 1) for _ in range(D + 1)]
    for p in range(D + 1):
        for q in range(p, D + 1):
            psi[p][q] = acb(sq[p] * sq[q] / two_pow)

    def get(p, q):
        return psi[p][q] if p <= q else psi[q][p]

    for beta, gamma in ((b1, g1), (b2, g2)):
        for p in range(D + 1):
            for q in range(p, D + 1):
                u = gamma * Cm[p][q]
                psi[p][q] = psi[p][q] * (u.cos() - I * u.sin())
        c, s = beta.cos(), beta.sin()
        ms = (-I) * s
        Mx = [[None] * (D + 1) for _ in range(D + 1)]
        for pp in range(D + 1):
            for p in range(D + 1):
                acc = acb(0)
                for k in range(max(0, p - pp), min(D - pp, p) + 1):
                    coef = arb(comb(D - pp, k) * comb(pp, p - k))
                    acc = acc + c ** (D + p - pp - 2 * k) \
                        * ms ** (2 * k + pp - p) * coef
                Mx[pp][p] = acc * (sq[pp] / sq[p])

        mid = [[None] * (D + 1) for _ in range(D + 1)]
        for i in range(D + 1):
            for j in range(D + 1):
                acc = acb(0)
                for t in range(D + 1):
                    acc = acc + Mx[i][t] * get(t, j)
                mid[i][j] = acc

        new = [[None] * (D + 1) for _ in range(D + 1)]
        for i in range(D + 1):
            for j in range(i, D + 1):
                acc = acb(0)
                for t in range(D + 1):
                    acc = acc + mid[i][t] * Mx[j][t]
                new[i][j] = acc
        psi = new

    tot = acb(0)
    for p in range(D + 1):
        z = psi[p][p]
        tot = tot + (z.real ** 2 + z.imag ** 2) * Cm[p][p]
        for q in range(p + 1, D + 1):
            z = psi[p][q]
            tot = tot + 2 * (z.real ** 2 + z.imag ** 2) * Cm[p][q]
    return tot / (D ** 2)


def kdd_ball(b1, g1, b2, g2):
    """Enclosure of f_{2,9}^K at p=2."""
    return kdd_form(b1, g1, b2, g2, acb(0, 1)).real


def kdd_dual(x):
    """(value, gradient) of f_{2,9}^K over a box."""
    res = kdd_form(x[0], x[1], x[2], x[3], acb(0, 1))
    return res.v.real, tuple(a.real for a in res.g)

def verify_b():
    """Certify b <= f_{2,9}^tree(witness)."""
    _set_prec(256)
    x = [acb(arb(float.fromhex(h))) for h in WITNESS_HEX]
    v = tree_acb(x)
    lo = flo(v)
    assert lo >= B, f"b verification failed: {lo} < {B}"
    return lo


def _ball(lo, hi):
    """(m, r) floats with [m-r, m+r] >= [lo, hi]."""
    m = 0.5 * (lo + hi)
    r = max(m - lo, hi - m)
    if r > 0.0:
        r = math.nextafter(r, math.inf)
    assert F(m) - F(r) <= F(lo) and F(m) + F(r) >= F(hi), "ball containment"
    return m, r


def cover(out_path, max_seconds=36000.0,
          max_boxes=200_000_000) -> dict[str, object]:
    """Run the covering search; returns stats dict. Writes certificate."""
    _set_prec(PREC)
    dom = domain()
    t0 = time.time()
    f = gzip.open(out_path, "wt")
    header = dict(kind="cover", version=1, d=D, p=2, prec=PREC,
                  b_hex=float.hex(B), dom_hex=[[float.hex(x), float.hex(y)]
                                                for x, y in dom],
                  witness_hex=WITNESS_HEX,
                  gap_skip=GAP_SKIP, w_safe=W_SAFE, min_w=MIN_W,
                  order="b1,g1,b2,g2",
                  tree="preorder; N k -> low subtree then high subtree")
    f.write(json.dumps(header) + "\n")

    stack = [tuple((x, y) for x, y in dom)]
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
            print(f" boxes={n['boxes']} stack={len(stack)} "
                  f"pn={n['pn']} pc={n['pc']} S={n['safe']} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        lo = [x for x, _ in box]
        hi = [y for _, y in box]
        mr = [_ball(x, y) for x, y in box]
        maxw = max(y - x for x, y in box)

        # 1. natural prune
        n["evals"] += 1
        val = tree_acb([acb(arb(m, r)) for m, r in mr])
        gap = fup(val) - B
        if gap < 0.0:
            f.write("P n\n")
            n["pn"] += 1
            continue

        contrib = [y - x for x, y in box]
        if gap <= GAP_SKIP:
            # 2. centred prune
            n["duals"] += 1
            _, g = tree_dual([Dual.var(arb(m, r), i)
                             for i, (m, r) in enumerate(mr)])
            gmag = [math.nextafter(max(abs(float(gi.upper())),
                                       abs(float(gi.lower()))), math.inf)
                    for gi in g]
            n["evals"] += 1
            tree_c = tree_acb([acb(arb(m)) for m, _ in mr])
            bound = tree_c
            contrib = [gmag[i] * mr[i][1] for i in range(4)]
            for i in range(4):
                bound = bound + g[i] * arb(0.0, mr[i][1])
            if fup(bound) < B:
                f.write("P c\n")
                n["pc"] += 1
                continue

            # 3. safe test (only near-optimal boxes get this far)
            if maxw <= W_SAFE:
                n["dduals"] += 1
                b1, g1, b2, g2 = (arb(m) for m, _ in mr)
                kdd_c = kdd_ball(b1, g1, b2, g2)
                _, gd = kdd_dual([Dual.var(arb(m, r), i)
                                 for i, (m, r) in enumerate(mr)])
                dbound = kdd_c - tree_c
                for i in range(4):
                    dbound = dbound + (gd[i] - g[i]) * arb(0.0, mr[i][1])
                sup_d = fup(dbound)
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
                        "delta_center": fup(kdd_c - tree_c),
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
    return dict(ok=True, stats=n, seconds=time.time() - t0,
                c=(-worst_safe if n["safe"] else None),
                worst_safe_sup_delta=worst_safe if n["safe"] else None,
                safe_boxes=safe_boxes)


def main():
    out = "certificate.jsonl.gz"
    wlo = verify_b()
    print(f"b verified: {B!r} <= certified point value {wlo!r}")
    print(f"d={D} b={B!r} prec={PREC}")
    res = cover(out, max_seconds=36000.0)
    res["b_hex"] = float.hex(B)
    res.pop("safe_boxes", None)
    json.dump(res, open(out.replace(".jsonl.gz", "_result.json"), "w"),
              indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

