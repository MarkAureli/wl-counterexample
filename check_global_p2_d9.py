"""Standalone checker for the global-optimality certificate of Theorem 3.1.

Verifies, with NO imports from the producer's modules, a certificate produced by
`certify_global_p2.py`:

    for every theta in the reduced fundamental domain with
    f_2(d; theta) >= L,   <C_e>(K_dd; theta) - f_2(d; theta) <= -c < 0.

Together with the symmetry lemma (S1: beta_k -> beta_k + pi/2; S2:
theta -> -theta; plus 2pi/pi periodicity — one-line operator identities, see
counterexample/README.md) the same inequality holds on the full torus, so
every global maximiser of the tree value witnesses the failure of the Fixed
Angle conjecture by at least c.

What is verified, and how:
  * the domain in the header equals the reduced fundamental domain, recomputed
    here from math.pi with outward rounding;
  * L <= f_2(d; witness), by a 256-bit point enclosure;
  * the certificate tree tiles the domain EXACTLY: the file only names split
    coordinates; every box is reconstructed by this checker from the domain by
    exact float-midpoint bisection, so there are no coordinates to trust;
  * every P leaf: sup f_2(box) < L, by natural ball evaluation, else by the
    first-order centred form (midpoint value + AD gradient enclosure);
  * every S leaf: sup [C_e - f_2](box) < 0 by the centred form; c is the
    worst (largest) such certified supremum, recomputed here, not read.
  * every enclosing ball is checked (in exact dyadic arithmetic) to contain
    its box.

Both scalar engines are transcribed fresh in this file from their defining
formulas: Marwaha's p=2 girth>5 closed form (arXiv:2101.05513v2, Sec. 2) and
the two-sided Dicke reduction of QAOA_2 on K_{d,d}.

Dependencies: python-flint only.  Usage:
    python3 check_global_p2_d9.py <certificate.jsonl.gz>
Exit code 0 iff every check passes.  All floats cross the file boundary as
exact binary64 hex.
"""

import gzip
import json
import math
import sys
from math import comb

from flint import ctx, arb, acb

PREC = 96


def fup(a):
    return math.nextafter(float(a.upper()), math.inf)


def flo(a):
    return math.nextafter(float(a.lower()), -math.inf)


# --------------------------------------------------------------------------
# forward-mode AD over acb
# --------------------------------------------------------------------------

class D4:
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v, self.g = v, g

    @staticmethod
    def const(v):
        z = acb(0)
        return D4(acb(v), (z, z, z, z))

    @staticmethod
    def var(v, i):
        g = [acb(0)] * 4
        g[i] = acb(1)
        return D4(acb(v), tuple(g))

    def _c(self, o):
        return o if isinstance(o, D4) else D4.const(o)

    def __add__(self, o):
        o = self._c(o)
        return D4(self.v + o.v, tuple(a + b for a, b in zip(self.g, o.g)))

    __radd__ = __add__

    def __neg__(self):
        return D4(-self.v, tuple(-a for a in self.g))

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) + (-self)

    def __mul__(self, o):
        o = self._c(o)
        return D4(self.v * o.v,
                  tuple(a * o.v + self.v * b for a, b in zip(self.g, o.g)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * (acb(1) / acb(o))

    def __pow__(self, n):
        n = int(n)
        if n == 0:
            return D4.const(1)
        if n == 1:
            return self
        pm1 = self.v ** (n - 1)
        return D4(pm1 * self.v, tuple((n * pm1) * a for a in self.g))

    def cos(self):
        return D4(self.v.cos(), tuple(-self.v.sin() * a for a in self.g))

    def sin(self):
        return D4(self.v.sin(), tuple(self.v.cos() * a for a in self.g))


# --------------------------------------------------------------------------
# engine 1: Marwaha closed form for f_2 on the d-regular tree
# f2 = 1/2 + c^2 r t y^E z - (c s / 2) alpha - (s^2 t z / 4) kappa
# --------------------------------------------------------------------------

def f2_marwaha(E, g1, g2, b1, b2, I):
    c, s = (2 * b2).cos(), (2 * b2).sin()
    r, t = (2 * b1).cos(), (2 * b1).sin()
    m, n = g2.cos(), g2.sin()
    y, z = g1.cos(), g1.sin()
    yE = y ** E
    A = (m * y - n * r * z) ** E
    B = (m * y + n * r * z) ** E
    P = (m + I * (n * t * yE * z)) ** E
    Q = (m - I * (n * t * yE * z)) ** E
    kappa = ((1 + r) * A - (1 - r) * B) * (P + Q)
    alpha = ((1 + r) * (-(m * r * z) - n * y) * A
             + (1 - r) * ((m * r * z) - n * y) * B
             + t * ((m * t * yE * z + I * n) * P
                    + (m * t * yE * z - I * n) * Q))
    return (0.5 + c * c * r * t * yE * z
            - (c * s) * alpha / 2
            - (s * s * t * z) * kappa / 4)


def f2_val(E, x):
    """x: 4 acb in order (g1,g2,b1,b2); returns arb enclosure."""
    return f2_marwaha(E, x[0], x[1], x[2], x[3], acb(0, 1)).real


def f2_grad(E, x):
    v = f2_marwaha(E, x[0], x[1], x[2], x[3], acb(0, 1))
    return v.v.real, tuple(a.real for a in v.g)


# --------------------------------------------------------------------------
# engine 2: QAOA_2 <C_e> on K_{d,d}, two-sided Dicke reduction
# state on (a,b) = (excitations left, right); Cm = a(d-b) + (d-a)b;
# mixer = product of single-side Wigner rotations in the Dicke basis
# --------------------------------------------------------------------------

def _mixer(d, cD, sD, I, wrap):
    """Row-stochastic-like Dicke mixer matrix M[b'][a'] with Dual/acb entries."""
    Dd = d + 1
    sq = [arb(comb(d, a)).sqrt() for a in range(Dd)]
    ms = (-I) * sD
    Mx = [[None] * Dd for _ in range(Dd)]
    for ib in range(Dd):
        for ia in range(Dd):
            acc = wrap(0)
            for k in range(max(0, ia - ib), min(d - ib, ia) + 1):
                coef = arb(comb(d - ib, k) * comb(ib, ia - k))
                acc = acc + (cD ** (d - ib + ia - 2 * k)) \
                    * (ms ** (ib - ia + 2 * k)) * coef
            Mx[ib][ia] = acc * (sq[ib] / sq[ia])
    return Mx


def _ce_generic(d, g1, g2, b1, b2, I, wrap):
    Dd = d + 1
    sq = [arb(comb(d, a)).sqrt() for a in range(Dd)]
    two_pow = arb(2) ** arb(d)
    Cm = [[a * (d - b) + (d - a) * b for b in range(Dd)] for a in range(Dd)]
    psi = [[wrap(sq[a] * sq[b] / two_pow) for b in range(Dd)]
           for a in range(Dd)]
    for g, b in ((g1, b1), (g2, b2)):
        for ia in range(Dd):
            for ib in range(Dd):
                u = g * Cm[ia][ib]
                psi[ia][ib] = psi[ia][ib] * (u.cos() - I * u.sin())
        Mx = _mixer(d, b.cos(), b.sin(), I, wrap)
        tmp = [[sum_(wrap, (Mx[i][t] * psi[t][j] for t in range(Dd)))
                for j in range(Dd)] for i in range(Dd)]
        psi = [[sum_(wrap, (tmp[i][t] * Mx[j][t] for t in range(Dd)))
                for j in range(Dd)] for i in range(Dd)]
    return psi, Cm


def sum_(wrap, it):
    acc = wrap(0)
    for v in it:
        acc = acc + v
    return acc


def ce_val(d, x):
    psi, Cm = _ce_generic(d, x[0], x[1], x[2], x[3], acb(0, 1), acb)
    tot = arb(0)
    Dd = d + 1
    for i in range(Dd):
        for j in range(Dd):
            z = psi[i][j]
            tot = tot + (z.real ** 2 + z.imag ** 2) * arb(Cm[i][j])
    return tot / arb(d ** 2)


def ce_grad(d, x):
    psi, Cm = _ce_generic(d, x[0], x[1], x[2], x[3], acb(0, 1), D4.const)
    Dd = d + 1
    tot_v = arb(0)
    tot_g = [arb(0)] * 4
    for i in range(Dd):
        for j in range(Dd):
            z = psi[i][j]
            w = arb(Cm[i][j])
            tot_v = tot_v + (z.v.real ** 2 + z.v.imag ** 2) * w
            for q in range(4):
                tot_g[q] = tot_g[q] + 2 * (z.v.real * z.g[q].real
                                           + z.v.imag * z.g[q].imag) * w
    dd2 = arb(d ** 2)
    return tot_v / dd2, tuple(t / dd2 for t in tot_g)


# --------------------------------------------------------------------------
# box machinery
# --------------------------------------------------------------------------

def enclosing_ball(lo, hi):
    """(m, r) floats with [m-r, m+r] >= [lo, hi], containment PROVEN exactly.

    The containment check is done in exact rational arithmetic (binary64 is a
    subset of Q), so it cannot false-alarm or false-pass at any depth.
    """
    from fractions import Fraction as F
    m = 0.5 * (lo + hi)
    r = max(m - lo, hi - m)
    if r > 0.0:
        r = math.nextafter(r, math.inf)
    assert F(m) - F(r) <= F(lo) and F(m) + F(r) >= F(hi), "ball containment"
    return m, r


def expected_domain(d):
    pi = math.nextafter(math.pi, math.inf)
    hpi = math.nextafter(math.pi / 2, math.inf)
    qpi = math.nextafter(math.pi / 4, math.inf)
    if d % 2 == 0:
        return [(0.0, hpi), (-hpi, hpi), (-qpi, qpi), (-qpi, qpi)]
    return [(0.0, pi), (-pi, pi), (-qpi, qpi), (-qpi, qpi)]


D = 9  # this checker verifies only the d=9 certificate of Theorem G


def check(path):
    fh = gzip.open(path, "rt")
    header = json.loads(fh.readline())
    assert header["kind"] == "wl-global-p2-cover" and header["p"] == 2
    assert header["order"] == "g1,g2,b1,b2"
    assert header["d"] == D, f"expected d={D}, header has d={header['d']}"
    d = D
    E = d - 1
    L = float.fromhex(header["L_hex"])
    dom = [(float.fromhex(a), float.fromhex(b))
           for a, b in header["domain_hex"]]
    exp = expected_domain(d)
    assert all(a == ea and b == eb for (a, b), (ea, eb) in zip(dom, exp)), \
        "header domain is not the reduced fundamental domain"

    # L <= f_2(d; witness), 256-bit point enclosure
    ctx.prec = 256
    wit = [acb(arb(float.fromhex(h))) for h in header["witness_hex"]]
    wlo = flo(f2_val(E, wit))
    assert wlo >= L, f"witness check failed: {wlo} < {L}"

    ctx.prec = PREC
    stack = [tuple(dom)]
    counts = {"N": 0, "P n": 0, "P c": 0, "S": 0}
    worst_safe = -math.inf
    nline = 1
    for line in fh:
        nline += 1
        tok = line.split()
        assert stack, f"line {nline}: tree exhausted early"
        box = stack.pop()
        if tok[0] == "N":
            k = int(tok[1])
            lo, hi = box[k]
            mid = 0.5 * (lo + hi)
            assert lo < mid < hi, f"line {nline}: degenerate split"
            hi_child = list(box)
            lo_child = list(box)
            lo_child[k] = (lo, mid)
            hi_child[k] = (mid, hi)
            stack.append(tuple(hi_child))
            stack.append(tuple(lo_child))
            counts["N"] += 1
            continue
        mr = [enclosing_ball(a, b) for a, b in box]
        ball = [acb(arb(m, r)) for m, r in mr]
        if tok[0] == "P":
            ok = fup(f2_val(E, ball)) < L
            if not ok:
                _, gv = f2_grad(E, [D4.var(arb(m, r), i)
                                    for i, (m, r) in enumerate(mr)])
                fc = f2_val(E, [acb(arb(m)) for m, _ in mr])
                # radius term formed IN arb: a float product rounds to nearest
                # and can shave the radius inward (referee defect E-G1)
                bound = fc
                for i in range(4):
                    bound = bound + gv[i] * arb(0.0, mr[i][1])
                ok = fup(bound) < L
            assert ok, f"line {nline}: P leaf fails, box={box}"
            counts["P " + tok[1]] += 1
        elif tok[0] == "S":
            pt = [acb(arb(m)) for m, _ in mr]
            fc = f2_val(E, pt)
            cc = ce_val(d, pt)
            _, gf = f2_grad(E, [D4.var(arb(m, r), i)
                                for i, (m, r) in enumerate(mr)])
            _, gc = ce_grad(d, [D4.var(arb(m, r), i)
                                for i, (m, r) in enumerate(mr)])
            # delta gradient as the arb difference of enclosures; radius term
            # in ball arithmetic (E-G1)
            bound = cc - fc
            for i in range(4):
                bound = bound + (gc[i] - gf[i]) * arb(0.0, mr[i][1])
            sup_d = fup(bound)
            assert sup_d < 0.0, f"line {nline}: S leaf fails, box={box}"
            worst_safe = max(worst_safe, sup_d)
            counts["S"] += 1
        else:
            raise AssertionError(f"line {nline}: bad record {tok!r} "
                                 "(ABORT lines mean an unfinished run)")
    assert not stack, f"tree incomplete: {len(stack)} boxes never discharged"
    fh.close()

    c = -worst_safe if counts["S"] else None
    print(f"certificate OK: d={d}, L={L!r}")
    print(f"  nodes: {counts}")
    if c is not None:
        print(f"  c = {c!r}  (= -max certified sup delta over S leaves)")
        print(f"THEOREM: every theta in the angle torus with "
              f"f_2({d};theta) >= {L!r}")
        print(f"  satisfies  <C_e>(K_{d},{d};theta) - f_2({d};theta) "
              f"<= {-c!r} < 0.")
        print(f"  In particular every global maximiser of f_2^tree({d}) "
              f"violates the Fixed Angle conjecture by at least {c:.6e}.")
    else:
        print("  no S leaves: {f_2 >= L} is empty on the domain")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check(args[0])
    print("PASS")
