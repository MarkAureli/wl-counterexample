"""Independent checker for the global-optimality certificate (mpmath.iv).

  * arithmetic is mpmath's inf-sup interval type (`mpmath.iv`)
  * stdlib + mpmath only.

What is verified
----------------
Certificate is gzipped text: line 1 a JSON header, then a preorder walk of
the box tree, one record per line:

    N <k>        internal node, split coordinate k, low subtree then high
    P n | P c    prune leaf: sup f_2(box) < b
    S <hex>      safe leaf: sup [f_{2,9}^K - f_{2,9}^tree](box) < 0

The file names split coordinates only; every box is reconstructed here from
the header domain by exact float-midpoint bisection. Checks performed:

  1. structural: the recorded preorder walk replays against explicit stack,
     every leaf discharges exactly one box, and the stack is empty at EOF.
     Any record other than N/P/S (e.g. "ABORT budget") is rejected. This
     proves the leaves tile the header domain exactly.
  2. domain: the header domain equals the reduced fundamental domain,
     recomputed here from math.pi with outward rounding.
  3. enclosing balls: for each box coordinate [lo,hi] the ball (m, r) with
     m = 0.5*(lo+hi), r = nextafter(max(m-lo, hi-m), +inf) is checked to
     contain [lo,hi] in EXACT rational arithmetic (fractions.Fraction).
  4. P leaves: sup f_{2,9}^tree(ball) < b by natural interval evaluation;
     if that fails, the first-order centred form
     f_{2,9}^tree(mid) + sum_i sup|\del f_{2,9}^tree /\del x_i|(ball) * r_i
     is used. One of the two must be < b.
  5. S leaves: c = -max_S sup delta is recomputed, never read from the file.
  6. witness: b <= inf f_{2,9}^tree(witness) at iv.dps = 40, read from the
     header (witness_hex/b_hex): cross-engine agreement between mpmath.iv
     value and the arb/flint value from certifier's verify_b.

Usage
-----
    python3 check.py <cert.jsonl.gz>
Exit status 0 iff every check passes (full run: every P and S leaf).
"""
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false

import argparse
import gzip
import json
import math
import sys
import time
from fractions import Fraction
from math import comb

from mpmath import iv, mp

D = 9

# --------------------------------------------------------------------------
# outward endpoint extraction
# --------------------------------------------------------------------------

def sup(x):
    """A binary64 >= sup(x) for an iv.mpf interval x."""
    return math.nextafter(float(mp.mpf(x.b)), math.inf)


def inf(x):
    """A binary64 <= inf(x) for an iv.mpf interval x."""
    return math.nextafter(float(mp.mpf(x.a)), -math.inf)


def mag(x):
    """A binary64 >= max |x| over an iv.mpf interval x."""
    return math.nextafter(max(abs(inf(x)), abs(sup(x))), math.inf)


def _sym(t):
    """The interval [-t, t] as iv.mpf, t a non-negative binary64."""
    return iv.mpf([-t, t])


# --------------------------------------------------------------------------
# complex interval scalars with optional forward-mode gradient
# --------------------------------------------------------------------------

_ZC = None     # iv.mpc(0) -- rebuilt whenever the working precision changes
_1C = None


def _reset_consts():
    global _ZC, _1C
    _ZC = iv.mpc(iv.mpf([0, 0]), iv.mpf([0, 0]))
    _1C = iv.mpc(iv.mpf([1, 1]), iv.mpf([0, 0]))


def set_dps(dps):
    iv.dps = dps
    mp.dps = max(mp.dps, dps + 10)
    _reset_consts()


class IVC:
    """A complex interval, optionally carrying 4 partial derivatives."""

    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = v
        self.g = g

    # -- constructors ------------------------------------------------------
    @staticmethod
    def const(v):
        return IVC(_as_mpc(v), None)

    @staticmethod
    def var(v, i):
        g = [_ZC] * 4
        g[i] = _1C
        return IVC(_as_mpc(v), tuple(g))

    def _c(self, o):
        return o if isinstance(o, IVC) else IVC(_as_mpc(o), None)

    # -- arithmetic --------------------------------------------------------
    def __add__(self, o):
        o = self._c(o)
        if self.g is None and o.g is None:
            return IVC(self.v + o.v, None)
        a, b = _grad(self), _grad(o)
        # `is _ZC` marks a partial that is EXACTLY the zero interval;
        return IVC(self.v + o.v,
                   tuple(y if x is _ZC else (x if y is _ZC else x + y)
                         for x, y in zip(a, b)))

    __radd__ = __add__

    def __neg__(self):
        if self.g is None:
            return IVC(-self.v, None)
        return IVC(-self.v,
                   tuple(_ZC if x is _ZC else -x for x in self.g))

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) + (-self)

    def __mul__(self, o):
        o = self._c(o)
        v = self.v * o.v
        if self.g is None and o.g is None:
            return IVC(v, None)
        a, b = _grad(self), _grad(o)
        sv, ov = self.v, o.v
        g = []
        for x, y in zip(a, b):
            xz, yz = x is _ZC, y is _ZC
            if xz and yz:
                g.append(_ZC)
            elif yz:
                g.append(x * ov)
            elif xz:
                g.append(sv * y)
            else:
                g.append(x * ov + sv * y)
        return IVC(v, tuple(g))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _as_mpc(o.v if isinstance(o, IVC) else o)
        if self.g is None:
            return IVC(self.v / o, None)
        return IVC(self.v / o,
                   tuple(_ZC if x is _ZC else x / o for x in self.g))

    def __pow__(self, n):
        n = int(n)
        if n == 0:
            return IVC.const(1)
        if n == 1:
            return self
        if self.g is None:
            return IVC(self.v ** n, None)
        pm1 = self.v ** (n - 1)
        f = pm1 * n
        return IVC(pm1 * self.v,
                   tuple(_ZC if x is _ZC else f * x for x in self.g))

    # -- transcendentals (real arguments in every call site here) ----------
    def cos(self):
        if self.g is None:
            return IVC(iv.cos(self.v), None)
        s = -iv.sin(self.v)
        return IVC(iv.cos(self.v),
                   tuple(_ZC if x is _ZC else s * x for x in self.g))

    def sin(self):
        if self.g is None:
            return IVC(iv.sin(self.v), None)
        c = iv.cos(self.v)
        return IVC(iv.sin(self.v),
                   tuple(_ZC if x is _ZC else c * x for x in self.g))


def _grad(a):
    return a.g if a.g is not None else (_ZC, _ZC, _ZC, _ZC)


def _as_mpc(v):
    if isinstance(v, type(_ZC)):
        return v
    if isinstance(v, type(_ZC.real)):
        return iv.mpc(v, iv.mpf([0, 0]))
    if isinstance(v, (int, float)):
        return iv.mpc(iv.mpf([v, v]), iv.mpf([0, 0]))
    return iv.mpc(v)


def _re(z):
    """Real part of an iv.mpc as iv.mpf."""
    return z.real


# --------------------------------------------------------------------------
# Marwaha's closed form
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
    c = (2 * b2).cos()
    s = (2 * b2).sin()
    r = (2 * b1).cos()
    t = (2 * b1).sin()
    m = g2.cos()
    n = g2.sin()
    y = g1.cos()
    z = g1.sin()
    yE = y ** (D - 1)
    A = (m * y - n * r * z) ** (D - 1)
    B = (m * y + n * r * z) ** (D - 1)
    q = n * t * yE * z
    P = (m + q * I) ** (D - 1)
    Q = (m - q * I) ** (D - 1)
    kappa = ((1 + r) * A - (1 - r) * B) * (P + Q)
    w = m * t * yE * z
    alpha = ((1 + r) * (-(m * r * z) - n * y) * A
             + (1 - r) * (m * r * z - n * y) * B
             + t * ((w + n * I) * P + (w - n * I) * Q))
    return (0.5 + c * c * r * t * yE * z
            - (c * s) * alpha / 2
            - (s * s * t * z) * kappa / 4)


def tree(x, I):
    """x: 4 IVC (no gradient) in order (b1,g1,b2,g2). Returns iv.mpf."""
    return _re(tree_form(x[0], x[1], x[2], x[3], I).v)


def tree_grad(x, I):
    """x: 4 IVC seeded as variables. Returns (iv.mpf, 4 iv.mpf)."""
    out = tree_form(x[0], x[1], x[2], x[3], I)
    return _re(out.v), tuple(_re(a) for a in out.g)


# --------------------------------------------------------------------------
# K_{d,d} per-edge engine
# --------------------------------------------------------------------------
#
#   psi0[p][q]  = sqrt(C(d,p) C(d,q)) / 2^d        (|+>^{2d} in Dicke basis)
#   Cm[p][q]    = d(p+q) - 2pq                     (cut value)
#   phase layer : psi[p][q] *= cos(gamma Cm[p][q]) - i sin(gamma Cm[p][q])
#   mixer       : M[p'][p] = sqrt(C(d,p')/C(d,p))
#                 * sum_k C(d-p',k) C(p',p-k)
#                   cos(beta)^(d+p-p'-2k) (-i sin beta)^(2k+p'-p),
#                 k = max(0, p-p') .. min(d-p', p')
#   psi <- M psi M^T ;   f_{2,d}^K = sum |psi[p][q]|^2 Cm[p][q] / d^2
# --------------------------------------------------------------------------

def _mixer(cb, sb, I):
    sq = [iv.sqrt(iv.mpf([comb(D, p), comb(D, p)])) for p in range(D + 1)]
    ms = sb * (-I)
    Mx = [[None] * (D+ 1) for _ in range(D + 1)]
    for pp in range(D + 1):
        for p in range(D + 1):
            acc = IVC.const(0)
            for k in range(max(0, p - pp), min(D - pp, p) + 1):
                coef = comb(D - pp, k) * comb(pp, p - k)
                acc = acc + ((cb ** (D + p - pp - 2 * k))
                             * (ms ** (2 * k + pp - p))
                             * IVC.const(iv.mpf([coef, coef])))
            Mx[pp][p] = acc * IVC.const(sq[pp] / sq[p])
    return Mx


def kdd_form(b1, g1, b2, g2, I):
    """Returns (psi as (d+1)x(d+1) IVC, Cm integer table)."""
    sq = [iv.sqrt(iv.mpf([comb(D, p), comb(D, p)])) for p in range(D + 1)]
    two_pow = iv.mpf([2, 2]) ** D
    Cm = [[D * (p + q) - 2 * p * q for q in range(D + 1)] for p in range(D + 1)]
    psi = [[IVC.const(sq[p] * sq[q] / two_pow) for q in range(D + 1)]
           for p in range(D + 1)]
    for gamma, beta in ((g1, b1), (g2, b2)):
        for p in range(D + 1):
            for q in range(D + 1):
                u = gamma * Cm[p][q]
                psi[p][q] = psi[p][q] * (u.cos() - u.sin() * I)
        Mx = _mixer(beta.cos(), beta.sin(), I)
        tmp = [[_dot(Mx[i], [psi[t][j] for t in range(D + 1)])
                for j in range(D + 1)] for i in range(D + 1)]
        psi = [[_dot(tmp[i], Mx[j]) for j in range(D + 1)] for i in range(D + 1)]
    return psi, Cm


def _dot(u, v):
    acc = IVC.const(0)
    for a, b in zip(u, v):
        acc = acc + a * b
    return acc


def kdd(d, x, I):
    psi, Cm = kdd_form(x[0], x[1], x[2], x[3], I)
    tot = iv.mpf([0, 0])
    for i in range(D + 1):
        for j in range(D + 1):
            z = psi[i][j].v
            w = Cm[i][j]
            tot = tot + (z.real ** 2 + z.imag ** 2) * iv.mpf([w, w])
    return tot / iv.mpf([D * D, D * D])


def kdd_grad(x, I):
    psi, Cm = kdd_form(x[0], x[1], x[2], x[3], I)
    tot = iv.mpf([0, 0])
    tg = [iv.mpf([0, 0])] * 4
    for p in range(D + 1):
        for q in range(D + 1):
            z = psi[p][q]
            zv = z.v
            w = iv.mpf([Cm[p][q], Cm[p][q]])
            tot = tot + (zv.real ** 2 + zv.imag ** 2) * w
            for i in range(4):
                di = z.g[i]
                tg[i] = tg[i] + (zv.real * di.real
                                 + zv.imag * di.imag) * (2 * w)
    dd2 = iv.mpf([D * D, D * D])
    return tot / dd2, tuple(t / dd2 for t in tg)


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------

def enclosing_ball(lo, hi):
    """(m, r) binary64 with [m-r, m+r] superset [lo, hi]."""
    m = 0.5 * (lo + hi)
    r = max(m - lo, hi - m)
    if r > 0.0:
        r = math.nextafter(r, math.inf)
    if not (Fraction(m) - Fraction(r) <= Fraction(lo)
            and Fraction(m) + Fraction(r) >= Fraction(hi)):
        raise AssertionError(f"[{m}+-{r}] does not contain [{lo},{hi}]")
    return m, r


def ball_ivc(m, r, grad_index=None):
    """The interval [m-r, m+r] as an IVC."""
    z = iv.mpf([m, m]) + _sym(r)
    return IVC.var(z, grad_index) if grad_index is not None \
            else IVC.const(z)


def expected_domain():
    """Reduced fundamental domain, every endpoint pushed outward one ulp."""
    pi = math.nextafter(math.pi, math.inf)
    qpi = math.nextafter(math.pi / 4, math.inf)
    return [(-qpi, qpi), (0.0, pi), (-qpi, qpi), (-pi, pi)]


# --------------------------------------------------------------------------
# leaf tests
# --------------------------------------------------------------------------

def check_prune(mr, b, I):
    """sup f_{2,9}^tree(box) < b, naturally, else by the centred form."""
    ball = [ball_ivc(m, r) for m, r in mr]
    nat = sup(tree(ball, I))
    if nat < b:
        return True, "n", nat
    dual = [ball_ivc(m, r, i) for i, (m, r) in enumerate(mr)]
    _, g = tree_grad(dual, I)
    mid = [IVC.const(iv.mpf([m, m])) for m, _ in mr]
    bound = tree(mid, I)
    for i in range(4):
        bound = bound + _sym(math.nextafter(mag(g[i]) * mr[i][1], math.inf))
    ctr = sup(bound)
    return ctr < b, "c", ctr


def check_safe(mr, I):
    """sup [f_{2,9}^K - f_{2,9}^tree](box) < 0 by the centred form."""
    dual = [ball_ivc(m, r, i) for i, (m, r) in enumerate(mr)]
    mid = [IVC.const(iv.mpf([m, m])) for m, _ in mr]
    _, gf = tree_grad(dual, I)
    _, gc = kdd_grad(dual, I)
    bound = kdd(D, mid, I) - tree(mid, I)
    for i in range(4):
        gm = math.nextafter(mag(gf[i]) + mag(gc[i]), math.inf)
        bound = bound + _sym(math.nextafter(gm * mr[i][1], math.inf))
    s = sup(bound)
    return s < 0.0, s


# --------------------------------------------------------------------------
# certificate replay
# --------------------------------------------------------------------------


def check(path):
    dps, wit_dps = 25, 40
    fh = gzip.open(path, "rt")
    header = json.loads(fh.readline())
    if header.get("kind") != "cover" or header.get("p") != 2:
        raise SystemExit(f"bad header kind/p: {header.get('kind')!r}")
    if header.get("order") != "b1,g1,b2,g2":
        raise SystemExit(f"unexpected variable order {header.get('order')!r}")
    if header.get("d") != D:
        raise SystemExit(f"expected d={D}, header has d={header.get('d')!r}")
    b = float.fromhex(header["b_hex"])
    dom = [(float.fromhex(x), float.fromhex(y))
           for x, y in header["dom_hex"]]
    if dom != [tuple(t) for t in expected_domain()]:
        raise SystemExit("header domain is not the reduced fundamental domain")
    print(f"header: d={D}  b={b!r}  domain OK (reduced fundamental domain)")

    set_dps(wit_dps)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    wh = header.get("witness_hex")
    if not wh:
        raise SystemExit("header has no witness_hex")
    wit = [IVC.const(iv.mpf([float.fromhex(h), float.fromhex(h)]))
           for h in wh]
    fw = tree(wit, I)
    lo = inf(fw)
    if lo < b:
        raise SystemExit(f"could not establish f_{2,9}^tree >= b={b!r}")
    mp.dps = wit_dps + 10
    marg = mp.mpf(fw.a) - mp.mpf(b)
    print(f"witness: b = {b!r} <= f_{{2,9}}^tree(witness) >= {lo!r} "
          f"(interval lower endpoint - b = {mp.nstr(marg, 6)}, "
          f"iv.dps={wit_dps})")

    set_dps(dps)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    stack = [tuple(dom)]
    counts = {"N": 0, "P n": 0, "P c": 0, "S": 0}
    checked = {"P": 0, "S": 0}
    modes = {"n": 0, "c": 0}
    worst_safe = -math.inf
    nline = 1
    t0 = time.time()
    for line in fh:
        nline += 1
        if nline % 250_000 == 0:
            print(f"  progress: {nline} records, verified P={checked['P']} "
                  f"S={checked['S']}, t={time.time() - t0:.0f}s", flush=True)
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
                raise SystemExit(f"line {nline}: degenerate split at coord {k}")
            child_lo = list(box)
            child_hi = list(box)
            child_lo[k] = (lo_k, mid)
            child_hi[k] = (mid, hi_k)
            stack.append(tuple(child_hi))
            stack.append(tuple(child_lo))
            counts["N"] += 1
            continue
        if head == "P":
            if len(tok) != 2 or tok[1] not in ("n", "c"):
                raise SystemExit(f"line {nline}: bad P record {tok!r}")
            counts["P " + tok[1]] += 1
            mr = [enclosing_ball(x, y) for x, y in box]
            ok, how, val = check_prune(mr, b, I)
            if not ok:
                raise SystemExit(f"line {nline}: P leaf FAILS, sup f_{{2,9}}^tree "
                                 f"<= {val!r} not < b={b!r}, box={box}")
            checked["P"] += 1
            modes[how] += 1
        elif head == "S":
            counts["S"] += 1
            mr = [enclosing_ball(x, y) for x, y in box]
            ok, s = check_safe(mr, I)
            if not ok:
                raise SystemExit(f"line {nline}: S leaf FAILS, sup delta "
                                 f"<= {s!r} not < 0, box={box}")
            worst_safe = max(worst_safe, s)
            checked["S"] += 1
        else:
            raise SystemExit(f"line {nline}: bad record {tok!r} "
                             "(an ABORT line means the run never finished)")
    fh.close()
    if stack:
        raise SystemExit(f"tree incomplete: {len(stack)} boxes never discharged")

    dt = time.time() - t0
    print(f"\nstructure: {nline-1} records replayed in {dt:.1f}s, "
          f"stack empty at EOF")
    print(f"  nodes: {counts}")
    print(f"  leaves verified: {checked['P']} P "
          f"(natural {modes['n']}, centred {modes['c']}), {checked['S']} S")
    if counts["S"]:
        c = -worst_safe
        print(f"\n  c = {c!r}   (= -max certified sup delta over S leaves, "
              "recomputed here)")
        print(f"THEOREM (mpmath.iv engine): every theta with "
              f"f_{{2,9}}^tree(theta) >= {b!r} satisfies")
        print(f"  f_{{2,9}}^K(theta) - f_{{2,9}}^tree(theta) <= {worst_safe!r} "
              f"< 0;  violation at least {c:.6e}.")
    else:
        print("\n  no S leaves: {f_{2,9}^tree >= b} is empty on the domain")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("certificate")
    args = ap.parse_args()
    t0 = time.time()
    check(args.certificate)
    print(f"\nPASS  ({time.time()-t0:.1f}s total, mpmath.iv only, "
          "no python-flint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

