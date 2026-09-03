"""Independent checker for the global-optimality certificate (mpmath.iv only).

This is a deliberately independent re-implementation of the verifier for the
certificates written by `certify_global_p2.py`.  It shares NO code with the
producer:

  * arithmetic is mpmath's inf-sup interval type (`mpmath.iv`), not arb balls;
  * both scalar engines are transcribed here from their defining formulas
    (Marwaha's p=2 girth>5 closed form for f_2 on the d-regular tree, and the
    two-sided Dicke reduction of QAOA_2 on K_{d,d});
  * python-flint is never imported.  stdlib + mpmath only.

What is verified
----------------
The certificate is gzipped text: line 1 a JSON header, then a preorder walk of
the box tree, one record per line:

    N <k>        internal node, split coordinate k, low subtree then high
    P n | P c    prune leaf: sup f_2(box) < b
    S <hex>      safe leaf: sup [<C_e> - f_2](box) < 0

The file names split coordinates ONLY, so every box is reconstructed here from
the header domain by exact float-midpoint bisection; there are no coordinates
in the file to trust.  Checks performed:

  1. structural: the recorded preorder walk replays against an explicit stack,
     every leaf discharges exactly one box, and the stack is empty at EOF.
     Any record other than N/P/S (e.g. "ABORT budget") is rejected.  This
     proves the leaves tile the header domain exactly.  Always done in full,
     even under --sample: it is pure float bookkeeping and costs seconds.
  2. domain: the header domain equals the reduced fundamental domain,
     recomputed here from math.pi with outward rounding.
  3. enclosing balls: for each box coordinate [lo,hi] the ball (m, r) with
     m = 0.5*(lo+hi), r = nextafter(max(m-lo, hi-m), +inf) is checked to
     contain [lo,hi] in EXACT rational arithmetic (fractions.Fraction);
     binary64 is a subset of Q so this cannot false-pass at any depth.
  4. P leaves: sup f_2(ball) < b by natural interval evaluation; if that fails
     (interval dependency is expression-shaped, so the arb and iv natural
     bounds differ slightly) the first-order centred form
        f_2(mid) + sum_i sup|d f_2/d x_i|(ball) * r_i
     is used, with the gradient enclosure from forward-mode dual numbers over
     iv.mpc.  One of the two must be < b.
  5. S leaves: centred form on delta = <C_e> - f_2,
        delta(mid) + sum_i (sup|d C_e/d x_i| + sup|d f_2/d x_i|) * r_i  <  0,
     and c = -max_S sup delta is RECOMPUTED here, never read from the file.
  6. witness: b <= inf f_2(d; witness) at iv.dps >= 40.

Calibration (always run, before any replay; loud abort on mismatch)
-------------------------------------------------------------------
  * delta = <C_e> - f_2 at the d=9 reference point must enclose
    -0.005636308471626929 to 1e-12 -- this exercises BOTH engines at once;
  * f_2(d=9; witness) >= b = 0x1.473bb99c3c5eap-1.
All angles enter as float.fromhex; no decimal angle string is ever parsed into
high precision (repo rule E3), and no angle is read from a document.

Usage
-----
    python3 check_global_p2_d9_mpiv.py <cert.jsonl.gz> [--sample N] [--dps 25]
Exit status 0 iff every check passes.
"""
# pyright: reportArgumentType=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import argparse
import gzip
import json
import math
import random
import sys
import time
from fractions import Fraction
from math import comb

from mpmath import iv, mp

# --------------------------------------------------------------------------
# calibration constants (exact binary64 hex; see module docstring)
# --------------------------------------------------------------------------

CAL_D = 9
CAL_POINT = ("-0x1.04f3d6213d819p-1", "-0x1.0b184b3c23969p-2",
             "-0x1.6f1f7f524975ep+1", "0x1.56f3f3e6b7d62p+1")
CAL_DELTA = -0.005636308471626929
CAL_TOL = 1e-12
CAL_B = float.fromhex("0x1.473bb99c3c5eap-1")          # 0.6391275408952286
CAL_WITNESS = ("-0x1.04f3d4e04111ap-1", "-0x1.0b184cde2ef9ap-2",
               "-0x1.6f1f7f7dd666ep+1", "0x1.56f3f40791132p+1")


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
#
# One class serves both roles: g is None for a plain value, or a 4-tuple of
# partial derivatives w.r.t. (b1, g1, b2, g2).  Keeping them in one class means
# the two scalar formulas below are written exactly once, so the value path and
# the derivative path cannot silently drift apart.
# --------------------------------------------------------------------------

_ZC = None      # iv.mpc(0) -- rebuilt whenever the working precision changes
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
        # `is _ZC` marks a partial that is EXACTLY the zero interval; skipping
        # those keeps the sparsity of the mixer (which depends on one beta
        # only) and is sound because 0 is exact in interval arithmetic.
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
# engine 1: Marwaha p=2 closed form for f_2 on the d-regular tree (E = d-1)
#
#   c = cos 2b2, s = sin 2b2, r = cos 2b1, t = sin 2b1,
#   m = cos g2, n = sin g2, y = cos g1, z = sin g1, yE = y^E
#   A = (m y - n r z)^E,  B = (m y + n r z)^E
#   P = (m + i n t yE z)^E,  Q = (m - i n t yE z)^E
#   kappa = ((1+r) A - (1-r) B) (P + Q)
#   alpha = (1+r)(-m r z - n y) A + (1-r)(m r z - n y) B
#           + t ((m t yE z + i n) P + (m t yE z - i n) Q)
#   f_2  = 1/2 + c^2 r t yE z - (c s / 2) alpha - (s^2 t z / 4) kappa
# --------------------------------------------------------------------------

def f2_form(E, b1, g1, b2, g2, I):
    c = (2 * b2).cos()
    s = (2 * b2).sin()
    r = (2 * b1).cos()
    t = (2 * b1).sin()
    m = g2.cos()
    n = g2.sin()
    y = g1.cos()
    z = g1.sin()
    yE = y ** E
    A = (m * y - n * r * z) ** E
    B = (m * y + n * r * z) ** E
    q = n * t * yE * z
    P = (m + q * I) ** E
    Q = (m - q * I) ** E
    kappa = ((1 + r) * A - (1 - r) * B) * (P + Q)
    w = m * t * yE * z
    alpha = ((1 + r) * (-(m * r * z) - n * y) * A
             + (1 - r) * (m * r * z - n * y) * B
             + t * ((w + n * I) * P + (w - n * I) * Q))
    return (0.5 + c * c * r * t * yE * z
            - (c * s) * alpha / 2
            - (s * s * t * z) * kappa / 4)


def f2_value(E, x, I):
    """x: 4 IVC (no gradient) in order (b1,g1,b2,g2).  Returns iv.mpf."""
    return _re(f2_form(E, x[0], x[1], x[2], x[3], I).v)


def f2_value_grad(E, x, I):
    """x: 4 IVC seeded as variables.  Returns (iv.mpf, 4 iv.mpf)."""
    out = f2_form(E, x[0], x[1], x[2], x[3], I)
    return _re(out.v), tuple(_re(a) for a in out.g)


# --------------------------------------------------------------------------
# engine 2: <C_e> for QAOA_2 on K_{d,d}, two-sided Dicke reduction
#
#   psi0[a][b]  = sqrt(C(d,a) C(d,b)) / 2^d          (|+>^{2d} in Dicke basis)
#   Cm[a][b]    = a(d-b) + (d-a)b                    (cut value)
#   phase layer : psi[a][b] *= cos(g Cm) - i sin(g Cm)
#   mixer       : M[b'][a'] = sqrt(C(d,b')/C(d,a'))
#                 * sum_k C(d-b',k) C(b',a'-k)
#                   cos(b)^(d-b'+a'-2k) (-i sin b)^(b'-a'+2k),
#                 k = max(0, a'-b') .. min(d-b', a')
#   psi <- M psi M^T ;   <C_e> = sum |psi[a][b]|^2 Cm[a][b] / d^2
# --------------------------------------------------------------------------

def _mixer(d, cb, sb, I):
    D = d + 1
    sq = [iv.sqrt(iv.mpf([comb(d, a), comb(d, a)])) for a in range(D)]
    ms = sb * (-I)
    Mx = [[None] * D for _ in range(D)]
    for ib in range(D):
        for ia in range(D):
            acc = IVC.const(0)
            for k in range(max(0, ia - ib), min(d - ib, ia) + 1):
                coef = comb(d - ib, k) * comb(ib, ia - k)
                acc = acc + ((cb ** (d - ib + ia - 2 * k))
                             * (ms ** (ib - ia + 2 * k))
                             * IVC.const(iv.mpf([coef, coef])))
            Mx[ib][ia] = acc * IVC.const(sq[ib] / sq[ia])
    return Mx


def ce_form(d, b1, g1, b2, g2, I):
    """Returns (psi as (d+1)x(d+1) IVC, Cm integer table)."""
    D = d + 1
    sq = [iv.sqrt(iv.mpf([comb(d, a), comb(d, a)])) for a in range(D)]
    two_pow = iv.mpf([2, 2]) ** d
    Cm = [[a * (d - b) + (d - a) * b for b in range(D)] for a in range(D)]
    psi = [[IVC.const(sq[a] * sq[b] / two_pow) for b in range(D)]
           for a in range(D)]
    for g, b in ((g1, b1), (g2, b2)):
        for ia in range(D):
            for ib in range(D):
                u = g * Cm[ia][ib]
                psi[ia][ib] = psi[ia][ib] * (u.cos() - u.sin() * I)
        Mx = _mixer(d, b.cos(), b.sin(), I)
        tmp = [[_dot(Mx[i], [psi[t][j] for t in range(D)])
                for j in range(D)] for i in range(D)]
        psi = [[_dot(tmp[i], Mx[j]) for j in range(D)] for i in range(D)]
    return psi, Cm


def _dot(u, v):
    acc = IVC.const(0)
    for a, b in zip(u, v):
        acc = acc + a * b
    return acc


def ce_value(d, x, I):
    psi, Cm = ce_form(d, x[0], x[1], x[2], x[3], I)
    D = d + 1
    tot = iv.mpf([0, 0])
    for i in range(D):
        for j in range(D):
            z = psi[i][j].v
            w = Cm[i][j]
            tot = tot + (z.real ** 2 + z.imag ** 2) * iv.mpf([w, w])
    return tot / iv.mpf([d * d, d * d])


def ce_value_grad(d, x, I):
    psi, Cm = ce_form(d, x[0], x[1], x[2], x[3], I)
    D = d + 1
    tot = iv.mpf([0, 0])
    tg = [iv.mpf([0, 0])] * 4
    for i in range(D):
        for j in range(D):
            z = psi[i][j]
            zv = z.v
            w = iv.mpf([Cm[i][j], Cm[i][j]])
            tot = tot + (zv.real ** 2 + zv.imag ** 2) * w
            for q in range(4):
                dq = z.g[q]
                tg[q] = tg[q] + (zv.real * dq.real
                                 + zv.imag * dq.imag) * (2 * w)
    dd2 = iv.mpf([d * d, d * d])
    return tot / dd2, tuple(t / dd2 for t in tg)


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------

def enclosing_ball(lo, hi):
    """(m, r) binary64 with [m-r, m+r] superset [lo, hi], PROVEN in exact Q."""
    m = 0.5 * (lo + hi)
    r = max(m - lo, hi - m)
    if r > 0.0:
        r = math.nextafter(r, math.inf)
    if not (Fraction(m) - Fraction(r) <= Fraction(lo)
            and Fraction(m) + Fraction(r) >= Fraction(hi)):
        raise AssertionError(f"ball [{m}+-{r}] does not contain [{lo},{hi}]")
    return m, r


def ball_ivc(m, r, grad_index=None):
    """The interval [m-r, m+r] as an IVC (outward rounded by iv arithmetic)."""
    z = iv.mpf([m, m]) + _sym(r)
    return IVC.var(z, grad_index) if grad_index is not None else IVC.const(z)


def expected_domain():
    """Reduced fundamental domain for d = 9, order (b1,g1,b2,g2), every endpoint
    pushed outward one ulp."""
    pi = math.nextafter(math.pi, math.inf)
    qpi = math.nextafter(math.pi / 4, math.inf)
    return [(-qpi, qpi), (0.0, pi), (-qpi, qpi), (-pi, pi)]


# --------------------------------------------------------------------------
# leaf tests
# --------------------------------------------------------------------------

def check_prune(E, mr, b, I):
    """sup f_2(box) < b, naturally if possible, else by the centred form."""
    ball = [ball_ivc(m, r) for m, r in mr]
    nat = sup(f2_value(E, ball, I))
    if nat < b:
        return True, "n", nat
    dual = [ball_ivc(m, r, i) for i, (m, r) in enumerate(mr)]
    _, g = f2_value_grad(E, dual, I)
    mid = [IVC.const(iv.mpf([m, m])) for m, _ in mr]
    bound = f2_value(E, mid, I)
    for i in range(4):
        bound = bound + _sym(math.nextafter(mag(g[i]) * mr[i][1], math.inf))
    ctr = sup(bound)
    return ctr < b, "c", ctr


def check_safe(d, E, mr, I):
    """sup [<C_e> - f_2](box) < 0 by the centred form; returns (ok, sup)."""
    dual = [ball_ivc(m, r, i) for i, (m, r) in enumerate(mr)]
    mid = [IVC.const(iv.mpf([m, m])) for m, _ in mr]
    _, gf = f2_value_grad(E, dual, I)
    _, gc = ce_value_grad(d, dual, I)
    bound = ce_value(d, mid, I) - f2_value(E, mid, I)
    for i in range(4):
        gm = math.nextafter(mag(gf[i]) + mag(gc[i]), math.inf)
        bound = bound + _sym(math.nextafter(gm * mr[i][1], math.inf))
    s = sup(bound)
    return s < 0.0, s


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate(dps=40):
    set_dps(dps)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    E = CAL_D - 1

    pt = [IVC.const(iv.mpf([float.fromhex(h), float.fromhex(h)]))
          for h in CAL_POINT]
    f2p = f2_value(E, pt, I)
    cep = ce_value(CAL_D, pt, I)
    delta = cep - f2p
    dlo, dhi = inf(delta), sup(delta)
    err = max(abs(dlo - CAL_DELTA), abs(dhi - CAL_DELTA))
    print(f"calibration 1 (both engines, d={CAL_D}):")
    print(f"  f_2   = [{inf(f2p)!r}, {sup(f2p)!r}]  width {sup(f2p)-inf(f2p):.3e}")
    print(f"  <C_e> = [{inf(cep)!r}, {sup(cep)!r}]  width {sup(cep)-inf(cep):.3e}")
    print(f"  delta = [{dlo!r}, {dhi!r}]  width {dhi-dlo:.3e}")
    print(f"  reference {CAL_DELTA!r}; max endpoint deviation {err:.3e}")
    if not (dlo - CAL_TOL <= CAL_DELTA <= dhi + CAL_TOL) or err > CAL_TOL:
        raise SystemExit(f"CALIBRATION 1 FAILED: delta enclosure "
                         f"[{dlo!r},{dhi!r}] vs {CAL_DELTA!r} (tol {CAL_TOL})")

    wit = [IVC.const(iv.mpf([float.fromhex(h), float.fromhex(h)]))
           for h in CAL_WITNESS]
    fw = f2_value(E, wit, I)
    lo = inf(fw)
    print(f"calibration 2 (witness): f_2 = [{lo!r}, {sup(fw)!r}]")
    print(f"  b = {CAL_B!r}; margin f_2 - b = {lo - CAL_B:.6e}")
    if lo < CAL_B:
        raise SystemExit(f"CALIBRATION 2 FAILED: {lo!r} < b = {CAL_B!r}")
    print("calibration OK\n")
    return dlo, dhi, lo


# --------------------------------------------------------------------------
# certificate replay
# --------------------------------------------------------------------------

def count_p_leaves(path):
    n = 0
    with gzip.open(path, "rt") as fh:
        fh.readline()
        for line in fh:
            if line[0] == "P":
                n += 1
    return n


D = 9  # this checker verifies only the d=9 certificate of Theorem G


def check(path, sample=None, dps=25, wit_dps=40):
    calibrate(max(wit_dps, 40))

    sel = None
    if sample is not None:
        t0 = time.time()
        npl = count_p_leaves(path)
        if sample < npl:
            sel = set(random.Random(0).sample(range(npl), sample))
        print(f"sampling {min(sample, npl)} of {npl} P leaves "
              f"(prescan {time.time()-t0:.1f}s)\n")

    fh = gzip.open(path, "rt")
    header = json.loads(fh.readline())
    if header.get("kind") != "wl-global-p2-cover" or header.get("p") != 2:
        raise SystemExit(f"bad header kind/p: {header.get('kind')!r}")
    if header.get("order") != "b1,g1,b2,g2":
        raise SystemExit(f"unexpected variable order {header.get('order')!r}")
    if header.get("d") != D:
        raise SystemExit(f"expected d={D}, header has d={header.get('d')!r}")
    d = D
    E = d - 1
    b = float.fromhex(header["b_hex"])
    dom = [(float.fromhex(x), float.fromhex(y))
           for x, y in header["domain_hex"]]
    if dom != [tuple(t) for t in expected_domain()]:
        raise SystemExit("header domain is not the reduced fundamental domain")
    print(f"header: d={d}  b={b!r}  domain OK (reduced fundamental domain)")

    set_dps(wit_dps)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    wh = header.get("witness_hex")
    if not wh:
        raise SystemExit("header has no witness_hex")
    wit = [IVC.const(iv.mpf([float.fromhex(h), float.fromhex(h)]))
           for h in wh]
    fw = f2_value(E, wit, I)
    lo = inf(fw)
    if lo < b:
        raise SystemExit(f"witness check failed: f_2 >= {lo!r} < b={b!r}")
    # E-G12: report the margin from the interval endpoint at working
    # precision; the outward-rounded binary64 endpoint prints 0 when b is
    # exactly that endpoint, misreading as "certified only up to equality".
    mp.dps = wit_dps + 10
    marg = mp.mpf(fw.a) - mp.mpf(b)
    print(f"witness: b = {b!r} <= f_2(witness) >= {lo!r} "
          f"(interval lower endpoint - b = {mp.nstr(marg, 6)}, "
          f"iv.dps={wit_dps})")

    set_dps(dps)
    I = iv.mpc(iv.mpf([0, 0]), iv.mpf([1, 1]))
    stack = [tuple(dom)]
    counts = {"N": 0, "P n": 0, "P c": 0, "S": 0}
    checked = {"P": 0, "S": 0}
    modes = {"n": 0, "c": 0}
    worst_safe = -math.inf
    p_index = 0
    nline = 1
    t0 = time.time()
    for line in fh:
        nline += 1
        # E-G13: without a heartbeat a third party cannot tell a multi-hour
        # full pass from a hang.
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
            i = p_index
            p_index += 1
            if sel is not None and i not in sel:
                continue
            mr = [enclosing_ball(x, y) for x, y in box]
            ok, how, val = check_prune(E, mr, b, I)
            if not ok:
                raise SystemExit(f"line {nline}: P leaf FAILS, sup f_2 "
                                 f"<= {val!r} not < b={b!r}, box={box}")
            checked["P"] += 1
            modes[how] += 1
        elif head == "S":
            counts["S"] += 1
            mr = [enclosing_ball(x, y) for x, y in box]
            ok, s = check_safe(d, E, mr, I)
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
    if sel is not None:
        print("  NOTE: --sample -- the tiling is fully verified, the P-leaf "
              "arithmetic is a sample")
    if counts["S"]:
        c = -worst_safe
        print(f"\n  c = {c!r}   (= -max certified sup delta over S leaves, "
              "recomputed here)")
        print(f"THEOREM (mpmath.iv engine): every theta with "
              f"f_2({d};theta) >= {b!r} satisfies")
        print(f"  <C_e>(K_{d},{d};theta) - f_2({d};theta) <= {worst_safe!r} "
              f"< 0;  violation at least {c:.6e}.")
    else:
        print("\n  no S leaves: {f_2 >= b} is empty on the domain")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("certificate")
    ap.add_argument("--sample", type=int, default=None,
                    help="verify all S leaves and only N sampled P leaves")
    ap.add_argument("--dps", type=int, default=25,
                    help="iv.dps for box arithmetic (default 25)")
    ap.add_argument("--witness-dps", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    check(args.certificate, sample=args.sample,
          dps=args.dps, wit_dps=args.witness_dps)
    print(f"\nPASS  ({time.time()-t0:.1f}s total, mpmath.iv only, "
          "no python-flint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
