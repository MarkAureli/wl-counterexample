"""Certified upper bounds on f_2^tree(d) = max over (g1,g2,b1,b2) in [-pi,pi]^4.

Interval branch-and-bound over the 4-torus with FLINT/arb ball arithmetic.
The scalar bounded is Marwaha's exact p=2 girth>5 per-edge value
(arXiv:2101.05513v2 Sec. 2), which on a d-regular graph of girth > 5 IS the
depth-2 tree value; that the closed form equals f_2^tree(d) is a proven
symbolic Fourier-coefficient identity (README.md §5), so it is used as a
*definition*, not an approximation needing separate numerical validation.

Rigour model
------------
* every arithmetic step is an arb/acb ball operation: FLINT rounds the
  midpoint and inflates the radius so the true value is always enclosed;
* transcendental constants (pi) come from `arb.pi()` at working precision,
  never from float math;
* box endpoints are exact binary64 (`arb(float)` is exact; `arb('0.4..')`
  is the decimal-string trap and is never used);
* radii are read with `.rad()`, never from `.str(n)` (CLAUDE.md rule 5);
* every float extracted from a ball is bumped outward with `math.nextafter`.

Inclusion function: natural ball evaluation, intersected with a first-order
mean-value (centred) form  f(X) <= f(c) + sum_i |df/dx_i(X)| * rad_i , with
the gradient enclosure produced by forward-mode automatic differentiation in
ball arithmetic.  The centred form is what makes this feasible: its
overestimate is O(w^2) near a maximiser instead of O(w).
"""

import math
import time

from flint import ctx, arb, acb

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
    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v = v
        self.d = d  # tuple of 4 acb

    # -- coercion ---------------------------------------------------------
    @staticmethod
    def const(v):
        return Dual(acb(v), (_Z, _Z, _Z, _Z))

    @staticmethod
    def var(v, i):
        d = [_Z, _Z, _Z, _Z]
        d[i] = acb(1)
        return Dual(acb(v), tuple(d))

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, tuple(a + b for a, b in zip(self.d, o.d)))
        return Dual(self.v + o, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, tuple(-a for a in self.d))

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v - o.v, tuple(a - b for a, b in zip(self.d, o.d)))
        return Dual(self.v - o, self.d)

    def __rsub__(self, o):
        return Dual(o - self.v, tuple(-a for a in self.d))

    def __mul__(self, o):
        if isinstance(o, Dual):
            u, w = self.v, o.v
            return Dual(u * w, tuple(a * w + u * b
                                     for a, b in zip(self.d, o.d)))
        return Dual(self.v * o, tuple(a * o for a in self.d))

    __rmul__ = __mul__

    def __truediv__(self, o):  # only by scalars here
        return Dual(self.v / o, tuple(a / o for a in self.d))

    def __pow__(self, n):
        n = int(n)
        if n == 0:
            return Dual.const(1)
        if n == 1:
            return self
        pm1 = self.v ** (n - 1)
        v = pm1 * self.v
        c = n * pm1
        return Dual(v, tuple(c * a for a in self.d))

    def cos(self):
        s, c = acb_sin_cos(self.v)
        return Dual(c, tuple(-s * a for a in self.d))

    def sin(self):
        s, c = acb_sin_cos(self.v)
        return Dual(s, tuple(c * a for a in self.d))


def acb_sin_cos(x):
    return x.sin(), x.cos()


# --------------------------------------------------------------------------
# Marwaha's closed form, written once, generic over acb and Dual
# --------------------------------------------------------------------------
#
#   f_2 = 1/2 + c^2 r t y^E z - (c s/2) alpha - (s^2 t z/4) kappa
#   c = cos 2b2, m = cos g2, r = cos 2b1, y = cos g1
#   s = sin 2b2, n = sin g2, t = sin 2b1, z = sin g1,   E = d-1
#   A = (m y - n r z)^E,  B = (m y + n r z)^E
#   P = (m + i n t y^E z)^E,  Q = (m - i n t y^E z)^E
#   kappa = ((1+r) A - (1-r) B) (P + Q)
#   alpha = (1+r)(-m r z - n y) A + (1-r)(m r z - n y) B
#           + t ((m t y^E z + i n) P + (m t y^E z - i n) Q)


def f2_form(E, g1, b1, g2, b2, I):
    """Generic; g1,b1,g2,b2 are acb balls or Duals, I is the acb unit imaginary."""
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


def f2_acb(E, x):
    """Natural ball evaluation.  x = 4 acb balls.  Returns arb (real part)."""
    return f2_form(E, x[0], x[2], x[1], x[3], acb(0, 1)).real


def f2_dual(E, x):
    """Value + gradient over a box.  Returns (arb value, 4 arb partials)."""
    v = f2_form(E, x[0], x[2], x[1], x[3], acb(0, 1))
    return v.v.real, tuple(a.real for a in v.d)


# variable order used everywhere: (g1, g2, b1, b2)


def _set_prec(prec):
    global _Z
    ctx.prec = prec
    _Z = acb(0)


# --------------------------------------------------------------------------
# branch and bound
# --------------------------------------------------------------------------


def _box_acb(mid, rad):
    return [acb(arb(m, r)) for m, r in zip(mid, rad)]


def _box_dual(mid, rad):
    return [Dual.var(arb(m, r), i) for i, (m, r) in enumerate(zip(mid, rad))]


def _point_acb(mid):
    return [acb(arb(m, 0.0)) for m in mid]


def bb(d, U, dom, prec=96, max_boxes=20_000_000, max_seconds=1800.0,
       verbose=False, log_every=500000, gap_skip=0.03):
    """Prove  max_{box in dom} f_2(d; .) <= U  by interval branch and bound.

    `dom` is a list of 4 (lo, hi) float pairs in variable order (g1,g2,b1,b2).

    Per box, in increasing cost:
      1. natural ball evaluation;
      2. forward-mode AD gradient enclosure over the box;
      3. MONOTONICITY REDUCTION -- if 0 is not in the enclosure of df/dx_i then
         f is strictly monotone in x_i on the box, so its supremum is attained
         on the corresponding face; the box is replaced by that face (exact,
         no relaxation).  This collapses dimensions and is what keeps the box
         count finite;
      4. first-order mean-value form  f(c) + sum_i sup|df/dx_i(X)| * rad_i.
    Boxes surviving all four are bisected on the coordinate with the largest
    sup|df/dx_i| * rad_i.

    Returns dict with 'ok', 'boxes', 'seconds', 'lb' (largest certified value
    seen at an evaluated point, a rigorous lower bound on the max).
    """
    _set_prec(prec)
    E = d - 1
    t0 = time.time()

    mid = [0.5 * (a + b) for a, b in dom]
    rad = [math.nextafter(0.5 * (b - a), math.inf) for a, b in dom]

    stack = [(mid, rad)]
    nboxes = 0
    nevals = 0
    nknife = 0
    ncollapse = 0
    peak = 1
    lb = -math.inf
    while stack:
        if nboxes > max_boxes or (nboxes % 2048 == 0
                                  and time.time() - t0 > max_seconds):
            return dict(ok=False, boxes=nboxes, evals=nevals,
                        seconds=time.time() - t0, lb=lb, peak_stack=peak,
                        reason="budget")
        m, r = stack.pop()
        nboxes += 1

        done = False
        contrib = list(r)
        for _sweep in range(4):
            # 1. cheap natural ball evaluation
            nevals += 1
            val = f2_acb(E, _box_acb(m, r))
            if fup(val) <= U:
                done = True
                break
            if all(ri == 0.0 for ri in r):
                lb = max(lb, flo(val))
                if flo(val) > U:
                    return dict(ok=False, boxes=nboxes, evals=nevals,
                                seconds=time.time() - t0, lb=lb,
                                peak_stack=peak, reason="U below true max",
                                witness=list(m))
                nknife += 1   # a point whose enclosure straddles U: only
                done = True   # possible if U is within ~1e-28 of f there;
                break         # counted, and makes the run not 'ok'

            # 1b. in the bulk the gradient work cannot pay for itself: if the
            # natural bound is far above U, bisect the widest side at once.
            if fup(val) - U > gap_skip and _sweep == 0:
                contrib = list(r)
                break

            # 2. gradient enclosure over the box
            _, g = f2_dual(E, _box_dual(m, r))
            gup = [float(gi.upper()) for gi in g]
            glo = [float(gi.lower()) for gi in g]

            # 3. monotonicity reduction to a face
            coll = False
            for i in range(4):
                if r[i] == 0.0:
                    continue
                if glo[i] > 0.0:
                    m[i], r[i] = m[i] + r[i], 0.0
                    coll = True
                elif gup[i] < 0.0:
                    m[i], r[i] = m[i] - r[i], 0.0
                    coll = True
            if coll:
                ncollapse += 1
                continue

            # 4. mean-value (centred) form
            fc = f2_acb(E, _point_acb(m))
            nevals += 1
            lb = max(lb, flo(fc))
            if flo(fc) > U:
                return dict(ok=False, boxes=nboxes, evals=nevals,
                            seconds=time.time() - t0, lb=lb, peak_stack=peak,
                            reason="U below true max", witness=list(m))
            # radius terms formed IN arb: the float product gi*r[i] rounds to
            # nearest and can shave the radius inward by <= 1e-20 (referee
            # defect E-G1, 2026-08-15; harmless against this file's 1e-8
            # slacks, but results recorded before this date were produced with
            # the float-product form). contrib stays float: split heuristic.
            bound = fc
            contrib = [0.0] * 4
            for i in range(4):
                gi = math.nextafter(max(abs(gup[i]), abs(glo[i])), math.inf)
                contrib[i] = gi * r[i]
                bound = bound + g[i] * arb(0.0, r[i])
            if fup(bound) <= U:
                done = True
            break

        if done:
            continue

        # 5. split
        k = max(range(4), key=lambda i: contrib[i])
        if r[k] == 0.0:
            k = max(range(4), key=lambda i: r[i])
        hr = math.nextafter(0.5 * r[k], math.inf)
        for sgn in (-1, 1):
            m2 = list(m)
            r2 = list(r)
            m2[k] = m[k] + sgn * 0.5 * r[k]
            r2[k] = hr
            stack.append((m2, r2))
        peak = max(peak, len(stack))
        if verbose and nboxes % log_every == 0:
            print(f"    d={d} boxes={nboxes} stack={len(stack)} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    return dict(ok=(nknife == 0), boxes=nboxes, evals=nevals,
                seconds=time.time() - t0, lb=lb, peak_stack=peak,
                collapses=ncollapse, knife_edge=nknife)


# --------------------------------------------------------------------------
# reduced search domain
# --------------------------------------------------------------------------


def domain(d):
    """Reduced fundamental domain, and the symmetries that justify it.

    Derived algebraically from the closed form and checked numerically at
    400-bit precision with the shifts applied exactly inside arb:

      S1  b_k -> b_k + pi/2 (k = 1,2, independently) is an exact invariance of
          f_2^tree: e^{-i(pi/2)B} = (-i)^n X^{otimes n}, which commutes with the
          cost operator and with every Z_uZ_v.  In the closed form it sends
          (r,t) -> (-r,-t) and (c,s) -> (-c,-s); c enters only as c^2, cs, s^2,
          and the (r,t) flip swaps A<->B and P<->Q, sends kappa -> -kappa with
          the prefactor t z absorbing the sign, and leaves alpha invariant.
          With the trivial pi-periodicity of cos 2b, sin 2b this gives period
          pi/2 in each beta.
      S2  (g,b) -> (-g,-b): sends z,n,t,s -> -z,-n,-t,-s, swaps P<->Q, sends
          alpha -> -alpha, which the prefactor -(cs/2) -> +(cs/2) cancels.
      S3  (even d only) g_k -> g_k + pi, k = 1,2 independently.  With E = d-1
          odd, y^E z is invariant, A,B -> -A,-B so kappa -> -kappa, cancelled by
          z -> -z in its prefactor; alpha is invariant.

    S1 confines each beta to [-pi/4, pi/4]; S2 confines gamma_1 to [0, .]; for
    even d, S3 halves both gamma ranges.

    ENDPOINTS ARE ROUNDED OUTWARD.  math.pi is the nearest binary64 to pi and
    lies BELOW it, so [0, math.pi] is strictly smaller than [0, pi] and the
    symmetry images of the reduced domain would not quite cover the torus.
    Every non-zero endpoint is therefore pushed one ulp away from zero, making
    the executed domain a strict superset of the exact fundamental domain at a
    cost of 2.2e-16 in width.
    """
    pi = math.nextafter(math.pi, math.inf)          # > pi
    hpi = math.nextafter(math.pi / 2, math.inf)     # > pi/2
    qpi = math.nextafter(math.pi / 4, math.inf)     # > pi/4
    if d % 2 == 0:
        return [(0.0, hpi), (-hpi, hpi), (-qpi, qpi), (-qpi, qpi)]
    return [(0.0, pi), (-pi, pi), (-qpi, qpi), (-qpi, qpi)]


# --------------------------------------------------------------------------
# float64 incumbent search (NOT part of the certificate; supplies U only)
# --------------------------------------------------------------------------


def f2_marwaha(D, g1, b1, g2, b2):
    """Marwaha, arXiv:2101.05513v2, Sec. 2, unnumbered Theorem: the exact p = 2
    per-edge value on a D-regular graph of girth > 5, which is the tree value.

    Transcribed from the arXiv PDF (the paper numbers neither this theorem nor
    any equation in its Section 2), NOT from any file in this repository:

        f_2 = 1/2 + c^2 r t y^{D-1} z - (cs/2) alpha - (s^2 t z/4) kappa
        c = cos 2b2, m = cos g2, r = cos 2b1, y = cos g1
        s = sin 2b2, n = sin g2, t = sin 2b1, z = sin g1
        kappa = ((1+r)(my-nrz)^{D-1} - (1-r)(my+nrz)^{D-1})
                * ((m+i n t y^{D-1} z)^{D-1} + (m - i n t y^{D-1} z)^{D-1})
        alpha = (1+r)(-mrz-ny)(my-nrz)^{D-1} + (1-r)(mrz-ny)(my+nrz)^{D-1}
                + t((m t y^{D-1} z + i n)(m + i n t y^{D-1} z)^{D-1}
                   + (m t y^{D-1} z - i n)(m - i n t y^{D-1} z)^{D-1})

    Marwaha's closed form; that it equals the depth-2 tree value is a theorem
    (README §6).  It is used for f_2^tree only, never for <C_e>(K_{d,d}):
    K_{d,d} has girth 4 and is OUTSIDE this theorem's hypothesis.
    """
    c = math.cos(2 * b2); m = math.cos(g2); r = math.cos(2 * b1); y = math.cos(g1)
    s = math.sin(2 * b2); n = math.sin(g2); t = math.sin(2 * b1); z = math.sin(g1)
    E = D - 1
    yE = y ** E
    A = (m * y - n * r * z) ** E
    B = (m * y + n * r * z) ** E
    P = (m + 1j * n * t * yE * z) ** E
    Q = (m - 1j * n * t * yE * z) ** E
    kappa = ((1 + r) * A - (1 - r) * B) * (P + Q)
    alpha = ((1 + r) * (-m * r * z - n * y) * A
             + (1 - r) * (m * r * z - n * y) * B
             + t * ((m * t * yE * z + 1j * n) * P
                    + (m * t * yE * z - 1j * n) * Q))
    return (0.5 + c * c * r * t * yE * z - (c * s / 2) * alpha
            - (s * s * t * z / 4) * kappa).real


def f2_float(d, g1, b1, g2, b2):
    return f2_marwaha(d, g1, b1, g2, b2)


def incumbent(d, seeds=(), n_starts=400, rng=None):
    import numpy as np
    from scipy.optimize import minimize
    if rng is None:
        rng = np.random.default_rng(20260815 + d)
    dom = domain(d)
    lo = np.array([a for a, _ in dom])
    hi = np.array([b for _, b in dom])

    def neg(x):
        return -f2_float(d, x[0], x[2], x[1], x[3])

    best = -math.inf
    bx = None
    starts = [np.array(s, dtype=float) for s in seeds]
    starts += [lo + (hi - lo) * rng.random(4) for _ in range(n_starts)]
    for x0 in starts:
        res = minimize(neg, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000,
                                    maxfev=20000))
        res = minimize(neg, res.x, method="Nelder-Mead",
                       options=dict(xatol=1e-13, fatol=1e-16, maxiter=20000,
                                    maxfev=20000))
        if -res.fun > best:
            best = -res.fun
            bx = res.x
    return best, [float(v) for v in bx]  # type: ignore[reportOptionalIterable]
