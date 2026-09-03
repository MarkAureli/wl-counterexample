"""Certified ball evaluation of f_{9,2}^{tree}.

FLINT/arb ball arithmetic over Marwaha's exact p=2 girth>5 per-edge value
(arXiv:2101.05513v2 Sec. 2), which on a d-regular graph of girth > 5 IS the
depth-2 tree value.

Rigour model
------------
* every arithmetic step is an arb/acb ball operation: FLINT rounds the
  midpoint and inflates the radius so the true value is always enclosed;
* transcendental constants (pi) come from `arb.pi()` at working precision,
  never from float math;
* box endpoints are exact binary64 (`arb(float)` is exact; `arb('0.4..')`
  is the decimal-string trap and is never used);
* radii are read with `.rad()`, never from `.str(n)`.
* every float extracted from a ball is bumped outward with `math.nextafter`.

Inclusion function: natural ball evaluation, intersected with a first-order
mean-value (centred) form  f(X) <= f(c) + sum_i |df/dx_i(X)| * rad_i , with
the gradient enclosure produced by forward-mode automatic differentiation in
ball arithmetic.
"""

import math

from flint import ctx, acb

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
            return Dual(self.v + o.v,
                        tuple(a + b for a, b in zip(self.d, o.d)))
        return Dual(self.v + o, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, tuple(-a for a in self.d))

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v - o.v,
                        tuple(a - b for a, b in zip(self.d, o.d)))
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
# Marwaha's closed form, generic over acb and Dual
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


def f2_form(E, b1, g1, b2, g2, I):
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
    """Natural acb ball evaluation. Returns arb (real part)."""
    return f2_form(E, x[0], x[1], x[2], x[3], acb(0, 1)).real


def f2_dual(E, x):
    """Value + gradient over a box.  Returns (arb value, 4 arb partials)."""
    v = f2_form(E, x[0], x[1], x[2], x[3], acb(0, 1))
    return v.v.real, tuple(a.real for a in v.d)


def _set_prec(prec):
    global _Z
    ctx.prec = prec
    _Z = acb(0)


# --------------------------------------------------------------------------
# reduced search domain
# --------------------------------------------------------------------------


def domain():
    """Reduced fundamental domain."""
    pi = math.nextafter(math.pi, math.inf)          # > pi
    qpi = math.nextafter(math.pi / 4, math.inf)     # > pi/4
    return [(-qpi, qpi), (0.0, pi), (-qpi, qpi), (-pi, pi)]

