"""Ball-input K_{d,d} per-edge engine, value and gradient, for wide boxes.

  ce_ball(d, b1, g1, b2, g2, prec)   -> arb enclosure of f_{2,9}^K
  ce_dual(d, x, prec)                -> arb value + 4 partials over a box

State lives in the (d+1)x(d+1) two-sided symmetric (Dicke) subspace;
Natural ball evaluation amplifies input radii too much on wide boxes, so
`ce_dual` supplies a gradient enclosure for the centred form
    sup delta(B) <= delta(mid) + sum_i sup|d delta/d x_i (B)| * rad_i.

psi[p][q] stays symmetric throughout (Cm and the mixer Mx are applied
identically to both indices), so only the upper triangle p<=q is stored;
the `mid` contraction still needs the full DxD psi, but `new` and the
final readout sum only touch the upper triangle.
"""

from math import comb

from flint import ctx, arb, acb

import f2_ub as M  # Dual class, _set_prec


def ce_ball(d, b1, g1, b2, g2, prec):
    """Enclosure of f_{2,9}^K at p=2; angles are arb balls, kept wide."""
    ctx.prec = prec
    D = d + 1
    binom = [arb(comb(d, p)) for p in range(D)]
    sq = [x.sqrt() for x in binom]
    Cm = [[d * (p + q) - 2 * p * q for q in range(D)] for p in range(D)]
    two_pow = arb(2) ** arb(d)

    psi = [[None] * D for _ in range(D)]
    for p in range(D):
        for q in range(p, D):
            psi[p][q] = acb(sq[p] * sq[q] / two_pow, arb(0))

    def get(p, q):
        return psi[p][q] if p <= q else psi[q][p]

    for beta, gamma in ((b1, g1), (b2, g2)):
        for p in range(D):
            for q in range(p, D):
                ph = acb(arb(0), -gamma * Cm[p][q]).exp()
                psi[p][q] = psi[p][q] * ph
        c, s = beta.cos(), beta.sin()
        ms = acb(arb(0), -s)
        Mx = [[acb(0)] * D for _ in range(D)]
        for pp in range(D):
            for p in range(D):
                acc = acb(0)
                for k in range(max(0, p - pp), min(d - pp, p) + 1):
                    coef = arb(comb(d - pp, k) * comb(pp, p - k))
                    acc = acc + acb(coef) * acb(c) ** (d + p - pp - 2 * k) \
                        * ms ** (2 * k + pp - p)
                Mx[pp][p] = acc * (sq[pp] / sq[p])

        mid = [[acb(0)] * D for _ in range(D)]
        for i in range(D):
            for j in range(D):
                acc = acb(0)
                for t in range(D):
                    acc = acc + Mx[i][t] * get(t, j)
                mid[i][j] = acc

        new = [[None] * D for _ in range(D)]
        for i in range(D):
            for j in range(i, D):
                acc = acb(0)
                for t in range(D):
                    acc = acc + mid[i][t] * Mx[j][t]
                new[i][j] = acc
        psi = new

    tot = arb(0)
    for p in range(D):
        z = psi[p][p]
        tot = tot + (z.real ** 2 + z.imag ** 2) * arb(Cm[p][p])
        for q in range(p + 1, D):
            z = psi[p][q]
            tot = tot + 2 * (z.real ** 2 + z.imag ** 2) * arb(Cm[p][q])
    return tot / arb(d ** 2)


def ce_dual(d, x, prec):
    """(value, gradient) of f_{2,9}^K over a box."""
    ctx.prec = prec
    betas, gammas = (x[0], x[2]), (x[1], x[3])
    D = d + 1
    I = acb(0, 1)
    binom = [arb(comb(d, p)) for p in range(D)]
    sq = [x.sqrt() for x in binom]
    Cm = [[d * (p + q) - 2 * p * q for q in range(D)] for p in range(D)]
    two_pow = arb(2) ** arb(d)

    psi = [[None] * D for _ in range(D)]
    for p in range(D):
        for q in range(p, D):
            psi[p][q] = M.Dual.const(sq[p] * sq[q] / two_pow)

    def get(p, q):
        return psi[p][q] if p <= q else psi[q][p]

    for beta, gamma in zip(betas, gammas):
        for p in range(D):
            for q in range(p, D):
                u = gamma * Cm[p][q]
                psi[p][q] = psi[p][q] * (u.cos() - I * u.sin())
        c, s = beta.cos(), beta.sin()
        ms = (-I) * s
        Mx = [[None] * D for _ in range(D)]
        for pp in range(D):
            for p in range(D):
                acc = M.Dual.const(0)
                for k in range(max(0, p - pp), min(d - pp, p) + 1):
                    coef = arb(comb(d - pp, k) * comb(pp, p - k))
                    acc = acc + (c ** (d + p - pp - 2 * k)) \
                        * (ms ** (2 * k + pp - p)) * coef
                Mx[pp][p] = acc * (sq[pp] / sq[p])

        mid = [[None] * D for _ in range(D)]
        for i in range(D):
            for j in range(D):
                acc = M.Dual.const(0)
                for t in range(D):
                    acc = acc + Mx[i][t] * get(t, j)
                mid[i][j] = acc

        new = [[None] * D for _ in range(D)]
        for i in range(D):
            for j in range(i, D):
                acc = M.Dual.const(0)
                for t in range(D):
                    acc = acc + mid[i][t] * Mx[j][t]
                new[i][j] = acc
        psi = new

    tot_v = arb(0)
    tot_d = [arb(0)] * 4
    for p in range(D):
        z = psi[p][p]
        zv = z.v
        w = arb(Cm[p][p])
        tot_v = tot_v + (zv.real ** 2 + zv.imag ** 2) * w
        for i in range(4):
            di = z.d[i]
            tot_d[i] = tot_d[i] + 2 * (zv.real * di.real + zv.imag * di.imag) * w
        for q in range(p + 1, D):
            z = psi[p][q]
            zv = z.v
            w = arb(Cm[p][q])
            tot_v = tot_v + 2 * (zv.real ** 2 + zv.imag ** 2) * w
            for i in range(4):
                di = z.d[i]
                tot_d[i] = tot_d[i] \
                    + 4 * (zv.real * di.real + zv.imag * di.imag) * w
    dd2 = arb(d ** 2)
    return tot_v / dd2, tuple(t / dd2 for t in tot_d)

