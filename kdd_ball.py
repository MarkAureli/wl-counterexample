"""Ball-input K_{d,d} per-edge engine, value and gradient, for wide boxes.

  ce_ball(d, b1, g1, b2, g2, prec)   -> arb enclosure of f_{2,9}^K
  ce_dual(d, x, prec)                -> (arb value, 4 arb partials) over a box

State lives in the (d+1)x(d+1) two-sided symmetric (Dicke) subspace;
Cm[a][b] = a(d-b) + (d-a)b is the total cut and <C_e> = <C>/d^2. Natural ball
evaluation amplifies input radii too much on wide boxes (phases e^{-i g Cm}
with Cm up to d^2, then two dense matrix products), so `ce_dual` supplies a
gradient enclosure for the centred form
    sup delta(B) <= delta(mid) + sum_i sup|d delta/d x_i (B)| * rad_i.

psi[a][b] stays symmetric throughout (Cm and the mixer Mx are applied
identically to both indices), so only the upper triangle a<=b is stored;
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
    binom = [arb(comb(d, a)) for a in range(D)]
    sq = [x.sqrt() for x in binom]
    Cm = [[a * (d - b) + (d - a) * b for b in range(D)] for a in range(D)]
    two_pow = arb(2) ** arb(d)

    psi = [[None] * D for _ in range(D)]
    for a in range(D):
        for b in range(a, D):
            psi[a][b] = acb(sq[a] * sq[b] / two_pow, arb(0))

    def get(a, b):
        return psi[a][b] if a <= b else psi[b][a]

    for g, b in ((g1, b1), (g2, b2)):
        for a in range(D):
            for bb in range(a, D):
                ph = acb(arb(0), -g * Cm[a][bb]).exp()
                psi[a][bb] = psi[a][bb] * ph
        c, s = b.cos(), b.sin()
        ms = acb(arb(0), -s)
        Mx = [[acb(0)] * D for _ in range(D)]
        for ib in range(D):
            for ia in range(D):
                acc = acb(0)
                for kk in range(max(0, ia - ib), min(d - ib, ia) + 1):
                    coef = arb(comb(d - ib, kk) * comb(ib, ia - kk))
                    acc = acc + acb(coef) * acb(c) ** (d - ib + ia - 2 * kk) \
                        * ms ** (ib - ia + 2 * kk)
                Mx[ib][ia] = acc * (sq[ib] / sq[ia])

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
    for a in range(D):
        z = psi[a][a]
        tot = tot + (z.real ** 2 + z.imag ** 2) * arb(Cm[a][a])
        for b in range(a + 1, D):
            z = psi[a][b]
            tot = tot + 2 * (z.real ** 2 + z.imag ** 2) * arb(Cm[a][b])
    return tot / arb(d ** 2)


def ce_dual(d, x, prec):
    """(value, gradient) of f_{2,9}^K over a box.

    x = [b1, g1, b2, g2] as f2_ub.Dual over acb balls.  Returns
    (arb value-enclosure, tuple of 4 arb partial-enclosures).
    """
    ctx.prec = prec
    gammas, betas = (x[1], x[3]), (x[0], x[2])
    D = d + 1
    I = acb(0, 1)
    binom = [arb(comb(d, a)) for a in range(D)]
    sq = [bb.sqrt() for bb in binom]
    Cm = [[a * (d - b) + (d - a) * b for b in range(D)] for a in range(D)]
    two_pow = arb(2) ** arb(d)

    psi = [[None] * D for _ in range(D)]
    for a in range(D):
        for b in range(a, D):
            psi[a][b] = M.Dual.const(sq[a] * sq[b] / two_pow)

    def get(a, b):
        return psi[a][b] if a <= b else psi[b][a]

    for g, b in zip(gammas, betas):
        for a in range(D):
            for bb in range(a, D):
                u = g * Cm[a][bb]
                psi[a][bb] = psi[a][bb] * (u.cos() - I * u.sin())
        c, s = b.cos(), b.sin()
        ms = (-I) * s
        Mx = [[None] * D for _ in range(D)]
        for ib in range(D):
            for ia in range(D):
                acc = M.Dual.const(0)
                for kk in range(max(0, ia - ib), min(d - ib, ia) + 1):
                    coef = arb(comb(d - ib, kk) * comb(ib, ia - kk))
                    acc = acc + (c ** (d - ib + ia - 2 * kk)) \
                        * (ms ** (ib - ia + 2 * kk)) * coef
                Mx[ib][ia] = acc * (sq[ib] / sq[ia])

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
    for a in range(D):
        z = psi[a][a]
        zv = z.v
        w = arb(Cm[a][a])
        tot_v = tot_v + (zv.real ** 2 + zv.imag ** 2) * w
        for q in range(4):
            dq = z.d[q]
            tot_d[q] = tot_d[q] + 2 * (zv.real * dq.real + zv.imag * dq.imag) * w
        for b in range(a + 1, D):
            z = psi[a][b]
            zv = z.v
            w = arb(Cm[a][b])
            tot_v = tot_v + 2 * (zv.real ** 2 + zv.imag ** 2) * w
            for q in range(4):
                dq = z.d[q]
                tot_d[q] = tot_d[q] \
                    + 4 * (zv.real * dq.real + zv.imag * dq.imag) * w
    dd2 = arb(d ** 2)
    return tot_v / dd2, tuple(t / dd2 for t in tot_d)

