This folder contains the main **Theorem** (§2–§4): a machine-checked interval-arithmetic
certificate that, at QAOA depth p = 2, degree d = 9, covers the entire parameter space

The certified tree-value threshold is **b = 0.6391275408952286** (`0x1.473bb99c3c5eap-1`),
and every parameter combination with tree value $\geq b$ is certified to perform worse on
$K_{9,9}$ by at least **c = 0.002920916205895164** (`0x1.7ed9af28483cfp-9`). The certificate
is a **3,237,923**-node binary box tree, machine-verified by an independent checker.


---

## 1. Reader's map

| section | content |
|---|---|
| §2 | Theorem |
| §3 | Proof shape |
| §4 | Verification record and exact reproduction commands |
| §5 | File manifest |

---

## 2. Theorem (= article Theorem `theorem:Counterexample`)

Let **b := `0x1.473bb99c3c5eap-1` = 0.6391275408952286** and **c := `0x1.7ed9af28483cfp-9` = 0.002920916205895164**. Then
&nbsp;&nbsp;**(i)** $b ≤ f_{2,9}^{tree,*}$, and
&nbsp;&nbsp;**(ii)** for every $(\bm{\beta}, \bm{\gamma}) \in \mathbb{R}^{4}$ with $f_{2,9}^{tree}(\bm{\beta}, \bm{\gamma}) ≥ b$ it is $f_{2,9}^{K}(\bm{\beta}, \bm{\gamma}) − f_{2,9}^{tree}(\bm{\beta}, \bm{\gamma}) \leq −c$.


**Provenance of b.** b is the outward-rounded lower endpoint of an arb point enclosure of
$f_{2,9}^{\text{tree},*}$ at the witness angles below and verified by the certifier.

**Witness point for part (i)**, $w = (\beta_{1}, \gamma_{1}, \beta_{2}, \gamma_{2})$, exact binary64 hex:

```
-0x1.04f3d4e04111ap-1, -0x1.0b184cde2ef9ap-2, -0x1.6f1f7f7dd666ep+1,  0x1.56f3f40791132p+1
```

$f_{2,9}^{\text{tree}}(w) − b$ = **+1.5810285419253478e-16** (= +1.42 ULP), rigorously positive in two independent engines.


---

## 3. Proof shape

**Box-tree covering.** (ii) is a covering certificate: a **3,237,923-node** binary box tree on $D = [-\pi/4, \pi/4] \times [0, \pi] \times [-\pi/4, \pi/4] \times [-\pi, \pi]$, every leaf discharged as:
- **P** ($\sup f_{2,9}^{\text{tree}} < b$ on the leaf; natural ball evaluation or centred form with AD gradient
  enclosures) — **1,605,938** leaves, split as 298,679 "P n" (natural bound sufficed) and
  1,307,259 "P c" (needed the centred/gradient form);
- **S** ($\sup δ < 0$ on the leaf, by the centred form) — **13,024** leaves;

with **c = −max_S sup δ**, formed outward at every step. 1,618,961 internal nodes +
1,618,962 leaves = 3,237,923.


---

## 4. Verification record and reproduction

| leg | engine | result |
|---|---|---|
| Certifier `certify.py` | arb, 96 bits | 3,237,923 boxes, 418 s, deterministic single-core |
| Checker `check.py` | mpmath.iv, second interval library, no FLINT | **PASS**, full unsampled pass, all 1,618,962 leaves, 7,082.6 s |

### Reproduce

```
python3 certify.py                      # produces certificate, ~420 s
python3 check.py certificate.jsonl.gz   # ~2 h, heartbeat every 250k records
```

---

## 5. File manifest

**Producer and its inputs**

- `certify.py`: FLINT-based certifier (§3, §4).
- `certificate.jsonl.gz`: certificate.
- `check.py`: mpmath.iv-based checker

