This repository contains the main **Theorem** (§2–§4) and **Lemma** (§5-§7):
machine-checked interval-arithmetic certificates that cover the entire parameter
space for $p = 2$ QAOA on $K_{9,9}$

For the Theorem: The certified tree threshold is **b = 0.6391275408952286**
(`0x1.473bb99c3c5eap-1`), and every parameter combination with tree value
$\geq b$ is certified to perform worse on $K_{9,9}$ by at least
**c = 0.002920916205895164** (`0x1.7ed9af28483cfp-9`). The certificate is a
**3,237,923**-node binary box tree, machine-verified by an independent checker.

For the Lemma: The certified tree upper bound is **u = 0.6391276594250365**
(`0x1.473bbd96662c7p-1`), and a parameter witness is certified to perform
strictly better than $u$ on $K_{9,9}$. The certidicate is a derivative of the
above Theorem certificate.


---

## 1. Reader's map

| section | content |
|---|---|
| §2 | Theorem |
| §3 | Theorem: Proof shape |
| §4 | Theorem: Verification record and reproduction |
| §5 | Lemma |
| §6 | Lemma: Proof shape |
| §7 | Lemma: Reproduction |
| §8 | File manifest |


---

## 2. Theorem (= article Theorem `theorem:Counterexample`)

Let **b := `0x1.473bb99c3c5eap-1` = 0.6391275408952286** and
**c := `0x1.7ed9af28483cfp-9` = 0.002920916205895164**. Then
  **i)** $b \leq f_{2,9}^{\text{tree},*}$ and
  **ii)** for every $(\beta, \gamma) \in \mathbb{R}^{4}$
with $f_{2,9}^{\text{tree}}(\beta, \gamma) \geq b$ it is $f_{2,9}^{K}(\beta, \gamma) − f_{2,9}^{\text{tree}}(\beta, \gamma) \leq −c$.


---

## 3. Theorem: Proof shape

**Provenance of b.** b is the outward-rounded lower endpoint of an arb point enclosure of
$f_{2,9}^{\text{tree},*}$ at the witness angles below and verified by the certifier.

**Witness point for part i).** $v = (\beta_{1}, \gamma_{1}, \beta_{2}, \gamma_{2})$, exact binary64 hex:

```
-0x1.04f3d4e04111ap-1, -0x1.0b184cde2ef9ap-2, -0x1.6f1f7f7dd666ep+1,  0x1.56f3f40791132p+1
```

$f_{2,9}^{\text{tree}}(v) − b$ = **+1.5810285419253478e-16** (= +1.42 ULP), rigorously positive in two independent engines.

**Box-tree covering for part ii).** a **3,237,923-node** binary box tree on $D = [-\pi/4, \pi/4] \times [0, \pi] \times [-\pi/4, \pi/4] \times [-\pi, \pi]$, every leaf discharged as:
- **P** ($\sup f_{2,9}^{\text{tree}} < b$ on the leaf; natural ball evaluation or centred form with AD gradient
  enclosures) — **1,605,938** leaves, split as 298,679 "P n" (natural bound sufficed) and
  1,307,259 "P c" (needed the centred/gradient form);
- **S** ($\sup δ < 0$ on the leaf, by the centred form) — **13,024** leaves;

with **c = −max_S sup δ**, formed outward at every step. 1,618,961 internal nodes + 1,618,962 leaves = 3,237,923.


---

## 4. Theorem: Verification record and reproduction

| leg | engine | result |
|---|---|---|
| Certifier `certify_theorem.py` | arb, 96 bits | 3,237,923 boxes |
| Checker `check_theorem.py` | mpmath.iv | full unsampled pass, all 1,618,962 leaves |

### Reproduce

```
python3 certify_theorem.py                      # produces certificate, ~420 s
python3 check_theorem.py certificate.jsonl.gz   # ~2 h, heartbeat every 250k records
```

---

## 5. Lemma (= article Lemma `lemma:NoCounterexample`)

Let **u := `0x1.473bbd96662c7p-1` = 0.6391276594250365**, $w = (\beta_{1}, \gamma_{1}, \beta_{2}, \gamma_{2})$ with
**beta1 := `0x1.0189541bf092fp-1` = 0.5030008586984759**, **gamma1 := `0x1.752b5bc9d2407p+1` = 2.9153856978850885**,
**beta2 := `-0x1.e0c856d2ed88ap-2` = -0.46951423323060537**, and **gamma2 := `0x1.6f478b15e56c7p+1` = 2.8693708283343287**.  Then
  **i)** $f_{2,9}^{tree,*} \leq u$ and
  **ii)** $f_{2,9}^{K}(w) > u$.


---

## 6. Lemma: Proof shape

**Box-tree covering for part i).** an additional certificate over the **S** leaves of the Theorem's certificate.
  The **P** leaves are already guaranteed to be upper-bounded in value by $b < u$, where **u = max_S sup f_{2,9}^tree**,
  formed outward at every step.

**Witness point for part ii).** evaluation of $f_{2,9}^{K}(v)$ in interval arithmetic and comparison of outward-rounded lower bound to $u$.


---

## 7. Lemma: Reproduction

```
python3 certify_lemma.py certificate.jsonl.gz   # ~3 s
python3 check_lemma.py certificate.jsonl.gz witness_result.json   # ~40 s
```

---

## 8. File manifest

**Producer and its inputs**

- `certify_theorem.py`: FLINT-based certifier (§3, §4).
- `certificate.jsonl.gz`: certificate.
- `check_theorem.py`: mpmath.iv-based checker for §3.
- `certify_lemma.py`: FLINT-based certifier (§5, §6).
- `check_lemma.py`: mpmath.iv-based checker for §5.

