# Computer-assisted certificates for the Nishimori boundaries paper

Companion code for Yan Ru Pei, *Vertical and reentrant ferromagnetic boundaries below the
Nishimori point* (2026).

The repository contains the programs, frozen inputs and certified outputs behind every
computer-assisted statement of the paper: the accumulation certificate of the
Sherrington-Kirkpatrick model (Section 5), the certificates for p = 3 and p = 4 (Section 8), the
triple-point certificates for 3 <= p <= 25 and for large p (Section 7 and Appendix B), and the
certified inputs of the decorated-lattice theorem (Section 9 and Appendix C). Section, theorem,
table and equation numbers refer to the paper; the LaTeX label follows in parentheses, because
numbers can change between versions.

## Reproduce

The default run needs Python 3.13 with mpmath, sympy and numpy. The full run also needs scipy
and python-flint 0.9.0 (FLINT/Arb 3.6.0). Versions are pinned in [requirements.txt](requirements.txt).

```sh
uv venv --python 3.13
uv pip install -r requirements.txt
.venv/bin/python run_all.py
```

The default run checks every file against [SHA256SUMS](SHA256SUMS), re-runs the 34 light
programs (exact, symbolic and interval certificates that finish in seconds, and the replays of
the frozen covers and grids) and checks the values printed in the paper's tables against the
stored outputs. It takes about two minutes. To re-run every program, including the Arb
certificates, the (L) searches and the (F) grids:

```sh
.venv/bin/python run_all.py --full
```

The full run re-runs 78 programs and takes about 15 minutes on a recent 8-core Linux machine
(two programs of `pspin/cert-p3/second-implementation/` start a pool of 10 worker processes);
the longest steps are the fixed-point certificate `cert_fixed.py`, the Monte Carlo cross-check and
the Arb certificates. `run_all.py --list` lists all programs and `--only NAME,...` runs a subset.
The programs run in a copy of the repository under `verification-output/`. Every re-run output is
compared with its stored copy by [checks/compare_outputs.py](checks/compare_outputs.py) (run times
and peak memory are masked) and then replaced by it, so each program runs on the stored inputs.
Successful runs end with a line of the form

```text
PASS: 221 stored files verified; 34 program runs matched their stored outputs.
```

All entrypoints reject `python -O`, because assertions carry decisions. Each program can also be
run by hand from its own directory with the command recorded in [checks/jobs.py](checks/jobs.py);
programs write their outputs next to themselves, so a manual run overwrites the stored copy
(`git checkout` restores it).

## What is checked

"Exact" means rational or symbolic arithmetic; "interval" means outward-rounded interval or ball
arithmetic. The paper rounds the certified enclosures outward (lower bounds down, upper bounds up)
in its tables.

### Sections 4 and 5: SK model

| paper statement | program | stored output | arithmetic |
|---|---|---|---|
| Lemma 5.8 (`lem:accumulator`), Section 5.6 (`sec:accum-cert`): B from the certificate P of eq. (14) (`eq:cert-P`) by eq. (15) (`eq:cert-B`) equals the displayed polynomial; (V1) by Bernstein subdivision (13 boxes), (V2), (V3), the Step 1 inequality 54l^4 > (8l-2)^3 | [accumulation/accum_cert.py](accumulation/accum_cert.py) with [certificate_P_1x2.json](accumulation/certificate_P_1x2.json) | `accum_cert.out` (ends with `ALL OK`) | exact rationals, standard library |
| identities of Lemmas 5.5 (`lem:omega`) and 5.6 (`lem:reduction`), the completion of the square behind (15), the constants of Lemma 5.8 and of (V2)-(V3) | [accumulation/accum_sym.py](accumulation/accum_sym.py) | `accum_sym.out` | SymPy |
| algebra and constants of the hand proofs of Lemma 4.8 (`lem:y`) (c),(d), Corollary 4.12 (`cor:envelope`) and Remark 4.14 (`rem:spiked`) | [sk/hand_checks.py](sk/hand_checks.py) | `hand_checks.out` | SymPy; prints values, asserts nothing |

Sections 3 and 4 contain no computer-assisted statement; `sk/hand_checks.py` is a recheck of hand
algebra, not an input of any proof.

### Section 8: p = 3 and p = 4 (Theorems 8.1 `thm:p3`, 8.2 `thm:p4`, Table 1 `tab:cert`)

Both theorems follow from Theorem 6.10 (`thm:reduction`) once the predicates (W), (L) and (F) are
certified. Each predicate has a primary certificate and a second implementation written separately
(`second-implementation/`); for (W) both are in `pspin/cert-warm/`.

| predicate / row of Table 1 | primary certificate | second implementation |
|---|---|---|
| (W): q_0, m_1; Phi(q_0) >= 5.0150255e-4 (p = 3), >= 2.2785439e-4 (p = 4); xi(m_1) - I(m_1); the margin; q_0 in Q_p and m_1^2 <= 1 - 2/p exactly | [pspin/cert-warm/warm_cert_A.py](pspin/cert-warm/warm_cert_A.py) (with `enc.py`, `iv.py`, `gauss.py`) -> `warm_cert_A.log`, `warm_cert_A_n2000.json` | [warm_cert_B.py](pspin/cert-warm/warm_cert_B.py) (with `rig.py`, `oned.py`) -> `warm_cert_B.log`, `warm_cert_B.json` |
| (L): covers of [m_lo, 1] by 75 (p = 3) and 67 (p = 4) intervals with trial measures; max_i U_i <= 2.0451151094 and <= 2.7532998672 | p = 3: [landscape_cert.py](pspin/cert-p3/landscape_cert.py) -> `landscape_cover.json`; replay of the frozen cover [verify_cover.py](pspin/cert-p3/verify_cover.py) -> `verify_cover.log`. p = 4: [landscape4_cert.py](pspin/cert-p4/landscape4_cert.py) -> `landscape4_cover.json`, `landscape4.log`; replay [verify_cover4.py](pspin/cert-p4/verify_cover4.py) -> `verify_cover4.log` | [L.py](pspin/cert-p3/second-implementation/L.py) -> `L_result.json`; [L4.py](pspin/cert-p4/second-implementation/L4.py) -> `L4_result.json`, `L4.log` |
| (F): trial mu*; Par(mu*,0) >= 2.04575395212 and >= 2.75458158138; certified min D >= -6.8e-5 and >= -4.9e-6; L_F = 2.0456862122 and 2.7545766959; cold margins 5.7e-4 and 1.27e-3 | p = 3: [fw_grid.py](pspin/cert-p3/fw_grid.py) (with `sg_lb.py`, `nested.py`) -> `fw_grid_n600.json`, `fw_grid_n600.log` (73 grid points); [assemble.py](pspin/cert-p3/assemble.py) -> `assembly_result.json`, `assemble.log`. p = 4: [fw_grid4.py](pspin/cert-p4/fw_grid4.py) (with `sg_lb4t.py`, `trap4.py`, `common4.py`) -> `fw4_grid.json`, `fw4_points.json`, `fw_grid4.log` (48 points); [assemble4.py](pspin/cert-p4/assemble4.py) -> `assembly4_result.json`, `assemble4.log` | p = 3: `fconst.py`, `fcells.py` (grid from `fgrid.py`) and [final.py](pspin/cert-p3/second-implementation/final.py) -> `F_const.json`, `F_cells_result.json`, `F_grid.json`, `FINAL_result.json` (min D >= -6.6e-7, L_F = 2.0457533010, margin 6.3e-4); `compare.py` -> `F_compare.json` (enclosure containment against the primary grid). p = 4: `fconst4.py`, `fcells4.py`, [final4.py](pspin/cert-p4/second-implementation/final4.py) -> `F_const4.json`, `F_cells4_result.json`, `FINAL4_result.json` (min D >= -1.5e-7, L_F = 2.7545814375, margin 1.28e-3) |
| quadrature validation | `test_gauss.py` (both p); [validate_trap4.py](pspin/cert-p4/validate_trap4.py) -> `validate_trap4.log` | `test_oned.py`; `sanity.py` -> `sanity_out.txt`; `sanity4.py` -> `sanity4_result.json`, `sanity4.log` |

For (L), g(m_lo) < 0 is an exact rational evaluation: g(1/4) = -159/128000 for p = 3 (by hand) and
g(2/5) = -2536/234375 for p = 4 (`g_MA` in `assembly4_result.json`). The directory
[pspin/cert-p4/spot-checks/](pspin/cert-p4/spot-checks) holds float spot checks of the p = 4
certificate (finite-difference derivatives, the landscape and small identities).

`cert-p3/warm.py`, `cert-p4/warm4.py`, `check_warm4_mp.py`, `second-implementation/W.py` and `W4.py`
certify an older warm predicate (a one-step replica-symmetry-breaking trial with Delta_w < 0) that
the paper does not use; they are kept because `assemble4.py`, `final.py` and `final4.py` read their
results and report Delta_w next to the (F)/(L) decision.

### Section 7 and Appendix B: the triple point (Theorems 7.5 `thm:nm-main`, 7.13 `thm:nm-nondeg`, 7.15 `thm:nm-LP`; Table 2 `tab:nm-cert`)

| paper statement | program | stored output | arithmetic |
|---|---|---|---|
| for 3 <= p <= 25: D_p has exactly one zero Lambda_c on (0, infinity) (sign scan, bracket of width 1/32, Krawczyk box of radius 2^-200; Proposition 7.2 `prop:nm-triple`); the 31 required signs per p in three codings; all cross-coding overlaps; relative radius of the kappa_p balls <= 3e-55 (Appendix B.5 `app:nm-cert`) | [pspin/kappa-cert/kappa_cert.py](pspin/kappa-cert/kappa_cert.py) | `kappa_cert.json` (every ball), `kappa_cert.log` | Arb |
| second run at 384 bits (tolerance 2^-350, Z = 28): every check passes and all balls overlap the 256-bit run; Nishimori's printed Tables I-II are correct roundings (75/75) | `kappa_cert.py --prec 384 --tol-bits 350 --zcut 28`; [compare.py](pspin/kappa-cert/compare.py) (float inputs in `inputs/`) | `kappa_cert_p384.json`, `kappa_cert_p384.log`, `compare.json`, `compare.log` | Arb |
| det S >= 2.55e-6, B''(mu) <= -1.559, (N3) margin >= 2.062, beta_1^2/2 <= 1.38629435 < 2 log 2 for all 3 <= p <= 25, in reduced and Nishimori-line-free codings; tests of python-flint's analytic flag | [pspin/cert-check/cert_extra.py](pspin/cert-check/cert_extra.py) (imports `../kappa-cert/kappa_cert.py`) | `cert_extra.json`, `cert_extra.log` | Arb |
| (N2) through the exact identities of Lemma 7.12 (`lem:nm-closed`), and det H = -(Omega^2/2) lambda (1 - Omega B) | [pspin/cert-check/ns_identities.py](pspin/cert-check/ns_identities.py) | `ns_identities.json` | SymPy |
| Table 2: kappa_p, lambda, det S, B''(mu), (N3) margin; phi_xx >= 0.4238; Section 7.3 (`sec:nm-route`): C_FM > 0 for every p = 3..25 (minimum 1.33e-8 at p = 25) | [pspin/triple-point/nm_inputs.py](pspin/triple-point/nm_inputs.py), from the stored balls | `nm_inputs.json`, `nm_inputs.log` | mpmath intervals (90 digits) on the Arb balls |
| a third route to (N2), (N3) from the stored balls; cell bounds for Remark 7.4 | [pspin/ball-checks/ball_checks.py](pspin/ball-checks/ball_checks.py) | `ball_checks.json` | mpmath intervals |
| Remark 7.4 (`rem:nm-chen`): the printed final bound of Lemma 6 of Zhou (2024) evaluates to -0.043; interval arithmetic on a finer cover gives a positive margin of 0.04 (not used by the paper) | [pspin/zhou-lemma6/zc_lemma6.py](pspin/zhou-lemma6/zc_lemma6.py); also `cert_extra.py` (block G) and [lemma6_alg.py](pspin/cert-check/lemma6_alg.py) | `zc_lemma6.json`, `zc_lemma6.log`, `lemma6_alg.json` | intervals, SymPy |
| Appendix B.1 (`app:nm-closed`), proof of Lemma 7.12: stationarity, the Hessian of U in (m, h, q), the resulting A_FM equals eq. (`eq:nm-A`) and the first coding of `kappa_cert.py` | [pspin/afm_sympy.py](pspin/afm_sympy.py) | `afm_sympy.json` (every difference 0) | SymPy |
| Theorem 7.15 (a)-(e), including 4.97e-6, 2.4e-6, 1.35e-7, 1.38629 and 0.94616 <= N_p kappa_p <= 0.96105; the constants of Appendix B.4 (`app:nm-LP`); cell enclosures of c_0, m_2, m_4; the second certificate of (N1)-(N3) for 7 <= p <= 25 (key `"P4 table"`) | [pspin/large-order/lp_bounds.py](pspin/large-order/lp_bounds.py) | `lp_bounds.json` | mpmath intervals (40 digits) |
| the sharp constants 0.94844 <= N_p kappa_p <= 0.95880: Arb enclosures of c_0 (which contains 4 pi log 2 - pi^3/4), m_2 and m_4, read by `lp_bounds.py`; a separate Arb recomputation of the constants of Appendix B.4-B.5 | [pspin/large-order/arb-constants/lpc_arb.py](pspin/large-order/arb-constants/lpc_arb.py) | `lpc_arb.json`, `lpc_arb.log` | Arb |
| exact identities of Lemma 7.12, of Proposition 7.14 (`prop:nm-J`, including c_0 = 4 pi log 2 - pi^3/4) and of the steps of Lemma 7.3 (`lem:nm-U`), with negative controls | [pspin/large-order/lp_identities.py](pspin/large-order/lp_identities.py) | `lp_identities.json` (56 identities; 5 negative controls fail as intended) | SymPy |
| Appendix B.2 (`app:nm-U`): the computer-assisted proof of the single crossing (interval enclosures on 242 cells) | [pspin/large-order/lp_unimodal.py](pspin/large-order/lp_unimodal.py) | `lp_unimodal.json` | mpmath intervals |

`lpc_arb.py` exits with status 1 by design: 78 of its 81 checks pass, and the three that fail
(E13a, T2, T3) test rounded constants from an earlier version of the argument (5-digit cell
bounds and the table `LP_TABLE`), which the paper does not use. The paper's constants come from
E13b, E17c and T4. `run_all.py` requires exactly these three failures.

### Section 9 and Appendix C.7: the decorated lattice (Theorem 9.1 `thm:lat-main`; Tables 3 `tab:lat-inputs` and 4 `tab:lat-bands`)

| input / claim | primary certificate | re-certifications |
|---|---|---|
| (C1) pbar_H(beta) < 1/2 on (0, beta_hot], beta_hot = 30847/20000 (and the cover by 102 Arb balls that does not use a monotonicity lemma) | [lattice/window/cert_window.py](lattice/window/cert_window.py) `certify` -> `out/claims.json` (key `hot`), `out/cert_window.txt` | [recheck-1/check_claims.py](lattice/window/recheck-1/check_claims.py) (with `fxi.py`, `model.py`) -> `logs/check_claims.log`, `logs/check_claims_out.json`; [recheck-2/iv_recheck.py](lattice/window/recheck-2/iv_recheck.py) -> `iv_recheck.out` |
| (C2) 1202 cells cover [beta_1, beta_2]; on each, F_H(beta, s_j) <= W_* for one rational s_j; the whole-cell check that avoids Lemma C.12 (`lem:lat-cell`) | `out/claims.json` (key `W_plus`: `cells`, `lemma_L_free_ball_check`) | recheck-1 (own covers and monotone brackets), recheck-2 |
| (C3) 9 W_*^2 < 1 and 2P(W_*) < 1 exactly | `out/claims.json` (key `thresholds`) | recheck-1, recheck-2, [exact-checks/thresholds_exact.py](lattice/window/exact-checks/thresholds_exact.py) -> `thresholds_exact.out` |
| (C4) pbar_H(beta) < 1/2 for all beta >= beta_c = 349277/50000, including [64, infinity) | [lattice/cert/cert_arb.py](lattice/cert/cert_arb.py) -> `out/cert_arb.json`, `out/cert_arb.txt` | [lattice/cert/cert_fixed.py](lattice/cert/cert_fixed.py) -> `out/cert_fixed.txt` (integer intervals scaled by 2^480) |
| (C5) v_H(beta_N) <= V = 20529504599/10^11; 1 - h(V) >= 0.332172142; 1 - 2P(V) >= 0.91700; v_P bracket | `out/cert_arb.txt` (lines `(VP)` and `(W)`) | `out/cert_fixed.txt`; `thresholds_exact.out`; [recheck-3/nishimori_iv.py](lattice/window/recheck-3/nishimori_iv.py) -> `nishimori_iv.out` |
| (C6) the chain bound of Lemma C.11 (`lem:lat-kappa`): kappabar_H >= 0.4914640678, hence kappahat_H^2 (1 - h(V)) >= 0.0802 (Theorem 9.1(d)) | [lattice/theory/h4_cert.py](lattice/theory/h4_cert.py) -> `h4_cert_b1000_k10.out`; the exact directed bound 0.4914640678396... from [h4_cert_digits.py](lattice/theory/h4_cert_digits.py) -> `h4_cert_digits.out` | [lattice/theory/fixed-point/h4_fixed.py](lattice/theory/fixed-point/h4_fixed.py) (384-bit fixed point, `fx.py`) -> `h4_fixed_b1000_k10.out` |
| (C7) v_H(beta_N(1/1000)) <= V_1; h(V_1) <= 0.864508105378 | `out/claims.json` (key `nishimori_line.p1`) | recheck-1, recheck-2 |
| (C8) v_H(beta_N(3/2500)) <= W_N = 236586916634239/10^15; 9 W_N^2 < 1; 1 - 2P(W_N) >= 0.770991 | `out/claims.json` (key `nishimori_line.display`, p = 3/2500) | recheck-1; `recheck-3/nishimori_iv.out`; [exact-checks/nishimori_reach.py](lattice/window/exact-checks/nishimori_reach.py) -> `nishimori_reach.out` (integer intervals); `recheck-3/crossing_checks.out` (C) |
| Table 4: band bounds on E Y and beta_N-unit endpoints | `out/claims.json` (`W_plus.bands`) | [recheck-2/bands_check.py](lattice/window/recheck-2/bands_check.py) -> `bands_check.out` (exact); recheck-1 |
| Remark C.13 (`rem:lat-sharp`): the criteria fail just outside | `out/claims.json` (`sharpness_W`, `hot.sharpness`); `out/cert_arb.txt` (line `(PT)`) | recheck-1; `out/cert_fixed.txt` |
| rational beta endpoints in units of beta_N (Theorem 9.1, Tables 3-4) | `out/claims.json` (`*_over_gamma`) | [exact-checks/display_checks.py](lattice/window/exact-checks/display_checks.py) -> `display_checks.out` |

The law of Proposition 9.3 (`prop:lat-law`) is proved in the paper. It is cross-checked, not
certified, by exhaustive enumeration of small gadgets ([cert/validate_small.py](lattice/cert/validate_small.py),
[cert/validate_certs_small.py](lattice/cert/validate_certs_small.py),
[recheck-2/bruteforce_checks.py](lattice/window/recheck-2/bruteforce_checks.py),
[recheck-1/bruteforce.py](lattice/window/recheck-1/bruteforce.py)) and by numerical decimation of the
full 20032-edge gadget on sample disorders ([lattice/decimation/decimation_check.py](lattice/decimation/decimation_check.py),
worst difference 6.7e-54). The class probabilities and limits quoted in Section 9 are in
`out/cert_arb.txt`, `out/limits_exact.txt` ([limits_exact.py](lattice/cert/limits_exact.py)) and the
(C6) outputs.

### Printed values

[expected/paper_tables.json](expected/paper_tables.json) lists 202 values printed in Tables 1-4 of
the paper, each with the stored output that implies it: a lower bound must not exceed the
certified lower endpoint, an upper bound must not be below the certified upper endpoint, each
Lambda_c entry of Table 2 must be the correct rounding of every point of its Arb ball, and
rationals must agree exactly. [checks/paper_tables.py](checks/paper_tables.py) checks every entry in
exact arithmetic (`expected/paper_tables.out`).

## Trust bases

- **Exact rationals.** `accum_cert.py`, `h4_cert.py`, the exact parts of `nishimori_reach.py`,
  `bands_check.py` and `thresholds_exact.py`, and all rational side conditions use Python integers
  and `fractions.Fraction` (standard library).
- **SymPy.** `accum_sym.py`, `afm_sympy.py`, `ns_identities.py`, `lemma6_alg.py`, `lp_identities.py`
  and the exact parts of `zc_lemma6.py` trust SymPy's simplification to zero and its exact root
  counting.
- **FLINT/Arb.** `kappa_cert.py`, `compare.py`, `cert_extra.py`, `lpc_arb.py`, `cert_arb.py`,
  `cert_window.py certify` and `validate_certs_small.py` trust FLINT/Arb 3.6.0 through the python-flint
  0.9.0 binary wheel: ball arithmetic, elementary functions, the error bounds of
  `acb_calc_integrate`, and Arb's guarantee that a printed ball contains the internal ball. The
  forwarding of the `analytic` flag was read in the python-flint source and is tested directly
  (`cert_extra.py`, block W). The lemmas L0-L3 and the growth bounds of the integrands are proved in
  Appendix B.5. Three codings of the formulas guard against transcription errors above about 1e-60.
- **mpmath interval arithmetic.** `lp_bounds.py` (40 digits), `lp_unimodal.py`, `nm_inputs.py` and
  `ball_checks.py` (90 digits, on the stored Arb balls), the second implementation for p = 3 and
  p = 4 (80-140 bits), `iv_recheck.py` (45 digits) and `check_claims.py` (256 bits) trust mpmath's
  directed rounding of `exp`, `log`, `sqrt` and `cos`.
- **IEEE double intervals.** The primary p = 3 and p = 4 certificates (`iv.py`) round every
  operation outward by one unit in the last place and widen `exp` and `log` by three units, which
  assumes that the platform libm errs by less than two units. Integrals use composite Simpson rules
  with order-4 interval Taylor remainders and analytic Gaussian tails; the p = 4 (F) certificate uses
  the trapezoid rule with the strip bound of Trefethen and Weideman (SIAM Rev. 2014, Thm 5.1) and the
  a priori bound gamma_{n-1} sum |x_i| on floating-point sums. The second implementation bounds its
  floating-point convolutions by Higham's forward error bound.
- **Fixed-point integers.** `cert_fixed.py` (integers scaled by 2^480), `fxi.py` (224 bits) and
  `fx.py` (384 bits) implement directed rounding on Python integers.

None of these arithmetic engines has been formally verified. The chains of the lattice inputs use
different engines. The analytic arguments, the probability and the infinite-volume steps are in
the paper; the programs certify the finite computations that the paper states.

## Floating-point evidence

Several programs evaluate quantities in floating point without error bounds: the float
comparisons of `compare.py` against `inputs/`, `endtoend_float.py` (Section 7.6, quoted for
orientation only), the spot checks in `pspin/cert-p4/spot-checks/`, `sanity*.py`, `test_*.py`,
`validate_trap4.py`, `validate_small.py`, `mc_crosscheck.py`, `decimation_check.py`,
`bruteforce*.py`, parts (A), (B), (D) of `crossing_checks.py`, part (5) of `nishimori_reach.py`,
`w_float_classform.py` and the `verify` and `lipcheck` modes
of `cert_window.py`. These files certify nothing. They are cross-checks of the certified values
and of the formulas coded in the certificates.

## Labels in program output

Some programs name the results they check by short labels. In the paper these are:

| label | paper |
|---|---|
| Theorem LP, LP(a)-(e) | Theorem 7.15 (`thm:nm-LP`) (a)-(e) |
| Theorem LP' | the second certificate of (N1)-(N3) for 7 <= p <= 25 (Appendix B.5) |
| Lemma U (in `pspin/`) | Lemma 7.3 (`lem:nm-U`) |
| Proposition J | Proposition 7.14 (`prop:nm-J`) |
| lemmas L0-L3 | the lemmas of Appendix B.5 |
| Lemma L | the samplewise bound on the beta-derivative of K_H behind the cell rule of Lemma C.12 (`lem:lat-cell`) |
| Lemma D | Lemma 9.10 (`lem:lat-degradation`) |
| Lemma H | a hot-side monotonicity criterion that the paper does not use; (C1) rests on the covers labelled `Lemma-H-free` |
| builder, generator | the primary certificate (as seen from the second implementation and the spot checks) |
| Route A | the uniqueness argument of Appendix B.5 for the zero of $D_p$ when $p\ge 26$: $Q$ is strictly increasing on $[35,\infty)$ and below $26$ on $(0,35]$ |

In the lattice programs, kappa and kappa_2 are the paper's chi and chi_2, m_c its y_c, P_c its
alpha_c, and 1 - 2p its vartheta; `cert_arb.py` numbers the classes (2, 1, 3) of Proposition 9.3 as
(1, 2, 3).

## Stored outputs and run environments

The stored outputs were produced with CPython 3.13.5, mpmath 1.3.0, sympy 1.14.0, numpy 2.2.6,
scipy 1.16.0 and python-flint 0.9.0, on Linux for the Arb programs and `check_claims.py`, and on
macOS or Linux for the others. Time stamps in the JSON files are UTC. Some outputs
were regenerated in September 2026 after editorial changes to the labels they print; every
certified number is unchanged. The JSON files of the Arb programs embed time stamps, so their
hashes change on a re-run; the balls do not.

Several programs use floating point (numpy and the platform libm) inside rigorous enclosures, so
their last digits can differ between platforms. `run_all.py` compares such outputs to a relative
tolerance of 1e-9 (1e-6 for floating-point evidence); every exact and Arb output must agree digit
for digit. The (L) searches are optimizations; their covers are compared for information only,
and the certificate is the replay of the frozen cover.

## Layout

```text
accumulation/                Section 5.6: certificate P and its exact check
sk/                          Section 4: SymPy recheck of hand algebra
pspin/cert-warm/             Section 8, predicate (W): primary and second implementation
pspin/cert-p3/, cert-p4/     Section 8, predicates (L) and (F); second-implementation/; p = 4 spot-checks/
pspin/kappa-cert/            Appendix B.5: Arb certificate of Lambda_c and the signs, 3 <= p <= 25
pspin/cert-check/            Appendix B.5: remaining conditions, python-flint tests, exact identities
pspin/triple-point/          Table 2 and C_FM > 0 from the stored balls
pspin/ball-checks/           a third route to (N2), (N3); cell bounds for Remark 7.4
pspin/zhou-lemma6/           Remark 7.4
pspin/afm_sympy.py           Appendix B.1: steps of the proof of Lemma 7.12
pspin/large-order/           Theorem 7.15, Appendix B.2-B.4; arb-constants/: Arb enclosures of c_0, m_2, m_4
pspin/float-evidence/        Section 7.6 (floating point, orientation only)
lattice/cert/                inputs (C4), (C5): Arb primary and fixed-point second implementation
lattice/window/              inputs (C1)-(C3), (C7), (C8); recheck-1/, recheck-2/, recheck-3/, exact-checks/
lattice/theory/              input (C6); fixed-point/: second implementation
lattice/decimation/          decimation of the full gadget (evidence for Proposition 9.3)
checks/                      job table, output comparison and printed-value check used by run_all.py
expected/                    values printed in the paper's tables and their sources
```

## Cite

Use [CITATION.cff](CITATION.cff) together with the commit hash recorded in the paper. This is a
computational companion to the paper, not a machine-checked formal proof.
