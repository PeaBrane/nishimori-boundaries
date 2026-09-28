"""Job table of run_all.py: every program of the repository, how to run it and what to compare.

Fields
  name    identifier (run_all.py --only NAME)
  dir     working directory, relative to the repository root
  cmd     script and arguments (the interpreter running run_all.py is prepended)
  stdout  file (relative to dir) receiving standard output; stderr likewise (None: kept in the run log)
  outputs stored files (relative to dir) compared after the run; defaults to [stdout]
  mode    "light" (default run) or "full" (--full only)
  kind    "exact": numbers must agree digit for digit;
          "float": numbers must agree to the relative tolerance rtol (default 1e-9) or the absolute
                   tolerance atol (default 1e-15), which absorb platform differences in libm and BLAS;
                   the program's own decisions must agree exactly;
          "decision": outputs are reported but not compared (adaptive searches); exit status only
  rc      expected exit status (default 0)
  env     extra environment variables
  fresh   files not copied into the working copy, so that the job recomputes them (--full only)
  post    ("copy", src, dst) steps inside dir after the job
  masks   extra regular expressions for volatile text (a group named v masks only that group)
  failed_checks  for lpc_arb.py: the check names (prefixes) expected to fail
  flint   True if the program needs python-flint
"""

W4 = "pspin/cert-p4/"
P3 = "pspin/cert-p3/"
# the per-point run time that ends each point record of the (F) grids
GRID3_TIME = r'\[\n\s+"[0-9/]+",\n\s+\d+,\n(?:\s+[-+0-9.eE]+,\n){4}\s+(?P<v>[0-9.eE+-]+)\n\s+\]'
GRID4_TIME = r'\[\n\s+"[0-9/]+",\n(?:\s+[-+0-9.eE]+,\n){4}\s+(?P<v>[0-9.eE+-]+)\n\s+\]'
POINTS4_TIME = r'\[\n(?:\s+[-+0-9.eE]+,\n){4}\s+(?P<v>[0-9.eE+-]+)\n\s+\]'
# float-derived exact rationals and enclosure widths of the second implementation's cell bounds
CELL_NOISE = [r'"min_D_lower_exact": "(?P<v>[^"]*)"', r'"max_[DG]_width": (?P<v>[-+0-9.eE]+)']

JOBS = [
    # ----------------------------------------------- Section 5 and Appendix A.5 (accumulation)
    dict(name="accum-cert", dir="accumulation", cmd=["accum_cert.py", "certificate_P_1x2.json"],
         stdout="accum_cert.out", mode="light", kind="exact"),
    dict(name="accum-sym", dir="accumulation", cmd=["accum_sym.py"], stdout="accum_sym.out",
         mode="light", kind="exact"),
    # ----------------------------------------- Section 4 and Appendices A.3-A.4 (hand algebra)
    dict(name="sk-hand-checks", dir="sk", cmd=["hand_checks.py"], stdout="hand_checks.out",
         mode="light", kind="exact"),
    # ---------------------------------------------------------------- Section 8: predicate (W)
    dict(name="warm-A", dir="pspin/cert-warm", cmd=["warm_cert_A.py"], stdout="warm_cert_A.log",
         outputs=["warm_cert_A.log", "warm_cert_A_n2000.json"], mode="light", kind="float"),
    dict(name="warm-B", dir="pspin/cert-warm", cmd=["warm_cert_B.py"], stdout="warm_cert_B.log",
         outputs=["warm_cert_B.log", "warm_cert_B.json"], mode="light", kind="exact"),
    # ---------------------------------------------------------------- Section 8: p = 3, primary
    dict(name="p3-test-gauss", dir=P3, cmd=["test_gauss.py"], stdout="test_gauss.out", mode="light", kind="float"),
    dict(name="p3-L-search", dir=P3, cmd=["landscape_cert.py"], stdout="landscape_cert.log",
         outputs=["landscape_cover.json"], mode="full", kind="decision"),
    dict(name="p3-F-grid", dir=P3, cmd=["fw_grid.py", "600", "6"], stdout="fw_grid_n600.log",
         outputs=["fw_grid_n600.json"], masks=[GRID3_TIME], mode="full", kind="float"),
    dict(name="p3-F-assemble", dir=P3, cmd=["assemble.py", "fw_grid_n600.json"], stdout="assemble.log",
         outputs=["assemble.log", "assembly_result.json"], mode="light", kind="float"),
    dict(name="p3-L-replay", dir=P3, cmd=["verify_cover.py"], stdout="verify_cover.log", mode="light", kind="float"),
    dict(name="p3-W-legacy", dir=P3, cmd=["warm.py"], stdout="warm.log", outputs=["warm_result.json"],
         mode="full", kind="decision"),
    # ---------------------------------------------------------------- Section 8: p = 3, second implementation
    dict(name="p3s-W-legacy", dir=P3 + "second-implementation", cmd=["W.py"], stdout="W.log",
         outputs=["W_result.json"], mode="full", kind="exact"),
    dict(name="p3s-L", dir=P3 + "second-implementation", cmd=["L.py"], stdout="L.log",
         outputs=["L_result.json"], mode="full", kind="float"),
    dict(name="p3s-F-const", dir=P3 + "second-implementation", cmd=["fconst.py"], stdout="fconst.log",
         outputs=["F_const.json"], mode="full", kind="exact"),
    dict(name="p3s-F-cells-coarse", dir=P3 + "second-implementation", cmd=["fcells.py"], stdout="fcells_coarse.log",
         fresh=["F_grid.json"], outputs=[], mode="full", kind="float"),
    dict(name="p3s-F-cells", dir=P3 + "second-implementation", cmd=["fcells.py"], stdout="fcells_log.txt",
         env={"FTOL": "5/1000000000"}, outputs=["fcells_log.txt", "F_cells_result.json", "F_grid.json"],
         masks=CELL_NOISE, mode="full", kind="float", rtol=1e-6),
    dict(name="p3s-final", dir=P3 + "second-implementation", cmd=["final.py"], stdout="final.log",
         outputs=["FINAL_result.json"], mode="light", kind="exact"),
    dict(name="p3s-compare", dir=P3 + "second-implementation", cmd=["compare.py"], stdout="compare.log",
         outputs=["F_compare.json"], mode="full", kind="float", rtol=1e-6),
    dict(name="p3s-sanity", dir=P3 + "second-implementation", cmd=["sanity.py"], stdout="sanity_out.txt",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="p3s-test-oned", dir=P3 + "second-implementation", cmd=["test_oned.py"], stdout="test_oned.out",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    # ---------------------------------------------------------------- Section 8: p = 4, primary
    dict(name="p4-test-gauss", dir=W4, cmd=["test_gauss.py"], stdout="test_gauss.out", mode="light", kind="float"),
    dict(name="p4-W-legacy", dir=W4, cmd=["warm4.py"], stdout="warm4.log",
         outputs=["warm4.log", "warm4_result.json"], mode="full", kind="decision"),
    dict(name="p4-W-legacy-mp", dir=W4, cmd=["check_warm4_mp.py"], stdout="check_warm4_mp.log",
         mode="light", kind="exact"),
    dict(name="p4-L-search", dir=W4, cmd=["landscape4_cert.py", "2.7533"], stdout="landscape4.log",
         outputs=["landscape4.log", "landscape4_cover.json"], mode="full", kind="decision"),
    dict(name="p4-F-grid", dir=W4, cmd=["fw_grid4.py"], stdout="fw_grid4.log",
         outputs=["fw_grid4.log", "fw4_grid.json", "fw4_points.json"], fresh=["fw4_points.json"],
         masks=[GRID4_TIME, POINTS4_TIME], mode="full", kind="float"),
    dict(name="p4-F-assemble", dir=W4, cmd=["assemble4.py"], stdout="assemble4.log",
         outputs=["assemble4.log", "assembly4_result.json"], mode="light", kind="float"),
    dict(name="p4-L-replay", dir=W4, cmd=["verify_cover4.py"], stdout="verify_cover4.log", mode="light", kind="float"),
    dict(name="p4-trap-validation", dir=W4, cmd=["validate_trap4.py"], stdout="validate_trap4.log",
         mode="light", kind="float", rtol=1e-6, atol=1e-12),
    # ---------------------------------------------------------------- Section 8: p = 4, second implementation
    dict(name="p4s-W-legacy", dir=W4 + "second-implementation", cmd=["W4.py"], stdout="W4.log",
         outputs=["W4.log", "W4_result.json"], mode="full", kind="float"),
    dict(name="p4s-L", dir=W4 + "second-implementation", cmd=["L4.py"], stdout="L4.log",
         outputs=["L4.log", "L4_result.json"], mode="full", kind="float"),
    dict(name="p4s-F-const", dir=W4 + "second-implementation", cmd=["fconst4.py"], stdout="fconst4.log",
         outputs=["fconst4.log", "F_const4.json"], mode="full", kind="float"),
    dict(name="p4s-F-cells-coarse", dir=W4 + "second-implementation", cmd=["fcells4.py"], stdout="fcells4_coarse.log",
         env={"FTOL": "1/1000000"}, fresh=["F_grid4.json"], outputs=["F_cells4_result_tol1e-6.json"],
         post=[("copy", "F_cells4_result.json", "F_cells4_result_tol1e-6.json")], masks=CELL_NOISE,
         mode="full", kind="float", rtol=1e-6),
    dict(name="p4s-F-cells", dir=W4 + "second-implementation", cmd=["fcells4.py"], stdout="fcells4.log",
         env={"FTOL": "1/10000000"}, outputs=["fcells4.log", "F_cells4_result.json", "F_grid4.json"],
         masks=CELL_NOISE, mode="full", kind="float", rtol=1e-6),
    dict(name="p4s-final", dir=W4 + "second-implementation", cmd=["final4.py"], stdout="final4.log",
         outputs=["final4.log", "FINAL4_result.json"], mode="light", kind="exact"),
    dict(name="p4s-sanity", dir=W4 + "second-implementation", cmd=["sanity4.py"], stdout="sanity4.log",
         outputs=["sanity4.log", "sanity4_result.json"], mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="p4-spot-fd", dir=W4 + "spot-checks", cmd=["fd_check.py"], stdout="fd_check.log",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="p4-spot-landscape", dir=W4 + "spot-checks", cmd=["land_check.py"], stdout="land_check.log",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="p4-spot-small", dir=W4 + "spot-checks", cmd=["small_checks.py"], stdout="small_checks.out",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    # ---------------------------------------------------------------- Section 7 and Appendix B (triple point)
    dict(name="afm-identities", dir="pspin", cmd=["afm_sympy.py"], stdout="afm_sympy.json", stderr="afm_sympy.err",
         outputs=["afm_sympy.json", "afm_sympy.err"], mode="light", kind="exact"),
    dict(name="kappa-256", dir="pspin/kappa-cert", cmd=["kappa_cert.py", "--pmin", "3", "--pmax", "25", "--out",
         "kappa_cert.json"], stderr="kappa_cert.log", outputs=["kappa_cert.json", "kappa_cert.log"],
         masks=[r'"platform": "[^"]*"'], mode="full", kind="exact", flint=True),
    dict(name="kappa-384", dir="pspin/kappa-cert", cmd=["kappa_cert.py", "--pmin", "3", "--pmax", "25", "--prec",
         "384", "--tol-bits", "350", "--zcut", "28", "--out", "kappa_cert_p384.json"], stderr="kappa_cert_p384.log",
         outputs=["kappa_cert_p384.json", "kappa_cert_p384.log"], masks=[r'"platform": "[^"]*"'],
         mode="full", kind="exact", flint=True),
    dict(name="kappa-compare", dir="pspin/kappa-cert", cmd=["compare.py", "kappa_cert.json", "--cert2",
         "kappa_cert_p384.json", "--results", "inputs/results.json", "--kappa-mp", "inputs/kappa_mp_result.json",
         "--out", "compare.json"], stdout="compare.log", outputs=["compare.log", "compare.json"],
         mode="full", kind="exact", flint=True),
    dict(name="nondegeneracy-arb", dir="pspin/cert-check", cmd=["cert_extra.py"], stdout="cert_extra.json",
         stderr="cert_extra.log", outputs=["cert_extra.json", "cert_extra.log"], mode="full", kind="exact", flint=True),
    dict(name="nondegeneracy-identities", dir="pspin/cert-check", cmd=["ns_identities.py"],
         stdout="ns_identities.json", stderr="ns_identities.err", outputs=["ns_identities.json", "ns_identities.err"],
         mode="light", kind="exact"),
    dict(name="lemma6-algebra", dir="pspin/cert-check", cmd=["lemma6_alg.py"], stdout="lemma6_alg.json",
         stderr="lemma6_alg.err", outputs=["lemma6_alg.json", "lemma6_alg.err"], mode="light", kind="exact"),
    dict(name="ball-checks", dir="pspin/ball-checks", cmd=["ball_checks.py"], stdout="ball_checks.json",
         stderr="ball_checks.err", outputs=["ball_checks.json", "ball_checks.err"], mode="light", kind="float"),
    dict(name="table2-inputs", dir="pspin/triple-point", cmd=["nm_inputs.py"], stdout="nm_inputs.log",
         outputs=["nm_inputs.log", "nm_inputs.json"], mode="light", kind="exact"),
    dict(name="zhou-lemma6", dir="pspin/zhou-lemma6", cmd=["zc_lemma6.py"], stdout="zc_lemma6.log",
         outputs=["zc_lemma6.log", "zc_lemma6.json"], mode="light", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="large-order-identities", dir="pspin/large-order", cmd=["lp_identities.py"], stdout="lp_identities.json",
         stderr="lp_identities.err", outputs=["lp_identities.json", "lp_identities.err"], mode="light", kind="exact"),
    dict(name="large-order-bounds", dir="pspin/large-order", cmd=["lp_bounds.py"], stdout="lp_bounds.json",
         stderr="lp_bounds.err", outputs=["lp_bounds.json", "lp_bounds.err"], mode="light", kind="exact"),
    dict(name="large-order-crossing", dir="pspin/large-order", cmd=["lp_unimodal.py"], stdout="lp_unimodal.json",
         stderr="lp_unimodal.err", outputs=["lp_unimodal.json", "lp_unimodal.err"], mode="light", kind="exact"),
    dict(name="large-order-arb", dir="pspin/large-order/arb-constants", cmd=["lpc_arb.py", "lpc_arb.json"],
         stdout="lpc_arb.log", outputs=["lpc_arb.log", "lpc_arb.json"], rc=1, failed_checks=["E13a", "T2", "T3"],
         mode="full", kind="exact", flint=True),
    dict(name="float-evidence-p345", dir="pspin/float-evidence", cmd=["endtoend_float.py", "3", "4", "5"],
         stdout="endtoend_p345.json", mode="full", kind="float", rtol=1e-4, atol=1e-12),
    # ---------------------------------------------------------------- Section 9 and Appendix C (lattice)
    dict(name="lattice-arb", dir="lattice/cert", cmd=["cert_arb.py", "out/cert_arb.json"], stdout="out/cert_arb.txt",
         outputs=["out/cert_arb.txt", "out/cert_arb.json"], mode="full", kind="exact", flint=True),
    dict(name="lattice-fixed-point", dir="lattice/cert", cmd=["cert_fixed.py"], stdout="out/cert_fixed.txt",
         mode="full", kind="exact"),
    dict(name="lattice-limits", dir="lattice/cert", cmd=["limits_exact.py"], stdout="out/limits_exact.txt",
         mode="full", kind="exact"),
    dict(name="lattice-small-gadgets", dir="lattice/cert", cmd=["validate_small.py"], stdout="out/validate_small.txt",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="lattice-small-certs", dir="lattice/cert", cmd=["validate_certs_small.py"],
         stdout="out/validate_certs_small.txt", mode="full", kind="float", rtol=1e-6, flint=True),
    dict(name="lattice-monte-carlo", dir="lattice/cert", cmd=["mc_crosscheck.py", "4000000"],
         stdout="out/mc_crosscheck.txt", mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="window-certify", dir="lattice/window", cmd=["cert_window.py", "certify", "out/claims.json"],
         stdout="out/cert_window.txt", outputs=["out/cert_window.txt", "out/claims.json"],
         mode="full", kind="exact", flint=True),
    dict(name="window-verify", dir="lattice/window", cmd=["cert_window.py", "verify", "out/claims.json"],
         stdout="out/verify.txt", mode="full", kind="exact"),
    dict(name="window-lipcheck", dir="lattice/window", cmd=["cert_window.py", "lipcheck"], stdout="out/lipcheck.txt",
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="recheck-1", dir="lattice/window/recheck-1", cmd=["check_claims.py", "../out/claims.json",
         "logs/check_claims_out.json", "--workers", "4"], stdout="logs/check_claims.log",
         outputs=["logs/check_claims.log", "logs/check_claims_out.json"], mode="full", kind="exact"),
    dict(name="recheck-1-bruteforce", dir="lattice/window/recheck-1", cmd=["bruteforce.py"],
         stdout="logs/bruteforce.log", mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="recheck-2", dir="lattice/window/recheck-2", cmd=["iv_recheck.py", "iv_recheck.out"],
         stdout="iv_recheck.stdout", outputs=["iv_recheck.out"], mode="full", kind="exact"),
    dict(name="recheck-2-bands", dir="lattice/window/recheck-2", cmd=["bands_check.py"], stdout="bands_check.out",
         mode="light", kind="exact"),
    dict(name="recheck-2-bruteforce", dir="lattice/window/recheck-2", cmd=["bruteforce_checks.py",
         "bruteforce_checks.out"], stdout="bruteforce_checks.stdout", outputs=["bruteforce_checks.out"],
         mode="full", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="recheck-3-crossings", dir="lattice/window/recheck-3", cmd=["crossing_checks.py"],
         stdout="crossing_checks.out", mode="light", kind="float", rtol=1e-6, atol=1e-12),
    dict(name="recheck-3-interval", dir="lattice/window/recheck-3", cmd=["nishimori_iv.py"], stdout="nishimori_iv.out",
         mode="light", kind="exact"),
    dict(name="window-thresholds", dir="lattice/window/exact-checks", cmd=["thresholds_exact.py"],
         stdout="thresholds_exact.out", mode="light", kind="exact"),
    dict(name="window-nishimori-line", dir="lattice/window/exact-checks", cmd=["nishimori_reach.py"],
         stdout="nishimori_reach.out", mode="light", kind="exact"),
    dict(name="window-displayed-numbers", dir="lattice/window/exact-checks", cmd=["display_checks.py"],
         stdout="display_checks.out", mode="light", kind="exact"),
    dict(name="window-class-form-float", dir="lattice/window/exact-checks", cmd=["w_float_classform.py"],
         stdout="w_float_classform.out", mode="full", kind="exact"),
    dict(name="chain-bound", dir="lattice/theory", cmd=["h4_cert.py", "1000", "10", "9", "10000", "2.15", "2.6"],
         stdout="h4_cert_b1000_k10.out", mode="light", kind="exact"),
    dict(name="chain-bound-digits", dir="lattice/theory", cmd=["h4_cert_digits.py"], stdout="h4_cert_digits.out",
         mode="light", kind="exact"),
    dict(name="chain-bound-fixed-point", dir="lattice/theory/fixed-point", cmd=["h4_fixed.py", "1000", "10", "9",
         "10000"], stdout="h4_fixed_b1000_k10.out", mode="light", kind="exact"),
    dict(name="paper-tables", dir="expected", cmd=["../checks/paper_tables.py"], stdout="paper_tables.out",
         mode="light", kind="exact"),
    dict(name="lattice-decimation", dir="lattice/decimation", cmd=["decimation_check.py", "1000", "10", "9", "10000"],
         stdout="decimation_check_b1000_k10.out", mode="full", kind="float", rtol=1e-6, atol=1e-12),
]
