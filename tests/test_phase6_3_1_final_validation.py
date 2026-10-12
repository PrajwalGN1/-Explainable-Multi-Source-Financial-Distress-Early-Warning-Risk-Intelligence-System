"""
Phase 6.3.1 Regression Test Suite:
Final Sequential Protocol Validation, Alpha-Spending Convention Verification,
and Integer Alert Capacity Enforcement.

Tests:
1. One-sided vs Two-sided spending-function selection and Z_{0.025} mathematical necessity.
2. Independent group-sequential boundary generation matching Lan-DeMets targets.
3. Overall Type I error control under null hypothesis (overall alpha <= 0.0500).
4. Integer alert-cap enforcement using floor(cohort_size * 0.08) for prospective (cap=21, actual=23, excess=2)
   and matured (cap=21, actual=19, excess=0) cohorts.
5. Consistency between reported metrics and stored artifacts.
6. Production artifact SHA-256 invariance against protected manifest.
"""

import os
import json
import math
import pytest
import numpy as np
from scipy import stats

from src.shadow.shadow_engine import compute_sha256
from src.shadow.sequential_protocol_final_validation import (
    validate_alpha_spending_convention,
    recalculate_sequential_boundaries,
    audit_integer_alert_capacity,
    run_phase6_3_1_final_validation
)

MANIFEST_PATH = "experiments/phase6/protected_artifacts_manifest.json"
DELIVERABLE_PATH = "results/phase6_3/sequential_protocol_final_validation.json"


@pytest.fixture(scope="module")
def convention_data():
    return validate_alpha_spending_convention()


@pytest.fixture(scope="module")
def boundary_data():
    return recalculate_sequential_boundaries()


@pytest.fixture(scope="module")
def capacity_data():
    return audit_integer_alert_capacity()


@pytest.fixture(scope="module")
def deliverable_json():
    if not os.path.exists(DELIVERABLE_PATH):
        run_phase6_3_1_final_validation()
    with open(DELIVERABLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# TEST 1: ONE-SIDED VS TWO-SIDED SPENDING-FUNCTION SELECTION
# ---------------------------------------------------------------------

def test_onesided_vs_twosided_spending_function_selection(convention_data):
    """
    Verifies that Lan-DeMets sfLDOF with Z_{0.025} = 1.95996 spends exactly alpha=0.0500 at t=1.0,
    and documents why substituting Z_{0.05} = 1.64485 into 2 - 2*Phi(...) would spend alpha=0.10.
    """
    fw = convention_data["testing_framework"]
    assert fw["significance_level"] == 0.05
    assert "one-sided" in fw["significance_level_description"].lower()
    assert fw["strict_non_inferiority_margin"] == 0.05
    assert fw["practical_non_inferiority_margin"] == 0.10

    math_analysis = convention_data["mathematical_spending_convention_analysis"]
    form_a = math_analysis["formula_A_lan_demets_sfldof"]

    # Verify Formula A spends exactly 0.0500 at t=1.0
    assert pytest.approx(form_a["spending_at_horizon_t_1"], abs=1e-6) == 0.050000
    assert form_a["spending_trajectory"]["t_0_3333"] == 0.000687
    assert form_a["spending_trajectory"]["t_0_6667"] == 0.016375
    assert form_a["spending_trajectory"]["t_1_0000"] == 0.050000

    # Verify Formula B (erroneous Z_0.05 substitution) spends 0.1000
    form_b = math_analysis["formula_B_naive_z005_error_demonstration"]
    assert pytest.approx(form_b["spending_at_horizon_t_1"], abs=1e-6) == 0.100000
    assert "REJECTED" in form_b["status"]


# ---------------------------------------------------------------------
# TEST 2: INDEPENDENT GROUP-SEQUENTIAL BOUNDARY GENERATION
# ---------------------------------------------------------------------

def test_independent_group_sequential_boundary_generation(boundary_data):
    """
    Verifies recalculated Lan-DeMets sequential boundaries:
    Look 1: b1 = 3.1993, nominal p = 0.000689, cumulative alpha = 0.000687
    Look 2: b2 = 2.1405, nominal p = 0.016157, cumulative alpha = 0.016375
    Look 3: b3 = 1.6946, nominal p = 0.045071, cumulative alpha = 0.050000
    """
    looks = boundary_data["recalculated_interim_looks"]
    assert len(looks) == 3

    # Look 1
    assert looks[0]["distress_events"] == 20
    assert pytest.approx(looks[0]["information_fraction"], abs=1e-3) == 0.3333
    assert pytest.approx(looks[0]["validated_critical_z"], abs=0.01) == 3.20
    assert pytest.approx(looks[0]["target_cumulative_alpha_spent"], abs=1e-4) == 0.000687
    assert pytest.approx(looks[0]["nominal_boundary_p_value"], abs=1e-4) == 0.000689

    # Look 2
    assert looks[1]["distress_events"] == 40
    assert pytest.approx(looks[1]["information_fraction"], abs=1e-3) == 0.6667
    assert pytest.approx(looks[1]["validated_critical_z"], abs=0.01) == 2.14
    assert pytest.approx(looks[1]["target_cumulative_alpha_spent"], abs=1e-4) == 0.016375
    assert pytest.approx(looks[1]["nominal_boundary_p_value"], abs=1e-4) == 0.016157

    # Look 3
    assert looks[2]["distress_events"] == 60
    assert pytest.approx(looks[2]["information_fraction"], abs=1e-3) == 1.0000
    assert pytest.approx(looks[2]["validated_critical_z"], abs=0.01) == 1.69
    assert pytest.approx(looks[2]["target_cumulative_alpha_spent"], abs=1e-4) == 0.050000
    assert pytest.approx(looks[2]["nominal_boundary_p_value"], abs=1e-4) == 0.045071


# ---------------------------------------------------------------------
# TEST 3: OVERALL TYPE I ERROR CONTROL
# ---------------------------------------------------------------------

def test_overall_type_I_error_control(boundary_data):
    """
    Verifies that overall probability of crossing any boundary under H_0
    is strictly controlled at alpha <= 0.0500 within simulation uncertainty bounds.
    """
    meta = boundary_data["simulation_metadata"]
    assert meta["simulation_draws"] == 20_000_000
    assert meta["reproducible_seed"] == 20261012

    tot_alpha = meta["overall_simulated_type_I_error"]
    assert pytest.approx(tot_alpha, abs=0.0005) == 0.050000
    assert meta["family_wise_error_requirement_satisfied"] is True

    ci = meta["confidence_interval_95pct"]
    assert ci[0] <= 0.0500 <= ci[1]


# ---------------------------------------------------------------------
# TEST 4: INTEGER ALERT-CAP ENFORCEMENT
# ---------------------------------------------------------------------

def test_integer_alert_cap_enforcement(capacity_data):
    """
    Verifies operational maximum calculation as floor(cohort_size * 0.08):
    - Prospective: N=270, floor(270*0.08)=21, actual=23, excess=2 alerts (CAPACITY_BREACH)
    - Matured: N=264, floor(264*0.08)=21, actual=19, excess=0 alerts (COMPLIANT)
    """
    prosp = capacity_data["prospective_cohort"]
    assert prosp["cohort_size_N"] == 270
    assert prosp["maximum_permitted_alerts"] == 21
    assert prosp["candidate_alerts"] == 23
    assert prosp["excess_alerts"] == 2
    assert prosp["capacity_verdict"] == "CAPACITY_BREACH"
    assert "Excess=2 alerts" in prosp["audit_statement"]

    mature = capacity_data["matured_cohort"]
    assert mature["cohort_size_N"] == 264
    assert mature["maximum_permitted_alerts"] == 21
    assert mature["candidate_alerts"] == 19
    assert mature["excess_alerts"] == 0
    assert mature["capacity_verdict"] == "COMPLIANT"
    assert "Excess=0 alerts" in mature["audit_statement"]


# ---------------------------------------------------------------------
# TEST 5: CONSISTENCY BETWEEN REPORTED METRICS AND STORED ARTIFACTS
# ---------------------------------------------------------------------

def test_consistency_reported_metrics_and_stored_artifacts(deliverable_json):
    """Verifies deliverable JSON artifact matches schema, parameters, and results."""
    assert deliverable_json["phase"] == "6.3.1"
    assert deliverable_json["objective"] == "FINAL_SEQUENTIAL_PROTOCOL_VALIDATION"

    # Boundaries consistency
    looks = deliverable_json["sequential_boundaries_recalculated"]["recalculated_interim_looks"]
    assert looks[0]["validated_critical_z"] == 3.1993
    assert looks[1]["validated_critical_z"] == 2.1405
    assert looks[2]["validated_critical_z"] == 1.6946

    # Capacity consistency
    cap = deliverable_json["integer_alert_capacity_enforcement"]
    assert cap["prospective_cohort"]["excess_alerts"] == 2
    assert cap["matured_cohort"]["excess_alerts"] == 0


# ---------------------------------------------------------------------
# TEST 6: PRODUCTION ARTIFACT SHA-256 INVARIANCE
# ---------------------------------------------------------------------

def test_phase6_3_1_protected_artifact_sha256_invariance():
    """Confirms 100% bitwise cryptographic invariance of all approved Phase 2 artifacts."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert os.path.exists(rel_path)
        actual_hash = compute_sha256(rel_path)
        assert actual_hash == expected_hash, f"Baseline integrity breach on {rel_path}!"
