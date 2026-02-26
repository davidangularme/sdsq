#!/usr/bin/env python3
"""
CST Prediction P8b: G-Ratio Gate Energy Scaling Test
=====================================================

Tests whether E_gate ∝ g·√(E_J,A / E_J,B) on IBM Quantum hardware.

Method:
  1. Pull public calibration data from an IBM backend
  2. Extract E_J for each qubit from frequency + anharmonicity:
       E_C = |anharmonicity|
       E_J = (f01 + E_C)^2 / (8 * E_C)
  3. For each CNOT pair (i,j), compute R_ij = √(E_J,i / E_J,j)
  4. Correlate R_ij with 1/gate_length(cx, [i,j])
  5. Report Pearson r and p-value

Requirements:
  - pip install qiskit-ibm-runtime numpy scipy matplotlib
  - IBM Quantum API token (free account: https://quantum.ibm.com)

Usage:
  export IBM_QUANTUM_TOKEN="your_token_here"
  python p8b_test.py

If no token available, the script runs with SYNTHETIC demo data
to show the analysis pipeline.
"""

import numpy as np
from scipy import stats
import os
import json

# ─── Configuration ───
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "")
BACKEND_NAME = "ibm_sherbrooke"  # 127-qubit Eagle processor


def extract_ej_from_calibration(frequency_ghz, anharmonicity_ghz):
    """
    Extract Josephson energy E_J from transmon parameters.
    
    From transmon physics (Koch et al. 2007):
      f01 ≈ √(8 E_J E_C) - E_C
      anharmonicity α ≈ -E_C
    
    Therefore:
      E_C = |α|
      E_J = (f01 + E_C)^2 / (8 E_C)
    """
    e_c = abs(anharmonicity_ghz)
    if e_c < 1e-6:  # avoid division by zero
        return None
    e_j = (frequency_ghz + e_c) ** 2 / (8 * e_c)
    return e_j


def fetch_ibm_data():
    """Attempt to fetch real IBM calibration data."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        if not IBM_TOKEN:
            print("No IBM_QUANTUM_TOKEN found. Using synthetic data.")
            return None
            
        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=IBM_TOKEN
        )
        
        backend = service.backend(BACKEND_NAME)
        props = backend.properties()
        config = backend.configuration()
        
        # Extract qubit data
        qubit_data = {}
        for i in range(config.n_qubits):
            freq = props.frequency(i) / 1e9  # Hz -> GHz
            anharm = props.anharmonicity(i) / 1e9 if props.anharmonicity(i) else None
            if freq and anharm:
                e_j = extract_ej_from_calibration(freq, anharm)
                if e_j:
                    qubit_data[i] = {
                        'frequency': freq,
                        'anharmonicity': anharm,
                        'E_J': e_j,
                        'E_C': abs(anharm)
                    }
        
        # Extract CX gate data
        cx_data = []
        for gate in props.gates:
            if gate.gate == 'cx':
                q0, q1 = gate.qubits
                gate_length = gate.parameters[1].value if len(gate.parameters) > 1 else None
                gate_error = gate.parameters[0].value if len(gate.parameters) > 0 else None
                if gate_length and q0 in qubit_data and q1 in qubit_data:
                    cx_data.append({
                        'q0': q0, 'q1': q1,
                        'gate_length_ns': gate_length * 1e9,
                        'gate_error': gate_error,
                        'E_J_0': qubit_data[q0]['E_J'],
                        'E_J_1': qubit_data[q1]['E_J'],
                    })
        
        return qubit_data, cx_data
        
    except Exception as e:
        print(f"Could not fetch IBM data: {e}")
        print("Using synthetic data instead.")
        return None


def generate_synthetic_data(n_qubits=127, seed=42):
    """
    Generate realistic synthetic transmon calibration data.
    
    Based on typical IBM Eagle processor parameters:
    - f01: 4.5-5.5 GHz
    - anharmonicity: -300 to -350 MHz
    - E_J/E_C ratio: ~50
    - CX gate lengths: 200-600 ns
    """
    rng = np.random.RandomState(seed)
    
    # Generate qubit parameters
    qubit_data = {}
    for i in range(n_qubits):
        freq = rng.uniform(4.5, 5.5)  # GHz
        anharm = -rng.uniform(0.28, 0.36)  # GHz (negative)
        e_j = extract_ej_from_calibration(freq, anharm)
        qubit_data[i] = {
            'frequency': freq,
            'anharmonicity': anharm,
            'E_J': e_j,
            'E_C': abs(anharm)
        }
    
    # Generate CX gate pairs (heavy-hex topology approximation)
    cx_data = []
    edges = []
    for i in range(n_qubits - 1):
        if rng.random() < 0.4:  # sparse connectivity
            edges.append((i, i + 1))
    # Add some longer-range connections
    for i in range(0, n_qubits - 12, 12):
        edges.append((i, i + 12))
    
    for q0, q1 in edges:
        if q0 in qubit_data and q1 in qubit_data:
            e_j_ratio = np.sqrt(qubit_data[q0]['E_J'] / qubit_data[q1]['E_J'])
            
            # CST prediction: gate_speed ∝ g * √(E_J,A / E_J,B)
            # So 1/gate_length ∝ R_ij (with noise)
            g_coupling = rng.uniform(0.03, 0.07)  # GHz
            
            # Model: gate_speed = g * R_ij * scale_factor + noise
            ideal_speed = g_coupling * e_j_ratio
            noise = rng.normal(0, 0.003)
            actual_speed = ideal_speed + noise
            gate_length_ns = 1.0 / actual_speed  # ns (simplified units)
            
            # Scale to realistic range
            gate_length_ns = gate_length_ns * 30  # scale to ~200-600 ns range
            
            cx_data.append({
                'q0': q0, 'q1': q1,
                'gate_length_ns': gate_length_ns,
                'gate_error': rng.uniform(0.005, 0.03),
                'E_J_0': qubit_data[q0]['E_J'],
                'E_J_1': qubit_data[q1]['E_J'],
            })
    
    return qubit_data, cx_data


def run_p8b_analysis(qubit_data, cx_data, data_source="unknown"):
    """
    Core P8b analysis: correlate G-ratio with inverse gate length.
    
    CST predicts: E_gate ∝ g · √(E_J,A / E_J,B)
    Therefore: 1/gate_length should correlate with R_ij = √(E_J,A / E_J,B)
    """
    print("=" * 65)
    print("  CST PREDICTION P8b: G-RATIO GATE ENERGY SCALING TEST")
    print(f"  Data source: {data_source}")
    print("=" * 65)
    print()
    
    # Compute R_ij and inverse gate length for each CX pair
    r_values = []
    inv_gate = []
    pair_labels = []
    
    for cx in cx_data:
        r_ij = np.sqrt(cx['E_J_0'] / cx['E_J_1'])
        r_values.append(r_ij)
        inv_gate.append(1.0 / cx['gate_length_ns'])
        pair_labels.append(f"({cx['q0']},{cx['q1']})")
    
    r_values = np.array(r_values)
    inv_gate = np.array(inv_gate)
    
    print(f"Number of CX gate pairs analyzed: {len(r_values)}")
    print(f"R_ij = sqrt(E_J,i / E_J,j) range: [{r_values.min():.4f}, {r_values.max():.4f}]")
    print(f"1/gate_length range: [{inv_gate.min():.6f}, {inv_gate.max():.6f}] (1/ns)")
    print()
    
    # ─── Pearson correlation ───
    pearson_r, pearson_p = stats.pearsonr(r_values, inv_gate)
    
    # ─── Spearman rank correlation (robust to outliers) ───
    spearman_r, spearman_p = stats.spearmanr(r_values, inv_gate)
    
    # ─── Linear fit ───
    slope, intercept, r_sq, p_val, std_err = stats.linregress(r_values, inv_gate)
    
    print("─── CORRELATION RESULTS ───")
    print(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman r = {spearman_r:+.4f}  (p = {spearman_p:.2e})")
    print(f"  Linear fit: R^2 = {r_sq**2:.4f}, slope = {slope:.6f} +/- {std_err:.6f}")
    print()
    
    # ─── CST verdict ───
    print("─── CST P8b VERDICT ───")
    if abs(pearson_r) > 0.9:
        print(f"  *** STRONG CONCORDANCE (r = {pearson_r:.4f}) ***")
        print("  CST prediction P8b is CONCORDANT with data.")
        print("  The G-ratio scaling E_gate ~ g*sqrt(E_J,A/E_J,B) is supported.")
    elif abs(pearson_r) > 0.7:
        print(f"  ** MODERATE CORRELATION (r = {pearson_r:.4f}) **")
        print("  Suggestive but not conclusive. Confounding variables possible.")
        print("  Recommend: control for coupling g variation between pairs.")
    elif abs(pearson_r) > 0.4:
        print(f"  * WEAK CORRELATION (r = {pearson_r:.4f}) *")
        print("  Insufficient to support or falsify. Need to isolate g dependence.")
    else:
        print(f"  NO SIGNIFICANT CORRELATION (r = {pearson_r:.4f})")
        print("  CST prediction P8b is NOT SUPPORTED by this data.")
        print("  Possible explanations: g variation dominates, or prediction incorrect.")
    
    print()
    print("─── QUBIT STATISTICS ───")
    e_j_values = [q['E_J'] for q in qubit_data.values()]
    e_c_values = [q['E_C'] for q in qubit_data.values()]
    freqs = [q['frequency'] for q in qubit_data.values()]
    print(f"  Qubits with valid data: {len(qubit_data)}")
    print(f"  E_J range: [{min(e_j_values):.2f}, {max(e_j_values):.2f}] GHz")
    print(f"  E_C range: [{min(e_c_values):.4f}, {max(e_c_values):.4f}] GHz")
    print(f"  E_J/E_C ratio: [{min(e_j_values)/max(e_c_values):.1f}, {max(e_j_values)/min(e_c_values):.1f}]")
    print(f"  Frequency range: [{min(freqs):.2f}, {max(freqs):.2f}] GHz")
    
    # ─── Generate plot ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: R_ij vs 1/gate_length
        ax1 = axes[0]
        ax1.scatter(r_values, inv_gate, alpha=0.6, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
        
        # Fit line
        x_fit = np.linspace(r_values.min(), r_values.max(), 100)
        y_fit = slope * x_fit + intercept
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Linear fit: r={pearson_r:.3f}, p={pearson_p:.1e}')
        
        ax1.set_xlabel(r'$R_{ij} = \sqrt{E_{J,i} / E_{J,j}}$', fontsize=12)
        ax1.set_ylabel(r'$1 / t_{gate}$ (1/ns)', fontsize=12)
        ax1.set_title('CST Prediction P8b: G-Ratio vs Gate Speed', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: E_J distribution
        ax2 = axes[1]
        ax2.hist(e_j_values, bins=25, color='steelblue', edgecolor='navy', alpha=0.7)
        ax2.set_xlabel(r'$E_J$ (GHz)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Josephson Energy Distribution (n={len(e_j_values)} qubits)', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = '/home/claude/p8b_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
        
    except ImportError:
        print("\n  (matplotlib not available, skipping plot)")
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r_sq**2,
        'n_pairs': len(r_values),
        'n_qubits': len(qubit_data),
        'data_source': data_source
    }


def main():
    print()
    print("CST P8b Test — Configuration Space Temporality")
    print("Prediction: E_gate proportional to g * sqrt(E_J,A / E_J,B)")
    print()
    
    # Try real IBM data first
    result = fetch_ibm_data()
    
    if result is not None:
        qubit_data, cx_data = result
        data_source = f"IBM Quantum ({BACKEND_NAME})"
    else:
        print("Generating synthetic calibration data (realistic IBM Eagle parameters)...")
        print(">>> To run with REAL data: export IBM_QUANTUM_TOKEN='your_token'")
        print()
        qubit_data, cx_data = generate_synthetic_data()
        data_source = "SYNTHETIC (IBM Eagle-like parameters)"
    
    results = run_p8b_analysis(qubit_data, cx_data, data_source)
    
    # Save results
    results_path = '/home/claude/p8b_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {results_path}")
    print()
    print("─── NEXT STEPS ───")
    if "SYNTHETIC" in data_source:
        print("  1. Get free IBM Quantum account: https://quantum.ibm.com")
        print("  2. Copy your API token from the dashboard")
        print("  3. Run: export IBM_QUANTUM_TOKEN='your_token'")
        print("  4. Re-run this script for REAL results")
    else:
        print("  1. If r > 0.9: Submit result to Physical Review Letters")
        print("  2. If r < 0.5: Investigate confounding variables (g variation)")
        print("  3. Repeat on multiple backends for robustness")


if __name__ == "__main__":
    main()
