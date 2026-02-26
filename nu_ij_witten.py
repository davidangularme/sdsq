#!/usr/bin/env python3
"""
CST: Relative Witten Index ν_ij — Inter-Processor Spectral Comparison
======================================================================

Concept (Catalyst AI, Tour 2):
  Define ν_ij = Z_i(β) - Z_j(β) where Z_i(β) = <Tr(e^{-β H_W})> averaged
  over all qubit pairs on processor i.
  
  Regress ν_ij against hardware metric differences (gate_length, T1, T2, error)
  between processors i and j.
  
  If ν_ij absorbs cross-device variability better than any single hardware metric,
  it proves that the Witten Laplacian encodes coupling geometry without measuring g.

Method:
  1. Extract calibration data from 6 IBM backends (real snapshots via FakeProvider)
  2. Compute Witten Laplacian H_W for each qubit pair (N=32 grid on T²)
  3. Compute heat kernel trace Z(β) = Σ_n e^{-β λ_n} at multiple β values
  4. PAIR-LEVEL: correlate Z(β) with gate_length, gate_error per pair
  5. PROCESSOR-LEVEL: compute ν_ij between all processor pairs
  6. Correlate ν_ij with Δ(metrics) between processors
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
from itertools import combinations

from qiskit_ibm_runtime.fake_provider import (
    FakeSherbrooke, FakeBrisbane, FakeOsaka, FakeKawasaki, FakeKyiv, FakeQuebec
)

# ─── Physics helpers ───

def extract_ej(freq_ghz, anharm_ghz):
    e_c = abs(anharm_ghz)
    if e_c < 1e-6: return None, None
    return (freq_ghz + e_c)**2 / (8 * e_c), e_c

def witten_spectrum(E_JA, E_JB, E_Jc, E_CA, E_CB, N=32, n_eigs=10):
    """Compute Witten Laplacian H_W spectrum for two coupled transmons on T²."""
    hbar_eff = np.sqrt(np.sqrt(8*E_CA/E_JA) * np.sqrt(8*E_CB/E_JB))
    dphi = 2*np.pi/N
    phi = np.linspace(0, 2*np.pi - dphi, N)
    PHI_A, PHI_B = np.meshgrid(phi, phi, indexing='ij')
    pa, pb = PHI_A.flatten(), PHI_B.flatten()
    ntot = N*N
    
    # Witten potential W = Φ/ℏ_eff
    # Gradients of W
    dW_dA = (1.0/hbar_eff)*(E_JA*np.sin(pa) + E_Jc*np.sin(pa-pb))
    dW_dB = (1.0/hbar_eff)*(E_JB*np.sin(pb) - E_Jc*np.sin(pa-pb))
    grad_W_sq = dW_dA**2 + dW_dB**2
    lap_W = (1.0/hbar_eff)*(E_JA*np.cos(pa) + E_JB*np.cos(pb) + 2*E_Jc*np.cos(pa-pb))
    V_eff = grad_W_sq - lap_W
    
    # Build sparse H_W = -Δ + V_eff
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            k = i*N + j
            for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ki = ((i+di)%N)*N + ((j+dj)%N)
                rows.append(k); cols.append(ki); vals.append(-1.0/dphi**2)
            rows.append(k); cols.append(k); vals.append(4.0/dphi**2 + V_eff[k])
    
    H = sparse.coo_matrix((vals,(rows,cols)), shape=(ntot,ntot)).tocsc()
    evals, _ = eigsh(H, k=n_eigs, which='SM')
    return np.sort(evals), hbar_eff

def heat_kernel_trace(evals, betas):
    """Compute Z(β) = Σ_n e^{-β λ_n} for array of β values."""
    return np.array([np.sum(np.exp(-beta * evals)) for beta in betas])

# ─── Data extraction ───

def extract_all_backends():
    """Extract calibration data from all 6 IBM backends."""
    backends_data = {}
    
    for BackendClass in [FakeSherbrooke, FakeBrisbane, FakeOsaka, 
                          FakeKawasaki, FakeKyiv, FakeQuebec]:
        b = BackendClass()
        p = b.properties()
        c = b.configuration()
        t = b.target
        bname = c.backend_name
        
        # Qubit properties
        qubit_info = {}
        for i in range(c.n_qubits):
            qp = p.qubit_property(i)
            freq = qp.get('frequency', (None,))[0]
            anharm = qp.get('anharmonicity', (None,))[0]
            t1 = qp.get('T1', (None,))[0]
            t2 = qp.get('T2', (None,))[0]
            if freq and anharm:
                fg = freq/1e9 if freq > 1e6 else freq
                ag = anharm/1e9 if abs(anharm) > 1e6 else anharm
                ej, ec = extract_ej(fg, ag)
                if ej:
                    qubit_info[i] = {
                        'f': fg, 'a': ag, 'ej': ej, 'ec': ec,
                        't1': t1 if t1 else 0, 't2': t2 if t2 else 0
                    }
        
        # 2-qubit gate pairs
        gn = 'ecr' if 'ecr' in t.operation_names else 'cx'
        pairs = []
        for qa in t.qargs:
            if len(qa) == 2:
                try:
                    ins = t[gn].get(qa)
                    if ins and ins.duration and ins.error < 0.5:
                        q0, q1 = qa
                        if q0 in qubit_info and q1 in qubit_info:
                            A, B = qubit_info[q0], qubit_info[q1]
                            gl = ins.duration * 1e9
                            if gl > 10:
                                pairs.append({
                                    'q0': q0, 'q1': q1,
                                    'gl': gl, 'error': ins.error,
                                    'ej_A': A['ej'], 'ej_B': B['ej'],
                                    'ec_A': A['ec'], 'ec_B': B['ec'],
                                    'f_A': A['f'], 'f_B': B['f'],
                                    't1_A': A['t1'], 't1_B': B['t1'],
                                    't2_A': A['t2'], 't2_B': B['t2'],
                                    'R_ij': np.sqrt(A['ej'] / B['ej']),
                                })
                except:
                    pass
        
        backends_data[bname] = {
            'pairs': pairs,
            'n_qubits': len(qubit_info),
            'gate_type': gn,
        }
        print(f"  {bname}: {len(qubit_info)} qubits, {len(pairs)} gate pairs ({gn})")
    
    return backends_data


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  CST: RELATIVE WITTEN INDEX ν_ij")
print("  Inter-Processor Spectral Comparison")
print("=" * 70)
print()

# 1. Extract data
print("─── STEP 1: Extracting IBM calibration data ───")
backends_data = extract_all_backends()
total_pairs = sum(len(d['pairs']) for d in backends_data.values())
print(f"\n  Total: {total_pairs} pairs across {len(backends_data)} backends\n")

# 2. Compute Witten spectra for all pairs
print("─── STEP 2: Computing Witten Laplacian spectra ───")
E_Jc = 0.005  # 5 MHz coupling (fixed, unknown true value)

# β values to probe different spectral scales
betas = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

t0 = time.time()
count = 0
for bname, bdata in backends_data.items():
    print(f"\n  Processing {bname} ({len(bdata['pairs'])} pairs)...")
    for pair in bdata['pairs']:
        try:
            evals, heff = witten_spectrum(
                pair['ej_A'], pair['ej_B'], E_Jc,
                pair['ec_A'], pair['ec_B'], N=32, n_eigs=10
            )
            Z = heat_kernel_trace(evals, betas)
            pair['evals'] = evals
            pair['heff'] = heff
            pair['Z'] = Z  # Z(β) at each β
            pair['gap01'] = evals[1] - evals[0]
            pair['gap02'] = evals[2] - evals[0]
            pair['lambda0'] = evals[0]
            pair['trace10'] = np.sum(evals)
            pair['valid'] = True
        except:
            pair['valid'] = False
        count += 1
        if count % 100 == 0:
            print(f"    {count}/{total_pairs}...")

elapsed = time.time() - t0
valid_count = sum(1 for bd in backends_data.values() for p in bd['pairs'] if p.get('valid'))
print(f"\n  Done in {elapsed:.1f}s. Valid spectra: {valid_count}/{total_pairs}")

# ═══════════════════════════════════════════════════════════════
# 3. PAIR-LEVEL ANALYSIS: Z(β) vs observables
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  ANALYSIS 1: PAIR-LEVEL — Z(β) vs gate observables")
print(f"  Does the heat kernel trace predict gate_length/error?")
print(f"{'=' * 70}\n")

# Collect all valid pairs
all_pairs = []
for bname, bdata in backends_data.items():
    for p in bdata['pairs']:
        if p.get('valid'):
            p['backend'] = bname
            all_pairs.append(p)

gl = np.array([p['gl'] for p in all_pairs])
err = np.array([p['error'] for p in all_pairs])
R = np.array([p['R_ij'] for p in all_pairs])
Z_matrix = np.array([p['Z'] for p in all_pairs])  # shape: (N_pairs, len(betas))

# Baseline
r_base_gl, _ = stats.pearsonr(R, gl)
r_base_err, _ = stats.pearsonr(R, err)
print(f"  Baseline: R_ij vs gate_length: r = {r_base_gl:+.4f}")
print(f"  Baseline: R_ij vs gate_error:  r = {r_base_err:+.4f}")
print()

# Z(β) correlations at each β
print(f"  {'β':>8s}  {'Z vs gl':>12s}  {'Z vs err':>12s}  {'Z vs R_ij':>12s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

best_beta_gl = 0
best_r_gl = 0
best_beta_err = 0
best_r_err = 0

for bi, beta in enumerate(betas):
    Z_col = Z_matrix[:, bi]
    r_gl, p_gl = stats.pearsonr(Z_col, gl)
    r_err, p_err = stats.pearsonr(Z_col, err)
    r_rij, _ = stats.pearsonr(Z_col, R)
    
    flag = " ***" if abs(r_gl) > 0.3 else " **" if abs(r_gl) > 0.15 else " *" if abs(r_gl) > 0.08 else ""
    print(f"  β={beta:6.2f}  r={r_gl:+.4f}{flag:>4s}  r={r_err:+.4f}       r={r_rij:+.4f}")
    
    if abs(r_gl) > abs(best_r_gl):
        best_r_gl = r_gl
        best_beta_gl = beta
    if abs(r_err) > abs(best_r_err):
        best_r_err = r_err
        best_beta_err = beta

print(f"\n  Best β for gate_length: β={best_beta_gl} (r={best_r_gl:+.4f})")
print(f"  Best β for gate_error:  β={best_beta_err} (r={best_r_err:+.4f})")

# Also try log(Z) and Z ratios
print(f"\n  ─── Derived spectral observables ───")
gap01 = np.array([p['gap01'] for p in all_pairs])
gap02 = np.array([p['gap02'] for p in all_pairs])
lam0 = np.array([p['lambda0'] for p in all_pairs])

derived = {
    'gap₀₁': gap01,
    'gap₀₂': gap02,
    'λ₀': lam0,
    'log(Z(0.1))': np.log(Z_matrix[:, 2] + 1e-30),
    'log(Z(1.0))': np.log(Z_matrix[:, 5] + 1e-30),
    'Z(0.1)/Z(1.0)': Z_matrix[:, 2] / (Z_matrix[:, 5] + 1e-30),
    'Z(0.01)-Z(10)': Z_matrix[:, 0] - Z_matrix[:, -1],
    'spectral entropy': np.array([
        -np.sum((np.exp(-0.5*p['evals'])/np.sum(np.exp(-0.5*p['evals']))) * 
                np.log(np.exp(-0.5*p['evals'])/np.sum(np.exp(-0.5*p['evals'])) + 1e-30))
        for p in all_pairs
    ]),
}

print(f"\n  {'Observable':25s}  {'vs gl':>10s}  {'vs err':>10s}")
print(f"  {'-'*25}  {'-'*10}  {'-'*10}")

for name, vals in derived.items():
    mask = np.isfinite(vals)
    if mask.sum() < 50:
        continue
    r_gl_d, p_gl_d = stats.pearsonr(vals[mask], gl[mask])
    r_err_d, _ = stats.pearsonr(vals[mask], err[mask])
    flag = " ***" if abs(r_gl_d) > 0.3 else " **" if abs(r_gl_d) > 0.15 else ""
    print(f"  {name:25s}  r={r_gl_d:+.4f}{flag}  r={r_err_d:+.4f}")

# ═══════════════════════════════════════════════════════════════
# 4. PROCESSOR-LEVEL: ν_ij relative Witten index
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  ANALYSIS 2: PROCESSOR-LEVEL — Relative Witten Index ν_ij")
print(f"  ν_ij = <Z_i(β)> - <Z_j(β)> between processors")
print(f"{'=' * 70}\n")

# Compute per-processor aggregates
proc_stats = {}
for bname, bdata in backends_data.items():
    valid_pairs = [p for p in bdata['pairs'] if p.get('valid')]
    if not valid_pairs:
        continue
    
    Z_avg = np.mean([p['Z'] for p in valid_pairs], axis=0)
    Z_std = np.std([p['Z'] for p in valid_pairs], axis=0)
    avg_gl = np.mean([p['gl'] for p in valid_pairs])
    avg_err = np.mean([p['error'] for p in valid_pairs])
    avg_t1 = np.mean([(p['t1_A'] + p['t1_B'])/2 for p in valid_pairs if p['t1_A'] and p['t1_B']])
    avg_t2 = np.mean([(p['t2_A'] + p['t2_B'])/2 for p in valid_pairs if p['t2_A'] and p['t2_B']])
    avg_gap01 = np.mean([p['gap01'] for p in valid_pairs])
    avg_R = np.mean([p['R_ij'] for p in valid_pairs])
    avg_lam0 = np.mean([p['lambda0'] for p in valid_pairs])
    
    proc_stats[bname] = {
        'Z_avg': Z_avg, 'Z_std': Z_std,
        'avg_gl': avg_gl, 'avg_err': avg_err,
        'avg_t1': avg_t1, 'avg_t2': avg_t2,
        'avg_gap01': avg_gap01, 'avg_R': avg_R,
        'avg_lam0': avg_lam0,
        'n_pairs': len(valid_pairs),
    }
    
    short = bname.replace('fake_', '')
    print(f"  {short:12s}: n={len(valid_pairs):3d}  "
          f"<gl>={avg_gl:6.1f}ns  <err>={avg_err:.4f}  "
          f"<gap₀₁>={avg_gap01:.2f}  <Z(1)>={Z_avg[5]:.4f}")

# Compute ν_ij for all processor pairs
proc_names = list(proc_stats.keys())
n_proc = len(proc_names)

print(f"\n  ─── ν_ij matrix (β = 1.0) ───")
print(f"\n  {'':15s}", end="")
for p in proc_names:
    print(f"  {p.replace('fake_',''):>10s}", end="")
print()

beta_idx = 5  # β=1.0

for i, pi in enumerate(proc_names):
    print(f"  {pi.replace('fake_',''):15s}", end="")
    for j, pj in enumerate(proc_names):
        nu = proc_stats[pi]['Z_avg'][beta_idx] - proc_stats[pj]['Z_avg'][beta_idx]
        print(f"  {nu:+10.4f}", end="")
    print()

# ν_ij vs Δ(hardware metrics)
print(f"\n  ─── ν_ij correlations with hardware metric differences ───\n")

# Build arrays for all processor pairs
nu_vals = {bi: [] for bi in range(len(betas))}
delta_gl = []
delta_err = []
delta_t1 = []
delta_t2 = []
delta_gap01 = []
pair_labels = []

for i, j in combinations(range(n_proc), 2):
    pi, pj = proc_names[i], proc_names[j]
    si, sj = proc_stats[pi], proc_stats[pj]
    
    for bi in range(len(betas)):
        nu_vals[bi].append(si['Z_avg'][bi] - sj['Z_avg'][bi])
    
    delta_gl.append(si['avg_gl'] - sj['avg_gl'])
    delta_err.append(si['avg_err'] - sj['avg_err'])
    delta_t1.append(si['avg_t1'] - sj['avg_t1'])
    delta_t2.append(si['avg_t2'] - sj['avg_t2'])
    delta_gap01.append(si['avg_gap01'] - sj['avg_gap01'])
    pair_labels.append(f"{pi.replace('fake_','')}-{pj.replace('fake_','')}")

delta_gl = np.array(delta_gl)
delta_err = np.array(delta_err)
delta_t1 = np.array(delta_t1)
delta_t2 = np.array(delta_t2)
delta_gap01 = np.array(delta_gap01)

n_pairs_proc = len(delta_gl)
print(f"  {n_pairs_proc} processor pairs")
print(f"  WARNING: Small sample (n={n_pairs_proc}), correlations are indicative only\n")

print(f"  {'β':>8s}  {'ν vs Δgl':>12s}  {'ν vs Δerr':>12s}  {'ν vs ΔT1':>12s}  {'ν vs ΔT2':>12s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

best_nu_r = 0
best_nu_beta = 0
best_nu_metric = ""

for bi, beta in enumerate(betas):
    nu = np.array(nu_vals[bi])
    
    # Spearman (more robust for small n)
    r_gl_s, p_gl_s = stats.spearmanr(nu, delta_gl)
    r_err_s, _ = stats.spearmanr(nu, delta_err)
    r_t1_s, _ = stats.spearmanr(nu, delta_t1)
    r_t2_s, _ = stats.spearmanr(nu, delta_t2)
    
    flag = " ***" if abs(r_gl_s) > 0.7 else " **" if abs(r_gl_s) > 0.5 else ""
    print(f"  β={beta:6.2f}  ρ={r_gl_s:+.4f}{flag:>4s}  ρ={r_err_s:+.4f}       "
          f"ρ={r_t1_s:+.4f}       ρ={r_t2_s:+.4f}")
    
    for metric_name, r_val in [('Δgl', r_gl_s), ('Δerr', r_err_s), ('ΔT1', r_t1_s), ('ΔT2', r_t2_s)]:
        if abs(r_val) > abs(best_nu_r):
            best_nu_r = r_val
            best_nu_beta = beta
            best_nu_metric = metric_name

print(f"\n  Best: ν(β={best_nu_beta}) vs {best_nu_metric}: ρ = {best_nu_r:+.4f}")

# ─── Also: ν_ij as spectral gap difference ───
print(f"\n  ─── Alternative: Δ(gap₀₁) between processors ───")
r_dgap_gl, p_dgap_gl = stats.spearmanr(delta_gap01, delta_gl)
r_dgap_err, _ = stats.spearmanr(delta_gap01, delta_err)
print(f"  Δ(gap₀₁) vs Δ(gl):  ρ = {r_dgap_gl:+.4f} (p = {p_dgap_gl:.2e})")
print(f"  Δ(gap₀₁) vs Δ(err): ρ = {r_dgap_err:+.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. DEVICE-AGNOSTIC SPECTRAL SIGNATURE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  ANALYSIS 3: DEVICE-AGNOSTIC SPECTRAL SIGNATURE")
print(f"  Can Witten spectra discriminate processors?")
print(f"{'=' * 70}\n")

# For each pair, compute a spectral fingerprint
# Then check if pairs from the same processor cluster together

from collections import defaultdict

# Spectral features per pair
features = np.column_stack([
    gap01,
    np.array([p['gap02'] for p in all_pairs]),
    lam0,
    Z_matrix[:, 2],  # Z(β=0.1)
    Z_matrix[:, 5],  # Z(β=1.0)
])

backend_labels = np.array([p['backend'] for p in all_pairs])
unique_backends = list(set(backend_labels))

# Simple discriminability: how well does spectral fingerprint separate backends?
# Use within-class vs between-class variance (Fisher criterion)
overall_mean = np.mean(features, axis=0)
S_W = np.zeros((features.shape[1], features.shape[1]))
S_B = np.zeros((features.shape[1], features.shape[1]))

for b in unique_backends:
    mask = backend_labels == b
    n_b = mask.sum()
    class_mean = np.mean(features[mask], axis=0)
    diff = features[mask] - class_mean
    S_W += diff.T @ diff
    mean_diff = (class_mean - overall_mean).reshape(-1, 1)
    S_B += n_b * (mean_diff @ mean_diff.T)

# Fisher discriminant ratio: tr(S_B) / tr(S_W)
fisher_ratio = np.trace(S_B) / (np.trace(S_W) + 1e-10)
print(f"  Fisher discriminant ratio (spectral features): {fisher_ratio:.4f}")
print(f"  (>1 means spectral features separate processors better than chance)\n")

# Per-backend spectral summary
print(f"  {'Backend':15s}  {'<gap₀₁>':>10s}  {'<λ₀>':>10s}  {'<Z(1)>':>10s}  {'std(Z(1))':>10s}")
print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for bname in unique_backends:
    mask = backend_labels == bname
    short = bname.replace('fake_', '')
    print(f"  {short:15s}  {gap01[mask].mean():10.2f}  {lam0[mask].mean():10.2f}  "
          f"{Z_matrix[mask, 5].mean():10.4f}  {Z_matrix[mask, 5].std():10.4f}")

# ═══════════════════════════════════════════════════════════════
# 6. SPECTRAL COLLAPSE TEST (Catalyst Tour 1)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  ANALYSIS 4: SPECTRAL COLLAPSE")
print(f"  Do spectra from different processors collapse onto one curve")
print(f"  when rescaled by a processor-dependent factor?")
print(f"{'=' * 70}\n")

# For each processor, compute the average spectral scale
# Then rescale each pair's spectrum by this scale
# If spectra collapse → universal structure exists

# Rescaling: λ_n → λ_n / <λ₁> (normalize by fundamental gap of that processor)
fig_collapse, ax_collapse = plt.subplots(1, 2, figsize=(14, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(unique_backends)))
cmap = dict(zip(unique_backends, colors))

# Raw spectra (unrescaled)
ax = ax_collapse[0]
for p in all_pairs:
    evals = p['evals']
    gaps = evals - evals[0]
    c = cmap[p['backend']]
    ax.scatter(range(len(gaps)), gaps, s=3, alpha=0.1, c=[c])

# Add processor means
for bname in unique_backends:
    mask = backend_labels == bname
    mean_evals = np.mean([all_pairs[i]['evals'] for i in np.where(mask)[0]], axis=0)
    mean_gaps = mean_evals - mean_evals[0]
    short = bname.replace('fake_', '')
    ax.plot(range(len(mean_gaps)), mean_gaps, 'o-', color=cmap[bname], 
            markersize=5, linewidth=2, label=short)

ax.set_xlabel('Eigenvalue index n')
ax.set_ylabel('λ_n - λ₀')
ax.set_title('Raw Spectra (unrescaled)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Rescaled spectra: normalize by gap₀₁
ax = ax_collapse[1]
rescaled_residuals = []

for p in all_pairs:
    evals = p['evals']
    gaps = evals - evals[0]
    scale = gaps[1] if gaps[1] > 1e-6 else 1.0
    rescaled = gaps / scale
    c = cmap[p['backend']]
    ax.scatter(range(len(rescaled)), rescaled, s=3, alpha=0.1, c=[c])

# Add processor means (rescaled)
for bname in unique_backends:
    mask = backend_labels == bname
    rescaled_means = []
    for i in np.where(mask)[0]:
        evals = all_pairs[i]['evals']
        gaps = evals - evals[0]
        scale = gaps[1] if gaps[1] > 1e-6 else 1.0
        rescaled_means.append(gaps / scale)
    mean_rescaled = np.mean(rescaled_means, axis=0)
    short = bname.replace('fake_', '')
    ax.plot(range(len(mean_rescaled)), mean_rescaled, 'o-', color=cmap[bname],
            markersize=5, linewidth=2, label=short)
    rescaled_residuals.append(mean_rescaled)

ax.set_xlabel('Eigenvalue index n')
ax.set_ylabel('(λ_n - λ₀) / (λ₁ - λ₀)')
ax.set_title('Rescaled Spectra (normalized by gap₀₁)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Measure collapse quality: coefficient of variation of rescaled means
if len(rescaled_residuals) > 1:
    rescaled_stack = np.array(rescaled_residuals)
    cv_per_level = np.std(rescaled_stack, axis=0) / (np.mean(rescaled_stack, axis=0) + 1e-10)
    mean_cv = np.mean(cv_per_level[2:])  # skip n=0,1 (trivially 0,1)
    print(f"  Collapse quality (CV of rescaled levels n≥2): {mean_cv:.4f}")
    print(f"  (0 = perfect collapse, >0.1 = no collapse)")
    print(f"  Per-level CV: {', '.join(f'{cv:.3f}' for cv in cv_per_level)}")

plt.suptitle('Spectral Collapse Test — Do Witten Spectra Show Universal Structure?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/spectral_collapse.png', dpi=150, bbox_inches='tight')

# ═══════════════════════════════════════════════════════════════
# 7. MAIN FIGURE: ν_ij analysis
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Z(β) profiles per processor
ax = axes[0, 0]
for bname in unique_backends:
    s = proc_stats[bname]
    short = bname.replace('fake_', '')
    ax.plot(betas, s['Z_avg'], 'o-', label=short, markersize=4)
    ax.fill_between(betas, s['Z_avg'] - s['Z_std'], s['Z_avg'] + s['Z_std'], alpha=0.1)
ax.set_xscale('log')
ax.set_xlabel('β (inverse temperature)')
ax.set_ylabel('<Z(β)>')
ax.set_title('Heat Kernel Traces by Processor')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: ν_ij vs Δ(gate_length) at best β
ax = axes[0, 1]
nu_best = np.array(nu_vals[list(betas).index(best_nu_beta)])
ax.scatter(nu_best, delta_gl, s=60, c='steelblue', edgecolors='navy', zorder=5)
for k, label in enumerate(pair_labels):
    ax.annotate(label, (nu_best[k], delta_gl[k]), fontsize=6, ha='center', va='bottom')
r_sp, p_sp = stats.spearmanr(nu_best, delta_gl)
ax.set_xlabel(f'ν_ij (β={best_nu_beta})')
ax.set_ylabel('Δ(gate_length) (ns)')
ax.set_title(f'ν_ij vs Δ(gl): Spearman ρ = {r_sp:+.3f}')
ax.grid(True, alpha=0.3)

# Plot 3: Z(β=1) vs gate_length (pair-level)
ax = axes[0, 2]
Z1 = Z_matrix[:, 5]
for bname in unique_backends:
    mask = backend_labels == bname
    short = bname.replace('fake_', '')
    ax.scatter(Z1[mask], gl[mask], s=8, alpha=0.4, label=short)
r_Z1_gl, p_Z1_gl = stats.pearsonr(Z1, gl)
ax.set_xlabel('Z(β=1.0)')
ax.set_ylabel('gate_length (ns)')
ax.set_title(f'Pair-level: Z(1) vs gl (r={r_Z1_gl:+.4f})')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 4: Spectral entropy vs gate_length
ax = axes[1, 0]
S_ent = derived['spectral entropy']
for bname in unique_backends:
    mask = backend_labels == bname
    short = bname.replace('fake_', '')
    ax.scatter(S_ent[mask], gl[mask], s=8, alpha=0.4, label=short)
r_ent, _ = stats.pearsonr(S_ent, gl)
ax.set_xlabel('Spectral Entropy S')
ax.set_ylabel('gate_length (ns)')
ax.set_title(f'Spectral Entropy vs gl (r={r_ent:+.4f})')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 5: ν_ij vs Δ(error) 
ax = axes[1, 1]
nu_1 = np.array(nu_vals[5])  # β=1.0
ax.scatter(nu_1, delta_err, s=60, c='coral', edgecolors='darkred', zorder=5)
for k, label in enumerate(pair_labels):
    ax.annotate(label, (nu_1[k], delta_err[k]), fontsize=6, ha='center', va='bottom')
r_sp_err, _ = stats.spearmanr(nu_1, delta_err)
ax.set_xlabel('ν_ij (β=1.0)')
ax.set_ylabel('Δ(gate_error)')
ax.set_title(f'ν_ij vs Δ(error): Spearman ρ = {r_sp_err:+.3f}')
ax.grid(True, alpha=0.3)

# Plot 6: Processor spectral fingerprint (radar-style)
ax = axes[1, 2]
proc_list = sorted(proc_stats.keys())
x_pos = range(len(proc_list))
avg_gls = [proc_stats[p]['avg_gl'] for p in proc_list]
avg_Z1s = [proc_stats[p]['Z_avg'][5] for p in proc_list]
short_names = [p.replace('fake_', '') for p in proc_list]

ax2 = ax.twinx()
bars = ax.bar(x_pos, avg_gls, alpha=0.6, color='steelblue', label='<gl> (ns)')
line = ax2.plot(x_pos, avg_Z1s, 'o-', color='darkred', markersize=8, linewidth=2, label='<Z(1)>')
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('avg gate_length (ns)', color='steelblue')
ax2.set_ylabel('<Z(β=1)>', color='darkred')
ax.set_title('Processor Profiles: gl vs Spectral Z')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.suptitle('CST: Relative Witten Index ν_ij — Inter-Processor Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/nu_ij_results.png', dpi=150, bbox_inches='tight')

# ═══════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  FINAL VERDICT")
print(f"{'=' * 70}")

print(f"""
  PAIR-LEVEL (n={len(all_pairs)} pairs):
    Best Z(β) predictor of gate_length: β={best_beta_gl} → r = {best_r_gl:+.4f}
    R_ij baseline:                              r = {r_base_gl:+.4f}
    Surplus: Δr = {abs(best_r_gl) - abs(r_base_gl):+.4f}

  PROCESSOR-LEVEL (n={n_pairs_proc} processor pairs):
    Best ν_ij predictor of Δ(gl): β={best_nu_beta} → ρ = {best_nu_r:+.4f}
    
  SPECTRAL COLLAPSE:
    CV after rescaling = {mean_cv:.4f}
    {'COLLAPSE DETECTED — universal spectral structure' if mean_cv < 0.05 else 'PARTIAL COLLAPSE' if mean_cv < 0.15 else 'NO COLLAPSE — spectra are processor-specific'}

  FISHER DISCRIMINABILITY:
    Ratio = {fisher_ratio:.4f}
    {'Witten spectrum DISCRIMINATES processors' if fisher_ratio > 1 else 'No discriminability'}
""")

# Save results
results = {
    'test': 'nu_ij relative Witten index',
    'n_backends': len(backends_data),
    'n_total_pairs': len(all_pairs),
    'n_processor_pairs': n_pairs_proc,
    'pair_level': {
        'best_beta_gl': float(best_beta_gl),
        'best_r_gl': float(best_r_gl),
        'baseline_R_gl': float(r_base_gl),
        'best_beta_err': float(best_beta_err),
        'best_r_err': float(best_r_err),
    },
    'processor_level': {
        'best_beta': float(best_nu_beta),
        'best_rho': float(best_nu_r),
        'best_metric': best_nu_metric,
    },
    'spectral_collapse': {
        'mean_cv': float(mean_cv),
        'per_level_cv': [float(x) for x in cv_per_level],
    },
    'fisher_ratio': float(fisher_ratio),
    'processor_stats': {
        k.replace('fake_', ''): {
            'n_pairs': v['n_pairs'],
            'avg_gl': float(v['avg_gl']),
            'avg_err': float(v['avg_err']),
            'avg_Z_beta1': float(v['Z_avg'][5]),
        }
        for k, v in proc_stats.items()
    }
}

with open('/home/claude/nu_ij_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved: /home/claude/nu_ij_results.json")
print(f"  Plots: /home/claude/nu_ij_results.png, spectral_collapse.png")
