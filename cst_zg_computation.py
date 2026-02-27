#!/usr/bin/env python3
"""
CST — Z_G(s) = Tr(G^{-s}) Numerical Computation
=================================================
Goal: Compute Z_G(s) for various constructions of G and check if
special values of s give known coupling constants.

Physical constants (PDG 2024):
  α_em^{-1} = 137.036
  α_w^{-1}  ≈ 29.5  (at M_Z scale)
  α_s^{-1}  ≈ 8.5   (at M_Z scale)
  Koide Q   = 2/3 = 0.6667

Particle masses in MeV/c²:
  Leptons: e=0.511, μ=105.66, τ=1776.86
  Quarks:  u=2.16, d=4.67, s=93.4, c=1270, b=4180, t=172760
  Bosons:  W=80377, Z=91188, H=125250, γ=0, g=0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Coupling constants (PDG) — our targets
ALPHA_EM_INV = 137.036  # at q=0
ALPHA_W_INV = 29.5      # at M_Z (sin²θ_W ≈ 0.231, so α_w ≈ α_em/sin²θ_W)
ALPHA_S_INV = 8.5       # at M_Z (α_s ≈ 0.118)

# Lepton masses (MeV)
m_e = 0.51100
m_mu = 105.6584
m_tau = 1776.86

# Quark masses (MeV) — current masses at 2 GeV
m_u = 2.16
m_d = 4.67
m_s = 93.4
m_c = 1270.0
m_b = 4180.0
m_t = 172760.0

# Boson masses (MeV) — only massive ones
m_W = 80377.0
m_Z = 91188.0
m_H = 125250.0

# ═══════════════════════════════════════════════════════════════
# Z_G(s) COMPUTATION
# ═══════════════════════════════════════════════════════════════

def Z_G(eigenvalues: np.ndarray, s: complex) -> complex:
    """
    Compute Z_G(s) = Tr(G^{-s}) = Σ λᵢ^{-s}
    For negative s: Z_G(s) = Σ λᵢ^{|s|}
    Handles s=0 specially: Z_G(0) = N (count of eigenvalues)
    """
    if abs(s) < 1e-12:
        return len(eigenvalues)
    
    # Filter out zero eigenvalues (massless particles)
    nonzero = eigenvalues[eigenvalues > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    
    result = np.sum(nonzero ** (-s))
    return np.real(result) if np.isreal(s) else result

def compute_koide(eigenvalues: np.ndarray) -> float:
    """Compute Koide ratio: Z_G(-1) / [Z_G(-1/2)]^2"""
    zg_neg1 = Z_G(eigenvalues, -1)      # = Σ λᵢ
    zg_neg_half = Z_G(eigenvalues, -0.5)  # = Σ √λᵢ
    return zg_neg1 / (zg_neg_half ** 2)

# ═══════════════════════════════════════════════════════════════
# BUILD DIFFERENT G-TENSOR EIGENVALUE SETS
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("CST — Z_G(s) = Tr(G^{-s}) NUMERICAL COMPUTATION")
print("=" * 80)

# --- Configuration 1: Leptonic sector only ---
leptons = np.array([m_e, m_mu, m_tau])

# --- Configuration 2: All charged fermions (no neutrinos) ---
charged_fermions = np.array([m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t])

# --- Configuration 3: Quarks with color (each ×3) ---
quarks_colored = np.array([m_u]*3 + [m_d]*3 + [m_s]*3 + [m_c]*3 + [m_b]*3 + [m_t]*3)
fermions_with_color = np.concatenate([np.array([m_e, m_mu, m_tau]), quarks_colored])

# --- Configuration 4: Full SM massive spectrum (fermions + bosons) ---
full_sm_massive = np.concatenate([
    np.array([m_e, m_mu, m_tau]),           # charged leptons
    quarks_colored,                          # quarks ×3 colors
    np.array([m_W, m_W, m_Z, m_H])         # W+, W-, Z, H
])

# --- Configuration 5: With spin DOF (×2 for fermion spins) ---
fermion_spins = np.concatenate([
    np.repeat([m_e, m_mu, m_tau], 2),        # leptons ×2 spins
    np.repeat(quarks_colored, 2),            # quarks ×2 spins ×3 colors
])
full_with_spins = np.concatenate([
    fermion_spins,
    np.array([m_W]*3 + [m_Z]*3 + [m_H])    # W±(3 pol) + Z(3 pol) + H(1)
])

# --- Configuration 6: With particle-antiparticle ---
full_with_anti = np.concatenate([
    np.repeat(fermion_spins, 2),             # fermions ×2 (particle + anti)
    np.array([m_W]*3 + [m_W]*3 + [m_Z]*3 + [m_H])  # W+, W-, Z, H with polarizations
])

configs = {
    "1. Leptons only (3)": leptons,
    "2. Charged fermions (9)": charged_fermions,
    "3. Fermions + color (21)": fermions_with_color,
    "4. Full SM massive (25)": full_sm_massive,
    "5. With spins (49)": full_with_spins,
    "6. With antiparticles (91)": full_with_anti,
}

# ═══════════════════════════════════════════════════════════════
# COMPUTATION AT SPECIAL VALUES
# ═══════════════════════════════════════════════════════════════

special_s = {
    "Z_G(0) [= dim(Γ)]": 0,
    "Z_G(-1/4) [→ α_s?]": -0.25,
    "Z_G(-1/3) [→ α_w?]": -1/3,
    "Z_G(-1/2)": -0.5,
    "Z_G(-1) [= Tr(G)]": -1,
    "Z_G(1/2)": 0.5,
    "Z_G(1) [= Tr(G⁻¹)]": 1,
    "Z_G(2)": 2,
}

print("\n" + "█" * 80)
print("PART 1: Z_G(s) AT SPECIAL VALUES")
print("█" * 80)
print(f"\nTargets: α_em⁻¹ = {ALPHA_EM_INV}, α_w⁻¹ = {ALPHA_W_INV}, α_s⁻¹ = {ALPHA_S_INV}")

for config_name, eigenvalues in configs.items():
    print(f"\n{'─' * 70}")
    print(f"Config: {config_name}")
    print(f"  Eigenvalues: {len(eigenvalues)} modes")
    print(f"  Mass range: [{eigenvalues.min():.4f}, {eigenvalues.max():.1f}] MeV")
    
    koide = compute_koide(eigenvalues[:3]) if len(eigenvalues) >= 3 else None
    print(f"  Koide (leptons): {koide:.6f}  (target: 0.666667, error: {abs(koide - 2/3)*100/( 2/3):.4f}%)")
    
    for s_name, s_val in special_s.items():
        val = Z_G(eigenvalues, s_val)
        # Check proximity to targets
        flag = ""
        if abs(val) > 1e-10:
            if abs(val - ALPHA_EM_INV) / ALPHA_EM_INV < 0.1:
                flag = f"  ← NEAR α_em⁻¹ ({abs(val - ALPHA_EM_INV)/ALPHA_EM_INV*100:.1f}% off)"
            elif abs(val - ALPHA_W_INV) / ALPHA_W_INV < 0.15:
                flag = f"  ← NEAR α_w⁻¹ ({abs(val - ALPHA_W_INV)/ALPHA_W_INV*100:.1f}% off)"
            elif abs(val - ALPHA_S_INV) / ALPHA_S_INV < 0.2:
                flag = f"  ← NEAR α_s⁻¹ ({abs(val - ALPHA_S_INV)/ALPHA_S_INV*100:.1f}% off)"
        print(f"    {s_name:30s} = {val:>20.6f}{flag}")

# ═══════════════════════════════════════════════════════════════
# PART 2: SWEEP s TO FIND WHERE α TARGETS ARE HIT
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("PART 2: SCANNING s ∈ [-2, 3] — WHERE DO TARGETS APPEAR?")
print("█" * 80)

s_grid = np.linspace(-2, 3, 5001)
targets = {"α_em⁻¹ = 137.036": 137.036, "α_w⁻¹ ≈ 29.5": 29.5, "α_s⁻¹ ≈ 8.5": 8.5}

for config_name, eigenvalues in configs.items():
    print(f"\n{'─' * 70}")
    print(f"Config: {config_name}")
    
    zg_values = np.array([Z_G(eigenvalues, s) for s in s_grid])
    
    for target_name, target_val in targets.items():
        # Find crossings
        crossings = []
        for i in range(len(s_grid) - 1):
            if (zg_values[i] - target_val) * (zg_values[i+1] - target_val) < 0:
                # Linear interpolation
                s_cross = s_grid[i] + (target_val - zg_values[i]) / (zg_values[i+1] - zg_values[i]) * (s_grid[i+1] - s_grid[i])
                crossings.append(s_cross)
        
        if crossings:
            print(f"  {target_name} hit at s = {', '.join(f'{c:.4f}' for c in crossings)}")
        else:
            # Find closest approach
            diffs = np.abs(zg_values - target_val)
            idx = np.argmin(diffs)
            print(f"  {target_name}: NOT hit. Closest: Z_G({s_grid[idx]:.3f}) = {zg_values[idx]:.4f} "
                  f"(off by {diffs[idx]/target_val*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# PART 3: NORMALIZED CONSTRUCTIONS
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("PART 3: NORMALIZED EIGENVALUES (masses / m_e)")
print("█" * 80)
print("Rationale: G might encode mass RATIOS, not absolute masses.")

for config_name, eigenvalues in configs.items():
    normalized = eigenvalues / m_e  # Normalize to electron mass
    
    print(f"\n{'─' * 70}")
    print(f"Config: {config_name} (normalized by m_e)")
    
    zg_values_norm = np.array([Z_G(normalized, s) for s in s_grid])
    
    for target_name, target_val in targets.items():
        crossings = []
        for i in range(len(s_grid) - 1):
            if np.isfinite(zg_values_norm[i]) and np.isfinite(zg_values_norm[i+1]):
                if (zg_values_norm[i] - target_val) * (zg_values_norm[i+1] - target_val) < 0:
                    s_cross = s_grid[i] + (target_val - zg_values_norm[i]) / (zg_values_norm[i+1] - zg_values_norm[i]) * (s_grid[i+1] - s_grid[i])
                    crossings.append(s_cross)
        
        if crossings:
            print(f"  {target_name} hit at s = {', '.join(f'{c:.4f}' for c in crossings[:5])}")
        else:
            diffs = np.abs(zg_values_norm - target_val)
            valid = np.isfinite(diffs)
            if valid.any():
                idx = np.argmin(diffs[valid])
                idx_orig = np.where(valid)[0][idx]
                print(f"  {target_name}: NOT hit. Closest: Z_G({s_grid[idx_orig]:.3f}) = {zg_values_norm[idx_orig]:.4f}")

# ═══════════════════════════════════════════════════════════════
# PART 4: FUNCTIONAL EQUATION TEST
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("PART 4: FUNCTIONAL EQUATION TEST — Z_G(s) vs Z_G(1-s)")
print("█" * 80)
print("Testing if Z_G(s)/Z_G(1-s) = X(s) has a simple form.")

s_test = np.linspace(0.01, 0.99, 99)  # Stay in (0,1) to avoid poles

for config_name, eigenvalues in [("Leptons only", leptons), 
                                   ("Leptons (normalized)", leptons/m_e),
                                   ("Full SM massive", full_sm_massive)]:
    print(f"\n  Config: {config_name}")
    
    ratios = []
    for s in s_test:
        zs = Z_G(eigenvalues, s)
        z1s = Z_G(eigenvalues, 1 - s)
        if abs(z1s) > 1e-30:
            ratios.append(zs / z1s)
        else:
            ratios.append(np.nan)
    
    ratios = np.array(ratios)
    valid = np.isfinite(ratios)
    if valid.any():
        r = ratios[valid]
        print(f"    Ratio Z_G(s)/Z_G(1-s) range: [{r.min():.6f}, {r.max():.6f}]")
        print(f"    Mean: {r.mean():.6f}, Std: {r.std():.6f}")
        if r.std() / abs(r.mean()) < 0.01:
            print(f"    → CONSTANT! Functional equation: Z_G(s) = {r.mean():.4f} · Z_G(1-s)")
        else:
            # Check if ratio is a simple function of s
            # Test: ratio = A^(2s-1) ?
            log_ratios = np.log(np.abs(r))
            s_valid = s_test[valid]
            coeffs = np.polyfit(2*s_valid - 1, log_ratios, 1)
            r_squared = 1 - np.sum((log_ratios - np.polyval(coeffs, 2*s_valid-1))**2) / np.sum((log_ratios - log_ratios.mean())**2)
            A = np.exp(coeffs[0])
            print(f"    → Test ratio = A^(2s-1): A = {A:.6f}, R² = {r_squared:.6f}")
            if r_squared > 0.99:
                print(f"    → EXPONENTIAL FE: Z_G(s) = {A:.4f}^(2s-1) · Z_G(1-s)")

# ═══════════════════════════════════════════════════════════════
# PART 5: KOIDE FOR ALL TRIPLETS
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("PART 5: KOIDE RATIO FOR DIFFERENT MASS TRIPLETS")
print("█" * 80)

triplets = {
    "Charged leptons (e, μ, τ)": [m_e, m_mu, m_tau],
    "Up quarks (u, c, t)": [m_u, m_c, m_t],
    "Down quarks (d, s, b)": [m_d, m_s, m_b],
    "Light quarks (u, d, s)": [m_u, m_d, m_s],
    "Heavy quarks (c, b, t)": [m_c, m_b, m_t],
    "Bosons (W, Z, H)": [m_W, m_Z, m_H],
    "Gen 1 (e, u, d)": [m_e, m_u, m_d],
    "Gen 2 (μ, c, s)": [m_mu, m_c, m_s],
    "Gen 3 (τ, t, b)": [m_tau, m_t, m_b],
}

for name, masses in triplets.items():
    m = np.array(masses)
    Q = np.sum(m) / (np.sum(np.sqrt(m)))**2
    error = abs(Q - 2/3) / (2/3) * 100
    marker = "★" if error < 1 else "●" if error < 10 else " "
    print(f"  {marker} {name:35s}  Q = {Q:.6f}  (error: {error:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# PART 6: PLOT Z_G(s) FOR ALL CONFIGS
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Z_G(s) raw masses
ax = axes[0, 0]
s_plot = np.linspace(-1.5, 2.5, 2000)
for config_name, eigenvalues in configs.items():
    vals = [Z_G(eigenvalues, s) for s in s_plot]
    ax.semilogy(s_plot, np.abs(vals), label=config_name, alpha=0.8)
ax.axhline(y=137.036, color='red', linestyle='--', alpha=0.5, label='α_em⁻¹ = 137')
ax.axhline(y=29.5, color='blue', linestyle='--', alpha=0.5, label='α_w⁻¹ ≈ 29.5')
ax.axhline(y=8.5, color='green', linestyle='--', alpha=0.5, label='α_s⁻¹ ≈ 8.5')
ax.axvline(x=0.5, color='purple', linestyle=':', alpha=0.3, label='Re(s) = 1/2')
ax.set_xlabel('s')
ax.set_ylabel('|Z_G(s)|')
ax.set_title('Z_G(s) = Tr(G⁻ˢ) — Raw Masses (MeV)')
ax.legend(fontsize=7)
ax.set_ylim(1e-3, 1e16)
ax.grid(True, alpha=0.3)

# Plot 2: Z_G(s) normalized masses
ax = axes[0, 1]
for config_name, eigenvalues in configs.items():
    norm = eigenvalues / m_e
    vals = [Z_G(norm, s) for s in s_plot]
    ax.semilogy(s_plot, np.abs(vals), label=config_name, alpha=0.8)
ax.axhline(y=137.036, color='red', linestyle='--', alpha=0.5, label='α_em⁻¹ = 137')
ax.axhline(y=29.5, color='blue', linestyle='--', alpha=0.5, label='α_w⁻¹ ≈ 29.5')
ax.axhline(y=8.5, color='green', linestyle='--', alpha=0.5, label='α_s⁻¹ ≈ 8.5')
ax.axvline(x=0.5, color='purple', linestyle=':', alpha=0.3)
ax.set_xlabel('s')
ax.set_ylabel('|Z_G(s)|')
ax.set_title('Z_G(s) — Normalized (masses / m_e)')
ax.legend(fontsize=7)
ax.set_ylim(1e-3, 1e16)
ax.grid(True, alpha=0.3)

# Plot 3: Functional equation ratio
ax = axes[1, 0]
s_fe = np.linspace(0.02, 0.98, 200)
for config_name, eigenvalues in [("Leptons", leptons), ("Fermions+color", fermions_with_color), 
                                   ("Full SM", full_sm_massive)]:
    ratios = []
    for s in s_fe:
        zs = Z_G(eigenvalues, s)
        z1s = Z_G(eigenvalues, 1-s)
        ratios.append(zs/z1s if abs(z1s) > 1e-30 else np.nan)
    ax.plot(s_fe, ratios, label=config_name, alpha=0.8)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0.5, color='purple', linestyle=':', alpha=0.3, label='s = 1/2')
ax.set_xlabel('s')
ax.set_ylabel('Z_G(s) / Z_G(1-s)')
ax.set_title('Functional Equation Test')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Koide ratio as function of generation mixing angle
ax = axes[1, 1]
# Plot Z_G(s) in neighborhood of s where it crosses targets
s_fine = np.linspace(-0.5, 1.5, 3000)
for config_name, eigenvalues in [("Full SM (normalized)", full_sm_massive/m_e),
                                   ("With antiparticles (normalized)", full_with_anti/m_e)]:
    vals = [Z_G(eigenvalues, s) for s in s_fine]
    ax.plot(s_fine, vals, label=config_name, alpha=0.8)
ax.axhline(y=137.036, color='red', linestyle='--', alpha=0.5, label='137.036')
ax.axhline(y=29.5, color='blue', linestyle='--', alpha=0.5, label='29.5')
ax.axhline(y=8.5, color='green', linestyle='--', alpha=0.5, label='8.5')
ax.axhline(y=0.667, color='orange', linestyle='--', alpha=0.5, label='2/3')
ax.set_xlabel('s')
ax.set_ylabel('Z_G(s)')
ax.set_title('Z_G(s) — Fine Scan Near Targets (normalized)')
ax.legend(fontsize=8)
ax.set_ylim(-10, 200)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/cst_zg_results.png', dpi=150, bbox_inches='tight')
print("\n\n[Plot saved to cst_zg_results.png]")

# ═══════════════════════════════════════════════════════════════
# PART 7: SUMMARY — THE VERDICT
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("PART 7: SUMMARY & VERDICT")
print("█" * 80)

# Compute key values for the most promising config
for config_name, eigenvalues in configs.items():
    norm = eigenvalues / m_e
    print(f"\n  {config_name}:")
    print(f"    Z_G(0)      = {Z_G(eigenvalues, 0):.1f}  (dim of Γ)")
    print(f"    Z_G(0) norm = {Z_G(norm, 0):.1f}")
    
    # Find s where Z_G = 137
    for s_try in s_grid:
        v = Z_G(norm, s_try)
        if abs(v - 137.036) < 0.5:
            print(f"    ★ Z_G({s_try:.4f}) [normalized] = {v:.3f}  ← NEAR 137!")
            break

print("\n" + "═" * 80)
print("KEY QUESTION: Does Z_G(s) at simple rational s-values give coupling constants?")
print("═" * 80)

# Check the specific claims from Catalyst
for config_name, eigenvalues in [("Leptons", leptons), ("Full SM massive", full_sm_massive), 
                                   ("With antiparticles", full_with_anti)]:
    print(f"\n  {config_name}:")
    norm = eigenvalues / m_e
    
    for s_val, target, target_name in [
        (0, 137.036, "α_em⁻¹"),
        (-1/3, 29.5, "α_w⁻¹"),
        (-1/4, 8.5, "α_s⁻¹"),
    ]:
        raw = Z_G(eigenvalues, s_val)
        normalized = Z_G(norm, s_val)
        print(f"    Z_G({s_val:>6.3f}) raw = {raw:>15.4f}  |  norm = {normalized:>15.4f}  |  target {target_name} = {target}")

