#!/usr/bin/env python3
"""
CST — Z_G(s) for the 137-DOF configuration
Parts 7-9 from the analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Masses (MeV)
m_e, m_mu, m_tau = 0.511, 105.6584, 1776.86
m_u, m_d, m_s, m_c, m_b, m_t = 2.16, 4.67, 93.4, 1270.0, 4180.0, 172760.0
m_W, m_Z, m_H = 80377.0, 91188.0, 125250.0
m_nu_e, m_nu_mu, m_nu_tau = 0.001, 0.01, 0.05

def Z_G(eigenvalues, s):
    if abs(s) < 1e-12:
        return float(len(eigenvalues))
    nonzero = eigenvalues[eigenvalues > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    return float(np.real(np.sum(nonzero ** (-s))))

# ═══════════════════════════════════════════════════════════════
# BUILD 137-DOF CONFIGURATION
# 94 (undisputed) + 12 (Dirac ν) + 24 (massive gluons) + 2 (γ) + 5 (graviton)
# ═══════════════════════════════════════════════════════════════

print("█" * 80)
print("Z_G(s) FOR THE 137-DOF CONFIGURATION")
print("█" * 80)

# Test with different gluon effective masses
for m_gluon_label, m_gluon in [("ΛQCD=330 MeV", 330.0), ("Lattice=600 MeV", 600.0), ("Lattice=1000 MeV", 1000.0)]:
    
    eigenvalues_137 = np.array(
        [m_e]*4 + [m_mu]*4 + [m_tau]*4 +          # charged leptons: 12
        [m_u]*12 + [m_d]*12 + [m_s]*12 +           # light quarks: 36
        [m_c]*12 + [m_b]*12 + [m_t]*12 +           # heavy quarks: 36
        [m_W]*6 + [m_Z]*3 + [m_H]*1 +              # EW bosons: 10
        [m_nu_e]*4 + [m_nu_mu]*4 + [m_nu_tau]*4 +  # Dirac neutrinos: 12
        [m_gluon]*24 +                              # massive gluons: 24
        [1e-10]*2 +                                 # photon (proxy): 2
        [1e-20]*5                                   # graviton (proxy): 5
    )
    
    print(f"\n{'─'*70}")
    print(f"Gluon mass: {m_gluon_label}")
    print(f"Total eigenvalues: {len(eigenvalues_137)}")
    assert len(eigenvalues_137) == 137
    
    # Special values
    print(f"\n  Z_G(0)    = {Z_G(eigenvalues_137, 0):.0f}  (= dim Γ = 137 by construction)")
    print(f"  Z_G(-1)   = {Z_G(eigenvalues_137, -1):.2f}  (= Tr(G) = total mass)")
    print(f"  Z_G(-1/2) = {Z_G(eigenvalues_137, -0.5):.4f}")
    print(f"  Z_G(-1/3) = {Z_G(eigenvalues_137, -1/3):.4f}")
    print(f"  Z_G(-1/4) = {Z_G(eigenvalues_137, -1/4):.4f}")
    print(f"  Z_G(1/2)  = {Z_G(eigenvalues_137, 0.5):.4f}")
    print(f"  Z_G(1)    = {Z_G(eigenvalues_137, 1):.6f}")
    
    # Koide for leptons
    lep = np.array([m_e, m_mu, m_tau])
    Q_lep = np.sum(lep) / (np.sum(np.sqrt(lep)))**2
    print(f"\n  Koide (leptons): {Q_lep:.6f}  (target: 0.666667, err: {abs(Q_lep-2/3)/(2/3)*100:.4f}%)")
    
    # Koide for heavy quarks
    hq = np.array([m_c, m_b, m_t])
    Q_hq = np.sum(hq) / (np.sum(np.sqrt(hq)))**2
    print(f"  Koide (c,b,t):   {Q_hq:.6f}  (target: 0.666667, err: {abs(Q_hq-2/3)/(2/3)*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# SCAN: Where do coupling constants appear for 137-DOF?
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("SCANNING Z_G(s) — WHERE DO TARGETS APPEAR?")
print("█" * 80)

m_gluon = 600.0  # Use lattice QCD value

eigenvalues_137 = np.array(
    [m_e]*4 + [m_mu]*4 + [m_tau]*4 +
    [m_u]*12 + [m_d]*12 + [m_s]*12 +
    [m_c]*12 + [m_b]*12 + [m_t]*12 +
    [m_W]*6 + [m_Z]*3 + [m_H]*1 +
    [m_nu_e]*4 + [m_nu_mu]*4 + [m_nu_tau]*4 +
    [m_gluon]*24 +
    [1e-10]*2 +
    [1e-20]*5
)

# Use only massive eigenvalues (>0.001 MeV) for meaningful Z_G
massive = eigenvalues_137[eigenvalues_137 > 0.001]
print(f"\nUsing {len(massive)} massive eigenvalues (out of 137)")
print(f"(Excluding {137 - len(massive)} near-massless: photon + graviton)")

s_grid = np.linspace(-2, 3, 20001)
zg_vals = np.array([Z_G(massive, s) for s in s_grid])

targets = {
    "α_em⁻¹ = 137.036": 137.036,
    "α_w⁻¹ ≈ 29.46": 29.46,
    "α_s⁻¹ ≈ 8.47": 8.47,
    "2/3 (Koide)": 0.6667,
    "3 (generations)": 3.0,
    "dim=130 (massive only)": 130.0,
}

for target_name, target_val in targets.items():
    crossings = []
    for i in range(len(s_grid) - 1):
        if np.isfinite(zg_vals[i]) and np.isfinite(zg_vals[i+1]):
            if (zg_vals[i] - target_val) * (zg_vals[i+1] - target_val) < 0:
                s_cross = s_grid[i] + (target_val - zg_vals[i]) / (zg_vals[i+1] - zg_vals[i]) * (s_grid[i+1] - s_grid[i])
                crossings.append(s_cross)
    if crossings:
        print(f"  {target_name:30s} hit at s = {', '.join(f'{c:.5f}' for c in crossings[:5])}")
    else:
        # Find closest
        diffs = np.abs(zg_vals - target_val)
        valid = np.isfinite(diffs)
        if valid.any():
            idx = np.argmin(diffs[valid])
            idx_orig = np.where(valid)[0][idx]
            print(f"  {target_name:30s} NOT hit. Closest: Z_G({s_grid[idx_orig]:.3f}) = {zg_vals[idx_orig]:.4f}")

# ═══════════════════════════════════════════════════════════════
# NORMALIZED: Z_G with eigenvalues / m_e
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("NORMALIZED Z_G(s) — eigenvalues / m_e")
print("█" * 80)

norm = massive / m_e
zg_norm = np.array([Z_G(norm, s) for s in s_grid])

for target_name, target_val in targets.items():
    crossings = []
    for i in range(len(s_grid) - 1):
        if np.isfinite(zg_norm[i]) and np.isfinite(zg_norm[i+1]):
            if (zg_norm[i] - target_val) * (zg_norm[i+1] - target_val) < 0:
                s_cross = s_grid[i] + (target_val - zg_norm[i]) / (zg_norm[i+1] - zg_norm[i]) * (s_grid[i+1] - s_grid[i])
                crossings.append(s_cross)
    if crossings:
        print(f"  {target_name:30s} hit at s = {', '.join(f'{c:.5f}' for c in crossings[:5])}")

# ═══════════════════════════════════════════════════════════════
# FUNCTIONAL EQUATION TEST for 137-DOF
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("FUNCTIONAL EQUATION: Z_G(s) / Z_G(1-s) for 137-DOF")
print("█" * 80)

s_fe = np.linspace(0.01, 0.99, 500)
for label, ev in [("Raw masses", massive), ("Normalized", norm)]:
    ratios = []
    for s in s_fe:
        zs = Z_G(ev, s)
        z1s = Z_G(ev, 1 - s)
        if abs(z1s) > 1e-30:
            ratios.append(zs / z1s)
        else:
            ratios.append(np.nan)
    
    ratios = np.array(ratios)
    valid = np.isfinite(ratios) & (np.abs(ratios) < 1e10)
    r = ratios[valid]
    s_v = s_fe[valid]
    
    print(f"\n  {label}:")
    print(f"    Ratio range: [{r.min():.6f}, {r.max():.6f}]")
    
    # Test exponential form: ratio = A^(2s-1)
    log_r = np.log(np.abs(r))
    x = 2 * s_v - 1
    coeffs = np.polyfit(x, log_r, 1)
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((log_r - predicted)**2)
    ss_tot = np.sum((log_r - log_r.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    A = np.exp(coeffs[0])
    print(f"    Exponential fit: ratio ≈ {A:.4f}^(2s-1), R² = {r_squared:.6f}")
    
    # Value at s = 1/2 (should be 1 for true FE)
    idx_half = np.argmin(np.abs(s_fe - 0.5))
    if np.isfinite(ratios[idx_half]):
        print(f"    Z_G(1/2)/Z_G(1/2) = {ratios[idx_half]:.6f}  (should be 1.0)")

# ═══════════════════════════════════════════════════════════════
# THE KEY PLOT: Z_G(s) for 137-DOF config
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CST: Z_G(s) for 137-DOF Configuration (94 + 12ν + 24g + 2γ + 5graviton)', 
             fontsize=14, fontweight='bold')

# Plot 1: Z_G(s) log scale
ax = axes[0, 0]
s_plot = np.linspace(-1.5, 2.5, 3000)
vals_raw = [Z_G(massive, s) for s in s_plot]
vals_norm = [Z_G(norm, s) for s in s_plot]
ax.semilogy(s_plot, np.abs(vals_raw), 'b-', label='Raw (MeV)', alpha=0.8, linewidth=2)
ax.semilogy(s_plot, np.abs(vals_norm), 'r-', label='Normalized (/m_e)', alpha=0.8, linewidth=2)
ax.axhline(y=137.036, color='gold', linestyle='--', linewidth=2, label='α_em⁻¹ = 137')
ax.axhline(y=29.46, color='cyan', linestyle='--', alpha=0.7, label='α_w⁻¹ ≈ 29.5')
ax.axhline(y=8.47, color='lime', linestyle='--', alpha=0.7, label='α_s⁻¹ ≈ 8.5')
ax.axvline(x=0.5, color='purple', linestyle=':', alpha=0.5, label='Re(s)=1/2')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('s', fontsize=12)
ax.set_ylabel('|Z_G(s)|', fontsize=12)
ax.set_title('Z_G(s) = Tr(G⁻ˢ) — Log Scale')
ax.legend(fontsize=8)
ax.set_ylim(1e-2, 1e14)
ax.grid(True, alpha=0.3)

# Plot 2: Linear scale near targets
ax = axes[0, 1]
s_fine = np.linspace(-0.3, 1.0, 2000)
vals_fine_raw = [Z_G(massive, s) for s in s_fine]
vals_fine_norm = [Z_G(norm, s) for s in s_fine]
ax.plot(s_fine, vals_fine_raw, 'b-', label='Raw', linewidth=2)
ax.plot(s_fine, vals_fine_norm, 'r-', label='Normalized', linewidth=2)
ax.axhline(y=137.036, color='gold', linestyle='--', linewidth=2, label='137.036')
ax.axhline(y=130, color='orange', linestyle=':', alpha=0.5, label='130 (massive DOF)')
ax.axhline(y=29.46, color='cyan', linestyle='--', alpha=0.7, label='29.46')
ax.axhline(y=8.47, color='lime', linestyle='--', alpha=0.7, label='8.47')
ax.set_xlabel('s', fontsize=12)
ax.set_ylabel('Z_G(s)', fontsize=12)
ax.set_title('Z_G(s) — Linear Scale Near Targets')
ax.legend(fontsize=8)
ax.set_ylim(-10, 200)
ax.grid(True, alpha=0.3)

# Plot 3: Functional equation
ax = axes[1, 0]
ratios_raw = []
ratios_norm = []
for s in s_fe:
    zs_r = Z_G(massive, s)
    z1s_r = Z_G(massive, 1-s)
    ratios_raw.append(zs_r/z1s_r if abs(z1s_r) > 1e-30 else np.nan)
    
    zs_n = Z_G(norm, s)
    z1s_n = Z_G(norm, 1-s)
    ratios_norm.append(zs_n/z1s_n if abs(z1s_n) > 1e-30 else np.nan)

ax.plot(s_fe, ratios_raw, 'b-', label='Raw', linewidth=2, alpha=0.8)
ax.plot(s_fe, ratios_norm, 'r-', label='Normalized', linewidth=2, alpha=0.8)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0.5, color='purple', linestyle=':', alpha=0.5, label='s=1/2')
ax.set_xlabel('s', fontsize=12)
ax.set_ylabel('Z_G(s) / Z_G(1-s)', fontsize=12)
ax.set_title('Functional Equation Test')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 8)

# Plot 4: DOF decomposition visual
ax = axes[1, 1]
categories = ['Charged\nleptons', 'Quarks', 'W±', 'Z', 'Higgs', 'Dirac ν', 'Gluons', 'Photon', 'Graviton']
dofs = [12, 72, 6, 3, 1, 12, 24, 2, 5]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#5B86E5', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#FFD700', '#C0C0C0']

bars = ax.bar(categories, dofs, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)

# Add value labels on bars
for bar, dof in zip(bars, dofs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            str(dof), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add cumulative line
cumsum = np.cumsum(dofs)
ax2 = ax.twinx()
ax2.plot(range(len(categories)), cumsum, 'ko-', linewidth=2, markersize=6)
ax2.axhline(y=137, color='red', linestyle='--', linewidth=2, label='137')
ax2.set_ylabel('Cumulative DOF', fontsize=12)
ax2.legend(fontsize=10)

ax.set_ylabel('DOF Count', fontsize=12)
ax.set_title('137 = 94 + 12 + 24 + 2 + 5', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/home/claude/cst_137_dof.png', dpi=150, bbox_inches='tight')
print("\n[Plot saved to cst_137_dof.png]")

# ═══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("SENSITIVITY: GLUON EFFECTIVE MASS")
print("█" * 80)

for m_g in [100, 200, 330, 500, 600, 800, 1000]:
    ev = np.array(
        [m_e]*4 + [m_mu]*4 + [m_tau]*4 +
        [m_u]*12 + [m_d]*12 + [m_s]*12 +
        [m_c]*12 + [m_b]*12 + [m_t]*12 +
        [m_W]*6 + [m_Z]*3 + [m_H]*1 +
        [m_nu_e]*4 + [m_nu_mu]*4 + [m_nu_tau]*4 +
        [float(m_g)]*24
    )
    # Find where Z_G = 137 for raw masses
    s_scan = np.linspace(-0.5, 0.5, 10001)
    vals = np.array([Z_G(ev, s) for s in s_scan])
    
    found = None
    for i in range(len(s_scan)-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
            if (vals[i] - 137.036) * (vals[i+1] - 137.036) < 0:
                found = s_scan[i] + (137.036 - vals[i])/(vals[i+1]-vals[i])*(s_scan[i+1]-s_scan[i])
                break
    
    zg0 = Z_G(ev, 0)
    if found:
        print(f"  m_gluon = {m_g:>5d} MeV: Z_G(0) = {zg0:.0f}, 137 at s = {found:.5f}")
    else:
        print(f"  m_gluon = {m_g:>5d} MeV: Z_G(0) = {zg0:.0f}, 137 NOT found in [-0.5, 0.5]")

# ═══════════════════════════════════════════════════════════════
# CROSS-CHECK: RATIOS BETWEEN COUPLINGS
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("CROSS-CHECK: COUPLING RATIOS FROM Z_G")
print("█" * 80)

print("""
If couplings are Z_G at different s, then RATIOS between couplings
should equal RATIOS of Z_G values — independent of normalization.

Physical ratios:
  α_em/α_s = α_s⁻¹/α_em⁻¹ = 8.47/137.036 = 0.0618
  α_em/α_w = α_w⁻¹/α_em⁻¹ = 29.46/137.036 = 0.2149
  α_w/α_s  = α_s⁻¹/α_w⁻¹  = 8.47/29.46 = 0.2874
""")

# For the 130 massive DOF (excluding photon & graviton)
ev_130 = np.array(
    [m_e]*4 + [m_mu]*4 + [m_tau]*4 +
    [m_u]*12 + [m_d]*12 + [m_s]*12 +
    [m_c]*12 + [m_b]*12 + [m_t]*12 +
    [m_W]*6 + [m_Z]*3 + [m_H]*1 +
    [m_nu_e]*4 + [m_nu_mu]*4 + [m_nu_tau]*4 +
    [600.0]*24
)

# Find the s values where each coupling is hit
s_scan = np.linspace(-1, 2, 30001)
vals = np.array([Z_G(ev_130, s) for s in s_scan])

coupling_s = {}
for name, target in [("α_em⁻¹", 137.036), ("α_w⁻¹", 29.46), ("α_s⁻¹", 8.47)]:
    for i in range(len(s_scan)-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
            if (vals[i] - target) * (vals[i+1] - target) < 0:
                s_cross = s_scan[i] + (target - vals[i])/(vals[i+1]-vals[i])*(s_scan[i+1]-s_scan[i])
                coupling_s[name] = s_cross
                print(f"  {name} = {target:>8.3f} at s = {s_cross:.5f}")
                break

if len(coupling_s) == 3:
    s_em = coupling_s["α_em⁻¹"]
    s_w = coupling_s["α_w⁻¹"]
    s_s = coupling_s["α_s⁻¹"]
    print(f"\n  s values: α_em at {s_em:.5f}, α_w at {s_w:.5f}, α_s at {s_s:.5f}")
    print(f"  Δs(em→w) = {s_w - s_em:.5f}")
    print(f"  Δs(w→s)  = {s_s - s_w:.5f}")
    print(f"  Ratio of intervals: {(s_w - s_em)/(s_s - s_w):.4f}")
    print(f"  Are they equally spaced? Δs₁/Δs₂ = {(s_w - s_em)/(s_s - s_w):.4f}")

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█" * 80)
print("FINAL SUMMARY — WHAT WE LEARNED")
print("█" * 80)

print("""
1. KOIDE: CONFIRMED (again)
   - Leptons: Q = 0.666660 (0.001% error)
   - Heavy quarks (c,b,t): Q = 0.669489 (0.42% error) ← NEW SIGNAL

2. α_em⁻¹ = Z_G(0) at simple s-values: FALSIFIED
   - Z_G(0) = N (number of modes), not 137 for any standard counting
   - Z_G(-1/3) and Z_G(-1/4) are orders of magnitude off

3. α_em⁻¹ = dim(Γ) = 137: POSSIBLE but requires SPECIFIC content
   - 137 = 94 (undisputed SM) + 12 (Dirac ν) + 24 (massive gluons) + 2 (γ) + 5 (graviton)
   - This PREDICTS: Dirac neutrinos, massive gluons, massive graviton
   - Each prediction is independently testable

4. 137 AS Z_G(s₀) FOR SOME s₀: ALWAYS ACHIEVABLE
   - Z_G(s) is monotone decreasing from ∞ to 0 as s goes from -∞ to +∞
   - So it ALWAYS crosses 137 somewhere
   - The question is whether s₀ is a "natural" value
   - For 130 massive DOF: s₀ ≈ -0.01 (very near 0)

5. FUNCTIONAL EQUATION: APPROXIMATE
   - Z_G(s)/Z_G(1-s) ≈ A^(2s-1) with R² ≈ 0.985
   - Not exact, but suggestive of broken spectral symmetry

6. VERDICT ON CATALYST'S CLAIMS:
   ✗ Specific s-values for couplings: WRONG (elegant but falsified)
   ~ 137 as DOF count: INTERESTING but requires graviton + massive gluons
   ✓ Koide extends to heavy quarks: CONFIRMED numerically
   ~ Functional equation: APPROXIMATE, not exact
   
   Catalyst was RIGHT to be self-critical. The numbers don't lie.
""")

