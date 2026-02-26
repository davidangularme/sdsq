"""
Spectral Fidelity of the Discrete Witten Laplacian on S¹:
Instanton Analysis vs. Numerical Discretization

Authors: F.D. Blum, Claude (Anthropic), Catalyst AI
Date: February 26, 2026

Purpose:
  Diagnose whether negative eigenvalues (λ₀ < 0) observed in transmon
  spectral analysis are artifacts of discretization or signals of physics.
  
  The Witten Laplacian H_W^(0) = d_W† d_W is positive semi-definite by
  construction. Any numerical λ < 0 must be an implementation artifact.
  This script maps exactly WHERE discretization breaks semi-positivity.

Method:
  1. Define double-well Morse potential W(x) = a·cos(x) on S¹ = [0, 2π)
  2. Construct H_W^(0) = -ℏ² d²/dx² + (W')² - ℏ·W'' analytically
  3. Compute exact low-lying spectrum via:
     a. Instanton methods (Coleman 1985) for tunnel splitting
     b. Harmonic approximation at minima for excited states
  4. Discretize H_W^(0) on uniform grid with N points, periodic BC
  5. Compare numerical vs analytical spectrum across N = 16 to 4096
  6. Track: minimum eigenvalue, spectral gap, convergence rate
  7. Identify resolution threshold below which λ₀ < 0 appears

References:
  - E. Witten, "Supersymmetry and Morse Theory" (1982)
  - S. Coleman, "Aspects of Symmetry" Ch. 7 (1985)
  - CST v14: DOI 10.5281/zenodo.18776120
  - SDSQ paper: DOI 10.5281/zenodo.18779189
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os

# ============================================================
# I. POTENTIAL AND WITTEN LAPLACIAN ON S¹
# ============================================================

class WittenLaplacianS1:
    """
    Witten Laplacian H_W^(0) on S¹ = [0, 2π) with periodic BC.
    
    For superpotential W(x), the Witten Laplacian on 0-forms is:
      H_W = -ℏ² d²/dx² + (W')² - ℏ·W''
    
    This is manifestly H_W = d_W† d_W ≥ 0 where d_W = ℏ·d/dx + W'.
    Semi-positivity is EXACT in the continuum. Any violation is
    a discretization artifact.
    """
    
    def __init__(self, W_func, dW_func, d2W_func, hbar=1.0):
        """
        W_func:   W(x) — superpotential
        dW_func:  W'(x) — first derivative
        d2W_func: W''(x) — second derivative
        hbar:     effective Planck constant
        """
        self.W = W_func
        self.dW = dW_func
        self.d2W = d2W_func
        self.hbar = hbar
    
    def V_eff(self, x):
        """Effective potential: V(x) = (W')² - ℏ·W'' """
        return self.dW(x)**2 - self.hbar * self.d2W(x)
    
    def build_matrix(self, N):
        """
        Discretize H_W on uniform grid with N points, periodic BC.
        
        Uses standard 3-point stencil for d²/dx²:
          d²f/dx² ≈ (f_{i+1} - 2f_i + f_{i-1}) / dx²
        
        Returns sparse matrix and grid points.
        """
        dx = 2 * np.pi / N
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # Kinetic term: -ℏ² d²/dx² with periodic BC
        diag_main = np.full(N, 2.0 * self.hbar**2 / dx**2)
        diag_off = np.full(N - 1, -self.hbar**2 / dx**2)
        
        H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], 
                         shape=(N, N), format='lil')
        
        # Periodic boundary conditions
        H[0, N-1] = -self.hbar**2 / dx**2
        H[N-1, 0] = -self.hbar**2 / dx**2
        
        # Potential term: V_eff(x_i) on diagonal
        V = self.V_eff(x)
        for i in range(N):
            H[i, i] += V[i]
        
        return H.tocsr(), x
    
    def build_matrix_factored(self, N):
        """
        Alternative: build H_W = d_W† d_W directly from the factored form.
        
        d_W = ℏ·d/dx + W'(x) discretized as matrix, then H = d_W^T · d_W.
        This GUARANTEES semi-positivity even after discretization.
        """
        dx = 2 * np.pi / N
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # d_W = ℏ·(forward difference) + W'(x)
        # Forward difference: (f_{i+1} - f_i) / dx with periodic BC
        dW_mat = sparse.lil_matrix((N, N))
        for i in range(N):
            dW_mat[i, i] = -self.hbar / dx + self.dW(x[i])
            dW_mat[i, (i + 1) % N] = self.hbar / dx
        
        dW_csr = dW_mat.tocsr()
        H = dW_csr.T @ dW_csr
        
        return H, x
    
    def spectrum(self, N, k=20, method='standard'):
        """Compute k lowest eigenvalues."""
        if method == 'standard':
            H, x = self.build_matrix(N)
        elif method == 'factored':
            H, x = self.build_matrix_factored(N)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        k = min(k, N - 2)
        eigenvalues = eigsh(H, k=k, which='SA', return_eigenvectors=False)
        eigenvalues.sort()
        
        return eigenvalues, x


# ============================================================
# II. DOUBLE-WELL POTENTIAL ON S¹
# ============================================================

def make_double_well(a):
    """
    Double-well Morse potential on S¹:
      W(x) = a · cos(x)
      W'(x) = -a · sin(x)
      W''(x) = -a · cos(x)
    
    Critical points: x = 0 (max of W, min of V_eff) and x = π (min of W)
    Morse indices: 0 at x=0, 0 at x=π → two minima of V_eff
    
    The effective potential V = (W')² - ℏW'' = a²sin²(x) + ℏa·cos(x)
    has minima near x = 0 and x = π (double well).
    """
    W = lambda x: a * np.cos(x)
    dW = lambda x: -a * np.sin(x)
    d2W = lambda x: -a * np.cos(x)
    return W, dW, d2W


def analytical_spectrum_double_well(a, hbar, n_levels=10):
    """
    Analytical predictions for the double-well W = a·cos(x) on S¹.
    
    Near x = 0: V_eff ≈ a²x² + ℏa → harmonic with ω₀ = a√2, offset ℏa
    Near x = π: V_eff ≈ a²(x-π)² - ℏa → harmonic with ω₀ = a√2, offset -ℏa
    
    Instanton splitting (Coleman):
      ΔE ~ (ω₀/π)^(1/2) · exp(-S_inst/ℏ)
    where S_inst = ∫ |W'| dx between minima = 2a (for cos potential)
    
    Returns dict with predictions and uncertainties.
    """
    # Harmonic frequency at each minimum
    # V_eff''(0) = 2a² - ℏa·(-1) = 2a² + ℏa  (at x=0, cos''(0) = -1)
    # More carefully: V = a²sin²x + ℏa·cos(x)
    # V'(x) = 2a²sin(x)cos(x) - ℏa·sin(x) = sin(x)(2a²cos(x) - ℏa)
    # V''(x) = cos(x)(2a²cos(x) - ℏa) + sin(x)(-2a²sin(x))
    # V''(0) = 2a² - ℏa
    # V''(π) = 2a² - ℏa  (by symmetry of cos)
    
    omega_sq_0 = 2 * a**2 - hbar * a  # V''(0) — curvature at x=0 minimum
    omega_sq_pi = 2 * a**2 + hbar * a  # V''(π) — curvature at x=π minimum
    
    # More careful: V(0) = 0 + ℏa, V(π) = 0 - ℏa
    # So x=π is the deeper minimum
    V_offset_0 = hbar * a
    V_offset_pi = -hbar * a  # But V_eff must be ≥ 0... 
    # Actually V_eff(π) = a²sin²(π) + ℏa·cos(π) = 0 - ℏa = -ℏa < 0!
    # This seems to violate semi-positivity but it doesn't —
    # the EIGENVALUES of H_W are ≥ 0, not V_eff pointwise.
    
    omega_0 = np.sqrt(abs(omega_sq_0)) if omega_sq_0 > 0 else 0
    omega_pi = np.sqrt(abs(omega_sq_pi)) if omega_sq_pi > 0 else 0
    
    # Harmonic approximation for each well
    levels_0 = [V_offset_0 + hbar * omega_0 * (n + 0.5) for n in range(n_levels)]
    levels_pi = [V_offset_pi + hbar * omega_pi * (n + 0.5) for n in range(n_levels)]
    
    # Instanton action: S = ∫₀^π |W'(x)| dx = ∫₀^π a|sin(x)| dx = 2a
    S_inst = 2 * a
    
    # Tunnel splitting (WKB/instanton)
    tunnel_splitting = np.sqrt(2 * max(omega_0, omega_pi) / np.pi) * np.exp(-S_inst / hbar)
    
    # Ground state energy (from supersymmetry): should be 0 if SUSY unbroken
    # On S¹ with W = a·cos(x), SUSY is unbroken if W has critical points
    # (which it does: x=0 and x=π). So E₀ = 0 exactly.
    E0_exact = 0.0
    
    return {
        'E0_exact': E0_exact,
        'omega_0': omega_0,
        'omega_pi': omega_pi,
        'V_offset_0': V_offset_0,
        'V_offset_pi': V_offset_pi,
        'S_instanton': S_inst,
        'tunnel_splitting': tunnel_splitting,
        'harmonic_levels_0': levels_0,
        'harmonic_levels_pi': levels_pi,
    }


# ============================================================
# III. CONVERGENCE STUDY
# ============================================================

def convergence_study(a_values, hbar_values, N_values, k=12):
    """
    Systematic study: for each (a, ℏ), compute spectrum at each N
    using both standard and factored discretizations.
    
    Track:
    - Minimum eigenvalue (should be ≥ 0)
    - Where it goes negative (standard method)
    - Whether factored method preserves positivity
    - Convergence rate of lowest eigenvalues
    """
    results = []
    
    for a in a_values:
        for hbar in hbar_values:
            W, dW, d2W = make_double_well(a)
            witten = WittenLaplacianS1(W, dW, d2W, hbar=hbar)
            analytical = analytical_spectrum_double_well(a, hbar)
            
            for N in N_values:
                print(f"  a={a:.1f}, ℏ={hbar:.2f}, N={N}...", end="", flush=True)
                
                # Standard discretization
                try:
                    eigs_std, _ = witten.spectrum(N, k=k, method='standard')
                    lambda_min_std = float(eigs_std[0])
                    eigs_std_list = eigs_std.tolist()
                except Exception as e:
                    lambda_min_std = None
                    eigs_std_list = []
                    print(f" STD ERROR: {e}", end="")
                
                # Factored discretization (guaranteed ≥ 0)
                try:
                    eigs_fac, _ = witten.spectrum(N, k=k, method='factored')
                    lambda_min_fac = float(eigs_fac[0])
                    eigs_fac_list = eigs_fac.tolist()
                except Exception as e:
                    lambda_min_fac = None
                    eigs_fac_list = []
                    print(f" FAC ERROR: {e}", end="")
                
                result = {
                    'a': a,
                    'hbar': hbar,
                    'N': N,
                    'dx': 2 * np.pi / N,
                    'lambda_min_standard': lambda_min_std,
                    'lambda_min_factored': lambda_min_fac,
                    'negative_standard': lambda_min_std is not None and lambda_min_std < -1e-12,
                    'negative_factored': lambda_min_fac is not None and lambda_min_fac < -1e-12,
                    'spectrum_standard': eigs_std_list[:k],
                    'spectrum_factored': eigs_fac_list[:k],
                    'E0_exact': analytical['E0_exact'],
                    'tunnel_splitting': analytical['tunnel_splitting'],
                    'S_instanton': analytical['S_instanton'],
                }
                results.append(result)
                print(" ✓")
    
    return results


# ============================================================
# IV. VISUALIZATION
# ============================================================

def plot_results(results, a_values, hbar_values, N_values, output_dir):
    """Generate comprehensive diagnostic figures."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.rcParams.update({
        'figure.facecolor': '#0a0a2e',
        'axes.facecolor': '#12123a',
        'text.color': '#ffffff',
        'axes.labelcolor': '#ffffff',
        'xtick.color': '#aaaaaa',
        'ytick.color': '#aaaaaa',
        'axes.edgecolor': '#333366',
        'grid.color': '#222255',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
    })
    
    # ---- FIGURE 1: λ_min vs N for both methods ----
    fig, axes = plt.subplots(len(a_values), len(hbar_values), 
                              figsize=(5*len(hbar_values), 4*len(a_values)),
                              squeeze=False)
    fig.suptitle('Minimum Eigenvalue vs Grid Resolution\n'
                 'Standard (●) vs Factored (▲) Discretization',
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    
    for i, a in enumerate(a_values):
        for j, hbar in enumerate(hbar_values):
            ax = axes[i][j]
            
            subset = [r for r in results if r['a'] == a and r['hbar'] == hbar]
            Ns = [r['N'] for r in subset]
            lmin_std = [r['lambda_min_standard'] for r in subset]
            lmin_fac = [r['lambda_min_factored'] for r in subset]
            
            ax.semilogx(Ns, lmin_std, 'o-', color='#ff6b6b', label='Standard', 
                       markersize=6, linewidth=1.5)
            ax.semilogx(Ns, lmin_fac, '^-', color='#4ecdc4', label='Factored',
                       markersize=6, linewidth=1.5)
            ax.axhline(y=0, color='#ffd93d', linestyle='--', linewidth=1, alpha=0.7,
                      label='Semi-positivity bound')
            
            # Shade negative region
            ax.axhspan(min(min(lmin_std), min(lmin_fac), -0.1), 0, 
                       alpha=0.15, color='#ff6b6b')
            
            ax.set_xlabel('Grid points N (log scale)')
            ax.set_ylabel('λ_min')
            ax.set_title(f'a = {a}, ℏ = {hbar}', fontsize=12, color='#4ecdc4')
            ax.legend(fontsize=9, loc='lower right', 
                     facecolor='#1a1a4e', edgecolor='#333366')
            ax.grid(True)
            
            # Mark first N where standard goes negative
            neg_Ns = [r['N'] for r in subset if r['negative_standard']]
            if neg_Ns:
                ax.axvline(x=min(neg_Ns), color='#ff6b6b', linestyle=':', 
                          alpha=0.5, linewidth=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{output_dir}/fig1_lambda_min_convergence.png', dpi=150, 
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Figure 1: λ_min convergence")
    
    # ---- FIGURE 2: Full spectrum convergence ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Spectral Convergence: First 10 Eigenvalues\n'
                 f'Double-well W = a·cos(x) on S¹',
                 fontsize=16, fontweight='bold', color='white')
    
    # Pick one representative case
    a_rep, hbar_rep = a_values[1], hbar_values[1]
    subset = [r for r in results if r['a'] == a_rep and r['hbar'] == hbar_rep]
    
    for method_idx, (method_key, method_label, color) in enumerate([
        ('spectrum_standard', 'Standard Discretization', '#ff6b6b'),
        ('spectrum_factored', 'Factored (d_W† d_W)', '#4ecdc4')
    ]):
        ax = axes[method_idx]
        
        for r in subset:
            N = r['N']
            eigs = r[method_key][:10]
            if len(eigs) > 0:
                alpha = 0.3 + 0.7 * (np.log2(N) - np.log2(min(N_values))) / \
                        (np.log2(max(N_values)) - np.log2(min(N_values)))
                ax.plot(range(len(eigs)), eigs, 'o-', color=color, alpha=alpha,
                       markersize=4, linewidth=1, label=f'N={N}' if N in [min(N_values), max(N_values)] else None)
        
        ax.axhline(y=0, color='#ffd93d', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Eigenvalue index n')
        ax.set_ylabel('λ_n')
        ax.set_title(f'{method_label}\na={a_rep}, ℏ={hbar_rep}', 
                    fontsize=12, color=color)
        ax.legend(fontsize=9, facecolor='#1a1a4e', edgecolor='#333366')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f'{output_dir}/fig2_spectrum_convergence.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Figure 2: Spectrum convergence")
    
    # ---- FIGURE 3: Effective potential landscape ----
    fig, axes = plt.subplots(len(a_values), len(hbar_values),
                              figsize=(5*len(hbar_values), 4*len(a_values)),
                              squeeze=False)
    fig.suptitle('Effective Potential V_eff(x) = (W\')² − ℏ·W\'\'\n'
                 'Shaded region: V_eff < 0 (does NOT imply λ < 0)',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    
    x_plot = np.linspace(0, 2*np.pi, 500)
    
    for i, a in enumerate(a_values):
        for j, hbar in enumerate(hbar_values):
            ax = axes[i][j]
            W, dW, d2W = make_double_well(a)
            
            V_eff = dW(x_plot)**2 - hbar * d2W(x_plot)
            Wprime = dW(x_plot)
            
            ax.plot(x_plot, V_eff, color='#a78bfa', linewidth=2, label='V_eff(x)')
            ax.plot(x_plot, Wprime**2, color='#4ecdc4', linewidth=1, 
                   linestyle='--', alpha=0.6, label="(W')²")
            ax.plot(x_plot, -hbar * d2W(x_plot), color='#ff6b6b', linewidth=1,
                   linestyle=':', alpha=0.6, label="-ℏW''")
            
            # Shade negative V_eff regions
            ax.fill_between(x_plot, V_eff, 0, where=(V_eff < 0),
                           alpha=0.2, color='#ff6b6b')
            ax.axhline(y=0, color='#ffd93d', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Mark critical points of W
            ax.axvline(x=0, color='#ffd93d', linestyle=':', alpha=0.3)
            ax.axvline(x=np.pi, color='#ffd93d', linestyle=':', alpha=0.3)
            ax.text(0.05, 0.95, f'a={a}, ℏ={hbar}', transform=ax.transAxes,
                   fontsize=10, color='#4ecdc4', va='top')
            
            ax.set_xlabel('x')
            ax.set_ylabel('V_eff(x)')
            ax.legend(fontsize=8, facecolor='#1a1a4e', edgecolor='#333366', loc='upper right')
            ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{output_dir}/fig3_effective_potential.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Figure 3: Effective potential")
    
    # ---- FIGURE 4: Negativity phase diagram ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Phase Diagram: Where Discretization Breaks Semi-Positivity\n'
                 'Standard method only (factored is always ≥ 0 by construction)',
                 fontsize=14, fontweight='bold', color='white')
    
    # For each (a, hbar), find critical N where λ_min first becomes positive
    for a in a_values:
        for hbar in hbar_values:
            subset = sorted([r for r in results if r['a'] == a and r['hbar'] == hbar],
                           key=lambda r: r['N'])
            Ns = [r['N'] for r in subset]
            is_neg = [r['negative_standard'] for r in subset]
            lmin = [r['lambda_min_standard'] for r in subset]
            
            color = plt.cm.viridis(hbar / max(hbar_values))
            marker_colors = ['#ff6b6b' if neg else '#4ecdc4' for neg in is_neg]
            
            ax.scatter(Ns, [a]*len(Ns), c=marker_colors, s=80, zorder=3,
                      edgecolors='white', linewidth=0.5)
            
            # Label
            if hbar == hbar_values[0]:
                ax.text(Ns[-1]*1.2, a, f'ℏ={hbar}', fontsize=9, color='#aaa',
                       va='center')
    
    ax.set_xscale('log')
    ax.set_xlabel('Grid resolution N', fontsize=12)
    ax.set_ylabel('Potential amplitude a', fontsize=12)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
               markersize=10, label='λ_min < 0 (ARTIFACT)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4',
               markersize=10, label='λ_min ≥ 0 (correct)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left',
             facecolor='#1a1a4e', edgecolor='#333366')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f'{output_dir}/fig4_negativity_phase_diagram.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Figure 4: Negativity phase diagram")
    
    # ---- FIGURE 5: Error scaling (convergence rate) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Convergence Rate: |λ₀(N) - λ₀(N_max)| vs dx²\n'
                 'Expected: O(dx²) for 3-point stencil',
                 fontsize=14, fontweight='bold', color='white')
    
    for method_idx, (method_key, method_label, color) in enumerate([
        ('lambda_min_standard', 'Standard', '#ff6b6b'),
        ('lambda_min_factored', 'Factored', '#4ecdc4')
    ]):
        ax = axes[method_idx]
        
        for a in a_values:
            for hbar in hbar_values:
                subset = sorted([r for r in results if r['a'] == a and r['hbar'] == hbar],
                               key=lambda r: r['N'])
                
                if not subset or subset[-1][method_key] is None:
                    continue
                    
                ref_val = subset[-1][method_key]  # Highest resolution as reference
                
                dxs = [r['dx'] for r in subset[:-1]]
                errors = [abs(r[method_key] - ref_val) for r in subset[:-1] 
                         if r[method_key] is not None]
                dxs = dxs[:len(errors)]
                
                if len(errors) > 2 and all(e > 0 for e in errors):
                    ax.loglog(dxs, errors, 'o-', color=color, alpha=0.7,
                             markersize=5, linewidth=1,
                             label=f'a={a}, ℏ={hbar}')
        
        # Reference line: O(dx²)
        dx_ref = np.array([0.01, 1.0])
        ax.loglog(dx_ref, 0.5*dx_ref**2, '--', color='#ffd93d', alpha=0.5,
                 linewidth=2, label='O(dx²)')
        
        ax.set_xlabel('dx = 2π/N')
        ax.set_ylabel('|λ₀(N) - λ₀(ref)|')
        ax.set_title(method_label, fontsize=12, color=color)
        ax.legend(fontsize=8, facecolor='#1a1a4e', edgecolor='#333366')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f'{output_dir}/fig5_convergence_rate.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Figure 5: Convergence rate")


def generate_summary_table(results, a_values, hbar_values, N_values):
    """Print summary table of key findings."""
    
    print("\n" + "="*90)
    print("SUMMARY: SPECTRAL FIDELITY OF DISCRETE WITTEN LAPLACIAN ON S¹")
    print("="*90)
    
    print(f"\n{'a':>5} {'ℏ':>6} {'N':>6} {'λ_min(std)':>14} {'λ_min(fac)':>14} {'neg(std)':>10} {'neg(fac)':>10}")
    print("-"*90)
    
    n_negative_std = 0
    n_negative_fac = 0
    
    for r in results:
        neg_std = "⚠ YES" if r['negative_standard'] else "✓ no"
        neg_fac = "⚠ YES" if r['negative_factored'] else "✓ no"
        
        if r['negative_standard']:
            n_negative_std += 1
        if r['negative_factored']:
            n_negative_fac += 1
        
        lmin_std = f"{r['lambda_min_standard']:.8f}" if r['lambda_min_standard'] is not None else "ERROR"
        lmin_fac = f"{r['lambda_min_factored']:.8f}" if r['lambda_min_factored'] is not None else "ERROR"
        
        print(f"{r['a']:5.1f} {r['hbar']:6.2f} {r['N']:6d} {lmin_std:>14} {lmin_fac:>14} {neg_std:>10} {neg_fac:>10}")
    
    total = len(results)
    print("-"*90)
    print(f"\nTotal configurations tested: {total}")
    print(f"Standard method — negative eigenvalues: {n_negative_std}/{total} ({100*n_negative_std/total:.1f}%)")
    print(f"Factored method — negative eigenvalues: {n_negative_fac}/{total} ({100*n_negative_fac/total:.1f}%)")
    
    # Find critical resolution for each (a, hbar)
    print(f"\n{'a':>5} {'ℏ':>6} {'N_critical (std)':>20} {'Conclusion':>40}")
    print("-"*75)
    for a in a_values:
        for hbar in hbar_values:
            subset = sorted([r for r in results if r['a'] == a and r['hbar'] == hbar],
                           key=lambda r: r['N'])
            neg_Ns = [r['N'] for r in subset if r['negative_standard']]
            pos_Ns = [r['N'] for r in subset if not r['negative_standard']]
            
            if neg_Ns and pos_Ns:
                N_crit = min(pos_Ns)
                conclusion = f"Semi-positivity restored at N ≥ {N_crit}"
            elif neg_Ns:
                conclusion = "⚠ Negative at ALL tested resolutions"
            else:
                conclusion = "✓ Always positive"
            
            N_crit_str = str(min(pos_Ns)) if pos_Ns and neg_Ns else "—"
            print(f"{a:5.1f} {hbar:6.2f} {N_crit_str:>20} {conclusion:>40}")
    
    print("\n" + "="*90)
    print("KEY FINDING: The factored discretization H = d_W† · d_W preserves")
    print("semi-positivity BY CONSTRUCTION at ALL resolutions.")
    print("Standard 3-point stencil may produce λ₀ < 0 at coarse grids.")
    print("This is an IMPLEMENTATION DIAGNOSTIC, not physics.")
    print("="*90)
    
    return {
        'n_negative_std': n_negative_std,
        'n_negative_fac': n_negative_fac,
        'total': total,
    }


# ============================================================
# V. MAIN
# ============================================================

if __name__ == '__main__':
    
    output_dir = '/home/claude/s1_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter ranges
    a_values = [1.0, 3.0, 10.0]          # Potential amplitude (weak → strong)
    hbar_values = [0.1, 0.42, 1.0, 2.0]  # ℏ_eff (note: 0.42 = transmon threshold!)
    N_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    
    print("="*60)
    print("WITTEN LAPLACIAN ON S¹ — SPECTRAL FIDELITY TEST")
    print("Double-well: W(x) = a·cos(x), periodic BC")
    print(f"Amplitudes: {a_values}")
    print(f"ℏ values: {hbar_values}")
    print(f"Grid sizes: {N_values}")
    print(f"Total configurations: {len(a_values)*len(hbar_values)*len(N_values)}")
    print("="*60 + "\n")
    
    # Run convergence study
    results = convergence_study(a_values, hbar_values, N_values)
    
    # Summary
    summary = generate_summary_table(results, a_values, hbar_values, N_values)
    
    # Save raw results
    results_file = f'{output_dir}/spectral_fidelity_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_results(results, a_values, hbar_values, N_values, output_dir)
    
    # Analytical predictions
    print("\n" + "="*60)
    print("ANALYTICAL PREDICTIONS (instanton + harmonic)")
    print("="*60)
    for a in a_values:
        for hbar in hbar_values:
            pred = analytical_spectrum_double_well(a, hbar)
            print(f"\na={a}, ℏ={hbar}:")
            print(f"  E₀ (exact, SUSY) = {pred['E0_exact']}")
            print(f"  S_instanton = {pred['S_instanton']:.4f}")
            print(f"  Tunnel splitting ≈ {pred['tunnel_splitting']:.2e}")
            print(f"  ω at x=0: {pred['omega_0']:.4f}")
            print(f"  ω at x=π: {pred['omega_pi']:.4f}")
    
    print("\n✅ All done. Figures in:", output_dir)
