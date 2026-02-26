"""
Negative controls for phase-shift isospectrality:
1) Dirichlet BC (breaks translational symmetry)
2) Non-uniform grid (breaks circulant structure)
Compare with periodic uniform (should show isospectrality)
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

a, hbar = 3.0, 0.42

def build_periodic_uniform(N, delta=0.0):
    """Standard Witten Laplacian, periodic BC, uniform grid"""
    dx = 2*np.pi/N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    dW = -a * np.sin(x + delta)
    d2W = -a * np.cos(x + delta)
    Veff = dW**2 - hbar * d2W
    
    diag_main = 2.0/dx**2 + Veff
    diag_off = -1.0/dx**2 * np.ones(N)
    H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N,N), format='lil')
    H[0, N-1] = -1.0/dx**2  # periodic BC
    H[N-1, 0] = -1.0/dx**2
    return H.tocsr(), x

def build_dirichlet(N, delta=0.0):
    """Standard Witten Laplacian, DIRICHLET BC (no periodicity)"""
    dx = 2*np.pi/(N+1)
    x = np.linspace(dx, 2*np.pi - dx, N)  # interior points only
    dW = -a * np.sin(x + delta)
    d2W = -a * np.cos(x + delta)
    Veff = dW**2 - hbar * d2W
    
    diag_main = 2.0/dx**2 + Veff
    diag_off = -1.0/dx**2 * np.ones(N-1)
    H = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N,N), format='csr')
    return H, x

def build_nonuniform(N, delta=0.0, seed=42):
    """Standard Witten Laplacian, periodic BC, NON-UNIFORM grid (jittered)"""
    rng = np.random.default_rng(seed)
    dx0 = 2*np.pi/N
    # Jitter each point by up to +/- 30% of dx
    x = np.sort(np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(-0.3*dx0, 0.3*dx0, N)) % (2*np.pi)
    x = np.sort(x)
    
    dW = -a * np.sin(x + delta)
    d2W = -a * np.cos(x + delta)
    Veff = dW**2 - hbar * d2W
    
    # Variable spacing
    dx_fwd = np.roll(x, -1) - x
    dx_fwd[-1] += 2*np.pi  # wrap-around
    dx_bwd = x - np.roll(x, 1)
    dx_bwd[0] += 2*np.pi
    dx_avg = 0.5 * (dx_fwd + dx_bwd)
    
    # Non-uniform Laplacian with periodic BC
    N_pts = len(x)
    rows, cols, vals = [], [], []
    for i in range(N_pts):
        ip = (i+1) % N_pts
        im = (i-1) % N_pts
        
        rows.append(i); cols.append(i)
        vals.append(1.0/(dx_avg[i]*dx_fwd[i]) + 1.0/(dx_avg[i]*dx_bwd[i]) + Veff[i])
        
        rows.append(i); cols.append(ip)
        vals.append(-1.0/(dx_avg[i]*dx_fwd[i]))
        
        rows.append(i); cols.append(im)
        vals.append(-1.0/(dx_avg[i]*dx_bwd[i]))
    
    H = sparse.csr_matrix((vals, (rows, cols)), shape=(N_pts, N_pts))
    return H, x

print("="*80)
print("NEGATIVE CONTROL: Phase-shift isospectrality test")
print("δ = 0.37·dx (non-commensurable)")
print("="*80)

for N in [64, 256]:
    dx = 2*np.pi/N
    delta = 0.37 * dx
    
    print(f"\n{'='*60}")
    print(f"N = {N},  δ = 0.37·dx = {delta:.6f} rad")
    print(f"{'='*60}")
    
    for label, builder in [("PERIODIC UNIFORM", build_periodic_uniform),
                            ("DIRICHLET BC", build_dirichlet),
                            ("NON-UNIFORM GRID", build_nonuniform)]:
        H0, x0 = builder(N, delta=0.0)
        Hd, xd = builder(N, delta=delta)
        
        # Matrix norm difference
        dH = Hd - H0
        dH_norm = sparse.linalg.norm(dH, 'fro') / sparse.linalg.norm(H0, 'fro')
        
        # Eigenvalues
        k = min(6, N-2)
        e0 = eigsh(H0, k=k, which='SA', return_eigenvectors=False); e0.sort()
        ed = eigsh(Hd, k=k, which='SA', return_eigenvectors=False); ed.sort()
        
        dl1 = abs(ed[1] - e0[1]) / max(abs(e0[1]), 1e-30)
        dl2 = abs(ed[2] - e0[2]) / max(abs(e0[2]), 1e-30)
        
        print(f"\n  {label}:")
        print(f"    ||ΔH||_F/||H||_F  = {dH_norm:.6e}")
        print(f"    |Δλ₁/λ₁|          = {dl1:.6e}")
        print(f"    |Δλ₂/λ₂|          = {dl2:.6e}")
        print(f"    λ₀(0)={e0[0]:.8e}  λ₀(δ)={ed[0]:.8e}")
        print(f"    λ₁(0)={e0[1]:.8e}  λ₁(δ)={ed[1]:.8e}")
        print(f"    λ₂(0)={e0[2]:.8e}  λ₂(δ)={ed[2]:.8e}")

