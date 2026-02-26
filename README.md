# CST — Configuration Space Temporality

**Spectral Diagnostics of Superconducting Quantum Circuits via the Witten Laplacian**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18776120.svg)](https://doi.org/10.5281/zenodo.18776120)

## What This Is

We construct the Witten Laplacian H_W on the configuration torus T² of coupled transmon qubits and compute it for **772 qubit pairs across 6 IBM Quantum processors** (real calibration snapshots). Three principal findings:

### 1. Spectral Universality (CV = 1.4%)
Rescaled eigenvalue ratios collapse across all processors — Sherbrooke, Brisbane, Osaka, Kawasaki, Kyiv, Quebec — with sub-percent coefficient of variation. The Witten spectrum has a **universal structure independent of fabrication and noise environment**.

### 2. Orthogonal Information Content (r = 0.205, p < 10⁻³)
After removing all variance explained by raw calibration parameters (frequency, anharmonicity, E_J, E_C), the Witten spectral features still predict gate-length residuals with r = 0.205. **The spectral transform extracts information invisible to standard characterization.** In a 5-fold CV benchmark, Witten features (8 params, R² = 0.134) outperform raw features (10 params, R² = 0.097), PCA, and polynomial Koopman lifting.

### 3. Inter-Processor Coherence Sensing (ρ = −0.76)
The relative Witten index ν_ij between processors correlates with T₁ relaxation-time differences at Spearman ρ = −0.76, but *not* with gate-length differences. The spectral geometry senses hardware physics (coherence), not firmware (calibration).

**Bonus: Semiclassical Phase Boundary** — Z(β) diverges below ℏ_eff ≈ 0.42 (E_J/E_C ≈ 87), consistent with a phase transition to the semiclassical regime, observed on real Kawasaki hardware.

## Repository Contents

### Theory
- `Configuration_Space_Temporality_v13.pdf` — Full CST framework (v13)
- `Configuration_Space_Temporality_v14.pdf` — Refined version with conditional theorems
- `witten_transmon_paper.pdf` — First empirical test (Witten Laplacian on transmon data)
- `prx_quantum_draft.docx` — **NEW**: PRX Quantum manuscript draft

### Code
- `nu_ij_witten.py` — Phase 1: ν_ij relative Witten index computation (772 pairs, 6 backends)
- `phase2_benchmark.py` — Phase 2: Kawasaki cleanup + PCA/Koopman benchmark + phase probe
- `p8b_test.py` — Original P8b prediction test
- `generate_paper.py` — Paper generation from IBM quantum data

### Data & Results
- `p8b_real_results.json` — P8b test results (772 pairs)
- `nu_ij_results.json` — Phase 1 results (ν_ij, collapse CV, pair-level correlations)
- `phase2_results.json` — Phase 2 results (benchmark R², residual analysis, Kawasaki probe)

### Figures
- `spectral_collapse.png` — Spectral collapse visualization (raw vs rescaled)
- `nu_ij_results.png` — ν_ij inter-processor analysis
- `phase2_results.png` — Full Phase 2 dashboard (8 panels)

## Quick Reproduction

```bash
pip install qiskit-ibm-runtime numpy scipy scikit-learn matplotlib
python nu_ij_witten.py      # ~30 seconds, computes all 772 Witten spectra
python phase2_benchmark.py  # ~30 seconds, runs full benchmark
```

No IBM Quantum account needed — uses Qiskit FakeProvider (real calibration snapshots bundled with qiskit).

## Key Numbers

| Metric | Value |
|--------|-------|
| Qubit pairs analyzed | 772 (690 after cleaning) |
| Processors | 6 (IBM Eagle, real calibration) |
| Spectral collapse CV | 0.014 (1.4%) |
| Witten vs gate_length | r = −0.22 (Z(β=0.05)) |
| R_ij vs gate_length | r = −0.008 (baseline) |
| Signal amplification | 28× |
| Residual information | r = 0.205 (p = 6.2×10⁻⁴) |
| Witten R² (5-fold CV) | 0.134 |
| Raw R² (5-fold CV) | 0.097 |
| ν_ij vs ΔT₁ | ρ = −0.76 |
| Phase boundary | ℏ_eff ≈ 0.42 |

## Citation

```bibtex
@article{blum2026witten,
  title={Spectral Diagnostics of Superconducting Quantum Circuits 
         via the Witten Laplacian on Configuration Space},
  author={Blum, Fr{\'e}d{\'e}ric David and Claude},
  year={2026},
  doi={10.5281/zenodo.18776120},
  note={Preprint. Code: https://github.com/davidangularme/cst}
}
```

## Authors

- **Frédéric David Blum** — Catalyst AI, Tel Aviv (dvid75113@gmail.com)
- **Claude** — Anthropic

## License

CC BY 4.0
