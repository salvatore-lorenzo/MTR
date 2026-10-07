# Mind the rank! Finite-resource effects in quantum linear regression

Code and data for the numerical results of

> G. Lo Monaco, S. Lorenzo, A. Ferraro, M. Paternostro, G. M. Palma, and L. Innocenti,
> *Mind the rank! Finite-resource effects in quantum linear regression*.

The simulations study pseudoinverse quantum linear regression (quantum extreme learning machines) when the feature probabilities are estimated from a finite number of measurement shots.

## Contents

Each folder in `python/` holds one experiment. It contains a simulation script that writes its results to `data/`, a `plot_*.py` script that turns `data/` into the figure, and the resulting PDF.

| Folder | Paper figure | Content |
|---|---|---|
| `python/N_scaling` | Fig. 2 | MSE vs. training shots $N$ (MUB, $M\to\infty$) |
| `python/ntrain_saturation` | Fig. 3 | MSE vs. training-set size $n_\mathrm{tr}$ |
| `python/bias_variance` | Fig. 4 | Training bias and variance vs. $n_\mathrm{tr}$ |
| `python/label_noise` | Fig. 5 | MSE vs. $N$ with training label noise |
| `python/noise_comparison` | Fig. SM1 | Multinomial shot noise vs. Gaussian approximation |
| `python/POVM_random` | Figs. SM2, SM3, SM5, SM6 | Random POVMs: MSE vs. $N$ and vs. $n_\mathrm{out}$ |
| `python/check_fit` | Fig. SM4 | Single-instance check of the MSE fit $f(N)$ |

`python/common.py` holds the shared routines (MUB and random POVMs, Haar-random states, shot-noise models, data I/O), and `python/markers.py` the marker styles used in the figures.

`mathematica/` contains the original Mathematica notebooks for Figs. 2, 3, 5 and SM2/SM3/SM5/SM6, of which the Python scripts are translations. The figures in the paper were produced with the Python code.

## Usage

Requirements: Python 3 with NumPy and Matplotlib (`pip install -r requirements.txt`; tested with Python 3.12, NumPy 2.4, Matplotlib 3.9), plus a LaTeX installation, since the figures use LaTeX text rendering.

To regenerate a figure from the stored data:

```bash
python python/N_scaling/plot_N_scaling.py
```

To rerun a simulation, which overwrites the files in its `data/` folder:

```bash
python python/N_scaling/N_scaling.py
```

All scripts can be run from any directory. The simulation parameters are set as constants at the top of each script. With the paper's parameters some simulations are slow, `ntrain_saturation` most of all. `noise_comparison` and `check_fit` are seeded and reproduce the stored data exactly; the other simulations are not seeded.

The Mathematica notebooks export their results to the directory given by the `SetDirectory[...]` call they contain, which must be adapted before running them.
