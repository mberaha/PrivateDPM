# MCMC for Bayesian nonparametric mixture modeling under differential privacy

This repository contains a C++ implementation of the methodology and algorithms described in the paper

> M. Beraha, S. Favaro, and V.A. Rao, *MCMC for Bayesian nonparametric mixture modeling under differential privacy*, Journal of Computational and Graphical Statistics


# Repository structure

The folder `privacy_experiments` contains the scripts to replicate the numerical experiments in the paper.

The source code used to implement the algorithms is avaiable on my fork of the `bayesmix` library [here](https://github.com/mberaha/bayesmix/tree/48975d84b02c108f41c808a114bf16e18dd1e320/src/privacy).

The Jupyter notebooks in the folder `privacy_experiments` can be used (after running the simulations) to generate the associated tables and plots.

# Instructions

1. Download this repository _in recursive mode_:

```
git clone --recursive git@github.com:mberaha/PrivateDPM.git
```

2. Build the desired executable. E.g., to run the experiment with Laplace differential privacy mechanism do

```
mkdir -p build
cd build
cmake ..
make laplace1d
```

All the other executables are listed in the `privacy_experiments/CMakeLists.txt` file

3. Run the executable

```
cd ..
./build/privacy_experiments/laplace1d
```



