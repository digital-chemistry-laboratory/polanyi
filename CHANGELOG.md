# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `opt_ts` and `opt_ts_ci` functions to optimise the TS from xtb command line
- If SCC doesn't converge in xtb runs, reruns with increase electronic temperature and then restarts with normal temperature
- Possibility of writing out the TS optimisation steps
- Possibility of giving in a reference reaction energy for the calculation of the energy shift
- Possibility of giving an adjacency matrix in the `setup_gfnff_topologies` function to write a neighbours list as input for xtb for the topology generation => requires [xtb bleeding edge version](https://github.com/grimme-lab/xtb/releases/tag/bleed)
- Possibility to give the charges of non-covalently bounded (NCI) fragments as input for the xtb calculations
- Implementation of fitting the coupling term to minimise the difference between EVB and reference (g-xTB and/or GFN2-xTB) energies
- Implementation of reaction path interpolation with EVB method

### Changed
- All xtb calculations now run from command line instead of using the deprecated `xtb-python` Python API
- `setup_gfnff_calculators` function renamed in `setup_gfnff_topologies`
- All functions optimising geometry with xtb now also return the final energy along with the optimised coordinates
- For PySCF optimisation: `e_g_function` returns the lowest EVB eigenvalue instead of the highest one and `ts_from_gfnff` a true transition state optimisation on the EVB ground state instead of a minimisation of the EVB excited state


### Removed
- All functions which previously used the deprecated `xtb-python` Python API

### Fixed
- Update the implementation of conical intersection optimisation to the version 1.0.1 of the geomeTRIC library
- Fix that `optimize_ci` was retuning the unmodified coordinates instead of the optimised ones
- Fix that some different xtb calculations were uncorrectly running in the same folder
- Add missing `--gfnff` keywords for xtb calculations (in `setup_gfnff_topologies` and `ts_from_gfnff` functions)
- Add missing `--grad` keyword for xtb calculations (in `e_g_function` function)

## [0.1.0] - 2025-04-03

- First release, as used in [Tartarus: A Benchmarking Platform for Realistic And Practical Inverse Molecular Design](https://arxiv.org/abs/2209.12487). Changelog use starts here.
