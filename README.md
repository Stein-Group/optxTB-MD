# optxTB-MD

Multiobjective optimization of GFN2-xTB parameters for molecular dynamics of liquid water and NaCl(aq) against revPBE0-D3 reference data. Two independent workflows are provided:

- **cluster**: gas phase water, Na+, Cl- clusters, energies E, forces F and partial charges Q (CM5 or Iterative Hirshfeld schemes from Multiwfn);
- **PBC**: periodic water and NaCl(aq) boxes, energies E, forces F and virials Xi.

Each workflow has two stages, **water (H/O parameters)** followed by **Na on top of the optimized water parameters**. Every stage is a Fisher information matrix (FIM) sensitivity analysis followed by an NSGA-III optimization (NSGA-II when only two objectives are selected, since it converges faster in this case).

## Repository structure

```
codes/                    shared Python modules (calculators, reference parsers, common.py helpers)
optimization/
  T1_partial_charge_schemes/      cluster, water stage   (FIM + NSGA on E, F, Reg)
  T_all_partial_charge_schemes/   cluster, Na stage      (FIM + NSGA on E, F, Q, Reg)
  T_pbc/                          PBC, water and Na stages (FIM + NSGA on E, F, Xi, Reg)
  analyze_pareto/                 Pareto analysis notebooks, point export and validation scan tests
parameter_files/          GFN2-xTB parameter files passed between stages or used in analysis
gfn2-xtb.toml             default GFN2-xTB parameters (tblite 0.5.x)
gfn2-xtb-060.toml         default GFN2-xTB parameters (tblite >= 0.6.0, cn = "dexp")
environment_py311.yml     conda environment of the cluster water stage
environment_tblite_env.yml conda environment of all other stages
structures/               input data (not tracked; layout below)
```

The objectives, charge scheme, dataset options, population size and number of generations are set in the "scientific settings" block at the top of each optimization script. NSGA adapts to the selected number of objectives.

## Software

| Stage | Python env | External programs |
|---|---|---|
| cluster water (`T1_partial_charge_schemes`) | `py311` | DFTB+ 25.1 built with tblite 0.5.0 (clusters and monomers) |
| cluster Na (`T_all_partial_charge_schemes`) | `tblite_env` | tblite 0.5.0 CLI (clusters), DFTB+ 25.1 built with tblite 0.5.0 (Na+,Cl-,water monomers), Multiwfn 2026.1.12 (CM5/Hirshfeld-I charges) |
| PBC (`T_pbc`) and scan tests | `tblite_env` | tblite 0.5.0 CLI |

### Installation requirements

- conda environments: `conda env create -f environment_py311.yml` and `conda env create -f environment_tblite_env.yml` (tblite 0.5.0, pymoo 0.6.1.6).
- DFTB+ 25.1 or later, compiled with tblite 0.5.0, `dftb+` first on `PATH` in both cluster stages. 
- Multiwfn 2026.1.12 (noGUI), set via `MULTIWFN_EXE`; its `settings.ini` must be next to the executable.
- xtb 6.7.0, only to regenerate the Molden templates (`--prepare-templates`, `XTB_EXE`). More recent versions of tblite can generate Molden files correctly without this.

### tblite version

All results were obtained with tblite 0.5.0, the current release when this work started (0.6.0 and 0.7.0 followed in May and July 2026). 

- **Motivation.** The shortcomings of the default GFN2-xTB molecular dynamics of water and NaCl(aq), which this code corrects by reparametrization, persist with tblite 0.7.0 (see [Citation](#citation)).
- **Code compatibility.** The command line options, gradient/virial, JSON output and restart file read by `tblite.interface.Result` are unchanged, so the workflow runs with tblite 0.7.0. Since tblite 0.6.0, the parameter files must use `cn = "dexp"` instead of `cn = "gfn"` (the old keyword is rejected). The base parameter file is selected automatically from `tblite --version` (`gfn2-xtb.toml` or `gfn2-xtb-060.toml`). The files in `parameter_files/` use `cn = "gfn"` and need the same change before they are passed to tblite >= 0.6.0.
- **Expected calculation changes.** For clusters, tblite 0.7.0 reproduces the tblite 0.5.0 energies and forces within numerical precision. For periodic structures, tblite ≥ 0.6.0 changed the Ewald electrostatics and its contributions to forces and stress, so the PBC-optimized parameters are specific to tblite 0.5.0.

## Data layout (`structures/`, not tracked)

```
structures/
  cluster_data/
    dataset/                      307 cluster xyz files (numbered 1_...xyz to 307_...xyz)
      dft_results_revpbe0/        Q-Chem revPBE0-D3/aug-cc-pVTZ outputs (E, F) for every xyz
      dft_multiwfn_chg_revpbe0/   cached reference charges, {CM5,hirshfeldI}/<stem>.chg
      oo_target_small/            O-O scans of the water hexamers (xyz + dft_results_revpbe0/)
      na_o_target/, na_h2o4_target/   Na-O scans (xyz + dft_results_revpbe0/ + dft_multiwfn_chg_revpbe0/)
    monomers/                     na/na.xyz, cl/cl.xyz, water/water.xyz (xTB monomer geometries)
      revpbe0/                    na.out, cl.out, water.out (revPBE0-D3 monomer references; water relaxed)
    molden_templates_gfn2/        Molden templates for the tblite -> Multiwfn charge step
    multiwfn_atmrad/              Multiwfn atomic densities (Hirshfeld-I)
    na_h2o_11_scan/, na_h2o4_tet_scan/, water_monomer_bde/, water_hexamers/
                                  Q-Chem reference scans used by the scan tests
    fchk/                         optional, see below
  periodic_data/
    dataset/                      CP2K revPBE0-D3 frames (extxyz with energy, forces, stress)
    isolated_atoms/               CP2K isolated atoms (only for reference energy_mode = "isolated_atoms")
    naO_shell_scan/               Na-O first-shell scan in NaCl(aq) boxes (references + generator)
    poscar_cache/                 created automatically from dataset/
    water/, NaCl/, unify_lw_nacl_extxyz.py   preparation of dataset/ from the raw trajectories
```

Which stage reads what:

- cluster water: `dataset/*.xyz`, `dataset/dft_results_revpbe0`, `dataset/oo_target_small`, `monomers/`;
- cluster Na: additionally `dataset/dft_multiwfn_chg_revpbe0`, `na_o_target`, `na_h2o4_target`, `molden_templates_gfn2`, `multiwfn_atmrad`;
- PBC: `periodic_data/dataset`, `naO_shell_scan/reference_naO_shell_scan` (Na stage), `isolated_atoms`.

The revPBE0 wavefunctions (`.fchk`) are only read if a reference charge is missing from `dft_multiwfn_chg_revpbe0`. They are expected in `$FCHK_ROOT/dft_results_revpbe0_fchk`, `$FCHK_ROOT/dft_results_revpbe0_fchk_na_o_target` and `$FCHK_ROOT/dft_results_revpbe0_fchk_na_h2o4_target` (default `FCHK_ROOT=structures/cluster_data/fchk`, e.g. a symlink to the scratch).

## Running

All scripts are started from the repository root and adapted for HPC usage:

```
python -u optimization/<stage>/<script>.py --run-id <name> --scratch-root <dir> --output-root <dir> --n-workers <N>
```

Without arguments, `--run-id` is generated from the date (and `SLURM_JOB_ID`), the scratch root is `$SCRATCH` (or `/tmp`) and the output root is the stage's `hpc_runs/`. 

### Cluster workflow

1. `optimization/T1_partial_charge_schemes/fim_EF_water_select_objectives.py` - FIM of the H/O parameters 
2. `optimization/T1_partial_charge_schemes/nsga3_EF_water.py` - NSGA-III on (E, F, Reg)
3. export the selected water point (see below) and set `FIXED_HO_TOML_PATH` in both Na scripts
4. `optimization/T_all_partial_charge_schemes/fim_EFQ_na_select_objectives.py` - FIM of the Na parameters
5. `optimization/T_all_partial_charge_schemes/nsga3_loss_regularized_Na.py` - NSGA-III on (E, F, Q, Reg), `type_charge = "CM5"` or `"hirshfeldI"`

### PBC workflow

1. `optimization/T_pbc/pbc_fim_water_select_objectives.py` - FIM of the H/O parameters
2. `optimization/T_pbc/nsga3_pbc_loss_regularized.py` - NSGA-III on (E, F, Xi, Reg)
3. export the selected water point and set `FIXED_HO_TOML_PATH` in both PBC Na scripts
4. `optimization/T_pbc/pbc_fim_Na.py` - FIM of the Na parameters
5. `optimization/T_pbc/nsga3_pbc_loss_regularized_Na.py` - NSGA-III on (E, F, Xi, Reg)

### Passing a Pareto point to the next stage

```
python optimization/analyze_pareto/export_pareto_point.py \
    optimization/<stage>/hpc_runs/<RUN_ID>/<algo>_results_<...>_<RUN_ID>.npz \
    --index <i> --output parameter_files/<name>.toml
```

### Pareto analysis notebooks

The notebooks in `optimization/analyze_pareto/` read the fronts in `optimization/<stage>/hpc_runs/`, plot them, select a point (ASF for water, sorting by Reg for Na) and write it to `parameter_files/`:

| Notebook | Front |
|---|---|
| `analyze_pymoo_cluster_water_reg.ipynb` | cluster water (E, F, Reg) |
| `analyze_pymoo_pbc_water_reg.ipynb`, `analyze_pymoo_pbc_water_reg_EF.ipynb` | PBC water (E, F, Xi, Reg) and (E, F, Reg) |
| `analyze_pymoo_cluster_Na_reg.ipynb`, `analyze_pymoo_cluster_Na_EF_reg.ipynb` | cluster Na (E, F, Q, Reg; Hirshfeld-I or CM5) and (E, F, Reg) |
| `analyze_pymoo_pbc_Na_reg.ipynb` | PBC Na (E, F, Xi, Reg) |
| `compare_optimal_solutions_water_na.ipynb` | parameter deviations of the selected water and Na points from GFN2-xTB |

### Validation scans

`optimization/analyze_pareto/test_pareto_water.py` and `test_pareto_na.py` evaluate selected Pareto points on the reference scans in `structures/cluster_data/`.

## Output

- `optimization/<stage>/hpc_runs/<RUN_ID>/`: `run_config.json`, `dataset_split.json`, initial parity plots, FIM (`fim_*.csv/png/json`, `fim_results_default_gfn2.npz`) or NSGA results (`<algo>_results_*.npz/.csv`, `generation_fronts/` written after every generation) and the run log `<script>_<RUN_ID>.out` (when started via SLURM).
- `$SCRATCH_ROOT/optxTB-MD/<RUN_ID>/work/`: all DFTB+, tblite and Multiwfn input and output files (parameter TOMLs, monomer runs, Molden files); can be deleted after the run.
- `optimization/analyze_pareto/pareto_tests/<RUN_ID>/`: scan-test results.

The GFN2-xTB base parameter file is selected by `tblite --version`: `gfn2-xtb.toml` for tblite 0.5.x (used here), `gfn2-xtb-060.toml` for tblite >= 0.6.0 (see "tblite version").

## Citation

The paper describing this work is in preparation. The reference will be added here on publication.
