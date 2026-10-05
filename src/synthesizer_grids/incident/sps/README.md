# Notes and Instructions


## BC03-2016

You need a working fortran compiler to convert the binary files into ascii. You can check this is available by running `which gfortran` at the command line.

## FSPS variable IMFs

`install_fsps_variable_imf.py` reads an IMF definition from YAML and can add
one IMF parameter as a grid axis. FSPS-specific configurations live in
`fsps/configs/`; see `variable_high_mass_slope.yaml` for the complete format
and both supported axis-sampling forms. `variable_single_slope.yaml` applies
one variable power-law slope across the complete IMF mass range.

```bash
export SPS_HOME=/path/to/fsps
python install_fsps_variable_imf.py \
    --grid-dir path/to/grid/dir \
    --config-file fsps/configs/variable_high_mass_slope.yaml
```

Generated names retain the standard SPS/IMF prefix and append
`_axis-{axis_name}`. The current `imf_type=2` implementation supports any one
of its three slopes or either outer mass limit as an axis. Its internal mass
boundaries remain fixed at 0.5 and 1.0 solar masses by FSPS.
