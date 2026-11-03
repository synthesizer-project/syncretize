"""
Create Cloudy input scripts by Sobol-sampling both incident grid parameters
(age, metallicity) and photoionization parameters.

Uses a single Sobol sequence across all parameters, then interpolates spectra
from the incident grid at the sampled age/metallicity points.

Alternatively, if the cloudy parameter file contains an `incident` block, the
incident spectra are computed directly from the named model (currently only
`relagn`) at each sampled point, and no incident grid is needed, e.g.:

    incident:
      model: relagn
      params:
        masses: [1.0e+6, 1.0e+10, 'log']
        accretion_rates_eddington: [1.0e-4, 200.0, 'log']
        spins: [0.0, 0.998, 'linear']   # or a scalar to fix it
        cosine_inclinations: [0.09, 1.0, 'linear']

WARNING: computing RELAGN spectra directly is slow, ~10-15 s per spectrum
per process (e.g. ~1.3 hours for 10,000 samples with --num-procs 28). For
more than a few hundred samples, run this script as a batch job on a full
node rather than on a login node.
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.stats import qmc
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from unyt import Angstrom, Hz, erg, s

import syncretize.cloudy.submission_scripts as submission_scripts
from syncretize.cloudy import cloudy17, cloudy23
from syncretize.parser import Parser


def parse_sobol_params(params, incident_grid=None, incident_config=None):
    """
    Parse parameters into incident, photoionization, and fixed categories.
    Automatically uses incident grid axes ranges when not specified in params.

    Args:
        params (dict): Parameter dictionary from YAML file
        incident_grid (Grid): Incident grid to extract axes ranges
        incident_config (dict): The `incident` block from the YAML file. If
            given, incident ranges are taken from it instead of the grid.

    Returns:
        tuple: (incident_params, photoionisation_params, fixed_params)
            - incident_params: dict mapping axes to (min, max, scale)
            - photoionisation_params: dict mapping params to (min, max, scale)
            - fixed_params: dict of fixed parameter values
    """
    incident_params = {}
    photoionisation_params = {}
    fixed_params = {}

    if incident_config is not None:
        # Scalars are fixed values; a zero-width range keeps them in the
        # saved parameters alongside the sampled ones
        for key, value in incident_config["params"].items():
            if isinstance(value, list) and len(value) == 3:
                incident_params[key] = tuple(value)
            else:
                incident_params[key] = (value, value, "linear")
        incident_axes = set(incident_params)
    else:
        # Get incident grid axes and their ranges
        incident_axes = set(incident_grid.axes)

        # Automatically add all incident grid axes with their full ranges
        for axis in incident_grid.axes:
            axis_values = getattr(incident_grid, axis)
            min_val = float(np.min(axis_values))
            max_val = float(np.max(axis_values))
            # Default to log scale for axes the grid itself treats as logged
            # (e.g. ages, metallicities, black hole masses, accretion rates)
            scale = (
                "log"
                if f"log10{axis}" in incident_grid._extract_axes
                or axis in ["ages", "age", "metallicities", "metallicity"]
                else "linear"
            )
            incident_params[axis] = (min_val, max_val, scale)

    for key, value in params.items():
        # Check if this is an incident grid parameter
        # Handle both singular (age, metallicity) and plural
        # (ages, metallicities)
        is_incident = False
        incident_key = None

        if key in ["age", "ages"] and (
            "ages" in incident_axes or "age" in incident_axes
        ):
            is_incident = True
            incident_key = "ages" if "ages" in incident_axes else "age"
        elif key in ["metallicity", "metallicities"] and (
            "metallicities" in incident_axes or "metallicity" in incident_axes
        ):
            is_incident = True
            incident_key = (
                "metallicities"
                if "metallicities" in incident_axes
                else "metallicity"
            )

        # Parse parameter value
        if isinstance(value, list) and len(value) == 3:
            min_val, max_val, scale = value

            if is_incident:
                # Override default with user-specified range
                incident_params[incident_key] = (min_val, max_val, scale)
            else:
                photoionisation_params[key] = (min_val, max_val, scale)

        # Handle nested abundance_scalings
        elif key == "abundance_scalings" and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list) and len(sub_value) == 3:
                    min_val, max_val, scale = sub_value
                    photoionisation_params[f"abundance_scalings.{sub_key}"] = (
                        min_val,
                        max_val,
                        scale,
                    )
                else:
                    if "abundance_scalings" not in fixed_params:
                        fixed_params["abundance_scalings"] = {}
                    fixed_params["abundance_scalings"][sub_key] = sub_value
        else:
            fixed_params[key] = value

    return incident_params, photoionisation_params, fixed_params


def generate_sobol_samples(
    incident_params, photoionisation_params, n_samples, seed=None
):
    """
    Generate Sobol sequence samples for all parameters.

    Args:
        incident_params (dict): Incident grid parameters with
            (min, max, scale)
        photoionisation_params (dict): Photoionization parameters with
            (min, max, scale)
        n_samples (int): Number of samples
        seed (int, optional): Random seed

    Returns:
        tuple: (incident_samples, photoionisation_samples)
            Each is a list of dicts containing sampled parameter values
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    # Combine all parameters
    all_params = {**incident_params, **photoionisation_params}
    n_dims = len(all_params)
    param_names = list(all_params.keys())

    # Generate Sobol samples - use power of 2 for optimal coverage
    n_sobol = int(2 ** np.ceil(np.log2(n_samples)))
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    unit_samples = sampler.random(n=n_sobol)[:n_samples]

    # Transform to parameter ranges
    samples = []
    for i in range(n_samples):
        sample = {}
        for j, param_name in enumerate(param_names):
            min_val, max_val, scale = all_params[param_name]

            if scale == "log":
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                value = 10 ** (
                    log_min + unit_samples[i, j] * (log_max - log_min)
                )
            elif scale == "linear":
                value = min_val + unit_samples[i, j] * (max_val - min_val)
            else:
                raise ValueError(f"Unknown scale: {scale}")

            sample[param_name] = value
        samples.append(sample)

    # Split into incident and photoionization samples
    incident_param_names = set(incident_params.keys())
    incident_samples = [
        {k: v for k, v in s.items() if k in incident_param_names}
        for s in samples
    ]
    photoionisation_samples = [
        {k: v for k, v in s.items() if k not in incident_param_names}
        for s in samples
    ]

    return incident_samples, photoionisation_samples


def compute_incident_spectra(model, incident_samples, num_procs=1):
    """
    Compute incident spectra directly from a model at each sampled point.

    Args:
        model (str): Name of the incident model (currently only 'relagn')
        incident_samples (list): List of dicts of incident parameter values
        num_procs (int): Number of processes to use

    Returns:
        tuple: (lam, lnu)
            - lam: wavelength array in Angstrom (increasing)
            - lnu: array of spectra in erg/s/Hz, shape (n_samples, n_lam)
    """
    if model != "relagn":
        raise ValueError(f"Unknown incident model: {model}")

    # Only needed for this model, so import here
    from multiprocessing import Pool

    from relagn import relagn

    from syncretize.incident.blackholes.relagn.generate_incident_grid import (
        compute_single_spectrum,
    )

    required = [
        "masses",
        "accretion_rates_eddington",
        "spins",
        "cosine_inclinations",
    ]
    missing = [key for key in required if key not in incident_samples[0]]
    if missing:
        raise ValueError(f"incident params missing for relagn: {missing}")

    lam = relagn().wave_grid[::-1]
    param_list = [tuple(s[key] for key in required) for s in incident_samples]

    with Pool(processes=num_procs) as pool:
        lnu = np.array(pool.starmap(compute_single_spectrum, param_list))

    return lam, lnu


def log10_ionising_photon_rate(lam, lnu):
    """
    Compute log10 of the HI ionising photon production rate of each spectrum.

    Args:
        lam (np.ndarray): Wavelength array in Angstrom
        lnu (np.ndarray): Spectra in erg/s/Hz, shape (n_lam,) or (n, n_lam)

    Returns:
        np.ndarray: log10 Q(HI) [s^-1] for each spectrum
    """
    sed = Sed(lam * Angstrom, lnu=lnu * erg / s / Hz)
    return np.log10(sed.calculate_ionising_photon_production_rate().value)


if __name__ == "__main__":
    parser = Parser(
        description="Create Cloudy inputs with Sobol-sampled parameters",
        cloudy_args=True,
        incident_grid_required=False,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of Sobol samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="Random seed for Sobol sampling",
    )

    parser.add_argument(
        "--grid-name",
        type=str,
        required=False,
        default=None,
        help="Override the output folder name (default: "
        "{incident_grid}_sobol-n{n_samples})",
    )

    parser.add_argument("--machine", type=str, required=False, default=None)

    parser.add_argument(
        "--cloudy-executable-path", type=str, required=False, default=None
    )

    parser.add_argument(
        "--no-striding",
        action="store_true",
        help="Disable striding in array jobs (use direct 1:1 mapping)",
    )

    parser.add_argument(
        "--stride-step",
        type=int,
        required=False,
        default=1000,
        help="Stride step for array jobs (default: 1000)",
    )

    parser.add_argument(
        "--time-per-model",
        type=int,
        required=False,
        default=10,
        help="Expected time per Cloudy model in minutes (default: 10)",
    )

    parser.add_argument(
        "--num-procs",
        type=int,
        required=False,
        default=1,
        help="Number of processes for computing incident spectra directly "
        "(only used with an `incident` block in the parameter file)",
    )

    args = parser.parse_args()

    incident_grid_name = args.incident_grid
    incident_grid_dir = args.grid_dir
    cloudy_output_dir = args.cloudy_output_dir
    cloudy_paramfile = args.cloudy_paramfile
    n_samples = args.n_samples
    seed = args.seed
    machine = args.machine
    cloudy_executable_path = args.cloudy_executable_path

    # Script generation needs a Cloudy executable; fail fast at the CLI rather
    # than emitting launchers that point at "None/cloudy.exe".
    if machine and not cloudy_executable_path:
        parser.error(
            "--cloudy-executable-path is required when --machine is set"
        )

    # Add extensions if needed
    if not cloudy_paramfile.endswith(".yaml"):
        cloudy_paramfile += ".yaml"

    # Load parameter file
    with open(cloudy_paramfile, "r") as file:
        all_params = yaml.safe_load(file)

    # An `incident` block means the incident spectra are computed directly
    # from a model rather than interpolated from an incident grid
    incident_config = all_params.pop("incident", None)
    if incident_config is None and incident_grid_name is None:
        parser.error(
            "--incident-grid is required unless the parameter file "
            "contains an `incident` block"
        )

    if incident_config is None:
        print(f"Incident grid: {incident_grid_name}")
    else:
        print(f"Incident model: {incident_config['model']} (computed directly)")
    print(f"Parameter file: {cloudy_paramfile}")
    print(f"Number of Sobol samples: {n_samples}")

    # Generate grid name (provenance like paramfile/version/seed -> HDF5 attrs)
    paramfile_base = Path(cloudy_paramfile).stem
    if incident_config is None:
        if not incident_grid_name.endswith(".hdf5"):
            incident_grid_name += ".hdf5"
        grid_base = Path(incident_grid_name).stem
    else:
        grid_base = incident_config["model"]
    new_grid_name = args.grid_name or f"{grid_base}_sobol-n{n_samples}"

    # Load incident grid
    if incident_config is None:
        incident_grid = Grid(
            incident_grid_name,
            grid_dir=incident_grid_dir,
            ignore_lines=True,
        )
    else:
        incident_grid = None

    # Parse parameters
    incident_params, photoionisation_params, fixed_params = parse_sobol_params(
        all_params, incident_grid, incident_config
    )

    print(f"\nIncident parameters: {list(incident_params.keys())}")
    print(f"Photoionization parameters: {list(photoionisation_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")

    # Generate Sobol samples
    incident_samples, photoionisation_samples = generate_sobol_samples(
        incident_params, photoionisation_params, n_samples, seed=seed
    )

    # Create output directory
    output_directory = f"{cloudy_output_dir}/{new_grid_name}"
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    Path(f"{output_directory}/logs").mkdir(exist_ok=True)

    # Determine Cloudy version
    cloudy_version = fixed_params["cloudy_version"]
    if cloudy_version == "c17.03":
        cloudy = cloudy17
    elif cloudy_version.startswith("c23"):
        cloudy = cloudy23
    else:
        raise ValueError(f"Unknown Cloudy version: {cloudy_version}")

    # Save grid parameters
    parameters_to_save = {
        "incident_n_models": int(n_samples),
        # Single photoionization model per incident
        "photoionisation_n_models": 1,
        "total_n_models": int(n_samples),
        "sampling_method": "sobol",
        "seed": seed,
        **fixed_params,
        "incident_params": {k: list(v) for k, v in incident_params.items()},
        "photoionisation_params": {
            k: list(v) for k, v in photoionisation_params.items()
        },
        "depletion_model": None,
    }

    with open(f"{output_directory}/grid_parameters.yaml", "w") as file:
        yaml.dump(parameters_to_save, file, default_flow_style=False)

    # Save the actual sampled parameter values to HDF5
    grid_hdf5_file = f"{output_directory}/grid.hdf5"
    with h5py.File(grid_hdf5_file, "w") as hf:
        # Create parameters group
        param_group = hf.create_group("parameters")

        # Save sampled parameter values for each model
        for i, (inc_sample, photo_sample) in enumerate(
            zip(incident_samples, photoionisation_samples)
        ):
            # Combine incident and photoionisation samples
            sample_params = {**inc_sample, **photo_sample}
            for key, value in sample_params.items():
                if key not in param_group:
                    # Create dataset on first encounter
                    param_group.create_dataset(
                        key, data=np.zeros(n_samples), compression="gzip"
                    )
                param_group[key][i] = float(value)

        # Save fixed parameters as attributes
        for key, value in fixed_params.items():
            if value is None or (isinstance(value, dict) and not value):
                continue
            if isinstance(value, dict):
                import json

                hf.attrs[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                hf.attrs[key] = np.array(value)
            else:
                hf.attrs[key] = value

        # Save grid metadata + provenance dropped from the folder name
        hf.attrs["n_samples"] = n_samples
        hf.attrs["sampling_method"] = "sobol"
        hf.attrs["incident_grid"] = grid_base
        hf.attrs["incident_generation"] = (
            "interpolated" if incident_config is None else "direct"
        )
        hf.attrs["cloudy_paramfile"] = paramfile_base
        if seed is not None:
            hf.attrs["seed"] = seed

    print(f"Saved parameter grid to {grid_hdf5_file}")

    shutil.copy(cloudy_paramfile, f"{output_directory}/")

    if incident_config is None:
        # Get wavelength array
        lam = incident_grid.lam

        # Collect the sampled values for every incident grid axis (in the
        # grid's native axis units), so this works for any grid (SPS, AGN,...)
        axes_values = {
            axis: np.array([s[axis] for s in incident_samples])
            for axis in incident_grid.axes
        }

        # Get interpolated spectra
        print("\nGenerating interpolated spectra...")
        interp = incident_grid.interpolate_grid_at_axes_value(
            spectra_type="incident",
            **axes_values,
        )
        lnu = interp["spectra"].lnu.value
    else:
        print(
            f"\nComputing {incident_config['model']} spectra directly "
            f"({args.num_procs} processes)..."
        )
        lam, lnu = compute_incident_spectra(
            incident_config["model"], incident_samples, args.num_procs
        )

    with h5py.File(grid_hdf5_file, "a") as hf:
        hf.create_dataset("incident_lam", data=lam, compression="gzip")
        hf.create_dataset("incident_lnu", data=lnu, compression="gzip")

    # Handle reference ionisation parameter if needed
    if (
        fixed_params.get("ionisation_parameter_model") == "ref"
        and incident_config is not None
    ):
        # The reference point is a set of incident parameter values
        reference = incident_config.get("reference")
        if reference is None:
            raise ValueError(
                "incident.reference (a dict of incident parameter values) "
                "is required for the 'ref' model with an `incident` block"
            )
        _, ref_lnu = compute_incident_spectra(
            incident_config["model"], [reference]
        )
        reference_log10_specific_ionising_lum = float(
            log10_ionising_photon_rate(lam, ref_lnu[0])
        )
        log10Q_samples = log10_ionising_photon_rate(lam, lnu)
    elif fixed_params.get("ionisation_parameter_model") == "ref":
        # Get reference point parameters
        ref_age = fixed_params.get("reference_age")
        ref_met = fixed_params.get("reference_metallicity")

        if ref_age is None or ref_met is None:
            raise ValueError(
                "reference_age and reference_metallicity required for "
                "'ref' model"
            )

        # Get reference grid point
        ref_dict = {}
        for axis in incident_grid.axes:
            if axis in ["ages", "age"]:
                ref_dict[axis] = ref_age
            elif axis in ["metallicities", "metallicity"]:
                ref_dict[axis] = ref_met

        ref_grid_point = incident_grid.get_grid_point(**ref_dict)
        reference_log10_specific_ionising_lum = float(
            incident_grid.log10_specific_ionising_lum["HI"][ref_grid_point]
        )

        # Per-sample specific ionising luminosity interpolated from the
        # incident grid (per unit grid normalisation). Q is in s^-1.
        log10Q_samples = np.asarray(
            interp["log10_specific_ionising_luminosity/HI"]
        )
    else:
        reference_log10_specific_ionising_lum = None
        log10Q_samples = None

    # Save ionisation parameter scaling if needed (for on-the-fly generation)
    if fixed_params.get("ionisation_parameter_model") == "ref":
        with h5py.File(grid_hdf5_file, "a") as hf:
            hf.attrs["reference_log10_specific_ionising_lum"] = (
                reference_log10_specific_ionising_lum
            )
            hf.create_dataset(
                "log10Q_samples",
                data=log10Q_samples,
                compression="gzip",
            )

    linelist_name = fixed_params.get("output_linelist", "linelist.dat")
    shutil.copyfile(linelist_name, f"{output_directory}/{linelist_name}")

    print(f"\nSaved parameter grid and incident spectra to {grid_hdf5_file}")
    print("Cloudy input files will be generated on-the-fly during execution")
    print(f"Output directory: {output_directory}")

    # Generate submission script (on-the-fly version)
    if machine:
        if machine == "cosma7":
            submission_scripts.cosma7_sobol_onthefly(
                new_grid_name=new_grid_name,
                n_samples=n_samples,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                time_per_model=args.time_per_model,
                use_striding=not args.no_striding,
                stride_step=args.stride_step,
            )
        else:
            getattr(submission_scripts, machine)(
                new_grid_name,
                number_of_incident_grid_points=n_samples,
                number_of_photoionisation_models=1,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory="4G",
                by_photoionisation_grid_point=False,
            )
