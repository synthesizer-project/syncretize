"""A module defining cloudy 17 helper utilities.

Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import os
import shutil

import numpy as np
from synthesizer.exceptions import InconsistentArguments, UnrecognisedOption
from synthesizer.photoionisation import calculate_Q_from_U
from synthesizer.synth_warnings import warn
from unyt import angstrom, c, cm, h, s, unyt_array

from syncretize.photoionisation import edge_is_resolved


class ShapeCommands:
    """A class for holding different cloudy shape commands.

    This class is used to create the shape commands for the cloudy input file.
    It contains methods for creating a cloudy input file using a tabulated SED
    or a cloudy AGN model. The class also contains methods for creating the
    shape commands for the AGN model and the SED model.
    """

    def table_sed(model_name, lam, lnu, output_dir="./"):
        """Create a cloudy input file using a tabulated SED.

        TODO: allow the user to instead specify nu and to automatically convert
        units if provided

        Args:
            model_name (str):
                User defined name of the model used for cloudy inputs
                and outputs.
            lam (array or unyt_array):
                Wavelength grid with or without units (via unyt)
            lnu (array or unyt_array):
                Spectral luminosity density
            output_dir (str):
                Output directory path

        Returns:
            list
                a list of strings with the cloudy input commands
        """
        # if lam is not a unyt_array assume it has units of angstrom and
        # convert to a unyt_array
        if not isinstance(lam, unyt_array):
            lam *= angstrom

        # Cloudy smears any ionisation edge the input doesn't sample closely,
        # which biases the Q normalisation of the outputs (#151). A smooth
        # SED has no edge to smear, so this is only a possible problem.
        if not edge_is_resolved(lam.to("Angstrom").value):
            warn(
                "The input SED has no wavelength points within 1 Angstrom of "
                "the HI ionisation edge. If it has an edge there, cloudy will "
                "smear it; regenerate the incident grid with sharp edges "
                "(see syncretize#151)."
            )

        # define frequency
        nu = c / lam

        # define energy
        E = h * nu

        # define energy in units of Rydbergs
        E_Ryd = E.to("Ry").value

        # get rid of negative/zero luminosities, which are unphysical and seem
        # to make cloudy break
        lnu[lnu <= 0.0] = 1e-100

        # save tabulated spectrum
        np.savetxt(
            f"{output_dir}/{model_name}.sed",
            np.array([E_Ryd[::-1], lnu[::-1]]).T,
        )

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(f'table SED "{model_name}.sed" \n')

        return shape_commands

    def cloudy_agn(TBB, aox=-1.4, auv=-0.5, ax=-1.35):
        """Define the cloudy AGN model.

        See 6.2 Hazy1.pdf.

        Args:
            model_name (str):
                User defined name of the model used for cloudy inputs and
                outputs
            TBB (float):
                The Big Bump temperature
            aox (float):
                The x-ray slope (default value from Calabro CEERS AGN model)
            auv (float):
                The uv-slope (default value from Calabro CEERS AGN model)
            ax (float):
                Slope normalisation

        Returns:
            list
                a list of strings with the cloudy input commands
        """
        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(
            f"AGN T = {TBB} k, a(ox) = {aox}, a(uv)= {auv} \
                              a(x)={ax} \n"
        )

        return shape_commands


def create_cloudy_input(
    model_name,
    shape_commands,
    abundances,
    output_dir="./",
    **kwargs,
):
    """Create a cloudy input file.

    Args:
        model_name (str):
            The model name. Used in the naming of the outputs.
        shape_commands (list):
            List of strings describing the cloudy input commands.
        abundances: (Abundances):
            A synthsizer Abundances object.
        output_dir (str):
            The output directory.
        **kwargs (**dict):
            Additional parameters to be passed to the cloudy input file.

    Returns:
        list
            A list of cloudy commands

    """
    default_params = {
        # ionisation parameter
        "ionisation_parameter": 0.01,
        # radius in parsecs (not log), only important for spherical geometry
        "radius": 0.01,
        # covering factor. Keep as 1 as it is more efficient to simply combine
        # SEDs to get != 1.0 values
        "covering_factor": False,
        # K, if not provided the command is not used
        "stop_T": False,
        # if not provided the command is not used
        "stop_efrac": False,
        # K, if not provided the command is not used
        "T_floor": False,
        # Hydrogen density
        "hydrogen_density": 10 ** (2.5),
        # redshift, only necessary if CMB heating included
        "z": 0.0,
        # include CMB heating
        "CMB": False,
        # include cosmic rays
        "cosmic_rays": False,
        # include metals
        "metals": True,
        # include dust grains (None, "Orion" or "ISM")
        "grains": None,
        # if True, don't scale the grain abundances to the depletion
        "no_grain_scaling": False,
        # the geometry
        "geometry": "planeparallel",
        # relative resolution the saved continuum spectra
        "resolution": 1.0,
        # output abundances
        "output_abundances": True,
        # output continuum
        "output_cont": True,
        # output full list of all available lines
        "output_lines": False,
        # output linelist
        "output_linelist": "linelist.dat",
    }

    # update default_params with kwargs
    params = default_params | kwargs

    # old approach for updated parameters
    # for key, value in list(kwargs.items()):
    #     params[key] = value

    # begin input list
    cinput = []

    # add spectral shape commands
    cinput += shape_commands

    # --- Define the chemical composition
    if params["metals"]:
        for ele in ["He"] + abundances.metals:
            cinput.append(
                (
                    f"element abundance {abundances.element_name[ele]} "
                    f"{abundances.gas[ele]} no grains\n"
                )
            )

        # In this case it would be inconsistent to turn on grains, so don't.

    else:
        cinput.append(f"element abundance helium {abundances.total['He']}\n")
        for ele in abundances.metals:
            cinput.append((f"elements {abundances.element_name[ele]} off \n"))

    """
    add graphite and silicate grains

    This old version does not actually conserve mass
    as the commands `abundances` and `grains` do not
    really talk with each other
    """
    # # graphite, scale by total C abundance relative to ISM
    # scale = 10**a_nodep['C']/2.784000e-04
    # cinput.append(f'grains Orion graphite {scale}'+'\n')
    # # silicate, scale by total Si abundance relative to ISM
    # # NOTE: silicates also rely on other elements.
    # scale = 10**a_nodep['Si']/3.280000e-05
    # cinput.append(f'grains Orion silicate {scale}'+'\n')

    """ specifies graphitic and silicate grains with a size
    distribution and abundance appropriate for those along
    the line of sight to the Trapezium stars in Orion. The
    Orion size distribution is deficient in small particles
    and so produces the relatively grey extinction observed
    in Orion (Baldwin et al., 1991). One problem with the
    grains approach is metals/element abundances do not talk
    to the grains command and hence there is issues with mass
    conservation (see cloudy documentation). To alleviate
    this one needs to make the orion or ism (in-built in cloudy)
    grain abundances consistent with the depletion values. Assume
    1 per cent of C is in PAH's.

    PAHs appear to exist mainly at the interface between the
    H+ region and the molecular clouds. Apparently PAHs are
    destroyed in ionized gas (Sellgren et al., 1990, AGN3
    section 8.5) by ionizing photons and by collisions with
    ions (mainly H+ ) and may be depleted into larger grains
    in molecular regions. Also assume the carbon fraction of
    PAHs from Abel+2008
    (https://iopscience.iop.org/article/10.1086/591505)
    assuming 1 percent of Carbon in PAHs. The factors in the
    denominators are the abundances of the carbon, silicon and
    PAH fractions when setting a value of 1 (fiducial abundance)
    for the orion and PAH grains.

    Another way is to scale the abundance as a function of the
    metallicity using the Z_PAH vs Z_gas relation from
    Galliano+2008
    (https://iopscience.iop.org/article/10.1086/523621,
    y = 4.17*Z_gas_sol - 7.085),
    which will again introduce issues on mass conservation.
    """

    # If grain physics are required, add this self-consistently
    if params["grains"] is None:
        f_graphite, f_Si, f_pah = 0, 0, 0

    else:
        if params["no_grain_scaling"] is False:
            delta_C = 10 ** abundances.total["C"] - 10 ** abundances.gas["C"]
            delta_PAH = 0.01 * (10 ** abundances.total["C"])
            delta_graphite = delta_C - delta_PAH
            delta_Si = (
                10 ** abundances.total["Si"] - 10 ** abundances.gas["Si"]
            )

            # Cloudy reads a scale factor <= 0 as a log, so grains without
            # depletion would silently get near-default abundances
            if delta_graphite <= 0 or delta_Si <= 0:
                raise InconsistentArguments(
                    "Can not scale grains without depletion of C and Si. "
                    "Use a depletion model or set no_grain_scaling."
                )

            # define the reference abundances for the different grain types
            # this should be the dust-phase abundance in the particular
            # reference environment.
            # Orion and ISM grains share the same underlying dust-phase
            # abundances, but Cloudy multiplies the abundance given to
            # "grains Orion" by 0.85 (parse_grain.cpp). The Orion
            # references include log10(0.85) to compensate.
            if params["grains"] == "Orion":
                reference_C_abund = -3.6259
                reference_Si_abund = -4.5547
            elif params["grains"] == "ISM":
                reference_C_abund = -3.5553
                reference_Si_abund = -4.4841
            else:
                raise UnrecognisedOption(
                    "Only Orion and ISM grain types are available in cloudy"
                )

            reference_PAH_abund = -5.526
            f_graphite = delta_graphite / (10 ** (reference_C_abund))
            f_Si = delta_Si / (10 ** (reference_Si_abund))
            f_pah = delta_PAH / (10 ** (reference_PAH_abund))

        else:
            f_graphite = 1.0
            f_Si = 1.0
            f_pah = 1.0

        command = (
            f"grains {params['grains']} graphite {f_graphite} \n"
            f"grains {params['grains']} silicate {f_Si} \n"
            f"grains PAH {f_pah}"
        )

        cinput.append(command + "\n")

    U = params["ionisation_parameter"]
    log10U = np.log10(U)

    # plane parallel geometry
    if params["geometry"] == "planeparallel":
        cinput.append(f"ionization parameter = {log10U:.3f}\n")

        # NOTE: I don't think the below is needed.
        # inner radius = 10^30 cm and thickness = 10^21.5 cm (==1 kpc) this is
        # essentially plane parallel geometry

        # cinput.append(f"radius 30.0 21.5\n")

    if params["geometry"] == "spherical":
        # in the spherical geometry case I think U is some average U, not U at
        # the inner face of the cloud.
        log10_specific_ionising_lum = np.log10(
            calculate_Q_from_U(
                U, params["hydrogen_density"] * cm**-3
            ).to_value(1 / s)
        )
        cinput.append(f"Q(H) = {log10_specific_ionising_lum}\n")
        cinput.append(f"radius {np.log10(params['radius'])} log parsecs\n")
        cinput.append("sphere\n")

    # add background continuum
    if params["cosmic_rays"]:
        cinput.append("cosmic rays, background\n")

    if params["CMB"]:
        cinput.append(f"CMB {params['z']}\n")

    # define hydrogend density
    if params["hydrogen_density"]:
        cinput.append(
            (
                f"hden {np.log10(params['hydrogen_density'])}"
                " log constant "
                "density\n"
            )
        )

    # covering factor
    if params["covering_factor"]:
        cinput.append(f"covering factor {params['covering_factor']} linear\n")

    # Processing commands
    cinput.append("iterate to convergence\n")
    if params["T_floor"]:
        cinput.append(f"set temperature floor {params['T_floor']} linear\n")

    if params["stop_T"]:
        cinput.append(f"stop temperature {params['stop_T']}K\n")

    if params["stop_efrac"]:
        cinput.append(f"stop efrac {params['stop_efrac']}\n")

    # --- output commands

    # cinput.append(f'print line vacuum\n')  # output vacuum wavelengths
    cinput.append(
        f"set continuum resolution {params['resolution']}\n"
    )  # set the continuum resolution
    cinput.append(f'save overview  "{model_name}.ovr" last\n')

    # output abundances
    if params["output_abundances"]:
        cinput.append(f'save last abundances "{model_name}.abundances"\n')

    # output continuum (i.e. spectra)
    if params["output_cont"]:
        cinput.append(
            (
                f'save last continuum "{model_name}.cont" '
                f"units Angstroms no clobber\n"
            )
        )
        # reflected continuum: its albedo column separates the incident
        # light scattered back towards the source from the nebular emission
        # (see read_continuum). Cloudy only predicts it for open geometries
        # (it aborts with "sphere", where nothing leaves the illuminated face).
        if params["geometry"] != "spherical":
            cinput.append(
                (
                    f'save last reflected continuum "{model_name}.refl" '
                    f"units Angstroms no clobber\n"
                )
            )
    # output lines
    if params["output_lines"]:
        cinput.append(
            (
                f'save last lines, array "{model_name}.lines" '
                "units Angstroms no clobber\n"
            )
        )

    # output linelist
    if params["output_linelist"]:
        # Same output files as the Cloudy 23 writer, so that grids from either
        # version are built the same way.

        # save intrinsic lines
        cinput.append(
            f'save linelist column absolute last units angstroms \
                  "{model_name}.intrinsic_elin" "linelist.dat"\n'
        )

        # save emergent lines
        cinput.append(
            f'save linelist column emergent absolute last units angstroms \
                  "{model_name}.emergent_elin" "linelist.dat"\n'
        )

        # make copy of linelist
        shutil.copyfile(
            params["output_linelist"], f"{output_dir}/linelist.dat"
        )

    # --- save input file
    open(f"{output_dir}/{model_name}.in", "w").writelines(cinput)

    return cinput


def read_lines(filename, extension="lines"):
    """Read a full line list from a cloudy output file.

    Args:
        filename (str):
            The cloudy filename
        extension (str):
            The extension of the file

    Returns:
        line_ids (list):
            A list of line identificaitons
        blends (list):
            A list containing flags whether the line is a blend
        wavelengths (list):
            A list of the line wavelengths
        intrinsic (list):
            A list of the intrinsic luminosities
        emergent (list):
            A list of emergent luminosities

    """
    # open file and read columns
    wavelengths, cloudy_ids, intrinsic, emergent = np.loadtxt(
        f"{filename}.{extension}",
        dtype=str,
        delimiter="\t",
        usecols=(0, 1, 2, 3),
    ).T

    wavelengths = wavelengths.astype(float)
    intrinsic = intrinsic.astype(float) - 7.0  # erg s^{-1}
    emergent = emergent.astype(float) - 7.0  # erg s^{-1}

    # make a new cloudy ID e.g. "H I 4861.33A"
    line_ids = np.array(
        [" ".join(list(filter(None, id.split(" ")))) for id in cloudy_ids]
    )

    # find out the length of the line id when split
    lenid = np.array(
        [len(list(filter(None, id.split(" ")))) for id in cloudy_ids]
    )

    # define a blend as a line with only two entries
    blends = np.ones(len(wavelengths), dtype=bool)
    blends[lenid == 3] = False

    return line_ids, blends, wavelengths, intrinsic, emergent


def convert_cloudy_wavelength(x):
    """Convert a wavelength string (e.g. 6562.81A, 1.008m) to a unyt_quantity.

    Args:
        x (str):
            A cloudy wavelength string

    Returns:
        float
            The wavelength in Angstroms
    """
    value = float(x[:-1])
    unit = x[-1]

    # if Angstroms
    if unit == "A":
        return value

    # if microns
    if unit == "m":
        return 1e4 * value


def read_linelist(filename, extension="emergent_elin"):
    """Read a line list from a cloudy output file.

    Args:
        filename (str):
            The cloudy filename
        extension (str):
            The extension of the file: "emergent_elin" (default) or
            "intrinsic_elin", as written by the input file generator.

    Returns:
        line_ids (list):
            A list of line identificaitons
        wavelengths (list):
            A list of the line wavelengths
        luminosities (list):
            A list of the luminosities
    """
    # read file
    with open(f"{filename}.{extension}", "r") as f:
        d = f.readlines()

    line_ids = []
    luminosities = []
    wavelengths = []

    for row in d:
        # ignore invalid rows
        if len(row.split("\t")) > 1:
            # split each row using tab character
            id, lum = row.split("\t")

            # reformat line id to be ELEMENT ION WAVELENGTH
            id = " ".join(list(filter(None, id.split(" "))))

            wavelength = convert_cloudy_wavelength(id.split(" ")[-1])

            # convert luminosity to float
            lum = float(lum)

            line_ids.append(id)
            wavelengths.append(wavelength)
            luminosities.append(lum)

    return np.array(line_ids), np.array(wavelengths), np.array(luminosities)


def read_wavelength(filename):
    """Extract the wavelength grid from cloudy .cont file.

    Note that the order will be reversed relative to the original file.

    Args:
        filename (str):
            The cloudy filename

    Returns:
        ndnarray
            The wavelength grid
    """
    lam = np.loadtxt(f"{filename}.cont", delimiter="\t", usecols=(0)).T
    lam = lam[::-1]  # reverse order

    return lam


def read_continuum(filename, return_dict=False, geometry=None):
    """Extract the spectra from a cloudy .cont file (and .refl, if present).

    The spectra include the light leaving both faces of the cloud, so that
    they are consistent with the emergent line luminosities of
    ``save line list emergent`` (which also sum both faces). The .cont
    columns are::

        0 = wavelength, 1 = incident, 2 = trans, 3 = DiffOut, 4 = net trans,
        5 = reflc, 6 = total, 7 = reflin, 8 = outlin

    with net trans = trans + DiffOut and total = net trans + reflc. reflc
    (light leaving the illuminated face) is the sum of the reflected
    incident continuum, the reflected diffuse continuum and the reflected
    lines (reflin). In a closed spherical geometry (``sphere``) nothing
    leaves the illuminated face: reflc = reflin = 0 and all emission is in
    DiffOut and outlin.

    The incident light scattered back towards the source is not nebular
    emission, so it is separated using the albedo (reflected incident /
    incident) column of the ``{filename}.refl`` file written by
    ``save last reflected continuum``. Cloudy only predicts this for open
    geometries, so no .refl file is written for ``geometry="spherical"``
    and ``reflected_incident`` is zero. For any other geometry a missing
    .refl file raises a warning: ``reflected_incident`` is then zero and
    the reflected incident light stays in ``nebular``.

    ``reflected_incident`` (albedo x incident) assumes full coverage.
    Cloudy multiplies reflc in the .cont file by the covering factor, but
    not the incident column or the .refl albedo, so a covering factor below
    one would make it too large. This cannot happen for runs normalised by
    the ionisation parameter (Cloudy forces the covering factor to one in
    that case) or for ``sphere`` (no .refl file); it would only matter for
    an open geometry normalised by luminosity (e.g. Q(H) and radius) with
    ``covering factor`` < 1. Partial coverage should instead be applied
    afterwards (e.g. via synthesizer's fesc).

    Args:
        filename (str):
            The cloudy filename (without the .cont extension).
        return_dict (bool):
            If True, return a dictionary of the spectra.
            If False, return a tuple of the spectra.
        geometry (str, optional):
            The Cloudy geometry of the run (the ``geometry`` parameter, e.g.
            "planeparallel" or "spherical"). Used only to decide whether a
            missing .refl file should raise a warning: it is expected for
            "spherical", and anything else (including None) is treated as
            an open, plane-parallel geometry.

    Returns:
        If ``return_dict`` is False, a tuple of 9 arrays, in this order:

        lam (np.ndarray of float):
            The wavelength grid (Angstrom, ascending)
        nu (np.ndarray of float):
            The frequency grid (Hz)
        incident (np.ndarray of float):
            The incident spectrum
        transmitted (np.ndarray of float):
            The transmitted spectrum (attenuated incident, outward)
        nebular (np.ndarray of float):
            The nebular spectrum, both faces, lines included
            (DiffOut + reflc - reflected_incident)
        nebular_continuum (np.ndarray of float):
            The nebular continuum spectrum (nebular - linecont)
        total (np.ndarray of float):
            The total spectrum (Cloudy's total = transmitted + nebular +
            reflected_incident)
        linecont (np.ndarray of float):
            The line contribution spectrum, both faces (outlin + reflin)
        reflected_incident (np.ndarray of float):
            Incident light scattered back towards the source (zero for
            'sphere' or when no .refl file is present)

        If ``return_dict`` is True, a dictionary keyed by "lam", "nu" and
        the spectrum names above, which also holds two spectra that are
        not in the tuple:

        reflected (np.ndarray of float):
            Light leaving the illuminated face (Cloudy's reflc)
        reflected_linecont (np.ndarray of float):
            Line contribution leaving the illuminated face (reflin)

        From these the outward-only spectra can be recovered (e.g. outward
        nebular = nebular + reflected_incident - reflected).

        Note: before the reflected spectra were added the tuple had 8
        values; reflected_incident is now appended as a 9th. Prefer
        ``return_dict=True``.

    """
    # ----- Open SED
    (
        lam,
        incident,
        transmitted,
        diffout,
        reflected,
        total,
        reflected_linecont,
        outlin,
    ) = np.loadtxt(
        f"{filename}.cont",
        delimiter="\t",
        usecols=(0, 1, 2, 3, 5, 6, 7, 8),
    ).T

    # reflected incident continuum = albedo x incident, from the .refl file
    # (columns: wavelength, cont, line, total, albedo, label). The .refl
    # "cont" column (= reflc - reflin) is not needed: the albedo is the ratio
    # of the reflected to the incident continuum in each cell.
    if os.path.exists(f"{filename}.refl"):
        refl = np.loadtxt(
            f"{filename}.refl", delimiter="\t", usecols=(0, 4), ndmin=2
        )
        if refl.shape[0] != lam.size or not np.allclose(
            refl[:, 0], lam, rtol=1e-4
        ):
            raise InconsistentArguments(
                f"{filename}.refl does not share the wavelength grid of "
                f"{filename}.cont"
            )
        reflected_incident = refl[:, 1] * incident
    else:
        # Nothing leaves the illuminated face of a closed sphere, so no .refl
        # file is expected there. Otherwise warn, with a fixed message (no
        # filename) so Python's warning filter shows it once rather than
        # once per model when building a grid.
        if geometry != "spherical":
            warn(
                "No reflected spectra output from cloudy found for "
                "plane-parallel geometry; total spectra will be "
                "underestimated"
            )
        reflected_incident = np.zeros_like(incident)

    # nebular emission leaving both faces (lines included). The reflected
    # nebular part (reflc - reflected incident) cannot be below the reflected
    # lines; clip there to absorb the 4-digit print precision of the files
    # (relevant where reflc is almost all reflected incident, e.g. X-rays).
    nebular = diffout + np.maximum(
        reflected - reflected_incident, reflected_linecont
    )

    # line contribution leaving both faces
    linecont = outlin + reflected_linecont

    # nebular continuum is the total nebular emission (nebular)
    # minus the line contribution (linecont)
    nebular_continuum = nebular - linecont

    # --- frequency
    lam = lam[::-1]  # reverse array
    lam_m = lam * 1e-10  # m
    nu = c.value / (lam_m)

    spec_dict = {"lam": lam, "nu": nu}

    for spec_type, sed in [
        ("incident", incident),
        ("transmitted", transmitted),
        ("nebular", nebular),
        ("nebular_continuum", nebular_continuum),
        ("total", total),
        ("linecont", linecont),
        ("reflected", reflected),
        ("reflected_linecont", reflected_linecont),
        ("reflected_incident", reflected_incident),
    ]:
        sed = sed[::-1]  # reverse array
        sed = sed / 10**7  # convert from W to erg
        sed = sed / nu  # convert from nu l_nu to l_nu
        spec_dict[spec_type] = sed

    if return_dict:
        return spec_dict
    else:
        return (
            lam,
            nu,
            spec_dict["incident"],
            spec_dict["transmitted"],
            spec_dict["nebular"],
            spec_dict["nebular_continuum"],
            spec_dict["total"],
            spec_dict["linecont"],
            spec_dict["reflected_incident"],
        )
