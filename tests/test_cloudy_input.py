"""Check the Cloudy input writer."""

import pytest


@pytest.mark.parametrize("value, expected", [(True, 3), (False, 0), (None, 0)])
def test_on_off_options(tmp_path, value, expected):
    """On/off options add their command only when true."""
    pytest.importorskip("synthesizer.abundances")
    from synthesizer.abundances import Abundances

    from syncretize.cloudy import cloudy23

    cinput = cloudy23.create_cloudy_input(
        "model",
        [],
        Abundances(metallicity=0.01),
        output_dir=tmp_path,
        copy_linelist=False,
        geometry="planeparallel",
        ionisation_parameter=0.01,
        CMB=value,
        cosmic_rays=value,
        constant_density=value,
    )
    commands = [
        line
        for line in cinput
        if line.startswith(("CMB", "cosmic rays", "constant density"))
    ]
    assert len(commands) == expected
