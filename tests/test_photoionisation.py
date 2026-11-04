"""Check the photoionisation utilities are syncretize's own copies."""

from pathlib import Path

import numpy as np

from syncretize import cloudy
from syncretize.photoionisation import calculate_Q_from_U, calculate_U_from_Q

SRC = Path(__file__).parents[1] / "src" / "syncretize"


def test_cloudy_modules_are_local():
    """The cloudy modules must not resolve to synthesizer's copies."""
    for module in (cloudy.cloudy17, cloudy.cloudy23, cloudy.cloudy25):
        assert module.__name__.startswith("syncretize.cloudy.")
    offenders = [
        str(p)
        for p in SRC.rglob("*.py")
        if "synthesizer.photoionisation" in p.read_text()
    ]
    assert not offenders, offenders


def test_q_u_round_trip():
    """calculate_U_from_Q inverts calculate_Q_from_U."""
    q = calculate_Q_from_U(1e-2, 100.0)
    assert np.isclose(calculate_U_from_Q(q, 100.0), 1e-2)
