"""Check syncretize uses its own cloudy modules."""

from pathlib import Path

from syncretize import cloudy

SRC = Path(__file__).parents[1] / "src" / "syncretize"


def test_cloudy_modules_are_local():
    """The cloudy modules must not resolve to synthesizer's copies."""
    for module in (cloudy.cloudy17, cloudy.cloudy23, cloudy.cloudy25):
        assert module.__name__.startswith("syncretize.cloudy.")
    offenders = [
        str(p)
        for p in SRC.rglob("*.py")
        if "synthesizer.photoionisation import cloudy" in p.read_text()
        or "synthesizer.photoionisation.cloudy" in p.read_text()
    ]
    assert not offenders, offenders
