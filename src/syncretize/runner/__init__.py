"""Run a grid from one config: incident grid, then Cloudy, then assembly.

The Runner (``syncretize.runner.runner``) is imported only when used, so the
parts that don't need it (e.g. ``syncretize watch``) start quickly without
importing synthesizer.
"""


def __getattr__(name):
    if name == "Runner":
        from syncretize.runner.runner import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
