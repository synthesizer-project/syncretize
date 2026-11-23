"""Perturb the Cloudy models that failed every attempt, to interpolate them.

A model that keeps failing may fail only because of its exact inputs. Once a
run's rounds have finished, the finish job writes, for each such model, runs
slightly to either side of it along each of its Cloudy axes (or, with no
Cloudy axes, along its ionisation parameter), and queues them. A side that
fails is tried again twice as far out, up to ``max_steps`` times. The
assembly then interpolates each pair of successful sides back to the model's
own parameters (``syncretize.cloudy.create_synthesizer_grid``); a model with
no successful pair stays failed. A model whose every neighbour on the grid
of Cloudy axes failed too isn't perturbed: it is most likely in a region
where Cloudy can't run at all, so it stays failed (with no parameters in
the plan). Once one of those neighbours can be interpolated (after a round
of perturbed runs), it is perturbed after all.

The plan and each side's latest run are kept in ``perturb.json`` beside the
Cloudy inputs:

    {"<i>_<j>": {"<parameter>": {
        "log": bool,          interpolated in log10 of the parameter
        "x": value,           the model's own value
        "minus": {"level": k, "name": "<j>p<a>-<k>", "x": value},
        "plus":  {...}}}}

A side at level k is ``step * 2**(k - 1)`` of the gap to the neighbouring
grid value on that side; at the edge of an axis, of the gap on its other
side. A parameter with no neighbouring values (the ionisation parameter when
it isn't an axis, or an axis of one value) is perturbed by
``log_step * 2**(k - 1)`` instead (dex, for a parameter in log10).
"""

import json
import math
import os

import yaml

#: The file holding the perturbation plan, in the Cloudy directory
PLAN = "perturb.json"


def is_perturbed(model):
    """Whether a model is a perturbed run (rather than a grid point).

    Args:
        model (str)
            The model, e.g. "3_12" (a grid point) or "3_12p0-1" (perturbed).

    Returns:
        bool
            True if it is a perturbed run.
    """
    return not model.partition("_")[2].isdigit()


def load(cloudy_dir):
    """The perturbation plan (empty if none).

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory, holding ``perturb.json``.

    Returns:
        dict
            The plan (see this module's docstring).
    """
    try:
        with open(os.path.join(cloudy_dir, PLAN)) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def _save(cloudy_dir, plan):
    """Write the perturbation plan, replacing the old one in one step.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory, to hold ``perturb.json``.
        plan (dict)
            The plan (see this module's docstring).
    """
    path = os.path.join(cloudy_dir, PLAN)
    with open(path + ".tmp", "w") as file:
        json.dump(plan, file, indent=1)
    os.replace(path + ".tmp", path)


def _neighbours(cloudy_dir):
    """Each photoionisation point's neighbours on the grid of Cloudy axes.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.

    Returns:
        dict
            Each photoionisation index's neighbours: the indices one grid
            value away along one axis (empty with no Cloudy axes known).
    """
    try:
        with open(os.path.join(cloudy_dir, "grid_points.json")) as file:
            points = json.load(file)["photoionisation"]
    except FileNotFoundError:
        return {}
    values = points["values"]
    grids = [
        sorted({v[n] for v in values}) for n in range(len(points["axes"]))
    ]
    places = [tuple(g.index(x) for g, x in zip(grids, v)) for v in values]
    where = {place: j for j, place in enumerate(places)}
    neighbours = {}
    for j, place in enumerate(places):
        neighbours[j] = []
        for n in range(len(place)):
            for step in (-1, 1):
                other = place[:n] + (place[n] + step,) + place[n + 1 :]
                if other in where:
                    neighbours[j].append(where[other])
    return neighbours


def _axes(cloudy_dir, model):
    """The parameters to perturb a model along, with their gaps.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        model (str)
            The model, e.g. "3_12".

    Returns:
        dict
            Each parameter's "log", "x" (in the interpolation's units: log10
            when "log") and "gaps" (to the lower and upper neighbouring grid
            values, None at an edge or for a parameter that isn't an axis).
    """
    try:
        with open(os.path.join(cloudy_dir, "grid_points.json")) as file:
            points = json.load(file)["photoionisation"]
    except FileNotFoundError:
        # Set up before grid_points.json was written: no Cloudy axes known
        points = {"axes": [], "values": []}
    incident_index, photoionisation_index = model.split("_")
    j = int(photoionisation_index)
    axes = {}
    for position, axis in enumerate(points["axes"]):
        values = sorted({v[position] for v in points["values"]})
        # Log-spaced positive axes (e.g. U, nH) are perturbed in log
        log = values[0] > 0 and values[-1] / values[0] > 10
        grid = [math.log10(v) for v in values] if log else values
        x = points["values"][j][position]
        x = math.log10(x) if log else x
        k = min(range(len(grid)), key=lambda n: abs(grid[n] - x))
        axes[axis] = {
            "log": log,
            "x": x,
            "gaps": [
                x - grid[k - 1] if k > 0 else None,
                grid[k + 1] - x if k < len(grid) - 1 else None,
            ],
        }
    if not axes:
        # No Cloudy axes: perturb the model's own ionisation parameter
        path = os.path.join(
            cloudy_dir, incident_index, f"{photoionisation_index}.yaml"
        )
        with open(path) as file:
            u = float(yaml.safe_load(file)["ionisation_parameter"])
        axes["ionisation_parameter"] = {
            "log": True,
            "x": math.log10(u),
            "gaps": [None, None],
        }
    return axes


def _side(model, position, parameter, side, level, settings):
    """Where a side at a level lies, and its run's name.

    Args:
        model (str)
            The model being perturbed, e.g. "3_12".
        position (int)
            The parameter's position among the model's, used in the name.
        parameter (dict)
            The parameter's plan: "log", "x" and "gaps" (see ``_axes``).
        side (str)
            "minus" or "plus".
        level (int)
            How far out the side is (1 is the first step).
        settings (dict)
            Cloudy.perturb: step, log_step, max_steps.

    Returns:
        dict
            The side's "level", "name" (e.g. "12p0-1") and "x" (in the
            interpolation's units).
    """
    sign = -1 if side == "minus" else 1
    gaps = parameter["gaps"]
    gap = gaps[0] if side == "minus" else gaps[1]
    # At an edge, the gap on the other side
    gap = gap if gap is not None else (gaps[1] if side == "minus" else gaps[0])
    if gap is None:
        offset = settings["log_step"] * 2 ** (level - 1)
    else:
        offset = settings["step"] * 2 ** (level - 1) * gap
    j = model.partition("_")[2]
    return {
        "level": level,
        "name": f"{j}p{position}{'-' if side == 'minus' else '+'}{level}",
        "x": parameter["x"] + sign * offset,
    }


def _value(parameter, x):
    """A value in the interpolation's units back in the parameter's own.

    Args:
        parameter (dict)
            The parameter's plan (see ``_axes``).
        x (float)
            The value, in log10 when the parameter's "log" is set.

    Returns:
        float
            The value of the parameter itself.
    """
    return 10**x if parameter["log"] else x


def advance(cloudy_dir, failed, states, settings, write_input):
    """Plan the next perturbed runs.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        failed (iterable of str)
            The models that have failed every attempt.
        states (dict)
            Each queued model's state ("todo", "claimed", "done", "failed").
        settings (dict)
            Cloudy.perturb: step, log_step, max_steps.
        write_input (callable)
            Writes a run's input: (cloudy_dir, model, name, changes).

    Returns:
        list of str
            The perturbed runs to queue.
    """
    plan = load(cloudy_dir)
    queue = []
    neighbours = _neighbours(cloudy_dir)

    def start(model, position, name, parameter, side, level):
        """Write a side's input, note it in the plan and queue it.

        Args:
            model (str)
                The model being perturbed, e.g. "3_12".
            position (int)
                The parameter's position among the model's.
            name (str)
                The parameter's name.
            parameter (dict)
                The parameter's plan, updated with the side's run.
            side (str)
                "minus" or "plus".
            level (int)
                How far out the side is (1 is the first step).
        """
        run = _side(model, position, parameter, side, level, settings)
        write_input(
            cloudy_dir, model, run["name"], {name: _value(parameter, run["x"])}
        )
        parameter[side] = run
        queue.append(f"{model.partition('_')[0]}_{run['name']}")

    # The failed models that can now be interpolated: both sides of one of
    # their parameters ran
    filled = {
        model
        for model, parameters in plan.items()
        if any(
            all(
                states.get(f"{model.partition('_')[0]}_{p[side]['name']}")
                == "done"
                for side in ("minus", "plus")
            )
            for p in parameters.values()
        )
    }

    # Grid points that have just failed every attempt, and those left
    # surrounded before
    for model in sorted(failed):
        if is_perturbed(model) or plan.get(model):
            continue
        # Surrounded by failures (none of them interpolated yet), there's
        # nothing it could be interpolated back from, so it's left failed
        # until one is
        incident_index, j = model.split("_")
        around = [f"{incident_index}_{n}" for n in neighbours.get(int(j), [])]
        if around and all(
            states.get(other) == "failed" and other not in filled
            for other in around
        ):
            plan[model] = {}
            continue
        plan[model] = _axes(cloudy_dir, model)
        for position, (name, parameter) in enumerate(plan[model].items()):
            for side in ("minus", "plus"):
                start(model, position, name, parameter, side, 1)

    # Sides that failed are tried again further out (and any planned but
    # never queued are queued)
    for model, parameters in plan.items():
        incident_index = model.partition("_")[0]
        for position, (name, parameter) in enumerate(parameters.items()):
            for side in ("minus", "plus"):
                run = parameter[side]
                queued = f"{incident_index}_{run['name']}"
                state = states.get(queued)
                if state is None and queued not in queue:
                    # Planned by a finish job that died before queuing it
                    queue.append(queued)
                elif (
                    state == "failed" and run["level"] < settings["max_steps"]
                ):
                    start(
                        model,
                        position,
                        name,
                        parameter,
                        side,
                        run["level"] + 1,
                    )

    _save(cloudy_dir, plan)
    return queue
