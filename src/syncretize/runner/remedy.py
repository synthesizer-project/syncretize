"""Run crashed Cloudy models again with numerical remedies.

Some crashes are numerical, not physical: Cloudy's assertion in grain
quantum heating at low density, say. After each round, the finish job runs
each grid point that crashed again with the first of the section's remedies
(``Cloudy.remedies``) whose ``match`` (a regular expression) is found in
why it last crashed, and which it hasn't tried yet. A remedy's ``changes``
are added to those of the remedies it tried before, so a later remedy
keeps what an earlier one fixed. The first remedied run to succeed stands
in for the grid point in the final grid, which flags it (see
``syncretize.cloudy.create_synthesizer_grid.QUALITY``). Only once no remedy
is left is a grid point failed (and perturbed, with ``Cloudy.perturb``).

The runs tried are kept in ``remedies.json`` beside the Cloudy inputs:

    {"<i>_<j>": [{"name": "<j>r<n>", "match": ..., "remedy": {...},
                  "changes": {...}}, ...]}

where the n-th run tried for a model is ``<j>r<n>``, ``remedy`` is its
remedy's changes and ``changes`` every change it ran with.
"""

import json
import os
import re

#: The file holding the remedies tried, in the Cloudy directory
PLAN = "remedies.json"


def is_grid_point(model):
    """Whether a model is a grid point (not a perturbed or remedied run).

    Args:
        model (str)
            The model, e.g. "3_12" (a grid point), "3_12p0-1" (perturbed) or
            "3_12r1" (remedied).

    Returns:
        bool
            True if it is a grid point.
    """
    return model.partition("_")[2].isdigit()


def is_remedy(model):
    """Whether a model is a remedied run, e.g. "3_12r1".

    Args:
        model (str)
            The model.

    Returns:
        bool
            True if it is a remedied run.
    """
    return re.fullmatch(r"\d+r\d+", model.partition("_")[2]) is not None


def load(cloudy_dir):
    """The remedies tried (empty if none).

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory, holding ``remedies.json``.

    Returns:
        dict
            Each model's runs tried (see this module's docstring).
    """
    try:
        with open(os.path.join(cloudy_dir, PLAN)) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def _save(cloudy_dir, plan):
    """Write the remedies tried, replacing the old file in one step.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        plan (dict)
            Each model's runs tried.
    """
    path = os.path.join(cloudy_dir, PLAN)
    with open(path + ".tmp", "w") as file:
        json.dump(plan, file, indent=1)
    os.replace(path + ".tmp", path)


def _run(model, run):
    """A tried run's queue name, e.g. "3_12r1"."""
    return f"{model.partition('_')[0]}_{run['name']}"


def filled(cloudy_dir, states):
    """The grid points a remedied run has filled.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        states (dict)
            Each queued model's state.

    Returns:
        dict
            Each such grid point's successful run, e.g. {"3_12": "3_12r1"}.
    """
    found = {}
    for model, runs in load(cloudy_dir).items():
        for run in runs:
            if states.get(_run(model, run)) == "done":
                found[model] = _run(model, run)
                break
    return found


def pending(cloudy_dir, states):
    """The grid points with a remedied run waiting or running.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        states (dict)
            Each queued model's state.

    Returns:
        set of str
            The grid points.
    """
    return {
        model
        for model, runs in load(cloudy_dir).items()
        if any(states.get(_run(model, r)) in ("todo", "claimed") for r in runs)
    }


def advance(cloudy_dir, states, reasons, remedies, write_input):
    """Queue the next remedy of each grid point that crashed.

    Args:
        cloudy_dir (str)
            The grid's Cloudy directory.
        states (dict)
            Each queued model's state.
        reasons (dict)
            Why each failed model (grid point or run) last crashed.
        remedies (list of dict)
            Cloudy.remedies: each a "match" and "changes".
        write_input (callable)
            Writes a run's input: (cloudy_dir, model, name, changes).

    Returns:
        list of str
            The remedied runs to queue.
    """
    plan = load(cloudy_dir)
    queue = []
    for model, state in sorted(states.items()):
        if state != "failed" or not is_grid_point(model):
            continue
        runs = plan.get(model, [])
        # One at a time: the last tried must have failed too
        if runs and states.get(_run(model, runs[-1])) != "failed":
            continue
        reason = reasons.get(_run(model, runs[-1]) if runs else model) or ""
        tried = [(r["match"], r["remedy"]) for r in runs]
        remedy = next(
            (
                r
                for r in remedies
                if (r["match"], r["changes"]) not in tried
                and re.search(r["match"], reason)
            ),
            None,
        )
        if remedy is None:
            continue
        run = {
            "name": f"{model.partition('_')[2]}r{len(runs) + 1}",
            "match": remedy["match"],
            "remedy": remedy["changes"],
            "changes": (runs[-1]["changes"] if runs else {})
            | remedy["changes"],
        }
        write_input(cloudy_dir, model, run["name"], run["changes"])
        plan[model] = runs + [run]
        queue.append(_run(model, run))
    _save(cloudy_dir, plan)
    return queue
