"""A queue of models on the filesystem, shared by every worker.

Each model is one small file, which moves between directories as its state
changes:

    todo/<model>                 waiting to run
    claimed/<model>@<owner>      being run by owner (e.g. "<jobid>.<loop>")
    done/<model>                 finished
    failed/<model>               failed; the file holds how many times

A worker claims a model by renaming it from ``todo/`` to ``claimed/``. A
rename within one filesystem is atomic, so when two workers try to claim the
same model only one succeeds, with no locks and no server.
"""

import os
import random

STATES = ("todo", "claimed", "done", "failed")


class Queue:
    """The model queue in one directory.

    Attributes:
        root (str)
            The queue's directory.
    """

    def __init__(self, root):
        """Use (and create if needed) the queue in root."""
        self.root = root
        for state in STATES:
            os.makedirs(self._path(state), exist_ok=True)

    def _path(self, state, name=""):
        return os.path.join(self.root, state, name)

    def _names(self, state):
        return os.listdir(self._path(state))

    def add(self, models):
        """Put models in todo, skipping any the queue already has.

        Adding is safe to repeat, so a run can be set up again after a
        crash without losing what has already run.

        Args:
            models (iterable of str)
                The model names (no "/" or "@").
        """
        known = self.models()
        for model in models:
            if "/" in model or "@" in model:
                raise ValueError(f"Model name {model!r} can't hold / or @")
            if model not in known:
                open(self._path("todo", model), "w").close()

    def models(self):
        """Every model in the queue, whatever its state."""
        claimed = {name.rpartition("@")[0] for name in self._names("claimed")}
        return claimed.union(
            *(self._names(state) for state in ("todo", "done", "failed"))
        )

    def claim(self, owner, n=1):
        """Claim up to n models for owner.

        Args:
            owner (str)
                Who is running them, e.g. "<jobid>.<loop>" (no "/").
            n (int)
                The most models to claim.

        Returns:
            list of str
                The models claimed; empty when todo is empty.
        """
        if "/" in owner:
            raise ValueError(f"Owner {owner!r} can't hold /")
        claimed = []
        # Workers start at random places in the listing so they rarely race
        # for the same file. Each claim lists todo/ once, which is fine up to
        # ~1e5 models per run.
        todo = self._names("todo")
        random.shuffle(todo)
        for model in todo:
            try:
                os.rename(
                    self._path("todo", model),
                    self._path("claimed", f"{model}@{owner}"),
                )
            except FileNotFoundError:
                continue  # another worker got there first
            claimed.append(model)
            if len(claimed) == n:
                break
        return claimed

    def done(self, model, owner):
        """Record that owner finished model."""
        os.rename(
            self._path("claimed", f"{model}@{owner}"),
            self._path("done", model),
        )

    def fail(self, model, owner, max_retries=None):
        """Record that owner's run of model failed.

        Args:
            max_retries (int)
                Put the model straight back in todo if it has failed at most
                this many times; None (or more failures) leaves it in failed.

        Returns:
            int
                How many times the model has failed.
        """
        claim = self._path("claimed", f"{model}@{owner}")
        attempts = _attempts(claim) + 1
        with open(claim, "w") as file:
            file.write(str(attempts))
        retry = max_retries is not None and attempts <= max_retries
        os.rename(claim, self._path("todo" if retry else "failed", model))
        return attempts

    def release(self, model, owner):
        """Put a model owner stopped back in todo, without an attempt."""
        os.rename(
            self._path("claimed", f"{model}@{owner}"),
            self._path("todo", model),
        )

    def requeue_failed(self, max_retries):
        """Put failed models that can still be retried back in todo.

        Args:
            max_retries (int)
                How many times a model may be rerun after failing.

        Returns:
            (list of str, list of str)
                The models requeued, and the models that have failed too
                often, which stay in failed.
        """
        requeued, final = [], []
        for model in self._names("failed"):
            path = self._path("failed", model)
            if _attempts(path) <= max_retries:
                os.rename(path, self._path("todo", model))
                requeued.append(model)
            else:
                final.append(model)
        return requeued, final

    def requeue_claims(self, is_running, fail=False):
        """Put models claimed by owners that have stopped back in todo.

        A model whose worker stopped (killed at its time limit, say) didn't
        fail, so its attempt count is unchanged.

        Args:
            is_running (callable)
                Takes an owner and returns whether it is still running.
            fail (bool)
                Record the models as failed instead, an attempt each.

        Returns:
            dict
                Each stopped owner's requeued (or failed) models.
        """
        requeued = {}
        for name in self._names("claimed"):
            model, _, owner = name.rpartition("@")
            if not is_running(owner):
                if fail:
                    self.fail(model, owner)
                else:
                    os.rename(
                        self._path("claimed", name), self._path("todo", model)
                    )
                requeued.setdefault(owner, []).append(model)
        return requeued

    def counts(self):
        """How many models are in each state."""
        return {state: len(self._names(state)) for state in STATES}


def _attempts(path):
    """The failures recorded in a model's file (0 if never failed)."""
    with open(path) as file:
        text = file.read().strip()
    return int(text) if text else 0
