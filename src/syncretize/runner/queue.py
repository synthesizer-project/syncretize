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

#: A claimer lists todo/ again once this many models in a row from its last
#: listing were gone (claimed by other workers since)
STALE_AFTER = 20


class Queue:
    """The model queue in one directory.

    Attributes:
        root (str)
            The queue's directory.
    """

    def __init__(self, root):
        """Use (and create if needed) the queue in root.

        Args:
            root (str)
                The queue's directory.
        """
        self.root = root
        self._listings = {}  # each claimer's shuffled listing of todo/
        for state in STATES:
            os.makedirs(self._path(state), exist_ok=True)

    def _path(self, state, name=""):
        """A path in one state's directory.

        Args:
            state (str)
                One of STATES.
            name (str)
                A file name in it, or "" for the directory itself.

        Returns:
            str
                The path.
        """
        return os.path.join(self.root, state, name)

    def names(self, state):
        """The file names in one state's directory.

        Args:
            state (str)
                One of STATES.

        Returns:
            list of str
                Its file names: the models, or "<model>@<owner>" for claims.
        """
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
        """Every model in the queue, whatever its state.

        Returns:
            set of str
        """
        return set(self.states())

    def states(self):
        """Each model's state.

        Returns:
            dict
                Each model's state (one of STATES), by model.
        """
        return {
            name.partition("@")[0]: state
            for state in STATES
            for name in self.names(state)
        }

    def claim(self, owner, n=1):
        """Claim up to n models for owner.

        Args:
            owner (str)
                Who is running them, e.g. "<jobid>.<loop>" (no "/").
            n (int)
                The most models to claim.

        Returns:
            list of str
                The models claimed; empty only when todo is empty.
        """
        if "/" in owner:
            raise ValueError(f"Owner {owner!r} can't hold /")
        # Listing todo/ for every claim, by thousands of workers at once,
        # is a storm of metadata on Lustre, so each claimer keeps its last
        # listing (shuffled, so workers rarely race for the same file) and
        # works through it, listing again only once it runs out or has gone
        # stale. Models put back in todo/ since are found then.
        listing = self._listings.setdefault(owner, [])
        claimed, misses, fresh = [], 0, False
        while len(claimed) < n:
            if not listing or (misses >= STALE_AFTER and not fresh):
                if fresh:
                    break  # todo/ is empty
                listing[:] = self.names("todo")
                random.shuffle(listing)
                misses, fresh = 0, True
                continue
            model = listing.pop()
            try:
                os.rename(
                    self._path("todo", model),
                    self._path("claimed", f"{model}@{owner}"),
                )
            except FileNotFoundError:
                misses += 1  # another worker got there first
                continue
            claimed.append(model)
            misses = 0
        return claimed

    def done(self, model, owner):
        """Record that owner finished model.

        Args:
            model (str)
                The model, e.g. "3_12".
            owner (str)
                The claimer that ran it, e.g. "12_3.0".
        """
        os.rename(
            self._path("claimed", f"{model}@{owner}"),
            self._path("done", model),
        )

    def fail(self, model, owner, max_retries=None):
        """Record that owner's run of model failed.

        Args:
            model (str)
                The model, e.g. "3_12".
            owner (str)
                The claimer that ran it, e.g. "12_3.0".
            max_retries (int)
                Put the model straight back in todo if it has failed at most
                this many times; None (or more failures) leaves it in failed.

        Returns:
            int
                How many times the model has failed.
        """
        claim = self._path("claimed", f"{model}@{owner}")
        attempts = _attempts(claim) + 1
        # Written over the old count, not after emptying the file, so a
        # worker killed mid-write can't leave it empty (no failures); the
        # count only grows, so the new one covers every old digit
        with open(claim, "r+") as file:
            file.write(str(attempts))
            file.truncate()
        retry = max_retries is not None and attempts <= max_retries
        os.rename(claim, self._path("todo" if retry else "failed", model))
        return attempts

    def release(self, model, owner):
        """Put a model owner stopped back in todo, without an attempt.

        Args:
            model (str)
                The model, e.g. "3_12".
            owner (str)
                The claimer that held it, e.g. "12_3.0".
        """
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
        for model in self.names("failed"):
            path = self._path("failed", model)
            if _attempts(path) <= max_retries:
                os.rename(path, self._path("todo", model))
                requeued.append(model)
            else:
                final.append(model)
        return requeued, final

    def requeue_claims(self, is_running):
        """Put models claimed by owners that have stopped back in todo.

        A model whose worker stopped (killed at its time limit, say) didn't
        fail, so its attempt count is unchanged.

        Args:
            is_running (callable)
                Takes an owner and returns whether it is still running.

        Returns:
            dict
                Each stopped owner's requeued models.
        """
        requeued = {}
        for name in self.names("claimed"):
            model, _, owner = name.rpartition("@")
            if not is_running(owner):
                os.rename(
                    self._path("claimed", name), self._path("todo", model)
                )
                requeued.setdefault(owner, []).append(model)
        return requeued

    def counts(self):
        """How many models are in each state.

        Returns:
            dict
                Each of STATES and its number of models.
        """
        return {state: len(self.names(state)) for state in STATES}


def _attempts(path):
    """The failures recorded in a model's file (0 if never failed).

    Args:
        path (str)
            The model's file, in any state's directory.

    Returns:
        int
            How many times it has failed.
    """
    with open(path) as file:
        text = file.read().strip()
    return int(text) if text else 0
