"""Tests for the Runner's filesystem model queue."""

import multiprocessing

from syncretize.runner.queue import Queue

MODELS = [f"{i}_{j}" for i in range(20) for j in range(25)]


def drain(root, owner, start, results):
    """Claim models two at a time until none are left, recording each."""
    queue = Queue(root)
    start.wait()  # every worker starts claiming at once
    while claimed := queue.claim(owner, n=2):
        for model in claimed:
            results.put((owner, model))
            queue.done(model, owner)


def test_concurrent_claims_never_share_a_model(tmp_path):
    Queue(tmp_path).add(MODELS)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    start = context.Barrier(8)
    workers = [
        context.Process(
            target=drain, args=(tmp_path, f"job.{n}", start, results)
        )
        for n in range(8)
    ]
    for worker in workers:
        worker.start()
    claims = [results.get(timeout=60) for _ in MODELS]
    for worker in workers:
        worker.join()

    assert sorted(model for _, model in claims) == sorted(MODELS)
    assert Queue(tmp_path).counts() == {
        "todo": 0,
        "claimed": 0,
        "done": len(MODELS),
        "failed": 0,
    }


def test_add_skips_known_models(tmp_path):
    queue = Queue(tmp_path)
    queue.add(["a", "b", "c"])
    (claimed,) = queue.claim("1.0")
    queue.done(claimed, "1.0")
    queue.claim("1.0")
    queue.add(["a", "b", "c", "d"])
    assert queue.counts() == {"todo": 2, "claimed": 1, "done": 1, "failed": 0}


def test_failures_are_retried_then_kept(tmp_path):
    queue = Queue(tmp_path)
    queue.add(["m"])
    for attempt in (1, 2, 3):
        assert queue.claim("1.0") == ["m"]
        assert queue.fail("m", "1.0") == attempt
        requeued, final = queue.requeue_failed(max_retries=2)
        if attempt <= 2:
            assert (requeued, final) == (["m"], [])
        else:
            assert (requeued, final) == ([], ["m"])
    assert queue.counts()["failed"] == 1


def test_stopped_workers_claims_are_requeued_without_an_attempt(tmp_path):
    queue = Queue(tmp_path)
    queue.add(["a", "b", "c"])
    first = queue.claim("1.0")
    queue.fail(first[0], "1.0")
    queue.requeue_failed(max_retries=2)
    stopped = queue.claim("1.0", n=3)
    running = queue.claim("2.0")
    assert running == []  # all three are with job 1

    requeued = queue.requeue_claims(lambda owner: owner.startswith("2."))
    assert sorted(requeued["1.0"]) == ["a", "b", "c"]
    assert sorted(stopped) == ["a", "b", "c"]

    # The earlier failure is still counted; the stop isn't
    queue.claim("2.0", n=3)
    assert queue.fail(first[0], "2.0") == 2
