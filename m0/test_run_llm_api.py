"""Unit tests of the API path of run_llm.py (GPT-4o): transient errors are retried with backoff, other
errors propagate, and a run that stops resumes from its output file without duplicates. A fake client stands in
for the OpenAI SDK (no network). Run: python test_run_llm_api.py"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import run_llm

HERE = Path(__file__).resolve().parent


class RateLimitError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


def response(token="Yes"):
    top = [NS(token=token, logprob=-0.1), NS(token="No" if token == "Yes" else "Yes", logprob=-2.5)]
    ch = NS(message=NS(content=token), logprobs=NS(content=[NS(top_logprobs=top)]))
    return NS(choices=[ch], system_fingerprint="fp_test", usage=NS(prompt_tokens=42))


class FakeClient:
    """create() raises the queued errors first, then answers; `fail_after` makes every call after that many
    successful calls raise a non-transient error (a stopped run)."""
    def __init__(self, errors=(), fail_after=None):
        self.errors, self.fail_after, self.calls, self.ok = list(errors), fail_after, 0, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.fail_after is not None and self.ok >= self.fail_after:
            raise BadRequestError("stopped")
        self.ok += 1
        return response("Yes" if len(kw["messages"][0]["content"]) % 2 else "No")


def chat(client):
    m = object.__new__(run_llm.OpenAIChat)
    m.c, m.model = client, "gpt-4o-test"
    return m


def test_retry_then_success():
    waits = []
    orig = run_llm.with_retry
    run_llm.with_retry = lambda fn, **kw: orig(fn, sleep=waits.append, **kw)
    try:
        c = FakeClient(errors=[RateLimitError("429"), RateLimitError("429"), type("InternalServerError", (Exception,), {})("500")])
        out = chat(c)(["Q?"])
    finally:
        run_llm.with_retry = orig
    assert c.calls == 4 and len(waits) == 3 and waits[0] < waits[1] < waits[2] <= 60, (c.calls, waits)
    assert out[0]["fallback"] is False and out[0]["lp_yes"] is not None and out[0]["system_fingerprint"] == "fp_test"


def test_non_transient_and_exhausted():
    c = FakeClient(errors=[BadRequestError("bad")])
    try:
        run_llm.with_retry(lambda: c.create(messages=[{"content": "x"}]), sleep=lambda s: None)
        raise AssertionError("a bad request must propagate")
    except BadRequestError:
        assert c.calls == 1
    c = FakeClient(errors=[RateLimitError("429")] * 5)
    try:
        run_llm.with_retry(lambda: c.create(messages=[{"content": "x"}]), tries=3, sleep=lambda s: None)
        raise AssertionError("retries must be bounded")
    except RateLimitError:
        assert c.calls == 3


def test_resume_after_stop():
    out = Path(tempfile.mkdtemp()) / "e10.jsonl"
    argv = ["run_llm.py", "--backend", "openai", "--model", "gpt-4o-test", "--instances", str(HERE / "data/m0_instances.jsonl"),
            "--limit", "2", "--serializations", "AL", "--out", str(out), "--name-report", str(out.parent / "nf.json"),
            "--names-from", ""]
    clients = [FakeClient(fail_after=5), FakeClient()]
    orig = run_llm.OpenAIChat.__init__

    def init(self, model, base_url=None):
        self.c, self.model = clients.pop(0), model
    run_llm.OpenAIChat.__init__ = init
    try:
        sys.argv = argv
        try:
            run_llm.main()
            raise AssertionError("the first run must stop")
        except BadRequestError:
            pass
        first = [json.loads(l) for l in open(out)]
        assert len(first) == 5, len(first)
        run_llm.main()
    finally:
        run_llm.OpenAIChat.__init__ = orig
    recs = [json.loads(l) for l in open(out)]
    keys = [(r["id"], r["ser"], r["tpl"]) for r in recs]
    assert len(keys) == len(set(keys)) and len(recs) > 5 and recs[:5] == first


if __name__ == "__main__":
    test_retry_then_success()
    test_non_transient_and_exhausted()
    test_resume_after_stop()
    print("test_run_llm_api: ok")
