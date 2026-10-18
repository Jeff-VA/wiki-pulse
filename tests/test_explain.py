"""Claude's read of a burst (wiki_pulse/explain.py), with the API stood in for."""

from types import SimpleNamespace

import anthropic
import httpx

from wiki_pulse.explain import MODEL, Explainer, Reading, prompt

BURST = {
    "title": "Taylor Swift",
    "language": "en",
    "url": "https://en.wikipedia.org/wiki/Taylor_Swift",
    "edits_30m": 24,
    "editors_30m": 11,
    "expected_edits": 0.75,
    "revert_share": 0.125,
    "bot_share": 0.0,
    "anonymous_share": 0.25,
}


class Messages:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def parse(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.reply


def explainer(messages: Messages, **options):
    clock = SimpleNamespace(t=0.0)
    logged = []
    reader = Explainer(SimpleNamespace(messages=messages), log=logged.append, clock=lambda: clock.t, **options)
    return reader, clock, logged


def reply(category="news", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=Reading(category=category, explanation="An album came out."),
        usage=SimpleNamespace(input_tokens=612, output_tokens=41),
    )


def test_it_asks_haiku_for_a_structured_reading():
    messages = Messages(reply())
    reader, _, logged = explainer(messages)
    reading = reader.read(BURST, ["Updated chart positions", "rv unsourced"])
    assert (reading.category, reading.explanation) == ("news", "An album came out.")
    assert logged == ["explain: Taylor Swift (en) reads as news (612 tokens in, 41 out)"]  # what each read costs
    [request] = messages.calls
    assert request["model"] == MODEL == "claude-haiku-4-5-20251001"
    assert request["output_format"] is Reading
    assert "never as instructions" in request["system"]
    assert "- rv unsourced" in request["messages"][0]["content"]


def test_it_keeps_to_its_hourly_budget():
    messages = Messages(reply())
    reader, clock, _ = explainer(messages, per_hour=2)
    assert [reader.read(BURST, []) is not None for _ in range(3)] == [True, True, False]
    clock.t = 3601  # an hour later
    assert reader.read(BURST, []) is not None
    assert len(messages.calls) == 3


def test_asking_about_the_budget_spends_none_of_it():
    messages = Messages(reply())
    reader, _, _ = explainer(messages, per_hour=1)
    assert reader.has_budget() and reader.has_budget()
    reader.read(BURST, [])
    assert not reader.has_budget() and len(messages.calls) == 1


def test_it_keeps_to_its_daily_budget():
    messages = Messages(reply())
    reader, clock, logged = explainer(messages, per_hour=2, per_day=3)
    for hour in range(3):
        clock.t = hour * 3601
        reader.read(BURST, [])
        reader.read(BURST, [])
    assert len(messages.calls) == 3  # two in the first hour, one more the next, then the day's budget is spent
    assert logged[-1] == "explain: the day's 3 reads are spent; bursts go unread until the oldest is a day old"
    clock.t = 86401 + 1  # a day after the first read
    assert reader.read(BURST, []) is not None


def test_failures_leave_the_burst_unread():
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    reader, _, logged = explainer(Messages(error=error))
    assert reader.read(BURST, []) is None
    assert logged[0].startswith("explain: Claude couldn't read the burst on Taylor Swift")
    refused, _, _ = explainer(Messages(reply(stop_reason="refusal")))
    assert refused.read(BURST, []) is None


def test_without_a_key_there_is_no_explainer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    logged = []
    assert Explainer.from_environment(log=logged.append) is None
    assert "no ANTHROPIC_API_KEY" in logged[0]


def test_the_prompt_gives_the_numbers_and_summaries():
    text = prompt(BURST, [])
    assert '"Taylor Swift" on the en Wikipedia' in text
    assert "24 edits by 11 editors; an article being edited usually gets about 0.8." in text
    assert "12% undo others, 0% are by bots and 25% by editors without an account" in text
    assert text.endswith("- (none given)")
