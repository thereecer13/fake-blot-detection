import json
from types import SimpleNamespace

from PIL import Image

from fake_blot_detection.detect import ClaudeDetector, run_detector


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def text_response(payload: dict):
    return SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        stop_details=None,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


REFUSAL = SimpleNamespace(
    stop_reason="refusal",
    model="claude-opus-5-5",
    stop_details=SimpleNamespace(category="other"),
    content=[],
)


def make_detector(responses):
    messages = FakeMessages(responses)
    return ClaudeDetector(client=SimpleNamespace(messages=messages)), messages


def make_images(tmp_path, n):
    (tmp_path / "images").mkdir()
    manifest = []
    for i in range(n):
        Image.new("L", (64, 32), 120).save(tmp_path / "images" / f"{i}.jpg")
        manifest.append({"image_id": str(i), "path": f"images/{i}.jpg", "label": "real"})
    return manifest


def test_judge_parses_answer_and_sends_image(tmp_path):
    manifest = make_images(tmp_path, 1)
    detector, messages = make_detector(
        [text_response({"reasoning": "smooth bands", "verdict": "ai", "probability_ai": 0.8})]
    )
    result = detector.judge(tmp_path / manifest[0]["path"])
    assert result["verdict"] == "ai" and result["probability_ai"] == 0.8
    content = messages.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_refusals_and_bad_answers_are_recorded_and_runs_resume(tmp_path):
    manifest = make_images(tmp_path, 3)
    out = tmp_path / "results.jsonl"
    detector, _ = make_detector(
        [
            text_response({"reasoning": "x", "verdict": "real", "probability_ai": 0.1}),
            REFUSAL,
            text_response({"reasoning": "x", "verdict": "ai", "probability_ai": 7}),  # out of range
        ]
    )
    results = run_detector(detector, manifest, tmp_path, out)
    assert results[0]["probability_ai"] == 0.1
    assert results[1]["stop_reason"] == "refusal" and results[1]["refusal_category"] == "other"
    assert "error" in results[2]

    # Re-running makes no new API calls: everything is already in the results file.
    detector2, messages2 = make_detector([])
    assert len(run_detector(detector2, manifest, tmp_path, out)) == 3
    assert messages2.calls == []
