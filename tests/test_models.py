import pytest
from pydantic import ValidationError

from triage.models import Verdict


def test_verdict_accepts_valid_output():
    verdict = Verdict.model_validate_json(
        '{"severity": "low", "escalate": false, "reason": "Matches CHG-12345", "evidence": ["CHG-12345"]}'
    )
    assert verdict.evidence == ("CHG-12345",)
    assert verdict.needs_human is False


@pytest.mark.parametrize("payload", [
    '{"severity": "critical", "escalate": true, "reason": "x"}',
    '{"severity": "low", "escalate": false, "reason": ""}',
    '{"severity": "low", "escalate": false, "reason": "x", "confidence": 0.9}',
])
def test_verdict_rejects_invalid_output(payload):
    with pytest.raises(ValidationError):
        Verdict.model_validate_json(payload)
