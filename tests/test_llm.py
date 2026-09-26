from triage.llm import TokenBudget, retry_delay

PER_MINUTE = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota. "
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 15\\nPlease retry in 42.096307241s.', "
    "'details': [{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}, {'retryDelay': '42s'}]}}]"
)
PER_DAY = PER_MINUTE.replace("PerMinute", "PerDay")


def test_per_minute_limits_are_waited_out():
    assert retry_delay(Exception(PER_MINUTE)) == 42.096307241


def test_daily_quota_and_unknown_errors_are_not():
    assert retry_delay(Exception(PER_DAY)) is None
    assert retry_delay(Exception("Error code: 429 - too many requests")) is None


def test_token_budget_waits_for_the_window_to_clear():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    budget = TokenBudget(10_000, clock=lambda: now[0], sleep=sleep)
    first, wait = budget.reserve(6_000)
    assert wait == 0
    now[0] = 20
    _, wait = budget.reserve(3_000)
    assert wait == 0
    budget.settle(first, 7_500)  # the provider counted more than estimated
    _, wait = budget.reserve(2_000)  # 7,500 + 3,000 + 2,000 > 10,000: wait for the first to expire
    assert slept and now[0] >= 60
