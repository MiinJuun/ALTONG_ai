def should_pass(urgency_score: int, relevance_score: int) -> bool:
    """
    Determine whether a notification should be passed immediately.

    Policy:
    - urgency >= 4  -> PASS
    - urgency == 3 and relevance >= 4 -> PASS
    - otherwise -> BLOCK
    """

    if not 1 <= urgency_score <= 5:
        raise ValueError("urgency_score must be between 1 and 5")

    if not 1 <= relevance_score <= 5:
        raise ValueError("relevance_score must be between 1 and 5")

    if urgency_score >= 4:
        return True

    if urgency_score == 3 and relevance_score >= 4:
        return True

    return False


def decision_label(urgency_score: int, relevance_score: int) -> str:
    return "PASS" if should_pass(urgency_score, relevance_score) else "BLOCK"