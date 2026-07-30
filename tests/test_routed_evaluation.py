from __future__ import annotations

from scripts.evaluate_routed import use_far_expert


def test_target_router_uses_inclusive_threshold() -> None:
    assert not use_far_expert(9.19, 9.2)
    assert use_far_expert(9.2, 9.2)
    assert use_far_expert(10.4, 9.2)
