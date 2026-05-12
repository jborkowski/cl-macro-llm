"""Tests for curriculum learning in GRPO training.

Imports `--self-test` mode of grpo_train so that trl/unsloth/datasets do
not need to be installed to run these tests. CurriculumSampler has no
ML deps; only the heavy imports are gated.
"""

import sys
import random
from pathlib import Path

# Make scripts/cloud importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "cloud"))

# Enable --self-test mode so grpo_train.py skips trl/unsloth imports
sys.argv = ["grpo_train.py", "--self-test"]

import grpo_train  # noqa: E402
from grpo_train import CurriculumSampler  # noqa: E402

import pytest  # noqa: E402


def _reset_step():
    """Restore _runtime['step'] to 0 — call in finally blocks."""
    grpo_train._runtime["step"] = 0


def test_sampler_step_0_only_basic():
    complexities = (
        ["basic"] * 10
        + ["intermediate"] * 10
        + ["complex"] * 10
        + ["advanced"] * 10
    )
    sampler = CurriculumSampler(complexities, max_steps=100)
    try:
        grpo_train._runtime["step"] = 0
        random.seed(0)
        basic_set = set(sampler.indices_by_tier[0])
        drawn = []
        for idx in sampler:
            drawn.append(idx)
            if len(drawn) >= 100:
                break
        # __iter__ yields exactly len(complexities)==40 indices; loop above
        # caps at 100 but len(drawn) will be 40. That's enough to assert.
        assert len(drawn) > 0
        for idx in drawn:
            assert idx in basic_set, (
                f"Index {idx} not in basic tier at step 0; "
                f"basic={basic_set}"
            )
    finally:
        _reset_step()


def test_sampler_step_max_all_tiers():
    complexities = (
        ["basic"] * 10
        + ["intermediate"] * 10
        + ["complex"] * 10
        + ["advanced"] * 10
    )
    sampler = CurriculumSampler(complexities, max_steps=100)
    try:
        grpo_train._runtime["step"] = 99
        random.seed(0)
        drawn: list[int] = []
        # Sampler yields exactly len(complexities)=40 per __iter__. Loop
        # iter() until we have at least 200 samples for statistical safety.
        while len(drawn) < 200:
            drawn.extend(list(iter(sampler)))
        tiers_seen = set()
        for tier_idx in range(4):
            tier_set = set(sampler.indices_by_tier[tier_idx])
            if any(d in tier_set for d in drawn):
                tiers_seen.add(tier_idx)
        assert tiers_seen == {0, 1, 2, 3}, (
            f"Expected all 4 tiers to appear in 200 draws at step=99, "
            f"got {tiers_seen}"
        )
    finally:
        _reset_step()


def test_unlock_tier_schedule_boundaries():
    complexities = (
        ["basic"] * 10
        + ["intermediate"] * 10
        + ["complex"] * 10
        + ["advanced"] * 10
    )
    sampler = CurriculumSampler(complexities, max_steps=100)
    assert sampler.unlock_tier(0) == 0
    assert sampler.unlock_tier(24) == 0
    assert sampler.unlock_tier(25) == 1
    assert sampler.unlock_tier(49) == 1
    assert sampler.unlock_tier(50) == 2
    assert sampler.unlock_tier(74) == 2
    assert sampler.unlock_tier(75) == 3
    assert sampler.unlock_tier(99) == 3
    assert sampler.unlock_tier(200) == 3  # over-shoot stays at max tier


def test_schedule_validation():
    # Non-monotonic schedule.
    with pytest.raises(ValueError):
        CurriculumSampler(
            ["basic"] * 4 + ["intermediate"],
            max_steps=100,
            schedule=(0, 0.5, 0.3, 0.8),
        )
    # schedule[0] != 0.
    with pytest.raises(ValueError):
        CurriculumSampler(
            ["basic"] * 4 + ["intermediate"],
            max_steps=100,
            schedule=(0.1, 0.25, 0.5, 0.75),
        )
    # Wrong length.
    with pytest.raises(ValueError):
        CurriculumSampler(
            ["basic"] * 4 + ["intermediate"],
            max_steps=100,
            schedule=(0, 0.25, 0.75),
        )


def test_basic_empty_raises():
    with pytest.raises(RuntimeError, match=r".*basic.*"):
        CurriculumSampler(
            ["intermediate"] * 5 + ["advanced"] * 5,
            max_steps=100,
        )


def test_unknown_complexity_buckets_advanced():
    sampler = CurriculumSampler(
        ["basic"] * 5 + ["???"] * 3 + ["unknown"] * 2,
        max_steps=100,
    )
    assert len(sampler.indices_by_tier[3]) == 5, (
        f"Expected 5 indices in advanced tier (3 '???' + 2 'unknown'), "
        f"got {len(sampler.indices_by_tier[3])}"
    )
    assert len(sampler.indices_by_tier[0]) == 5, (
        f"Expected 5 indices in basic tier, "
        f"got {len(sampler.indices_by_tier[0])}"
    )


def test_iter_length_is_total_dataset_size():
    complexities = (
        ["basic"] * 4
        + ["intermediate"] * 3
        + ["complex"] * 2
        + ["advanced"] * 1
    )
    sampler = CurriculumSampler(complexities, max_steps=100)
    try:
        grpo_train._runtime["step"] = 99
        assert len(list(iter(sampler))) == 10
        assert len(sampler) == 10
    finally:
        _reset_step()


def test_resume_midpoint_first_iter_uses_correct_tier():
    complexities = (
        ["basic"] * 10
        + ["intermediate"] * 10
        + ["complex"] * 10
        + ["advanced"] * 10
    )
    sampler = CurriculumSampler(complexities, max_steps=100)
    try:
        # Set step BEFORE first iter() call (resume scenario).
        grpo_train._runtime["step"] = 50
        random.seed(0)
        drawn: list[int] = []
        while len(drawn) < 200:
            drawn.extend(list(iter(sampler)))
        advanced_set = set(sampler.indices_by_tier[3])
        for idx in drawn:
            assert idx not in advanced_set, (
                f"Advanced index {idx} drawn at step=50 — advanced tier "
                f"should not be unlocked until step 75."
            )
        tiers_seen = set()
        for tier_idx in (0, 1, 2):
            tier_set = set(sampler.indices_by_tier[tier_idx])
            if any(d in tier_set for d in drawn):
                tiers_seen.add(tier_idx)
        assert tiers_seen == {0, 1, 2}, (
            f"Expected tiers 0, 1, 2 to all appear in 200 draws at "
            f"step=50, got {tiers_seen}"
        )
    finally:
        _reset_step()
