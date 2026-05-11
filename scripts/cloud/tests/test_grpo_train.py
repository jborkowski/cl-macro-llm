#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.4"]
# ///
"""Unit tests for the pure-Python helpers in scripts/cloud/grpo_train.py.

These tests run locally on Mac (no GPU, no unsloth, no SBCL): the heavy
imports — unsloth, trl, torch, datasets, macro_gym — are stubbed in
sys.modules BEFORE grpo_train is loaded. Anything that touches those at
import time gets a no-op. The reward path is exercised against a mock
MacroEnv injected via monkeypatch.

Run:
    uv run scripts/cloud/tests/test_grpo_train.py
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path


# ─── stubs for heavy deps that aren't available locally ───────────────

def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeFLM:
    """Stand-in for unsloth.FastLanguageModel — never actually invoked
    in these tests since main()/_run_baseline aren't exercised."""

    @staticmethod
    def from_pretrained(*_a, **_kw):
        raise NotImplementedError("unsloth is stubbed for tests")

    @staticmethod
    def for_inference(*_a, **_kw):
        pass


class _FakeGRPOConfig:
    def __init__(self, **kw):
        self.kw = kw


class _FakeGRPOTrainer:
    def __init__(self, *_a, **_kw):
        raise NotImplementedError("trl is stubbed for tests")


_fake_module("unsloth", FastLanguageModel=_FakeFLM)
_fake_module("trl", GRPOConfig=_FakeGRPOConfig, GRPOTrainer=_FakeGRPOTrainer)
_fake_module("torch", inference_mode=contextlib.nullcontext)


class _FakeDataset:
    """Minimal Dataset replacement covering the surface load_katas uses."""

    def __init__(self, rows=None):
        self._rows = list(rows or [])

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self._rows]
        return self._rows[key]

    def train_test_split(self, test_size, seed=42):
        n = max(1, int(round(len(self._rows) * test_size)))
        return {
            "train": _FakeDataset(self._rows[:-n]),
            "test":  _FakeDataset(self._rows[-n:]),
        }


_fake_module("datasets", Dataset=_FakeDataset)


# Now we can safely import grpo_train.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # scripts/cloud
import grpo_train  # noqa: E402


# ─── tests: extract_defmacro ──────────────────────────────────────────

def test_extract_defmacro_simple():
    src = "(defmacro foo (x) `(list ,x ,x))"
    assert grpo_train.extract_defmacro(src) == src


def test_extract_defmacro_strips_think():
    text = (
        "<think>plan: bind expr, then list it twice</think>\n"
        "(defmacro foo (x) `(list ,x))"
    )
    got = grpo_train.extract_defmacro(text)
    assert got is not None
    assert "<think>" not in got
    assert got.startswith("(defmacro")


def test_extract_defmacro_truncated_returns_none():
    # Never closes — must NOT pretend it's a valid macro.
    assert grpo_train.extract_defmacro("(defmacro foo (x) `(list ,x") is None


def test_extract_defmacro_nested_parens():
    src = "(defmacro foo (x) (let ((y (+ x 1))) `(list ,x ,y)))"
    assert grpo_train.extract_defmacro(src) == src


def test_extract_defmacro_handles_strings_with_parens():
    # A ')' inside a string literal must not close the form prematurely.
    src = '(defmacro foo (x) `(format t "(hi ~A)" ,x))'
    assert grpo_train.extract_defmacro(src) == src


def test_extract_defmacro_handles_escaped_quote_in_string():
    src = '(defmacro foo (x) `(format t "say \\")\\"" ,x))'
    got = grpo_train.extract_defmacro(src)
    assert got == src


def test_extract_defmacro_finds_after_prose_and_fences():
    text = (
        "Sure! Here is a macro:\n"
        "```lisp\n"
        "(defmacro inc (x) `(1+ ,x))\n"
        "```\n"
        "Done."
    )
    assert grpo_train.extract_defmacro(text) == "(defmacro inc (x) `(1+ ,x))"


def test_extract_defmacro_case_insensitive_head():
    src = "(DEFMACRO foo (x) `(list ,x))"
    assert grpo_train.extract_defmacro(src) == src


def test_extract_defmacro_no_match_returns_none():
    assert grpo_train.extract_defmacro("just some prose") is None


# ─── tests: _completion_to_text ───────────────────────────────────────

def test_completion_to_text_string_passthrough():
    assert grpo_train._completion_to_text("hello") == "hello"


def test_completion_to_text_messages_picks_assistant():
    msgs = [
        {"role": "user",      "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert grpo_train._completion_to_text(msgs) == "a"


def test_completion_to_text_messages_picks_last_assistant():
    msgs = [
        {"role": "assistant", "content": "first"},
        {"role": "user",      "content": "follow-up"},
        {"role": "assistant", "content": "second"},
    ]
    assert grpo_train._completion_to_text(msgs) == "second"


def test_completion_to_text_falls_back_to_last_when_no_assistant():
    msgs = [{"role": "user", "content": "only user"}]
    assert grpo_train._completion_to_text(msgs) == "only user"


def test_completion_to_text_empty_list():
    assert grpo_train._completion_to_text([]) == ""


# ─── tests: load_katas ────────────────────────────────────────────────

def _write_kata(d: Path, instruction: str = "do thing",
                category: str = "control-flow") -> None:
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({
        "instruction": instruction,
        "category": category,
        "complexity": "basic",
        "quality_score": 0.9,
    }))


def test_load_katas_two_level_layout(tmp_path):
    _write_kata(tmp_path / "cl-ds" / "k001")
    _write_kata(tmp_path / "creative" / "k002", category="anaphoric")
    # An ignored dir (leading underscore) and an empty subdir.
    (tmp_path / "_skipme").mkdir()
    (tmp_path / "empty-sub").mkdir()

    ds = grpo_train.load_katas(tmp_path)
    assert len(ds) == 2
    names = sorted(Path(p).name for p in ds["kata_path"])
    assert names == ["k001", "k002"]
    assert sorted(set(ds["category"])) == ["anaphoric", "control-flow"]


def test_load_katas_one_level_layout(tmp_path):
    # metadata.json directly under kata_root/<dir>/
    _write_kata(tmp_path / "k001")
    _write_kata(tmp_path / "k002", category="dsl")
    ds = grpo_train.load_katas(tmp_path)
    assert len(ds) == 2
    assert sorted(set(ds["category"])) == ["control-flow", "dsl"]


def test_load_katas_skips_blank_instruction(tmp_path):
    d = tmp_path / "kbad"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"instruction": "   "}))
    _write_kata(tmp_path / "kgood")
    ds = grpo_train.load_katas(tmp_path)
    assert len(ds) == 1
    assert Path(ds["kata_path"][0]).name == "kgood"


def test_load_katas_empty_root(tmp_path):
    assert len(grpo_train.load_katas(tmp_path)) == 0


# ─── tests: macro_gym_reward (with mock env) ──────────────────────────

class _MockEnv:
    """Reward = 1.0 if defmacro contains 'good', else -0.4."""

    def __init__(self, kata_dir):
        self.kata_dir = kata_dir
        self.closed = False

    def step(self, defmacro):
        r = 1.0 if "good" in defmacro else -0.4
        return None, r, True, False, {}

    def close(self):
        self.closed = True


def _install_mock_macro_gym(env_cls):
    mod = types.ModuleType("macro_gym")
    mod.MacroEnv = env_cls
    sys.modules["macro_gym"] = mod


def test_macro_gym_reward_happy_path(monkeypatch):
    _install_mock_macro_gym(_MockEnv)
    grpo_train._env_cache.clear()
    completions = [
        "<think>plan</think>\n(defmacro good (x) `(list ,x))",
        "(defmacro bad (x) `(broken",   # truncated → -0.2
        "no macro at all here",          # no defmacro → -0.2
    ]
    rewards = grpo_train.macro_gym_reward(
        prompts=[None, None, None],
        completions=completions,
        kata_path=["/tmp/a", "/tmp/b", "/tmp/c"],
    )
    assert rewards == [1.0, -0.2, -0.2]


def test_macro_gym_reward_clamps_to_upper_bound(monkeypatch):
    class _CrazyEnv:
        def __init__(self, kata_dir): pass
        def step(self, defmacro):
            return None, 99.0, True, False, {}
        def close(self): pass

    _install_mock_macro_gym(_CrazyEnv)
    grpo_train._env_cache.clear()
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_path=["/tmp/x"],
    )
    assert rewards == [1.5]


def test_macro_gym_reward_clamps_to_lower_bound(monkeypatch):
    class _NegEnv:
        def __init__(self, kata_dir): pass
        def step(self, defmacro):
            return None, -99.0, True, False, {}
        def close(self): pass

    _install_mock_macro_gym(_NegEnv)
    grpo_train._env_cache.clear()
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_path=["/tmp/x"],
    )
    assert rewards == [-0.5]


def test_macro_gym_reward_step_error_penalised(monkeypatch):
    class _BoomEnv:
        def __init__(self, kata_dir): pass
        def step(self, defmacro):
            raise RuntimeError("sbcl died")
        def close(self): pass

    _install_mock_macro_gym(_BoomEnv)
    grpo_train._env_cache.clear()
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_path=["/tmp/x"],
    )
    assert rewards == [-0.3]


def test_macro_gym_reward_mismatched_lengths_raises(monkeypatch):
    _install_mock_macro_gym(_MockEnv)
    grpo_train._env_cache.clear()
    import pytest
    with pytest.raises(RuntimeError, match="reward fn"):
        grpo_train.macro_gym_reward(
            prompts=[None, None],
            completions=["(defmacro a () nil)", "(defmacro b () nil)"],
            kata_path=["/tmp/x"],
        )


def test_macro_gym_reward_completion_as_messages(monkeypatch):
    _install_mock_macro_gym(_MockEnv)
    grpo_train._env_cache.clear()
    msgs = [
        {"role": "user",      "content": "write the macro"},
        {"role": "assistant", "content": "(defmacro good (x) `(list ,x))"},
    ]
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=[msgs],
        kata_path=["/tmp/x"],
    )
    assert rewards == [1.0]


def test_env_cache_eviction_fifo(monkeypatch):
    """When the cache fills, the oldest entry should be closed and dropped."""
    instantiated: list[str] = []

    class _CountingEnv:
        def __init__(self, kata_dir):
            instantiated.append(kata_dir)
            self.kata_dir = kata_dir
            self.closed = False
        def step(self, defmacro):
            return None, 0.0, True, False, {}
        def close(self):
            self.closed = True

    _install_mock_macro_gym(_CountingEnv)
    grpo_train._env_cache.clear()
    monkeypatch.setattr(grpo_train, "_ENV_CACHE_MAX", 2)

    a = grpo_train._get_env("/k/a")
    b = grpo_train._get_env("/k/b")
    assert "/k/a" in grpo_train._env_cache and "/k/b" in grpo_train._env_cache
    c = grpo_train._get_env("/k/c")   # evicts /k/a
    assert "/k/a" not in grpo_train._env_cache
    assert a.closed is True
    assert "/k/b" in grpo_train._env_cache
    assert "/k/c" in grpo_train._env_cache
    assert instantiated == ["/k/a", "/k/b", "/k/c"]


# ─── module-level env knobs sanity ────────────────────────────────────

def test_grpoconfig_unsloth_defaults_visible():
    """The Unsloth long-context recommended defaults must be wired
    into module-level constants so main() picks them up."""
    assert grpo_train.LOSS_TYPE == "grpo"
    assert abs(grpo_train.EPSILON - 0.2) < 1e-9
    assert abs(grpo_train.EPSILON_HIGH - 0.28) < 1e-9
    assert abs(grpo_train.DELTA - 1.5) < 1e-9
    assert grpo_train.MASK_TRUNCATED_COMPLETIONS is True


def test_unsloth_vllm_standby_is_set_at_import():
    import os
    # The module set it on import; even if the test env didn't have it,
    # importing grpo_train (above) should have planted the default.
    assert os.environ.get("UNSLOTH_VLLM_STANDBY") == "1"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
