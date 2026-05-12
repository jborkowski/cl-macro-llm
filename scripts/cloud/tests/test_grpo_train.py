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


class _FakeTrainerCallback:
    """transformers.TrainerCallback stand-in — just an empty base class."""
    pass


_fake_module("transformers", TrainerCallback=_FakeTrainerCallback)


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


# Fake macro_gym module — exposes the v0.3 API (get_grader,
# shutdown_grader, MacroGrader) that grpo_train.py imports at module load.
# Per-test, swap `_mock_grader._handler` to control what verdicts come back.

class _MockGrader:
    """In-test grader. `_handler` (if set) is called as
    handler(kata_id, macro_src) -> Result dict; otherwise returns a
    zero-reward verdict."""

    def __init__(self):
        self._handler = None

    def grade(self, kata_id, macro_src, timeout=None):
        if self._handler is None:
            return {"reward": 0.0, "passed": 0, "total": 0, "error": None}
        return self._handler(kata_id, macro_src)

    def grade_batch(self, items, max_workers=None, timeout=None):
        return [self.grade(kid, src) for kid, src in items]

    def reward_fn(self, prompts, completions, *, kata_ids, timeout=None,
                  **kwargs):
        # mirrors the real grader's reward_fn signature
        return [
            float(self.grade(k, c).get("reward", -0.1))
            for k, c in zip(kata_ids, completions)
        ]

    def close(self):
        pass


# Stash the singleton grader on the fake module itself so that, even if
# pytest's collector + the script entry point both end up importing this
# test module (different module objects, different module globals), every
# `from macro_gym import get_grader` resolves to the SAME grader instance.
def _get_grader():
    return sys.modules["macro_gym"]._mock_grader


def _shutdown_grader():
    pass


_fake_module(
    "macro_gym",
    get_grader=_get_grader,
    shutdown_grader=_shutdown_grader,
    MacroGrader=_MockGrader,
)
sys.modules["macro_gym"]._mock_grader = _MockGrader()
# Also stub macro_gym.env which load_katas() reaches for the KATAS_DIR
# symlink directory.
import tempfile  # noqa: E402
_mg_env = types.ModuleType("macro_gym.env")
_mg_env.KATAS_DIR = Path(tempfile.mkdtemp(prefix="test-macro-gym-katas-"))
sys.modules["macro_gym.env"] = _mg_env


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

# ─── tests: macro_gym_reward (against mocked grader) ──────────────────
#
# In v0.3, the trainer delegates entirely to macro_gym.MacroGrader via
# get_grader() singleton. Tests swap _mock_grader._handler per-test to
# control what verdicts come back; grpo_train.py never sees the real grader.

def _set_grader_handler(handler):
    """Install a (kata_id, macro_src) -> verdict-dict handler on the mock.

    Resolves via `sys.modules["macro_gym"]` so it works whether this test
    module is loaded once or twice (script entry vs pytest collector).
    """
    sys.modules["macro_gym"]._mock_grader._handler = handler


def _reset_grader(monkeypatch=None):
    sys.modules["macro_gym"]._mock_grader._handler = None


def test_macro_gym_reward_happy_path(monkeypatch):
    """Defmacro containing 'good' gets reward 1.0; truncated or missing
    defmacro feeds an empty string to the grader (the trainer's
    extract_defmacro returns None there, we pass '' downstream which the
    grader treats as a syntax error → -0.1)."""
    _set_grader_handler(lambda kid, src:
                        {"reward": 1.0 if "good" in src else -0.1})
    completions = [
        "<think>plan</think>\n(defmacro good (x) `(list ,x))",
        "(defmacro bad (x) `(broken",   # truncated → extract returns None → ''
        "no macro at all here",          # no defmacro → extract returns None → ''
    ]
    rewards = grpo_train.macro_gym_reward(
        prompts=[None, None, None],
        completions=completions,
        kata_ids=["ka", "kb", "kc"],
    )
    assert rewards == [1.0, -0.1, -0.1]


def test_macro_gym_reward_passes_grader_value_through(monkeypatch):
    """Trainer no longer client-side-clamps rewards — the grader's reward
    scale (-0.1 / 0.0 / 0.1-0.9 / 1.0) is authoritative."""
    _set_grader_handler(lambda kid, src: {"reward": 0.7})
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_ids=["kx"],
    )
    assert rewards == [0.7]


def test_macro_gym_reward_handles_grader_error_field(monkeypatch):
    """If the grader returns an error-tagged Result, the reward field still
    drives the trainer (typically -0.1 from the grader for protocol failures)."""
    _set_grader_handler(lambda kid, src: {
        "reward": -0.1, "passed": 0, "total": 0,
        "error": {"type": "timeout", "message": "macroexpand timeout"},
    })
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_ids=["kx"],
    )
    assert rewards == [-0.1]


def test_macro_gym_reward_missing_reward_field_defaults(monkeypatch):
    """If the grader returns a verdict with no reward (malformed Result),
    the trainer falls back to -0.1 (syntax-error tier)."""
    _set_grader_handler(lambda kid, src: {})   # empty verdict
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro x () nil)"],
        kata_ids=["kx"],
    )
    assert rewards == [-0.1]


def test_macro_gym_reward_mismatched_lengths_raises(monkeypatch):
    import pytest
    _reset_grader()
    with pytest.raises(RuntimeError, match="reward fn"):
        grpo_train.macro_gym_reward(
            prompts=[None, None],
            completions=["(defmacro a () nil)", "(defmacro b () nil)"],
            kata_ids=["ka"],
        )


def test_macro_gym_reward_completion_as_messages(monkeypatch):
    """Chat-format completions (list of role/content dicts) → _completion_to_text
    extracts the assistant turn before defmacro extraction."""
    _set_grader_handler(lambda kid, src:
                        {"reward": 1.0 if "good" in src else -0.1})
    msgs = [
        {"role": "user",      "content": "write the macro"},
        {"role": "assistant", "content": "(defmacro good (x) `(list ,x))"},
    ]
    rewards = grpo_train.macro_gym_reward(
        prompts=[None],
        completions=[msgs],
        kata_ids=["kx"],
    )
    assert rewards == [1.0]


def test_macro_gym_reward_empty_returns_empty(monkeypatch):
    _reset_grader()
    assert grpo_train.macro_gym_reward(
        prompts=[], completions=[], kata_ids=[],
    ) == []


def test_macro_gym_reward_extracts_from_fenced_block(monkeypatch):
    """A defmacro inside ```lisp fences (common LLM output shape) gets
    pulled out before the grader sees it."""
    seen_src = []
    def _h(kid, src):
        seen_src.append(src)
        return {"reward": 0.9}
    _set_grader_handler(_h)
    completion = (
        "Sure, here's the macro:\n"
        "```lisp\n"
        "(defmacro foo (x) `(1+ ,x))\n"
        "```\n"
        "Done."
    )
    rewards = grpo_train.macro_gym_reward(
        prompts=[None], completions=[completion], kata_ids=["kx"],
    )
    assert rewards == [0.9]
    assert seen_src == ["(defmacro foo (x) `(1+ ,x))"]


def test_sample_dump_writes_jsonl(monkeypatch, tmp_path):
    """When LOG_SAMPLES_EVERY and step align, the reward fn writes a JSONL."""
    _set_grader_handler(lambda kid, src:
                        {"reward": 1.0 if "good" in src else -0.1})
    monkeypatch.setitem(grpo_train._runtime, "step", 50)
    monkeypatch.setitem(grpo_train._runtime, "log_samples_every", 25)
    monkeypatch.setitem(grpo_train._runtime, "n_samples_per_dump", 2)
    monkeypatch.setitem(grpo_train._runtime, "output_dir", tmp_path)

    completions = [
        "(defmacro good (x) `(list ,x))",
        "(defmacro good-two (x) `(list ,x))",
        "no macro",
    ]
    grpo_train.macro_gym_reward(
        prompts=[{"role": "user", "content": f"q{i}"} for i in range(3)],
        completions=completions,
        kata_ids=["ka", "kb", "kc"],
    )
    dump = tmp_path / "samples-step-00050.jsonl"
    assert dump.exists()
    lines = [json.loads(ln) for ln in dump.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    for entry in lines:
        assert entry["step"] == 50
        assert "completion" in entry
        assert "reward" in entry


def test_sample_dump_on_demand(monkeypatch, tmp_path):
    """sample_now=True forces a dump even mid-cycle."""
    _set_grader_handler(lambda kid, src:
                        {"reward": 1.0 if "good" in src else -0.1})
    monkeypatch.setitem(grpo_train._runtime, "step", 7)   # mid-cycle
    monkeypatch.setitem(grpo_train._runtime, "log_samples_every", 25)
    monkeypatch.setitem(grpo_train._runtime, "sample_now", True)
    monkeypatch.setitem(grpo_train._runtime, "n_samples_per_dump", 1)
    monkeypatch.setitem(grpo_train._runtime, "output_dir", tmp_path)

    grpo_train.macro_gym_reward(
        prompts=[None],
        completions=["(defmacro good (x) `(list ,x))"],
        kata_ids=["ka"],
    )
    assert (tmp_path / "samples-step-00007.jsonl").exists()
    # sample_now should auto-clear after firing
    assert grpo_train._runtime["sample_now"] is False


# ─── tests: runtime callbacks ─────────────────────────────────────────

class _FakeState:
    def __init__(self, step):
        self.global_step = step


class _FakeControl:
    def __init__(self):
        self.should_save = False
        self.should_training_stop = False


def test_metrics_logger_appends_jsonl(tmp_path):
    cb = grpo_train.MetricsLoggerCallback(tmp_path / "metrics.jsonl")
    cb.on_log(None, _FakeState(10), _FakeControl(),
              logs={"loss": 0.5, "reward": 0.3})
    cb.on_log(None, _FakeState(20), _FakeControl(),
              logs={"loss": 0.4, "reward": 0.4})
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {"step": 10, "loss": 0.5, "reward": 0.3}
    assert parsed[1] == {"step": 20, "loss": 0.4, "reward": 0.4}


def test_metrics_logger_skips_empty(tmp_path):
    cb = grpo_train.MetricsLoggerCallback(tmp_path / "metrics.jsonl")
    cb.on_log(None, _FakeState(1), _FakeControl(), logs=None)
    cb.on_log(None, _FakeState(2), _FakeControl(), logs={})
    assert not (tmp_path / "metrics.jsonl").exists()


def test_runtime_control_stop(tmp_path):
    ctl = tmp_path / "control.json"
    ctl.write_text(json.dumps({"stop": True}))
    args = types.SimpleNamespace(temperature=0.9)
    cb = grpo_train.RuntimeControlCallback(ctl, args)
    control = _FakeControl()
    cb.on_step_begin(args, _FakeState(42), control)
    assert control.should_training_stop is True
    assert control.should_save is True
    assert grpo_train._runtime["step"] == 42


def test_runtime_control_save_now(tmp_path):
    ctl = tmp_path / "control.json"
    ctl.write_text(json.dumps({"save_now": True}))
    args = types.SimpleNamespace(temperature=0.9)
    cb = grpo_train.RuntimeControlCallback(ctl, args)
    control = _FakeControl()
    cb.on_step_begin(args, _FakeState(15), control)
    assert control.should_save is True
    assert control.should_training_stop is False


def test_runtime_control_sample_now_and_temp(tmp_path):
    grpo_train._runtime["sample_now"] = False
    ctl = tmp_path / "control.json"
    ctl.write_text(json.dumps({"sample_now": True, "temperature": 0.5,
                               "log_samples_every": 10}))
    args = types.SimpleNamespace(temperature=0.9)
    cb = grpo_train.RuntimeControlCallback(ctl, args)
    cb.on_step_begin(args, _FakeState(3), _FakeControl())
    assert grpo_train._runtime["sample_now"] is True
    assert args.temperature == 0.5
    assert grpo_train._runtime["log_samples_every"] == 10


def test_runtime_control_skips_unchanged_file(tmp_path):
    ctl = tmp_path / "control.json"
    ctl.write_text(json.dumps({"save_now": True}))
    args = types.SimpleNamespace(temperature=0.9)
    cb = grpo_train.RuntimeControlCallback(ctl, args)

    c1 = _FakeControl()
    cb.on_step_begin(args, _FakeState(1), c1)
    assert c1.should_save is True

    # Second call without mtime change: should NOT re-fire save_now.
    c2 = _FakeControl()
    cb.on_step_begin(args, _FakeState(2), c2)
    assert c2.should_save is False


def test_runtime_control_missing_file_is_noop(tmp_path):
    ctl = tmp_path / "does-not-exist.json"
    args = types.SimpleNamespace(temperature=0.9)
    cb = grpo_train.RuntimeControlCallback(ctl, args)
    c = _FakeControl()
    cb.on_step_begin(args, _FakeState(99), c)
    assert c.should_save is False
    assert c.should_training_stop is False
    assert grpo_train._runtime["step"] == 99   # step still updated


# ─── module-level env knobs sanity ────────────────────────────────────

def test_grpoconfig_unsloth_defaults_visible():
    """The Unsloth long-context recommended defaults must be wired
    into module-level constants so main() picks them up."""
    assert grpo_train.LOSS_TYPE == "grpo"
    assert abs(grpo_train.EPSILON - 0.2) < 1e-9
    assert abs(grpo_train.EPSILON_HIGH - 0.28) < 1e-9
    assert abs(grpo_train.DELTA - 1.5) < 1e-9
    assert grpo_train.MASK_TRUNCATED_COMPLETIONS is True


def test_unsloth_env_defaults_planted_at_import():
    import os
    # Module sets these on import. STANDBY is OFF (use_vllm=False, so the
    # rollout-weight swap dance buys nothing and silently strips
    # PYTORCH_ALLOC_CONF, which is the thing actually keeping us off the
    # OOM cliff at the chunked log-softmax allocation).
    assert os.environ.get("UNSLOTH_VLLM_STANDBY") == "0"
    assert "expandable_segments:True" in os.environ.get("PYTORCH_ALLOC_CONF", "")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
