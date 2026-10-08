"""Tests for BES forward evolutionary search over tool-call trajectories.

These exercise the inference-time selection wired into ``ffmperative.ffmp`` via
``trajectory_search.search_best_trajectory``. They deliberately import the
existing (non-new) ``ffmperative.interpretor`` module so the test runs the same
extract -> search pipeline the call site uses.
"""

import inspect

from ffmperative.interpretor import extract_function_calls
from ffmperative.trajectory_search import (
    recombine,
    score_trajectory,
    search_best_trajectory,
)

# Minimal tool registry: search only needs the tool names (the keys).
TOOLS = {
    "get_video_info": lambda **k: None,
    "sample_video": lambda **k: None,
    "compress_video": lambda **k: None,
}


def test_extract_then_select_prefers_grounded_trajectory():
    """The full pipeline: extract candidates with the interpretor, then search.

    A rollout whose tool calls are all known and grounded should beat one that
    references a variable the trajectory never bound.
    """
    good_text = (
        "Thought: inspect then sample.\n"
        "get_video_info(input='in.mp4')\n"
        "sample_video(video=get_video_info(input='in.mp4'))"
    )
    bad_text = "sample_video(video=missing_var)"

    good = extract_function_calls(good_text, TOOLS)
    bad = extract_function_calls(bad_text, TOOLS)
    assert "get_video_info" in good  # interpretor really extracted the calls

    best = search_best_trajectory([bad, good], TOOLS)
    assert "get_video_info" in best
    assert "missing_var" not in best


def test_score_trajectory_gives_dense_per_step_feedback():
    code = "get_video_info(input='in.mp4')\nsample_video(video=missing_var)"
    score = score_trajectory(code, set(TOOLS))
    assert score.num_steps == 2
    assert score.num_valid == 1  # second step references an unbound name
    assert 0.0 < score.score < 1.0


def test_recombine_splices_partial_trajectories():
    """Evolution operator chains steps no single rollout produced end-to-end."""
    candidate_a = "clip = sample_video(input='a.mp4')"
    candidate_b = "out = compress_video(video=clip)"

    evolved = recombine([candidate_a, candidate_b], set(TOOLS))
    assert "sample_video" in evolved
    assert "compress_video" in evolved  # depends on `clip` bound by candidate_a


def test_unknown_tool_calls_are_not_grounded():
    score = score_trajectory("not_a_tool(x=1)", set(TOOLS))
    assert score.num_valid == 0
    assert score.score == 0.0


def test_call_site_is_wired():
    """The ffmp call site imports and exposes the search entry point."""
    import ffmperative
    from ffmperative import trajectory_search

    assert ffmperative.search_best_trajectory is trajectory_search.search_best_trajectory
    assert "num_candidates" in inspect.signature(ffmperative.ffmp).parameters
