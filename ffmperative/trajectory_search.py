"""Forward evolutionary search over candidate tool-call trajectories.

Adapted from *Self-Improving Language Models with Bidirectional Evolutionary
Search* (BES, arXiv:2605.28814). BES argues that best-of-N sampling is limited
because (a) candidates are selected on a single sparse end-of-trajectory signal
and (b) every candidate comes from one autoregressive rollout, so exploration
stays confined to a narrow region of model probability mass.

This module brings BES's *forward candidate evolution* and its *dense
intermediate feedback* idea to FFMPerative's inference path. Each candidate the
agent emits is a trajectory of tool calls. We score every trajectory with a
per-step verifier (the BES "checkable subgoal": each step is validated, not just
the final result) and add an evolution operator that recombines groundable
partial trajectories from different rollouts into a candidate that no single
rollout produced.

The search is purely inference-time: no training, no checkpoints, no gradients.
It selects/recombines the best already-generated tool-call sequence before it is
handed to ``interpretor.evaluate``.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TrajectoryScore:
    """Dense scoring of a single candidate tool-call trajectory."""

    code: str
    steps: List[str] = field(default_factory=list)  # source of the groundable steps, in order
    num_steps: int = 0
    num_valid: int = 0
    defined: Set[str] = field(default_factory=set)

    @property
    def score(self) -> float:
        """Fraction of checkable subgoals satisfied, biased toward depth.

        A trajectory that grounds more tool calls (more satisfied subgoals)
        scores higher, mirroring BES's preference for trajectories that make
        verifiable progress over those that merely parse.
        """
        if self.num_steps == 0:
            return 0.0
        coverage = self.num_valid / self.num_steps
        return coverage + 0.01 * self.num_valid


def _statements(code: str) -> List[ast.stmt]:
    """Parse a candidate into top-level statements, tolerating empty input."""
    if not code or not code.strip():
        return []
    try:
        return ast.parse(code).body
    except SyntaxError:
        return []


def _call_node(node: ast.stmt) -> Optional[ast.Call]:
    """Return the tool ``Call`` a statement represents, if any."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return node.value
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        return node.value
    return None


def _called_name(call: ast.Call) -> Optional[str]:
    return call.func.id if isinstance(call.func, ast.Name) else None


def _referenced_names(call: ast.Call) -> Set[str]:
    """Variable names a call loads from prior state (its subgoal dependencies)."""
    names: Set[str] = set()
    for child in ast.walk(call):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    # The callee itself is a Name(Load); it is a tool, not a state dependency.
    if isinstance(call.func, ast.Name):
        names.discard(call.func.id)
    return names


def _assigned_names(node: ast.stmt) -> Set[str]:
    out: Set[str] = set()
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.add(target.id)
    return out


def _step_is_grounded(node: ast.stmt, tool_names: Set[str], defined: Set[str]) -> bool:
    """A step is a checkable subgoal: a known tool call whose inputs are bound."""
    call = _call_node(node)
    if call is None:
        return False
    name = _called_name(call)
    if name is None or name not in tool_names:
        return False
    return _referenced_names(call).issubset(defined)


def score_trajectory(code: str, tool_names: Set[str]) -> TrajectoryScore:
    """Walk a candidate, validating each step against accumulated state.

    Produces the dense feedback signal: how many tool-call subgoals are
    well-formed and have their dependencies satisfied by earlier steps.
    """
    result = TrajectoryScore(code=code)
    defined: Set[str] = set()
    for node in _statements(code):
        result.num_steps += 1
        if _step_is_grounded(node, tool_names, defined):
            result.num_valid += 1
            result.steps.append(ast.unparse(node))
            defined |= _assigned_names(node)
    result.defined = defined
    return result


def recombine(candidate_codes: List[str], tool_names: Set[str]) -> str:
    """Evolution operator: splice groundable steps from several rollouts.

    Greedily assembles a new trajectory by repeatedly appending, from the pool
    of all candidates' steps, the next tool call whose dependencies are already
    satisfied and that contributes a new bound variable. The result can chain
    partial trajectories that no single rollout produced end-to-end.
    """
    pool: List[ast.stmt] = []
    for code in candidate_codes:
        pool.extend(_statements(code))

    defined: Set[str] = set()
    chosen: List[str] = []
    seen: Set[str] = set()
    used = [False] * len(pool)

    progress = True
    while progress:
        progress = False
        for idx, node in enumerate(pool):
            if used[idx] or not _step_is_grounded(node, tool_names, defined):
                continue
            src = ast.unparse(node)
            if src in seen:
                used[idx] = True
                continue
            produced = _assigned_names(node)
            # Keep a step if it binds something new or is a terminal effect call.
            if produced and produced.issubset(defined):
                used[idx] = True
                continue
            used[idx] = True
            seen.add(src)
            chosen.append(src)
            defined |= produced
            progress = True
    return "\n".join(chosen)


def search_best_trajectory(
    candidate_codes: List[str],
    tools: Dict[str, object],
    recombine_enabled: bool = True,
) -> str:
    """Select (and optionally evolve) the strongest tool-call trajectory.

    This is the BES forward search: score every candidate with the dense
    verifier, additionally synthesise a recombined candidate, and return the
    highest-scoring tool-call sequence as source ready for evaluation.
    """
    cleaned = [c for c in candidate_codes if c and c.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1 and not recombine_enabled:
        return cleaned[0]

    tool_names = set(tools.keys())
    scored = [score_trajectory(c, tool_names) for c in cleaned]

    if recombine_enabled:
        evolved = recombine(cleaned, tool_names)
        if evolved.strip():
            scored.append(score_trajectory(evolved, tool_names))

    # Highest dense score wins; ties break toward the earlier (original) rollout.
    best = max(range(len(scored)), key=lambda i: (scored[i].score, -i))
    return scored[best].code
