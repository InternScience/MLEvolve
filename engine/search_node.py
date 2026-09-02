"""SearchNode: solution tree node (code, execution, evaluation, search metadata)."""

import copy
import difflib
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dataclasses_json import DataClassJsonMixin
from engine.executor import ExecutionResult
from config import SearchConfig
from utils.metric import MetricValue
from utils.response import trim_long_string

logger = logging.getLogger("MLEvolve")


@dataclass(eq=False)
class SearchNode(DataClassJsonMixin):
    """Solution tree node: code, execution results, evaluation, and search metadata."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore
    prompt_input: str | None = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["SearchNode"] = field(default=None, kw_only=True)
    children: set["SearchNode"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    is_valid: bool = field(default=None, kw_only=True)  # type: ignore

    # ---- search / MCTS ----
    stage: Literal["root", "improve", "debug", "draft", "fusion_draft", "evolution", "fusion"]
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    is_terminal: bool = field(default=False, kw_only=True)
    _uct: float = field(default=0.0, kw_only=True)
    local_best_node: Optional["SearchNode"] = field(default=None, kw_only=True)
    is_debug_success: bool = field(default=False, kw_only=True)
    continue_improve: bool = field(default=False, kw_only=True)
    improve_failure_depth: int = field(default=0, kw_only=True)
    lock: bool = field(default=False, kw_only=True)
    child_count_lock: bool = threading.Lock()
    expected_child_count: int = field(default=0, kw_only=True)
    finish_time: str = field(default=None, kw_only=True)
    created_time: str = field(default=None, kw_only=True)

    # ---- Bayesian sampling ----
    alpha: int = field(default=1, kw_only=True)
    beta: int = field(default=1, kw_only=True)

    # ---- branch management ----
    branch_id: Optional[int] = field(default=None, kw_only=True)
    from_topk: bool = field(default=False, kw_only=True)
    code_summary: Optional[str] = field(default=None, kw_only=True)
    work_dir: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)
        if self.stage not in ["root", "improve", "debug", "draft", "fusion_draft", "evolution", "fusion"]:
            raise ValueError(f"Invalid stage: {self.stage}")

    # ---- base node properties ----

    @property
    def stage_name(self) -> str:
        """Inferred stage based on parent relationship."""
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __eq__(self, other):
        return isinstance(other, SearchNode) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

    # ---- search methods ----

    
    def update_beta(self, success: bool):
        if success: 
            self.alpha += 1
        else:
            self.beta += 1
            
    def p_mean(self):
        return self.alpha / (self.alpha + self.beta)
    
    
    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) value of the current node.
        UCT = Q + c * sqrt(ln(N) / n), where:
        - Q = total_reward / visits (average reward)
        - c = exploration_constant (exploration constant, default is sqrt(2))
        - N = parent_visits (number of visits to the parent node)
        - n = visits (number of visits to the current node)
        """
        parent_visits: int | None = None
        if self.parent:
            parent_visits = self.parent.visits
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have the highest priority
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * (math.log(parent_visits) / self.visits) ** 0.5
        self._uct = exploitation + exploration
        return self._uct

    def reached_child_limit(self, scfg: SearchConfig, for_topk: bool = False) -> bool:
        """Whether this node has reached its child limit (draft/improve/debug). for_topk uses higher limit."""
        with self.child_count_lock:
            if self.step == 0:
                regular_draft_count = sum(1 for child in self.children if child.stage == "draft")
                # expected_child_count includes in-flight children; estimate in-flight drafts
                in_flight = max(0, self.expected_child_count - len(self.children))
                regular_expected = regular_draft_count + in_flight
                logger.info(f"[reached_child_limit] node {self.id} regular_draft_count={regular_draft_count}, in_flight={in_flight}, limit={scfg.num_drafts}")
                return regular_expected >= scfg.num_drafts
            else:
                if self.is_buggy:
                    if self.has_no_bug_child():
                        return True
                    else:
                        return self.expected_child_count >= scfg.num_bugs
                else:
                    if for_topk:
                        topk_max_improves = getattr(scfg, 'topk_max_improves', 10)
                        return self.expected_child_count >= topk_max_improves
                    else:
                        regular_expected = sum(
                            1 for child in self.children
                            if not getattr(child, 'from_topk', False)
                        )
                        regular_expected += (self.expected_child_count - len(self.children))
                        return regular_expected >= scfg.num_improves

    
    def update(self, result, add=True):
        if add:
            self.visits += 1
            self.total_reward += result
        
    def has_no_bug_child(self):
        for child in self.children:
            if not child.is_buggy:
                return True
        return False

    @property
    def num_children(self):
        return len(self.children)

    def fetch_child_memory(self, include_code=False):
        """Build memory string from children for the model (include draft nodes; optionally include code diff)."""
        logger.info("fetch_child_memory")
        summary = []

        candidates = [n for n in self.children
                      if n.is_buggy is not None or n.stage == "draft"]

        # The metric direction is carried by MetricValue.maximize. Without it a
        # minimize metric (RMSE, MAE, logloss) is ranked backwards and the model
        # is shown the worst sibling as "Attempt #1". Same convention as
        # engine/conditions.py: fall back to maximize when unknown.
        maximize = next(
            (n.metric.maximize for n in candidates
             if n.metric is not None and n.metric.maximize is not None),
            True,
        )
        missing = float('-inf') if maximize else float('inf')

        sorted_children = sorted(
            candidates,
            key=lambda n: (
                n.is_buggy is False,
                n.is_buggy is not None,
                (n.metric.value if (n.metric and n.metric.value is not None)
                 else missing) * (1 if maximize else -1)
            ),
            reverse=True
        )

        for idx, n in enumerate(sorted_children, 1):
            summary_part = f"Attempt #{idx}:\n"
            summary_part += f"Design: {n.plan}\n"

            if include_code and self.code and n.code:
                code_diff = self._compute_code_diff(self.code, n.code)
                if code_diff:
                    summary_part += f"Code Changes:\n{code_diff}\n"
                else:
                    summary_part += f"Code Changes: (minimal or formatting changes only)\n"

            if n.is_buggy is None:
                summary_part += f"Status: Code generated, execution pending (will run in parallel with other drafts).\n"
            elif n.is_buggy is True:
                summary_part += f"Results: The implementation of this design has bugs.\n"
                summary_part += f"Insight: Using a different approach may not result in the same bugs as the above approach.\n"
            else:
                if n.analysis:
                    summary_part += f"Results: {n.analysis}\n"
                if n.metric and n.metric.value is not None:
                    metric_display = self._format_metric_change(n)
                    summary_part += f"Validation Metric: {metric_display}\n"
                if hasattr(n, 'exec_time') and n.exec_time is not None:
                    summary_part += f"Execution Time: {n.exec_time:.2f}s\n"

            summary.append(summary_part)

        if len(summary) == 0:
            summary.append("")
        else:
            total_attempts = len(sorted_children)
            pending = [n for n in sorted_children if n.is_buggy is None]
            executed = [n for n in sorted_children if n.is_buggy is not None]
            successful = [n for n in executed if n.is_buggy is False]

            stats_parts = []
            if pending:
                stats_parts.append(f"{len(pending)} pending execution")
            if executed:
                stats_parts.append(f"{len(executed)} executed")
                if successful:
                    values = [n.metric.value for n in successful
                              if n.metric and n.metric.value is not None]
                    best_metric = max(values) if maximize else min(values)
                    stats_parts.append(f"{len(successful)} successful (best: {best_metric:.4f})")
                else:
                    stats_parts.append(f"0 successful (all failed or buggy)")

            stats = f"Summary: {total_attempts} total attempts - " + ", ".join(stats_parts)
            summary.insert(0, stats + "\n")

        return "\n-------------------------------\n".join(summary)

    def _format_metric_change(self, node) -> str:
        """Format metric change for display (respects maximize/minimize)."""
        if not node.metric or node.metric.value is None:
            return "N/A"

        current_val = node.metric.value

        if (node.parent and
            hasattr(node.parent, 'is_buggy') and
            node.parent.is_buggy is False and
            node.parent.metric and
            node.parent.metric.value is not None):

            parent_val = node.parent.metric.value
            raw_change = current_val - parent_val

            if hasattr(node.metric, 'maximize'):
                if node.metric.maximize:
                    improvement = raw_change
                    direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                else:
                    improvement = -raw_change
                    direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            else:
                improvement = raw_change
                direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"

            return f"{parent_val:.4f} → {current_val:.4f} ({improvement:+.4f} {direction})"
        else:
            return f"{current_val:.4f}"

    def _compute_code_diff(self, parent_code: str, child_code: str, context_lines: int = 3) -> str:
        """Compute formatted diff between parent and child code."""
        parent_lines = parent_code.splitlines(keepends=True)
        child_lines = child_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            parent_lines,
            child_lines,
            fromfile='Parent Code',
            tofile='Modified Code',
            lineterm='',
            n=context_lines
        )

        diff_lines = list(diff)
        if not diff_lines:
            return ""

        formatted_diff = []
        for line in diff_lines[2:]:
            if line.startswith('@@'):
                continue
            elif line.startswith('+') and not line.startswith('+++'):
                formatted_diff.append(f"  + {line[1:]}")
            elif line.startswith('-') and not line.startswith('---'):
                formatted_diff.append(f"  - {line[1:]}")
            elif not line.startswith(('---', '+++')):
                if len(formatted_diff) < 100:
                    formatted_diff.append(f"    {line}")

        if len(formatted_diff) > 100:
            formatted_diff = formatted_diff[:100]
            formatted_diff.append("  ... (diff truncated, too many changes)")

        return '\n'.join(formatted_diff) if formatted_diff else ""

    def fetch_parent_memory(self, include_code=False):
        logger.info("fetch_parent_memory")
        summary = []
        if self.parent is not None and self.parent.is_buggy is not None and self.parent.is_buggy is False:
            summary_part = f"Design: {self.parent.plan}\n"
            if include_code:
                summary_part += f"Code: {self.parent.code}\n"
            summary_part += f"Results: {self.parent.analysis}\n"
            summary_part += f"Validation Metric: {self.parent.metric.value}\n"
            if hasattr(self.parent, 'exec_time') and self.parent.exec_time is not None:
                summary_part += f"Execution Time: {self.parent.exec_time:.2f}s\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
    
    def add_expected_child_count(self):
        with self.child_count_lock:
            self.expected_child_count += 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")
            
            
    def sub_expected_child_count(self):
        with self.child_count_lock:
            self.expected_child_count -= 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('child_count_lock', None) 
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.child_count_lock = threading.Lock()
    
    def generate_node_trajectory(self, need_code=False) -> str:
        """Return formatted trajectory string for this node."""
        summary_part = f""
        if hasattr(self, 'branch_id') and self.branch_id:
            summary_part += f"Branch ID: {self.branch_id}\n"

        summary_part += f"Stage: {self.stage.upper()}\n"
        if self.plan:
            summary_part += f"Design: {self.plan}\n"

        if self.code and need_code:
            summary_part += f"Code: {self.code}\n"

        if self.is_buggy is True:
            summary_part += f"Results: The implementation of this design has bugs.\n"
            if self.analysis:
                summary_part += f"Analysis: {self.analysis}\n"
        elif self.is_buggy is False:
            if self.analysis:
                summary_part += f"Results: {self.analysis}\n"
            if self.metric and self.metric.value is not None:
                metric_display = self._format_metric_change(self)
                summary_part += f"Validation Metric: {metric_display}\n"
            if hasattr(self, 'exec_time') and self.exec_time is not None:
                summary_part += f"Execution Time: {self.exec_time:.2f}s\n"

        else:
            summary_part += f"Results: Step not yet executed.\n"
            logger.warning(f"Node {self.id} is not executed.")
        
        return summary_part
    
    def get_root_to_current_trajectory(self, max_steps: int = None, llm_summary_threshold: int = 5) -> str:
        """Return formatted trajectory from root to this node (optionally limited to max_steps)."""
        trajectory = self._get_trajectory_raw(max_steps)
        return self._get_trajectory_full(trajectory)
    
    def _get_trajectory_raw(self, max_steps: int = None) -> List[str]:
        """Collect raw trajectory steps from this node up to root."""
        trajectory = []
        current = self
        while current and current.parent:
            step_trajectory = current.generate_node_trajectory()
            trajectory.append(step_trajectory)
            current = current.parent
            if max_steps and len(trajectory) >= max_steps:
                break
        return list(reversed(trajectory))
    
    def _get_trajectory_full(self, trajectory: List[str]) -> str:
        """Format trajectory as Step 1: ..., Step 2: ..."""
        trajectory_parts = []
        
        for i, step_trajectory in enumerate(trajectory):
            step_header = f"Step {i+1}:"
            step_info = f"{step_header}\n{step_trajectory}"
            trajectory_parts.append(step_info)
        
        return "\n-------------------------------\n".join(trajectory_parts)
    
    
# ---------------------------------------------------------------------------
# Journal — ordered collection of SearchNodes forming the solution tree
# ---------------------------------------------------------------------------

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[SearchNode] = field(default_factory=list)

    def __getitem__(self, idx: int) -> SearchNode:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: SearchNode) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[SearchNode]:
        """Return a list of nodes representing initial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def good_nodes(self) -> list[SearchNode]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good=True) -> None | SearchNode:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)


def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    path = [node_id]
    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}
    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id
    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    journal_copy = copy.deepcopy(journal)
    journal_copy.nodes = [n for n in journal_copy.nodes if n.id in path]
    for n in journal_copy.nodes:
        n._term_out = "<OMITTED>"
        n.exc_stack = "<OMITTED>"
    return journal_copy


def filter_for_best_path(journal: Journal, best_node: str) -> Journal:
    path_to_best = get_path_to_node(journal, best_node)
    return filter_on_path(journal, path_to_best)


def filter_for_longest_path(journal: Journal) -> Journal:
    longest_path = get_longest_path(journal)
    return filter_on_path(journal, longest_path)


def filter_journal(journal: Journal) -> Journal:
    best_node = journal.get_best_node(only_good=True)
    if best_node is not None:
        return filter_for_best_path(journal, best_node.id)
    else:
        return filter_for_longest_path(journal)

