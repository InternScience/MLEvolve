"""configuration and setup utils"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Hashable, cast
import datetime
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

# Lazy import to avoid circular dependency with engine.search_node
# Journal and filter_journal are imported where needed via _get_journal_classes()
def _get_journal_classes():
    from engine.search_node import Journal, filter_journal
    return Journal, filter_journal

from utils import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("MLEvolve")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str

@dataclass
class DecayConfig:
    exploration_constant: float
    lower_bound: float
    alpha: float
    phase_ratios: list
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    topk_max_improves: int
    max_improve_failure: int
    parallel_search_num: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    top_candidates_size: int
    stagnation_window: int
    num_gpus: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    topk_early_k: int
    topk_early_max_per_branch: int
    topk_late_k: int
    topk_late_max_per_branch: int
    force_backprop_late_threshold: float
    force_backprop_late_prob: float
    force_backprop_mid_threshold: float
    force_backprop_mid_modulo: int
    recent_best_window: int
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int
    use_stagnation_detection: bool = True

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    initial_drafts: int
    seed: int
    data_preview: bool
    code: StageConfig
    feedback: StageConfig
    check_data_leakage: bool
    fusion_vs_evolution_prob: float
    branch_fusion_trigger_prob: float
    max_fusion_drafts: int
    use_global_memory: bool
    memory_similarity_threshold: float
    memory_embedding_device: str
    memory_embedding_model_path: str
    search: SearchConfig
    decay: DecayConfig
    use_diff_mode: bool = True
    use_stepwise_generation: bool = True
    use_evolution: bool = True
    use_fusion: bool = True
    use_aggregation: bool = True
    # Bounds for the data-leakage trigger (see agents/triggers.py).
    leakage_max_threshold: float = 0.999
    leakage_min_threshold: float = 1e-6
@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str


@dataclass
class ColdstartConfig:
    use_coldstart: bool
    task_json_path: str
    model_json_path: str
    description: str


@dataclass
class InitSolutionConfig:
    use: bool = False


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str
    exp_id: str

    torch_hub_dir: str
    pretrain_model_dir: str

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str

    coldstart: ColdstartConfig

    use_grading_server: bool = True
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg

def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    # generate experiment name and prefix with consecutive index
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.exp_name = f"{timestamp}_{cfg.exp_name or coolname.generate_slug(3)}"

    # If log_dir and workspace_dir point to the same path, treat it as a unified
    # "runs" root and place logs/workspace under the per-run directory
    if top_log_dir == top_workspace_dir:
        runs_root = top_log_dir
        runs_root.mkdir(parents=True, exist_ok=True)
        per_run_root = (runs_root / cfg.exp_name).resolve()
        cfg.log_dir = (per_run_root / "logs").resolve()
        cfg.workspace_dir = (per_run_root / "workspace").resolve()
    else:
        top_log_dir.mkdir(parents=True, exist_ok=True)
        top_workspace_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    Journal, filter_journal = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
