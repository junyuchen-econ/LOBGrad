"""
execution/
----------
Downstream optimal-execution evaluation for the LOBGrad benchmark.

Modules
-------
config      : ExecutionConfig — hyperparameters for the execution experiment
cost        : Book-walking execution cost function
schedule    : TWAP, Almgren-Chriss, parametric, cost-weighted schedule constructors
rollouts    : RolloutSource ABC + concrete sources (model, bootstrap, real-future)
mpc         : Receding-horizon MPC algorithm + episode runner
benchmark   : Multi-model / multi-window orchestrator with regret + bootstrap CI
simulate    : JointTimeGradGenerator — extends TimeGrad to model (Δmid, ΔQ)
"""

from .config import ExecutionConfig
from .cost import book_walk_cost, schedule_cost, implementation_shortfall
from .schedule import (
    twap_schedule, almgren_chriss_schedule, parametric_schedule,
    cost_weighted_schedule,
)
from .simulate import JointTimeGradGenerator

__all__ = [
    "ExecutionConfig",
    "book_walk_cost",
    "schedule_cost",
    "implementation_shortfall",
    "twap_schedule",
    "almgren_chriss_schedule",
    "parametric_schedule",
    "cost_weighted_schedule",
    "JointTimeGradGenerator",
]
