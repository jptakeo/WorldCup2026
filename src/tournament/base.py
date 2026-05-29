"""Shared tournament types and abstract simulator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GroupStanding:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Higher is better for all components."""
        return (self.points, self.goal_diff, self.goals_for)


@dataclass
class TournamentResult:
    """Accumulated results across many simulations."""

    counts: int = 0
    first_place: dict[str, int] = field(default_factory=dict)
    second_place: dict[str, int] = field(default_factory=dict)
    third_place: dict[str, int] = field(default_factory=dict)
    group_stage: dict[str, int] = field(default_factory=dict)
    round_of_32: dict[str, int] = field(default_factory=dict)
    round_of_16: dict[str, int] = field(default_factory=dict)
    quarterfinals: dict[str, int] = field(default_factory=dict)
    semifinals: dict[str, int] = field(default_factory=dict)
    final: dict[str, int] = field(default_factory=dict)
    champion: dict[str, int] = field(default_factory=dict)


class TournamentSimulator(ABC):
    """Abstract base for frequentist and Bayesian tournament simulators."""

    @abstractmethod
    def simulate(self, n: int) -> TournamentResult:
        """Run *n* Monte Carlo simulations and return accumulated stage counts."""
