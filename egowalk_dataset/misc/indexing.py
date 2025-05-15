from typing import List
from dataclasses import dataclass


@dataclass
class StartingSequence:
    start_idx: int
    n_steps: int
    step: int = 1
    
    def __post_init__(self):
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if self.n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if self.step < 0:
            raise ValueError("step must be non-negative")


@dataclass
class EndingSequence:
    end_idx: int
    n_steps: int
    step: int = 1

    def __post_init__(self):
        if self.end_idx < 0:
            raise ValueError("end_idx must be non-negative")
        if self.n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if self.step < 0:
            raise ValueError("step must be non-negative")


@dataclass
class IndependentSequence:
    indices: List[int]

    def __post_init__(self):
        if len(self.indices) == 0:
            raise ValueError("indices must not be empty")
        if any(idx < 0 for idx in self.indices):
            raise ValueError("all indices must be non-negative")
