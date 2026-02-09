from dataclasses import dataclass
from structures.problem import Problem
from structures.graph import Graph
from typing import (Tuple, Iterable)

@dataclass(frozen=True)
class MoveAction:
    from_node: str
    to_node: str

    def __repr__(self) -> str:
        return f"move({self.from_node} -> {self.to_node})"
    
@dataclass(frozen=True)
class RoutingState:
    location: str

class RoutingProblem(Problem[RoutingState, MoveAction]):
    def __init__(self, graph: Graph, start: str, goal: str):
        self.G = graph
        self.start = start
        self.goal = goal

        if start not in self.G.nodes:
            raise ValueError(f"Start node {start} not in graph")
        if goal not in self.G.nodes:
            raise ValueError(f"Goal node {goal} not in graph")
    
    def initial_state(self) -> RoutingState:
        return RoutingState(location=self.start)
    
    def is_goal(self, state: RoutingState) -> bool:
        return state.location == self.goal
    
    def successors(self, state: RoutingState) -> Iterable[Tuple[MoveAction, RoutingState, float]]:
        for neighbor, cost in self.G.neighbors(state.location):
            action = MoveAction(from_node=state.location, to_node=neighbor)
            new_state = RoutingState(location=neighbor)
            yield (action, new_state, cost)

    def predecessors(self, state):
        for neighbor, cost in self.G.neighbors(state.location):
            action = MoveAction(from_node=state.location, to_node=neighbor)
            new_state = RoutingState(location=neighbor)
            yield (action, new_state, cost)
