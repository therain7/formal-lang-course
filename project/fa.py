import itertools
import operator
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, NamedTuple, Optional, Self, cast

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import csr_array, kron

from project.bidict import BiDict


class AdjacencyMatrixFA:
    """
    Finite automaton implementation based on sparse adjacency matrices.
    Can be constructed from `pyformlang`'s `NondeterministicFiniteAutomaton`

    Attributes:
        states_count: amount of states in automaton
        states: map of states' names to indicies
        adj: boolean adjacency matrices for respective transition symbols
        start_states: indices of start states
        final_states: indices of final states
    """

    @classmethod
    def from_intersection(cls, fa1: Self, fa2: Self) -> Self:
        """
        Construct intersection of finite automata.
        `states` attribute will contain keys of the following kind:
        `State((st1, st2))`, where `st1` & `st2` are states' names
        of `fa1` & `fa2` respectively
        """
        inter = cls(None)

        # use kronecker product to construct intersection
        inter.states_count = fa1.states_count * fa2.states_count

        for st1, st2 in itertools.product(fa1.states.keys(), fa2.states.keys()):
            idx1, idx2 = fa1.states[st1], fa2.states[st2]
            inter_idx = fa2.states_count * idx1 + idx2

            if idx1 in fa1.start_states and idx2 in fa2.start_states:
                inter.start_states.add(inter_idx)
            if idx1 in fa1.final_states and idx2 in fa2.final_states:
                inter.final_states.add(inter_idx)

            inter.states[State((st1, st2))] = inter_idx

        for sym, adj1 in fa1.adj.items():
            if (adj2 := fa2.adj.get(sym)) is None:
                continue

            inter.adj[sym] = cast(csr_array, kron(adj1, adj2, format="csr"))

        return inter

    def __init__(self, fa: Optional[NondeterministicFiniteAutomaton]):
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()

        # completely empty automaton
        if fa is None:
            self.states_count = 0
            self.states = BiDict()
            self.adj = {}
            return

        self.states: BiDict[State, int] = BiDict()
        for idx, st in enumerate(fa.states):
            self.states[st] = idx

            if st in fa.start_states:
                self.start_states.add(idx)
            if st in fa.final_states:
                self.final_states.add(idx)

            idx += 1

        self.states_count = len(self.states)
        self.adj: dict[Symbol, csr_array] = defaultdict(
            lambda: csr_array((self.states_count, self.states_count), dtype=bool)
        )

        for st1, sym, st2 in fa._transition_function.get_edges():
            self.adj[sym][self.states[st1], self.states[st2]] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        class Conf(NamedTuple):
            word: list[Symbol]
            state: int

        word = list(word)
        stack = [Conf(word, start) for start in self.start_states]

        while len(stack) != 0:
            conf = stack.pop()

            if not conf.word:
                if conf.state in self.final_states:
                    # 1 successful path is enough to accept
                    return True
                continue

            if (adj := self.adj.get(conf.word[0])) is None:
                continue

            for next_state in range(self.states_count):
                if adj[conf.state, next_state]:
                    stack.append(Conf(conf.word[1:], next_state))

        return False

    def transitive_closure(self) -> csr_array:
        """
        Returns transitive closure for automaton states.
        Get indices from `states` attribute to index the matrix
        """
        tc = csr_array((self.states_count, self.states_count), dtype=bool)
        tc.setdiag(True)

        if not self.adj:
            return tc

        tc: csr_array = reduce(operator.add, self.adj.values(), tc)
        while True:
            tc = cast(csr_array, (prev := tc) @ tc)
            if prev.nnz == tc.nnz:
                break

        return tc

    def is_empty(self) -> bool:
        """Returns whether language recognized by automaton is empty"""
        tc = self.transitive_closure()
        return not any(
            tc[start, final]
            for start, final in itertools.product(self.start_states, self.final_states)
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    return AdjacencyMatrixFA.from_intersection(automaton1, automaton2)


@dataclass(frozen=True)
class RSMState:
    sym: Symbol
    state: State


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()  # type: ignore

    for sym, box in rsm.boxes.items():
        dfa: DeterministicFiniteAutomaton = box.dfa

        for start in dfa.start_states:
            nfa.add_start_state(State(RSMState(sym, start)))
        for final in dfa.final_states:
            nfa.add_final_state(State(RSMState(sym, final)))

        for st1, lbl, st2 in dfa._transition_function.get_edges():
            nfa.add_transition(
                State(RSMState(sym, st1)), lbl, State(RSMState(sym, st2))
            )

    return nfa
