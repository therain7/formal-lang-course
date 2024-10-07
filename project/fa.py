import itertools
import operator
from collections import defaultdict
from functools import reduce
from typing import Any, Iterable, NamedTuple, Optional, Self, cast

import numpy as np
from numpy import bool_
from numpy.typing import NDArray
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_array, kron


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
        `(st1, st2)`, where `st1` & `st2` are states' names of `fa1` & `fa2` respectively
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

            inter.states[(st1, st2)] = inter_idx

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
            self.states = {}
            self.adj = {}
            return

        graph = fa.to_networkx()
        self.states_count = graph.number_of_nodes()
        self.states: dict[Any, int] = {st: i for (i, st) in enumerate(graph.nodes)}

        for st, ddict in graph.nodes(data=True):
            if ddict.get("is_start"):
                self.start_states.add(self.states[st])
            if ddict.get("is_final"):
                self.final_states.add(self.states[st])

        transitions: dict[Symbol, NDArray[bool_]] = defaultdict(
            lambda: np.zeros((self.states_count, self.states_count), dtype=bool_)
        )
        for idx1, idx2, sym in (
            (self.states[st1], self.states[st2], Symbol(lbl))
            for st1, st2, lbl in graph.edges(data="label")
            if lbl
        ):
            transitions[sym][idx1, idx2] = True

        # convert to sparse matrices
        self.adj: dict[Symbol, csr_array] = {
            sym: csr_array(matrix) for (sym, matrix) in transitions.items()
        }

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

    def transitive_closure(self) -> NDArray[bool_]:
        """
        Returns transitive closure for automaton states.
        Get indices from `states` attribute to index the matrix
        """
        if not self.adj:
            return np.diag(np.ones(self.states_count, dtype=bool_))

        matrices = list(self.adj.values())
        sum: csr_array = reduce(operator.add, matrices[1:], matrices[0])
        sum.setdiag(True)  # make reflexive

        # compute transitive closure using matrix exponentiation
        tc = sum.toarray()
        for pow in range(2, self.states_count + 1):
            prev = tc
            tc = np.linalg.matrix_power(prev, pow)
            if np.array_equal(prev, tc):
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
