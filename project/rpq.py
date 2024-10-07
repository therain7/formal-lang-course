import itertools

import numpy as np
from networkx import MultiDiGraph
from scipy.sparse import lil_array, vstack

from project.fa import AdjacencyMatrixFA, intersect_automata
from project.graph_tools import graph_to_nfa
from project.regex_tools import regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes else all_nodes
    final_nodes = final_nodes if final_nodes else all_nodes

    graph_mfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_dfa = regex_to_dfa(regex)
    regex_mfa = AdjacencyMatrixFA(regex_dfa)

    inter_mfa = intersect_automata(graph_mfa, regex_mfa)
    inter_tc = inter_mfa.transitive_closure()

    return {
        (start, final)
        for start, final in itertools.product(start_nodes, final_nodes)
        for regex_start, regex_final in itertools.product(
            regex_dfa.start_states, regex_dfa.final_states
        )
        if inter_tc[
            inter_mfa.states[(start, regex_start)],
            inter_mfa.states[(final, regex_final)],
        ]
    }


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes else all_nodes
    final_nodes = final_nodes if final_nodes else all_nodes

    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))

    starts = list(start_nodes)
    dfa_start = next(iter(dfa.start_states))

    front_st = []
    for st in starts:
        fr = lil_array((dfa.states_count, nfa.states_count), dtype=np.bool_)
        fr[dfa_start, nfa.states[st]] = True
        front_st.append(fr)

    front = vstack(front_st, format="lil")
    visited = front

    symbols = nfa.adj.keys() & dfa.adj.keys()
    dfa_transposed = {sym: m.transpose() for (sym, m) in dfa.adj.items()}

    while front.count_nonzero():
        sym_fronts = []
        for sym in symbols:
            new_front = front @ nfa.adj[sym]
            sym_fronts.append(
                vstack(
                    [
                        dfa_transposed[sym]
                        @ new_front[dfa.states_count * i : dfa.states_count * (i + 1)]
                        for i in range(len(starts))
                    ],
                    format="lil",
                )
            )

        front = sum(sym_fronts) > visited
        visited += front

    result = set()
    for st_idx, st in enumerate(starts):
        visited_st = visited[  # type: ignore
            dfa.states_count * st_idx : dfa.states_count * (st_idx + 1)
        ]
        for dfa_fi, fi in itertools.product(dfa.final_states, final_nodes):
            if visited_st[dfa_fi, nfa.states[fi]]:  # type: ignore
                result.add((st, fi))

    return result
