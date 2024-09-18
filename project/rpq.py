import itertools

from networkx import MultiDiGraph

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
