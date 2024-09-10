from dataclasses import dataclass

import cfpq_data
from networkx import MultiDiGraph, nx_pydot
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
)


@dataclass
class GraphMetadata:
    node_count: int
    edge_count: int
    edge_labels: set


def get_graph_metadata(graph_name: str) -> GraphMetadata:
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    return GraphMetadata(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        edge_labels={lbl for _, _, lbl in graph.edges(data="label")},
    )


def build_save_2cycles_graph(
    cycle1_nodes: int,
    cycle1_labels: str,
    cycle2_nodes: int,
    cycle2_labels: str,
    output_path: str,
) -> None:
    """Constructs graph with two cycles and saves it in a DOT file"""
    graph = cfpq_data.labeled_two_cycles_graph(
        n=cycle1_nodes,
        m=cycle2_nodes,
        labels=(cycle1_labels, cycle2_labels),
    )
    nx_pydot.to_pydot(graph).write_raw(output_path)


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    nfa: NondeterministicFiniteAutomaton = (
        NondeterministicFiniteAutomaton.from_networkx(
            graph
        ).remove_epsilon_transitions()
    )

    all_nodes = set(int(n) for n in graph.nodes)
    actual_starts = start_states if start_states else all_nodes
    actual_finals = final_states if final_states else all_nodes

    for st in actual_starts:
        nfa.add_start_state(State(st))

    for st in actual_finals:
        nfa.add_final_state(State(st))

    return nfa
