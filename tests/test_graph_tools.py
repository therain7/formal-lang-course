from pathlib import Path
from tempfile import NamedTemporaryFile

import cfpq_data
import pytest
from networkx import MultiDiGraph, nx_pydot
from networkx.utils import graphs_equal
from pyformlang.finite_automaton import State

from project.graph_tools import (
    GraphMetadata,
    build_save_2cycles_graph,
    get_graph_metadata,
    graph_to_nfa,
)

EXPECTED_PATH = Path(__file__).parent / "expected"


graph_meta_testdata = [
    ("wc", GraphMetadata(node_count=332, edge_count=269, edge_labels={"d", "a"})),
    (
        "pathways",
        GraphMetadata(
            node_count=6238,
            edge_count=12363,
            edge_labels={"subClassOf", "narrower", "imports", "type", "label"},
        ),
    ),
]


@pytest.mark.parametrize("graph_name,expected", graph_meta_testdata)
def test_get_graph_metadata(graph_name: str, expected: GraphMetadata):
    actual = get_graph_metadata(graph_name)
    assert actual == expected


two_cycles_testdata = [
    (10, "fst", 20, "snd", EXPECTED_PATH / "two_cycles1.dot"),
    (1, "c1", 41, "c2", EXPECTED_PATH / "two_cycles2.dot"),
]


@pytest.mark.parametrize(
    "nodes1,labels1,nodes2,labels2,expected_path", two_cycles_testdata
)
def test_build_save_2cycles_graph(
    nodes1: int,
    labels1: str,
    nodes2: int,
    labels2: str,
    expected_path: Path,
):
    with NamedTemporaryFile() as tmp:
        build_save_2cycles_graph(nodes1, labels1, nodes2, labels2, tmp.name)
        actual: MultiDiGraph = nx_pydot.read_dot(tmp.name)

    expected: MultiDiGraph = nx_pydot.read_dot(expected_path)

    assert graphs_equal(actual, expected)


def states_to_ints(states: set[State]):
    return set(int(st.value) for st in states)


@pytest.mark.parametrize("graph_name", ["wc", "pathways"])
def test_graph_to_nfa(graph_name: str):
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    graph_meta = get_graph_metadata(graph_name)
    nfa = graph_to_nfa(graph, set(), set())

    start_states = states_to_ints(nfa.start_states)
    final_states = states_to_ints(nfa.final_states)
    all_states = states_to_ints(nfa.states)

    assert start_states == final_states == all_states
    assert len(all_states) == graph_meta.node_count

    assert nfa.symbols == graph_meta.edge_labels


def test_2cycles_graph_to_nfa():
    nodes1, label1 = 10, "fst"
    nodes2, label2 = 20, "snd"

    with NamedTemporaryFile() as tmp:
        build_save_2cycles_graph(nodes1, label1, nodes2, label2, tmp.name)
        graph: MultiDiGraph = nx_pydot.read_dot(tmp.name)
    nfa = graph_to_nfa(graph, set(), set())

    start_states = states_to_ints(nfa.start_states)
    final_states = states_to_ints(nfa.final_states)
    all_states = states_to_ints(nfa.states)

    assert start_states == final_states == all_states
    assert len(all_states) == nodes1 + nodes2 + 1

    assert nfa.symbols == {label1, label2}
