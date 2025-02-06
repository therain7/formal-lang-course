from dataclasses import dataclass
from typing import Optional

import networkx as nx
from pyformlang.finite_automaton import State, Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.fa import RSMState


@dataclass(frozen=True)
class GSSNode:
    rsm: RSMState
    node: int


@dataclass(frozen=True)
class Conf:
    rsm: RSMState
    gss: GSSNode
    node: int


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes else all_nodes
    final_nodes = final_nodes if final_nodes else all_nodes

    gss = nx.MultiDiGraph()
    paths: set[tuple[int, int]] = set()

    def process_term(conf: Conf, rtransitions: dict[Symbol, State]) -> set[Conf]:
        return {
            Conf(
                rsm=RSMState(sym=conf.rsm.sym, state=rtransitions[term]),
                gss=conf.gss,
                node=gnext,
            )
            for _, gnext, lbl in graph.out_edges(conf.node, data="label")
            if (term := Symbol(lbl)) in rtransitions
        }

    def process_nonterm(conf: Conf, rtransitions: dict[Symbol, State]) -> set[Conf]:
        r: set[Conf] = set()

        for nonterm, rreturn_state in rtransitions.items():
            if nonterm not in rsm.labels:  # symbol is a terminal
                continue

            rnext = RSMState(sym=nonterm, state=rsm.boxes[nonterm].dfa.start_state)
            gsnext = GSSNode(
                rsm=rnext,
                node=conf.node,
            )

            gss.add_node(gsnext)  # does nothing if node already exists
            rreturn = RSMState(sym=conf.rsm.sym, state=rreturn_state)
            gss.add_edge(gsnext, conf.gss, label=rreturn)

            # if possible reuse previous calls' results
            # without making a call explicitly
            if returns := gss.nodes[gsnext].get("returns"):
                r |= {
                    Conf(rsm=rreturn, gss=conf.gss, node=greturn) for greturn in returns
                }
                continue

            r.add(Conf(rsm=rnext, gss=gsnext, node=conf.node))

        return r

    def process_return(conf: Conf) -> set[Conf]:
        if conf.rsm.state not in rsm.boxes[conf.rsm.sym].dfa.final_states:
            return set()

        # remember previous calls results
        gss.nodes[conf.gss].setdefault("returns", set()).add(conf.node)

        if conf.gss.rsm.sym == rsm.initial_label:
            paths.add((conf.gss.node, conf.node))

        return {
            Conf(rsm=rreturn, gss=gsreturn, node=conf.node)
            for _, gsreturn, rreturn in gss.out_edges(conf.gss, data="label")
        }

    processed: set[Conf] = set()
    pending: set[Conf] = set()

    for start in start_nodes:
        rstate = RSMState(
            sym=rsm.initial_label,
            state=rsm.boxes[rsm.initial_label].dfa.start_state,
        )
        gsnode = GSSNode(
            rsm=rstate,
            node=start,
        )

        gss.add_node(gsnode)
        pending.add(Conf(rsm=rstate, gss=gsnode, node=start))

    while pending:
        conf = pending.pop()
        processed.add(conf)

        rtransitions = rsm.boxes[
            conf.rsm.sym
        ].dfa._transition_function._transitions.setdefault(conf.rsm.state, {})

        new = process_term(conf, rtransitions)
        new |= process_nonterm(conf, rtransitions)
        new |= process_return(conf)

        pending |= new - processed

    return {(v1, v2) for v1, v2 in paths if v1 in start_nodes and v2 in final_nodes}
