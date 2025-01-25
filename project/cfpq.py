from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Optional, cast

import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_array

from project.grammar_tools import ReversedProds, cfg_to_weak_normal_form


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes else all_nodes
    final_nodes = final_nodes if final_nodes else all_nodes

    cfg = cfg_to_weak_normal_form(cfg)
    prods = ReversedProds(cfg.productions)

    @dataclass(frozen=True)
    class HellingsEdge:
        start: int
        var: Variable
        end: int

    edges = {
        HellingsEdge(v1, var, v2)
        for (v1, v2, lbl) in graph.edges(data="label")
        if (term := Terminal(lbl)) in prods.terminals
        for var in prods.terminals[term]
    } | {
        HellingsEdge(v, cast(Variable, var), v)
        for v in all_nodes
        for var in cfg.get_nullable_symbols()
    }

    def eval_new(e1: HellingsEdge, e2: HellingsEdge) -> set[HellingsEdge]:
        if e1.end == e2.start and (body := (e1.var, e2.var)) in prods.bodies:
            return {
                e
                for var in prods.bodies[body]
                if (e := HellingsEdge(e1.start, var, e2.end)) not in edges
            }

        return set()

    queue: list[HellingsEdge] = list(edges)
    while queue:
        e1 = queue.pop(0)

        new = set()
        for e2 in edges:
            new |= eval_new(e1, e2)
            new |= eval_new(e2, e1)

        edges |= new
        queue.extend(list(new))

    return {
        (e.start, e.end)
        for e in edges
        if e.var == cfg.start_symbol and e.start in start_nodes and e.end in final_nodes
    }


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    nodes = {int(n): idx for (idx, n) in enumerate(graph.nodes)}
    start_nodes = start_nodes if start_nodes else set(nodes.keys())
    final_nodes = final_nodes if final_nodes else set(nodes.keys())

    cfg = cfg_to_weak_normal_form(cfg)
    prods = ReversedProds(cfg.productions)

    matrices: dict[Variable, csr_array] = defaultdict(
        lambda: csr_array((n := len(nodes), n), dtype=bool)
    )

    for v1, v2, lbl in graph.edges(data="label"):
        for hd in prods.terminals.get(Terminal(lbl), set()):
            matrices[hd][nodes[v1], nodes[v2]] = True

    for var in cfg.get_nullable_symbols():
        matrices[cast(Variable, var)].setdiag(True)

    updated_vars = list(cfg.variables)
    while updated_vars:
        updated = updated_vars.pop(0)

        for body, heads in prods.bodies.items():
            if updated not in body:
                continue

            delta: csr_array = matrices[body[0]] @ matrices[body[1]]
            for hd in heads:
                old = matrices[hd]
                matrices[hd] += delta

                if old.nnz < matrices[hd].nnz:
                    updated_vars.append(hd)

    if cfg.start_symbol not in matrices:
        return set()

    return {
        (start, final)
        for (start, final) in product(start_nodes, final_nodes)
        if matrices[cfg.start_symbol][nodes[start], nodes[final]]
    }
