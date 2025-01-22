from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, cast

import networkx as nx
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.grammar_tools import cfg_to_weak_normal_form


class ReversedProds:
    def __init__(self, productions: Iterable[Production]):
        self.terminals: dict[Terminal, set[Variable]] = defaultdict(set)
        self.bodies: dict[tuple[Variable, Variable], set[Variable]] = defaultdict(set)

        for prod in productions:
            if len(prod.body) == 1 and isinstance(term := prod.body[0], Terminal):
                self.terminals[term].add(prod.head)
                continue

            if (
                len(prod.body) == 2
                and isinstance(var1 := prod.body[0], Variable)
                and isinstance(var2 := prod.body[1], Variable)
            ):
                self.bodies[(var1, var2)].add(prod.head)


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
        new = set()

        if e1.end == e2.start and (body := (e1.var, e2.var)) in prods.bodies:
            new |= {
                e
                for var in prods.bodies[body]
                if (e := HellingsEdge(e1.start, var, e2.end)) not in edges
            }

        return new

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
