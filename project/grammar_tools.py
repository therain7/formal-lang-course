from collections import defaultdict
from typing import Iterable, cast

from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nf_cfg = cfg.to_normal_form()
    nullables = cfg.get_nullable_symbols()

    # add epsilon rules back
    new_productions = nf_cfg.productions | {
        Production(cast(Variable, var), [Epsilon()]) for var in nullables
    }

    return CFG(
        variables=nf_cfg.variables,
        terminals=nf_cfg.terminals,
        start_symbol=nf_cfg.start_symbol,
        productions=new_productions,
    )


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
