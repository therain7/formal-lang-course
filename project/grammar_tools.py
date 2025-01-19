from typing import cast

from pyformlang.cfg import CFG, Epsilon, Production, Variable


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
