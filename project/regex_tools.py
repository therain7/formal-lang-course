from pyformlang.finite_automaton import DeterministicFiniteAutomaton, EpsilonNFA
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    nfa = Regex(regex).to_epsilon_nfa()
    if not isinstance(nfa, EpsilonNFA):
        raise ValueError("Unexpected regex-nfa conversion error")

    dfa: DeterministicFiniteAutomaton = nfa.to_deterministic()
    return dfa.minimize()
