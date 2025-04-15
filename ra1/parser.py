# Gabriel Prost Gomes Pereira

# Gramatica original:
# FORMULA         ::= CONSTANTE
#                   | PROPOSICAO
#                   | FORMULAUNARIA
#                   | FORMULABINARIA
#
# CONSTANTE       ::= "true" | "false"
# PROPOSICAO      ::= [0–9][0–9a–z]*
#
# FORMULAUNARIA   ::= ABREPAREM OPERADORUNARIO FORMULA FECHAPAREM
# FORMULABINARIA  ::= ABREPAREM OPERADORBINARIO FORMULA FORMULA FECHAPAREM
#
# ABREPAREM       ::= "("
# FECHAPAREM      ::= ")"
#
# OPERADORUNARIO  ::= "\neg"
# OPERADORBINARIO ::= "\wedge" | "\vee" | "\rightarrow" | "\leftrightarrow"

# Gramatica implementada:
# FORMULA             ::= CONSTANTE
#                       | PROPOSICAO
#                       | FORMULAPAREM
#
# CONSTANTE           ::= "true" | "false"
# PROPOSICAO          ::= [0–9][0–9a–z]*
#
# FORMULAPAREM        ::= ABREPAREM RESTFORMULA
# RESTFORMULA         ::= RESTFORMULAUNARIA
#                       | RESTFORMULABINARIA
# RESTFORMULAUNARIA   ::= OPERADORUNARIO FORMULA FECHAPAREM
# RESTFORMULABINARIA  ::= OPERADORBINARIO FORMULA FORMULA FECHAPAREM
#
# ABREPAREM           ::= "("
# FECHAPAREM          ::= ")"
#
# OPERADORUNARIO      ::= "\neg"
# OPERADORBINARIO     ::= "\wedge" | "\vee" | "\rightarrow" | "\leftrightarrow"
from __future__ import annotations

from __future__ import annotations


from tokenizer import tokenize
from symbol import (
    FORMULA,
    CONSTANTE,
    PROPOSICAO,
    FORMULAPAREM,
    RESTFORMULA,
    RESTFORMULAUNARIA,
    RESTFORMULABINARIA,
    ABREPAREM,
    FECHAPAREM,
    OPERATORUNARIO,
    OPERATORBINARIO,
    Constante,
    EndOfStream,
    Proposicao,
    AbreParem,
    FechaParem,
    OperadorUnario,
    OperadorBinario,
    SymbolTerminal,
    SymbolNonTerminal,
    Symbol,
)

parsing_table: dict[
    tuple[type[SymbolNonTerminal], type[SymbolTerminal]], list[type[Symbol]]
] = {
    (FORMULA, Constante): [CONSTANTE],
    (FORMULA, Proposicao): [PROPOSICAO],
    (FORMULA, AbreParem): [FORMULAPAREM],
    (CONSTANTE, Constante): [Constante],
    (PROPOSICAO, Proposicao): [Proposicao],
    (FORMULAPAREM, AbreParem): [ABREPAREM, RESTFORMULA],
    (RESTFORMULA, OperadorUnario): [RESTFORMULAUNARIA],
    (RESTFORMULA, OperadorBinario): [RESTFORMULABINARIA],
    (RESTFORMULAUNARIA, OperadorUnario): [OPERATORUNARIO, FORMULA, FECHAPAREM],
    (RESTFORMULABINARIA, OperadorBinario): [
        OPERATORBINARIO,
        FORMULA,
        FORMULA,
        FECHAPAREM,
    ],
    (ABREPAREM, AbreParem): [AbreParem],
    (FECHAPAREM, FechaParem): [FechaParem],
    (OPERATORUNARIO, OperadorUnario): [OperadorUnario],
    (OPERATORBINARIO, OperadorBinario): [OperadorBinario],
}


def syntactic_analysis(tokens: list[SymbolTerminal], verbose: bool = False) -> None:
    def local_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    local_print("tokens:", ", ".join(map(str, tokens)))
    stack: list[type[Symbol]] = [EndOfStream, FORMULA]  # S$
    pos = 0
    while stack:
        local_print("\nstacks:", ", ".join(map(lambda s: s.__name__, reversed(stack))))
        svalue = stack.pop()
        token = tokens[pos]
        if issubclass(svalue, SymbolTerminal):
            if type(token) is svalue:
                pos += 1
                local_print(f"{token = },", "pop", svalue.__name__)
                if isinstance(token, EndOfStream):
                    local_print("input accepted")
            else:
                raise ValueError("bad terminal on input:", str(token))
        elif issubclass(svalue, SymbolNonTerminal):
            local_print(f"svalue = {svalue.__name__}, {token = !s}")
            rule = parsing_table[svalue, type(token)]
            local_print("rule = ", ", ".join(map(lambda s: s.__name__, rule)))
            stack.extend(reversed(rule))


def validate(code: str, verbose: bool = False) -> bool:
    try:
        tokens = list(tokenize(code))
        syntactic_analysis(tokens, verbose=verbose)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    code = r"(\wedge 5asd (\vee (\neg false) true) )"
    validate(code)
