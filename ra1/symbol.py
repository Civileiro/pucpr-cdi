# Gabriel Prost Gomes Pereira

from __future__ import annotations

import re
from typing import Self


class SymbolBase:
    def same_type_as(self, other: SymbolBase) -> bool:
        return type(self) is type(other)

    def __repr__(self) -> str:
        return self.__class__.__name__


class SymbolNonTerminal(SymbolBase): ...


class FORMULA(SymbolNonTerminal): ...


class CONSTANTE(SymbolNonTerminal): ...


class PROPOSICAO(SymbolNonTerminal): ...


class FORMULAPAREM(SymbolNonTerminal): ...


class RESTFORMULA(SymbolNonTerminal): ...


class RESTFORMULAUNARIA(SymbolNonTerminal): ...


class RESTFORMULABINARIA(SymbolNonTerminal): ...


class ABREPAREM(SymbolNonTerminal): ...


class FECHAPAREM(SymbolNonTerminal): ...


class OPERATORUNARIO(SymbolNonTerminal): ...


class OPERATORBINARIO(SymbolNonTerminal): ...


class SymbolTerminal(SymbolBase):
    pattern: str
    _skip: bool = False
    _regex: re.Pattern | None = None

    def __init__(self, content: str):
        self.content = content

    @classmethod
    def regex(cls) -> re.Pattern:
        if cls._regex is None:
            cls._regex = re.compile(rf"(?P<main>{cls.pattern})")
        return cls._regex

    @classmethod
    def try_consume(cls, s: str, pos: int) -> Self | None:
        match = cls.regex().match(s, pos=pos)
        if match is None:
            return None
        return cls(content=match.group("main"))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r})"


type Token = SymbolTerminal

type Symbol = SymbolTerminal | SymbolNonTerminal

token_name_type: dict[str, type[SymbolTerminal]] = {}


def add_token(cls: type[SymbolTerminal]) -> type[SymbolTerminal]:
    token_name_type[cls.__name__] = cls
    return cls


@add_token
class Constante(SymbolTerminal):
    pattern = r"true|false"


@add_token
class OperadorUnario(SymbolTerminal):
    pattern = r"\\neg"


@add_token
class OperadorBinario(SymbolTerminal):
    pattern = r"\\(wedge|vee|rightarrow|leftrightarrow)"


@add_token
class Proposicao(SymbolTerminal):
    pattern = r"[0-9][0-9a-z]*"


@add_token
class AbreParem(SymbolTerminal):
    pattern = r"\("


@add_token
class FechaParem(SymbolTerminal):
    pattern = r"\)"


@add_token
class Espaco(SymbolTerminal):
    pattern = r"\s+"
    _skip = True


class EndOfStream(SymbolTerminal):
    pattern = r"$"

    def __init__(self):
        super().__init__(content="$")
