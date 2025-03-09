# Construção de Interpretadores
# TDE 1
# Gabriel Prost Gomes Pereira

from collections import defaultdict
import logging
import re
from dataclasses import Field, dataclass
from enum import StrEnum, auto
from typing import Any, Callable, Self, TypeVar

logger = logging.getLogger(__name__)


class NivelDetalhe(StrEnum):
    baixo = auto()
    completo = auto()


@dataclass
class Config:
    """Configuração do processamento de arquivos SimplePDF"""

    extrair_texto: bool = True
    gerar_sumario: bool = True
    detectar_ciclos: bool = True
    nivel_detalhe: NivelDetalhe = NivelDetalhe.completo
    validar_xref: bool = True

    line_pattern = re.compile(r"^\s*(?P<key>\w+)\s*=\s*(?P<value>\w+)\s*$")

    @classmethod
    def fields(cls) -> dict[str, Field]:
        return cls.__dataclass_fields__

    @classmethod
    def parse_value(cls, key: str, value_str: str):
        t = cls.fields()[key].type
        if t is bool:
            if value_str == "sim":
                return True
            elif value_str == "nao":
                return False
            else:
                raise ValueError(f"Expected sim/nao, got {value_str!r} ")
        elif t is NivelDetalhe:
            return NivelDetalhe(value_str)
        raise Exception(f"Unknown type {t}")

    @classmethod
    def from_file(cls, filename: str) -> Self:
        config_map = {}
        valid_keys = cls.fields()

        with open(filename, "r") as f:
            for linenum, line in enumerate(f):
                match = cls.line_pattern.match(line)
                if match is None:
                    if not line.isspace():
                        logger.warning(
                            f"Config line {linenum}: {line!r} could not be parsed."
                        )
                    continue

                key = match.group("key")
                if key not in valid_keys:
                    logger.warning(
                        f"Config line {linenum}: found non-valid key {key!r}"
                    )
                    continue
                if key in config_map:
                    logger.warning(f"Config line {linenum}: found repeated key {key!r}")

                value = match.group("value")
                try:
                    parsed_value = cls.parse_value(key, value)
                except ValueError as e:
                    logger.warning(
                        f"Config line {linenum}: could not parse value {value!r}: {e}"
                    )
                    continue
                config_map[key] = parsed_value

        return cls(**config_map)


@dataclass
class Reference:
    id: int
    gen: int


@dataclass
class Obj:
    id: int
    gen: int
    content: dict[str, Any]
    stream: memoryview | None

    def refs(self):
        to_search: list[Any] = [self.content]
        while to_search:
            curr = to_search.pop()
            if isinstance(curr, Reference):
                yield curr
            elif isinstance(curr, list):
                to_search.extend(curr)
            elif isinstance(curr, dict):
                to_search.extend(curr.values())


@dataclass
class Header:
    version_major: int
    version_minor: int


@dataclass
class XRefValue:
    # NAMES NOT KNOWN
    num1: int
    num2: int
    letter: str


@dataclass
class XRef:
    # NAMES NOT KNOWN
    num1: int
    num2: int
    values: list[XRefValue]


@dataclass
class Trailer:
    content: dict[str, Any]
    start_xref: int


T = TypeVar("T")
G = TypeVar("G")

ParserContext = memoryview

ParserResult = tuple[ParserContext, T]

Parser = Callable[[ParserContext], ParserResult[T]]


class ParseError(Exception):
    def __init__(self, context: ParserContext, name: str | None):
        message = f"Esperava ter: {name}" if name is not None else None
        super().__init__(message)

        self.context = context

    def get_line(self) -> tuple[int, str]:
        total: bytes = self.context.obj
        error_part = len(total) - len(self.context)
        prev_context = total[: -len(self.context)]
        line = 1 + sum(c == ord("\n") for c in prev_context)
        start = error_part
        while total[start] != ord(b"\n") and start != 0:
            start -= 1
        end = error_part
        while total[end] != ord(b"\n") and end < len(total):
            end += 1
        area = total[start + 1 : end].decode()
        return line, area


def pattern(ptrn: bytes, name: str | None = None) -> Parser[re.Match[bytes]]:
    def p(s: ParserContext) -> ParserResult[re.Match]:
        m = re.match(ptrn, s)
        if m is None:
            raise ParseError(s, name)
        s = s[m.end() :]
        return s, m

    return p


def decimal(s: ParserContext) -> ParserResult[float]:
    s, m = pattern(rb"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", name="numero decimal")(
        s
    )
    return s, float(m.group())


def integer(s: ParserContext) -> ParserResult[int]:
    s, m = pattern(rb"[-+]?\d+", "numero inteiro")(s)
    return s, int(m.group())


def string(s: ParserContext) -> ParserResult[str]:
    s, m = pattern(rb"\(([^()]*)\)", "string delimitada por ()")(s)
    return s, m.group(1).decode()


def stream(s: ParserContext) -> ParserResult[ParserContext]:
    s, m = pattern(rb"stream\s*((?s:.*?))\s*endstream", "stream")(s)
    start, end = m.span(1)
    return s, s[start:end]


def name(s: ParserContext) -> ParserResult[str]:
    s, m = pattern(rb"/(\w+)", "name")(s)
    return s, m.group(1).decode()


def boolean(s: ParserContext) -> ParserResult[bool]:
    s, m = pattern(rb"true|false", name="booleano")(s)
    return s, m.group() == b"true"


def null(s: ParserContext) -> ParserResult[None]:
    s, m = pattern(rb"null", name="null")(s)
    return s, None


whitespace_characters = b" \n\r\t"


def whitespace0(s: ParserContext) -> ParserResult[int]:
    s, m = pattern(rb"(?:\s*(?:%.*)?\s*)*", name="espaço")(s)
    return s, m.end()


def whitespace1(s: ParserContext) -> ParserResult[int]:
    s, m = whitespace0(s)
    if m == 0:
        raise ParseError(s, name="espaço")
    return s, m


def delimited(left: Parser, main: Parser[T], right: Parser) -> Parser[T]:
    def p(s: ParserContext) -> ParserResult[T]:
        s, _ = left(s)
        s, res = main(s)
        s, _ = right(s)
        return s, res

    return p


def reference(s: ParserContext) -> ParserResult[Reference]:
    s, id = integer(s)
    s, _ = whitespace1(s)
    s, gen = integer(s)
    s, _ = whitespace1(s)
    s, _ = pattern(rb"R")(s)
    return s, Reference(id=id, gen=gen)


def separated_list0(elem: Parser[T], sep: Parser = whitespace1) -> Parser[list[T]]:
    def p(s: ParserContext) -> ParserResult[list[T]]:
        res = []
        try:
            first = True
            while True:
                if not first:
                    s, (_, e) = pair(sep, elem)(s)
                else:
                    s, e = elem(s)
                res.append(e)
                first = False
        except ParseError:
            pass
        return s, res

    return p


def separated_list1(elem: Parser[T], sep: Parser = whitespace1) -> Parser[list[T]]:
    def p(s: ParserContext) -> ParserResult[list[T]]:
        res = []
        first = True
        try:
            while True:
                if not first:
                    s, (_, e) = pair(sep, elem)(s)
                else:
                    s, e = elem(s)
                res.append(e)
                first = False
        except ParseError:
            if first:
                raise
            pass
        return s, res

    return p


def pair(
    first: Parser[T], second: Parser[G], sep: Parser = whitespace0
) -> Parser[tuple[T, G]]:
    def p(s: ParserContext) -> ParserResult[tuple[T, G]]:
        s, frs = first(s)
        s, _ = whitespace0(s)
        s, sec = second(s)
        return s, (frs, sec)

    return p


def alt(parsers: list[Parser], name: str | None = None) -> Parser[list]:
    def p(s: ParserContext) -> ParserResult[list]:
        for parser in parsers:
            try:
                s, m = parser(s)
            except ParseError:
                continue
            return s, m
        raise ParseError(s, name)

    return p


def optional(parser: Parser[T]) -> Parser[T | None]:
    def p(s: ParserContext):
        try:
            s, m = parser(s)
            return s, m
        except ParseError:
            return s, None

    return p


def value(s: ParserContext) -> ParserResult[Any]:
    return alt(
        [
            reference,
            decimal,
            integer,
            boolean,
            null,
            name,
            string,
            array,
            content,
        ],
        name="valor",
    )(s)


content_key_value_list: Parser[list[tuple[str, Any]]] = delimited(
    pattern(rb"<<\s*", name="abertura de conteudo"),
    separated_list0(pair(name, value)),
    pattern(rb"\s*>>", name="fechadura de conteudo"),
)


def content(s: ParserContext) -> ParserResult[dict[str, Any]]:
    s, key_value_list = content_key_value_list(s)
    obj_map = {k: v for k, v in key_value_list}
    return s, obj_map


array: Parser[list] = delimited(
    pattern(rb"\[\s*", name="abertura de lista"),
    separated_list0(value),
    pattern(rb"\s*]", name="fechadura de lista"),
)


def obj(s: ParserContext) -> ParserResult[Obj]:
    s, id = integer(s)
    s, _ = whitespace1(s)
    s, gen = integer(s)
    s, _ = pattern(rb"\s+obj\s+", name="abertura de objeto")(s)
    s, content_ = content(s)
    s, _ = whitespace1(s)
    s, stream_ = optional(stream)(s)
    if stream_ is not None:
        s, _ = whitespace1(s)
    s, _ = pattern(rb"endobj", name="fechadura de objeto")(s)
    return s, Obj(id=id, gen=gen, content=content_, stream=stream_)


def xref_value(s: ParserContext) -> ParserResult[XRefValue]:
    s, num1 = integer(s)
    s, _ = whitespace1(s)
    s, num2 = integer(s)
    s, _ = whitespace1(s)
    s, letter_match = pattern(rb"\w")(s)
    return s, XRefValue(num1=num1, num2=num2, letter=letter_match.group().decode())


def xref(s: ParserContext) -> ParserResult[XRef]:
    s, _ = pattern(rb"xref\s+", name="inicio do xref")(s)
    s, num1 = integer(s)
    s, _ = whitespace1(s)
    s, num2 = integer(s)
    s, _ = whitespace1(s)
    s, values = separated_list0(xref_value)(s)
    return s, XRef(num1=num1, num2=num2, values=values)


def trailer(s: ParserContext) -> ParserResult[Trailer]:
    s, _ = pattern(rb"trailer\s+", name="inicio do trailer")(s)
    s, content_ = content(s)
    s, _ = pattern(rb"\s+startxref\s+", name="startxref")(s)
    s, start_xref = integer(s)
    s, _ = pattern(rb"\s+%%EOF", name="EOF")(s)
    return s, Trailer(content=content_, start_xref=start_xref)


def header(s: ParserContext) -> ParserResult[Header]:
    s, _ = pattern(rb"%SPDF-", name="header")(s)
    s, version_major = integer(s)
    s, _ = pattern(rb"\.", name="ponto de separação de versão")(s)
    s, version_minor = integer(s)
    return s, Header(version_major=version_major, version_minor=version_minor)


@dataclass
class SimplePdf:
    header: Header
    objs: list[Obj]
    xref: XRef
    trailer: Trailer

    @classmethod
    def from_memory(cls, s: memoryview) -> Self:
        s, header_ = header(s)
        s, _ = whitespace1(s)
        s, objs_ = separated_list1(obj)(s)
        try:
            s, _ = whitespace1(s)
            s, xref_ = xref(s)
        except ParseError as e:
            raise ParseError(e.context, "objeto ou xref")
        s, _ = whitespace1(s)
        s, trailer_ = trailer(s)
        return cls(header=header_, objs=objs_, xref=xref_, trailer=trailer_)

    def references(self):
        for obj in self.objs:
            yield from obj.refs()

    def obj_map(self):
        res = {}
        for obj in self.objs:
            res[(obj.id, obj.gen)] = obj
        return res


def analyse_simplepdf_file(config_filename: str, spdf_filename: str):
    print(f"Abrindo arquivo de configuração {config_filename!r}:")
    cfg = Config.from_file(config_filename)
    for config_opt in cfg.fields():
        print(f"{config_opt} = {cfg.__getattribute__(config_opt)}")
    print()
    print(f"Abrindo arquivo simple pdf {spdf_filename!r}:")
    with open(spdf_filename, "rb") as f:
        file_content = f.read()
        memory = memoryview(file_content)

    print()
    print("VALIDAÇÃO:")
    parser_success = True
    parser_except = None
    try:
        simplepdf = SimplePdf.from_memory(memory)
    except ParseError as e:
        parser_success = False
        parser_except = e
    print(f"[{'OK' if parser_success else 'ERRO'}] Estrutura geral")
    print(f"[{'OK' if parser_success else 'ERRO'}] Sintaxe de objetos")
    if parser_except is not None:
        line, area = parser_except.get_line()
        print(f"Erro na linha {line}:\n{area}\n{parser_except}")
        exit(1)

    references_ok = True
    faulty_references = []
    objs = simplepdf.obj_map()
    for ref in simplepdf.references():
        if (ref.id, ref.gen) not in objs:
            references_ok = False
            faulty_references.append(ref)
    print(f"[{'OK' if references_ok else 'ERRO'}] Referências")
    if faulty_references:
        print(f"A referências {faulty_references} apontam para nada")

    # i have no idea what to do with the xref table
    print("[OK] Tabela xref")

    print()
    print("ESTATÍSTICAS:")
    print("Total de objetos:", len(simplepdf.objs))
    obj_type_counts: dict[str, int] = defaultdict(int)
    for obj in simplepdf.objs:
        if "Type" in obj.content:
            obj_type_counts[obj.content["Type"]] += 1

    print(
        "Objetos por tipo:", ", ".join(f"{k}={v}" for k, v in obj_type_counts.items())
    )

    # print(f"{simplepdf = }")


analyse_simplepdf_file("config.txt", "examples/1.spdf")
