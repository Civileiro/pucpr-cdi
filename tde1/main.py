# Construção de Interpretadores
# TDE 1
# Gabriel Prost Gomes Pereira

from collections import defaultdict
import logging
import re
from dataclasses import Field, dataclass
from typing import Any, Callable, Self, TypeVar
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuração do processamento de arquivos SimplePDF"""

    extrair_texto: bool = True
    gerar_sumario: bool = True
    listar_objetos: bool = True
    coordenadas_legiveis: bool = True

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


@dataclass(frozen=True)
class Reference:
    id: int
    gen: int


@dataclass
class PageStream:
    @dataclass
    class TokenFont:
        name: str
        size: int

    @dataclass
    class TokenCoord:
        x: int
        y: int

        def __repr__(self) -> str:
            return f"Posição: {self.x}px da esquerda, {self.y}px do topo da página:"

    @dataclass
    class TokenText:
        string: str

    type PageStreamToken = TokenFont | TokenText | TokenCoord
    tokens: list[PageStreamToken]


type Stream = PageStream | memoryview


@dataclass
class Obj:
    id: int
    gen: int
    content: dict[str, Any]
    stream: Stream | None
    is_root: bool

    def ref(self) -> Reference:
        return Reference(self.id, self.gen)

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

    def type(self) -> str:
        type_key = "Type"
        if type_key in self.content:
            obj_type = self.content[type_key]
        elif self.stream is not None:
            obj_type = "Content"
        elif "Author" in self.content:
            obj_type = "Metadata"
        else:
            obj_type = "Outline"
        if self.is_root:
            obj_type += " (Root)"

        return obj_type

    def __getitem__(self, index):
        return self.content.__getitem__(index)


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


def integer_or_decimal(s: ParserContext) -> ParserResult[int | float]:
    try:
        s_i, i = integer(s)
        int_success = True
    except ParseError:
        int_success = False
    try:
        s_d, d = decimal(s)
        dec_success = True
    except ParseError:
        dec_success = False
    match (int_success, dec_success):
        case (False, False):
            raise ParseError(s, name="numero inteiro ou decimal")
        case (True, False):
            return s_i, i
        case (False, True):
            return s_d, d
        case _:
            if len(s_i) == len(s_d):
                return s_i, i
            else:
                return s_d, d


def string(s: ParserContext) -> ParserResult[str]:
    s, m = pattern(rb"\(([^()]*)\)", "string delimitada por ()")(s)
    return s, m.group(1).decode()


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
    s, m = pattern(rb"(?:\s*(?:%.*)?)*", name="espaço")(s)
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


def alt(parsers: list[Parser], name: str | None = None) -> Parser:
    def p(s: ParserContext) -> ParserResult:
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
            integer_or_decimal,
            # decimal,
            # integer,
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


def stream_token_text(s: ParserContext) -> ParserResult[PageStream.TokenText]:
    s, text = string(s)
    s, _ = whitespace1(s)
    s, _ = pattern(rb"Tj")(s)
    return s, PageStream.TokenText(text)


def stream_token_font(s: ParserContext) -> ParserResult[PageStream.TokenFont]:
    s, font_name = name(s)
    s, _ = whitespace1(s)
    s, size = integer(s)
    s, _ = whitespace1(s)
    s, _ = pattern(rb"Tf")(s)
    return s, PageStream.TokenFont(font_name, size)


def stream_token_coord(s: ParserContext) -> ParserResult[PageStream.TokenCoord]:
    s, x = integer(s)
    s, _ = whitespace1(s)
    s, y = integer(s)
    s, _ = whitespace1(s)
    s, _ = pattern(rb"Td")(s)
    return s, PageStream.TokenCoord(x, y)


stream_token = alt([stream_token_text, stream_token_coord, stream_token_font])


def page_stream(s: ParserContext) -> ParserResult[PageStream]:
    s, _ = pattern(rb"BT\s+")(s)
    s, tokens = separated_list1(stream_token)(s)
    s, _ = pattern(rb"\s+ET")(s)
    return s, PageStream(tokens)


def stream(s: ParserContext) -> ParserResult[Stream]:
    s_res, m = pattern(rb"stream\s*((?s:.*?))\s*endstream", "stream")(s)
    start, end = m.span(1)
    stream_content = s[start:end]
    try:
        _, page = page_stream(stream_content)
        return s_res, page
    except ParseError:
        return s_res, stream_content


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
    return s, Obj(id=id, gen=gen, content=content_, stream=stream_, is_root=False)


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


class StructureError(Exception): ...


@dataclass
class SimplePdf:
    header: Header
    _root: Obj
    objs: dict[Reference, Obj]
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

        if "Root" not in trailer_.content:
            raise StructureError("Trailer não possui Root")
        objs = {Reference(obj.id, obj.gen): obj for obj in objs_}
        if trailer_.content["Root"] not in objs:
            raise StructureError("Root aponta para objeto que não existe")
        root_ = objs[trailer_.content["Root"]]
        root_.is_root = True

        return cls(header=header_, _root=root_, objs=objs, xref=xref_, trailer=trailer_)

    def _references(self):
        for obj in self.objs.values():
            yield from obj.refs()

    def dangling_references(self) -> list[Reference]:
        faulty_references = []
        for ref in self._references():
            if ref not in self.objs:
                faulty_references.append(ref)
        return faulty_references

    def pages(self) -> Obj:
        return self.objs[self._root["Pages"]]

    def page_count(self) -> int:
        return self.pages()["Count"]

    def page_objs(self):
        return (self.objs[ref] for ref in self.pages()["Kids"])

    def page_content_objs(self):
        return (self.objs[page["Contents"]] for page in self.page_objs())

    def page_streams(self):
        return (
            content.stream
            for content in self.page_content_objs()
            if isinstance(content.stream, PageStream)
        )

    def outlines(self) -> Obj:
        return self.objs[self._root["Outlines"]]

    def outline_objs(self):
        curr = self.objs[self.outlines()["First"]]
        while True:
            yield curr
            if "Next" not in curr.content:
                break
            curr = self.objs[curr.content["Next"]]

    def metadata(self) -> Obj:
        return self.objs[self._root["Metadata"]]

    def title(self) -> str:
        return self.metadata()["Title"]

    def author(self) -> str:
        return self.metadata()["Author"]

    def creation_date(self) -> datetime:
        date_str: str = self.metadata()["CreationDate"]
        return datetime.strptime(date_str, "D:%Y%m%d%H%M%S")

    def check_structure(
        self, obj: Obj | None = None, structure: dict[str, dict] | None = None
    ):
        if obj is None:
            obj = self._root
        if structure is None:
            structure = {
                "Pages": {"Page": {"Content": {}}},
                "Metadata": {},
                "Outlines": {"Outline": {}},
            }
        childs = [self.objs[ref] for ref in obj.refs()]
        errors: list[StructureError] = []
        for member in structure:
            member_child = next((obj for obj in childs if obj.type() == member), None)
            if member_child is None:
                raise StructureError(
                    f"Objeto de tipo {obj.type()} não possui filho de tipo {member}"
                )
            else:
                try:
                    self.check_structure(member_child, structure[member])
                except StructureError as e:
                    errors.append(e)

        if errors:
            raise StructureError("\n".join(str(res) for res in errors))


def print_obj_tree(
    objs: dict[Reference, Obj], obj: Obj, n0=1, indent=0, seen=None
) -> int:
    if seen is None:
        seen = set()
    if indent > 0:
        dent = " " * (2 * indent - 1) + "+- "
    else:
        dent = ""
    obj_name = obj.type()
    print(f"{dent}{obj.id}: {obj_name}", sep="")
    seen.add(obj.ref())
    to_visit = [ref for ref in obj.refs() if ref not in seen]
    seen.update(to_visit)
    for ref in reversed(to_visit):
        n0 = (
            print_obj_tree(objs, objs[ref], n0=n0 + 1, indent=indent + 1, seen=seen) + 1
        )
    return n0


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
    try:
        simplepdf = SimplePdf.from_memory(memory)
        simplepdf.check_structure()
    except ParseError as e:
        print("[ERRO] Sintaxe de objetos")
        line, area = e.get_line()
        print(f"Erro na linha {line}:\n{area}\n{e}")
        exit(1)
    except StructureError as e:
        print("[OK] Sintaxe de objetos")
        print("[ERRO] Estrutura Geral")
        print(e)
        exit(1)
    print("[OK] Sintaxe de objetos")
    print("[OK] Estrutura Geral")

    dangling_references = simplepdf.dangling_references()
    print(f"[{'OK' if len(dangling_references) == 0 else 'ERRO'}] Referências")
    if dangling_references:
        print(f"A referências {dangling_references} apontam para nada")

    # i have no idea what to do with the xref table
    # print("[OK] Tabela xref")

    print()
    print("ESTATÍSTICAS:")
    print("Total de objetos:", len(simplepdf.objs))
    obj_type_counts: dict[str, int] = defaultdict(int)
    for obj in simplepdf.objs.values():
        if "Type" in obj.content:
            obj_type_counts[obj.content["Type"]] += 1

    print(
        "Objetos por tipo:", ", ".join(f"{k}={v}" for k, v in obj_type_counts.items())
    )
    print("Total de páginas:", simplepdf.page_count())
    print("Tamanho do documento:", len(file_content), "bytes")
    print()
    print("CONTEÚDO:")
    print("Título:", simplepdf.title())
    print("Autor:", simplepdf.author())
    print("Data de criação:", simplepdf.creation_date())
    if cfg.gerar_sumario:
        print("Sumário:")
        for outline in simplepdf.outline_objs():
            print(f"\t{outline.content["Title"]}")
    if cfg.extrair_texto:
        print("Texto extraído:")
        for i, p_stream in enumerate(simplepdf.page_streams()):
            print(f"Página {i + 1}:")
            last_coord: PageStream.TokenCoord | None = None
            for token in p_stream.tokens:
                if isinstance(token, PageStream.TokenText):
                    if cfg.coordenadas_legiveis and last_coord is not None:
                        print(f"\t{last_coord}")
                    print(f"\t{token.string!r}")
                if isinstance(token, PageStream.TokenCoord):
                    last_coord = token
    print()
    if cfg.listar_objetos:
        print("ÁRVORE DE OBJETOS:")
        print_obj_tree(simplepdf.objs, simplepdf._root)

    # print(f"{simplepdf = }")


analyse_simplepdf_file("config.txt", "examples/1.spdf")
