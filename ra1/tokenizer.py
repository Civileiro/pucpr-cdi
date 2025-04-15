# Gabriel Prost Gomes Pereira

from __future__ import annotations
from typing import Iterator
import re

from symbol import EndOfStream, Token, token_name_type


token_regex = re.compile(
    "|".join(
        rf"(?P<{token_type.__name__}>{token_type.pattern})"
        for token_type in token_name_type.values()
    )
)


def next_token(code: str, pos: int) -> Token | None:
    match = token_regex.match(code, pos=pos)
    if match is None:
        return None
    token_type = next(
        token_type
        for token_name, token_type in token_name_type.items()
        if match.group(token_name) is not None
    )
    return token_type(content=match.group())


def tokenize(code: str) -> Iterator[Token]:
    pos = 0
    while pos < len(code):
        token = next_token(code, pos)
        if token is None:
            raise Exception("couldnt tokenize")
        pos += len(token.content)
        if token._skip:
            continue
        yield token
    yield EndOfStream()
