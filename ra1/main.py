# Gabriel Prost Gomes Pereira

import sys

from parser import validate


def main(filename: str):
    with open(filename, "r") as f:
        _count = f.readline()
        lines = f.readlines()

    for line in lines:
        if validate(line):
            print("válido")
        else:
            print("inválido")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print("USE: python main.py <FILE>")
        filename = "exemplo1.txt"
    else:
        filename = sys.argv[1]
    main(filename)
