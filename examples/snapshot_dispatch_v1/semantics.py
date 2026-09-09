"""Fresh finite fixtures shared by capture and frozen-source reconstruction."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from fractions import Fraction
from pathlib import PurePosixPath

from examples.publication_snapshot_v1.common import pack


@dataclass
class Record:
    value: int
    hidden: int = field(compare=False)


class Number(IntEnum):
    ONE = 1


class Text(str, Enum):
    WORD = "word"


class Integer(int):
    pass


class String(str):
    pass


class Blob(bytes):
    pass


class Real(float):
    pass


class Ratio(Fraction):
    pass


class PathValue(PurePosixPath):
    pass


class Opaque:
    pass


def primitive_input(spec, frozen):
    p = frozen["primitive_vector"]
    vector = (p["string"], p["integer"], p["boolean"], bytes.fromhex(p["bytes_hex"]),
              p["none"], float.fromhex(p["float_hex"]))
    return [vector * spec["width"] for _ in range(spec["history"])]


def observe(name, snapshot):
    """Return complete encoded outputs or ordered failures, with actual hook traces."""
    trace, calls, witnesses = [], [], []

    def take(value, project=None, opaque=None):
        try:
            result = snapshot(value, project=project)
        except Exception as error:  # noqa: BLE001
            calls.append({"error": {"type": type(error).__name__, "message": str(error)}})
            return None
        if opaque is not None:
            witnesses.append(result[2] is opaque)
            # This declared leaf is opaque. Never invoke its spoofable equality.
            calls.append({"opaque": {"tag": list(result[:2]), "same_object": result[2] is opaque}})
        else:
            calls.append({"snapshot": pack(result)})
        return result

    if name == "primitives":
        take(("λ:字", 2**130 + 7, True, False, 0, 1, b"\x00\xff\x80", None, 0.0, -0.0, 1.5))
    elif name.startswith("nonfinite-") and name in ("nonfinite-nan", "nonfinite-positive", "nonfinite-negative"):
        take(float({"nonfinite-nan": "nan", "nonfinite-positive": "inf", "nonfinite-negative": "-inf"}[name]))
    elif name == "primitive-subclasses":
        for value in (Integer(1), String("s"), Blob(b"b"), Real(1.25)):
            take(value)
    elif name == "enum-values":
        take((Number.ONE, Text.WORD))
    elif name == "fraction-path-subclasses":
        take((Fraction(2, 6), Ratio(-2, 7), PurePosixPath("a/b"), PathValue("a/c")))
    elif name == "dataclass-fields":
        value = Record(1, 2)
        take(value)
        value.hidden = 3
        take(value)
    elif name == "dataclass-class":
        take(Record)
    elif name == "metadata-change":
        @dataclass
        class Dynamic:
            value: int
        value = Dynamic(3)
        take(value)
        Dynamic.__module__, Dynamic.__qualname__ = "changed_module", "ChangedName"
        take(value)
    elif name == "metaclass-spoof":
        class Spoof(type):
            def __eq__(self, other):
                trace.append(["eq", other.__name__])
                return other is int

            def __hash__(self):
                trace.append(["hash"])
                return 1

        class Pretender(metaclass=Spoof):
            pass
        value = Pretender()
        take(value, opaque=value)
    elif name == "alias-fresh-tuples":
        alias = [None, "s"]
        result = take([alias, alias, None, None])
        if result is not None:
            rows = result[2]
            witnesses.extend((rows[0] is not rows[1], rows[2] is not rows[3],
                              rows[0][2][0] is not rows[1][2][0]))
    elif name == "ordered-dictionary":
        a, b = Opaque(), Opaque()
        labels = {id(a): "key", id(b): "value"}

        def project(item):
            trace.append(labels[id(item)])
            return labels[id(item)]
        take({a: b, "tail": 7}, project)
    elif name == "unordered-containers":
        take(({1, 2, 3}, frozenset(("a", "b"))))
    elif name == "list-cycle":
        value = []
        value.append(value)
        take(value)
    elif name == "dict-cycle":
        value = {}
        value["self"] = value
        take(value)
    elif name.startswith("projection-"):
        value = Opaque()

        def project(item):
            trace.append(type(item).__name__)
            if name == "projection-success":
                return 7
            if name == "projection-decline":
                return NotImplemented
            if name == "projection-error":
                raise LookupError("declared projection failure")
            if name == "projection-nested":
                return {"result": (3, -0.0, None)}
            if name == "projection-cycle":
                return [item]
            raise ValueError("unknown projection fixture")
        take(value, project)
    elif name == "nonfinite-before-unsupported":
        take([float("nan"), object()])
    elif name == "unsupported-before-nonfinite":
        take([object(), float("nan")])
    elif name == "fraction-before":
        Fraction.register(list)
        take([1])
    elif name == "fraction-during":
        def project(item):
            trace.append("register-list")
            Fraction.register(list)
            return [1]
        take(Opaque(), project)
    else:
        raise ValueError("unknown semantic fixture: " + name)
    return {"name": name, "calls": calls, "trace": trace, "identity_witnesses": witnesses}
