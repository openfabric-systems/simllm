"""Admit complete captures, reconstruct inputs and separate count evidence."""

from copy import deepcopy
from pathlib import Path

from examples.publication_snapshot_v1 import checks as legacy
from examples.publication_snapshot_v1 import worker as legacy_worker
from examples.publication_snapshot_v1.common import (
    original_nodes,
    pack,
    profile_rows,
    sha,
    snapshot_functions,
    unpack,
)
from examples.snapshot_dispatch_v1.common import (
    binding_witnesses,
    dataclass_functions,
    runtime_identity,
)
from examples.snapshot_dispatch_v1.semantics import observe, primitive_input

Evidence, GuardFailure = legacy.Evidence, legacy.GuardFailure
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def counts(rows, root):
    functions = snapshot_functions(root)
    visit = functions["visit"]
    dataclass_origins = dataclass_functions()
    result = {name: 0 for name in ("visit", "value_snapshot", "is_dataclass", "fields", "isfinite", "hex", "visit_isinstance")}
    for row in rows:
        name = row["function"][2]
        if name in ("visit", "value_snapshot") and row["function"] == functions[name] or name in ("is_dataclass", "fields") and row["function"] == dataclass_origins[name]["identity"]:
            result[name] += row["calls"]
        elif name == "<built-in method math.isfinite>":
            result["isfinite"] += row["calls"]
        elif name == "<method 'hex' of 'float' objects>":
            result["hex"] += row["calls"]
        elif name == "<built-in method builtins.isinstance>":
            result["visit_isinstance"] += sum(values[1] for caller, values in row["callers"] if caller == visit)
    return result


def package_calls(rows, root):
    result = {}
    for row in rows:
        file = Path(row["function"][0])
        if root not in file.parents or "simllm" not in file.relative_to(root).parts[:1]:
            continue
        # Only snapshot source lines move. All other function identities stay exact.
        relative = file.relative_to(root).as_posix()
        line = 0 if relative == "simllm/core/value_snapshot.py" else row["function"][1]
        key = f"{relative}:{line}:{row['function'][2]}"
        result[key] = result.get(key, 0) + row["calls"]
    return result


def normalized_reader(row):
    result = deepcopy(row["snapshot"])
    for index, label in ((2, "sink"), (3, "config"), (4, "provider")):
        result[2][index] = {"process_identity": label}
    result[2][7] = {"process_bindings": row["bindings"]}
    return result


def admit(data, arm, root, output, frozen, old, sources, reference, evidence):
    key = arm
    evidence.equal(key + ":arm", data["arm"], arm)
    evidence.equal(key + ":mode", data["mode"], "main")
    evidence.equal(key + ":runtime-before", data["runtime_before"], runtime_identity())
    evidence.equal(key + ":runtime-after", data["runtime_after"], data["runtime_before"])
    evidence.equal(key + ":sources-before", data["sources_before"], sources)
    evidence.equal(key + ":sources-after", data["sources_after"], sources)
    evidence.equal(key + ":worker-sha", data["worker_sha256"], sha(HERE / "worker.py"))
    evidence.equal(key + ":semantic-domain", [row["name"] for row in data["semantics"]], frozen["semantics"])
    for row in data["semantics"]:
        evidence.equal(key + ":semantic:" + row["name"], row, observe(row["name"], reference))
        evidence.check(key + ":identity:" + row["name"], all(value is True for value in row["identity_witnesses"]))
    domain = [spec["id"] for spec in frozen["cases"]]
    workdir = output.parent / "sink-work"
    vectors = {}
    for group in ("primitives", "readers"):
        evidence.equal(key + ":" + group + ":domain", [row["id"] for row in data[group]], domain)
        suffix = "primitive" if group == "primitives" else "reader"
        for row, spec in zip(data[group], frozen["cases"], strict=True):
            label = key + ":" + suffix + ":" + spec["id"]
            evidence.equal(label + ":spec", row["spec"], spec)
            actual = profile_rows(output / (spec["id"] + "-" + suffix + ".pstats"))
            evidence.equal(label + ":full-profile", row["profiles"], actual)
            evidence.equal(label + ":unchanged-before", row["snapshot"], row["before"])
            evidence.equal(label + ":unchanged-after", row["snapshot"], row["after"])
            decoded = unpack(row["snapshot"])
            original_nodes(decoded)
            observed = counts(actual, root)
            evidence.equal(label + ":one-snapshot", observed["value_snapshot"], 1)
            evidence.equal(label + ":all-visits", observed["visit"], original_nodes(decoded))
            if group == "primitives":
                evidence.equal(label + ":complete-input", row["snapshot"], pack(reference(primitive_input(spec, frozen))))
                h, q = spec["history"], spec["width"]
                expected = {"value_snapshot": 1, "visit": 1 + h + 6*h*q, "is_dataclass": h + 1,
                            "fields": 0, "isfinite": h*q, "hex": h*q,
                            "visit_isinstance": 3*(h+1) + (6*h*q if arm == "before" else 0)}
                evidence.equal(label + ":dispatch-vector", observed, expected)
                vectors[spec["id"]] = observed
            else:
                evidence.equal(label + ":outer-shape", [list(decoded[:2]), len(decoded[2])], [["builtins", "tuple"], 8])
                slots = decoded[2]
                evidence.equal(label + ":state", pack(slots[0]), pack(legacy.expected_state(old, workdir)[0]))
                evidence.equal(label + ":publications", pack(slots[1]), pack(legacy.expected_publications(
                    spec, old, legacy_worker.SyntheticRow.__module__)))
                names = list(legacy.BASE_BINDINGS)
                names[8:8] = old["new_helpers"]
                evidence.equal(label + ":bindings", row["bindings"], names)
                ids, bindings = row["identities"], row["binding_identities"]
                evidence.check(label + ":identity-domain", len(ids) == 3 and len(set(ids)) == 3
                               and all(type(value) is int and value > 0 for value in ids))
                evidence.check(label + ":binding-domain", len(bindings) == len(names) and all(
                    len(entry) == 3 and entry[0] == ids[0] and all(type(value) is int and value > 0 for value in entry)
                    for entry in bindings))
                evidence.equal(label + ":persistent-method-identities", [entry[1:] for entry in bindings],
                               [entry[1:] for entry in data["readers"][0]["binding_identities"]])
                flat = ids + [value for entry in bindings for value in entry[1:]]
                evidence.equal(label + ":binding-alias-graph", len(set(flat)), 3 + 2*len(names))
                expected_sink = legacy_worker.make_sink(old, workdir)
                expected_witnesses = binding_witnesses(expected_sink, names, ROOT)
                witnesses = row["binding_witnesses"]
                evidence.equal(label + ":witness-domain", len(witnesses), len(names))
                for index, (witness, expected_witness) in enumerate(zip(witnesses, expected_witnesses, strict=True)):
                    expected_witness.update(owner_id=bindings[index][0], function_id=bindings[index][1], code_id=bindings[index][2])
                    evidence.equal(label + ":source-witness:" + str(index), witness, expected_witness)
                evidence.equal(label + ":identity-slots", pack(slots[2:5]), pack(reference(tuple(ids))[2]))
                evidence.equal(label + ":binding-slots", pack(slots[7]), pack(reference(tuple(tuple(entry) for entry in bindings))))
                first = data["jobs"][0]["job"]
                evidence.equal(label + ":record", pack(slots[5]), first["records"][2][0])
                evidence.equal(label + ":simulation", pack(slots[6]), first["prepared_simulation"])
    evidence.equal(key + ":job-domain", [row["job"]["id"] for row in data["jobs"]], [spec["id"] for spec in old["jobs"]])
    for row, spec in zip(data["jobs"], old["jobs"], strict=True):
        label = key + ":job:" + spec["id"]
        evidence.equal(label + ":full-profile", row["profiles"], profile_rows(output / (spec["id"] + "-job.pstats")))
        legacy.check_job(row["job"], spec, old, workdir, evidence, label)
    first, second = (next(row["job"]["service_ps"][0] for row in data["jobs"] if row["job"]["id"] == name)
                     for name in ("l1-h0", "l2-h0"))
    evidence.check(key + ":layer-bound", first < second <= 2*first + 1000)
    fixture = next(row["job"] for row in data["jobs"] if row["job"]["id"] == "l1-h1")
    for group, names in (("mutations", old["common_corruptions"]), ("helpers", old["new_helper_corruptions"])):
        legacy.check_mutations(data[group], names, old, "after", evidence, key + ":" + group, fixture)
    return vectors


def compare(data, roots, frozen, evidence):
    before, after = data["before"], data["after"]
    for left, right, spec in zip(before["primitives"], after["primitives"], frozen["cases"], strict=True):
        key = "paired:primitive:" + spec["id"]
        a, b = counts(left["profiles"], roots["before"]), counts(right["profiles"], roots["after"])
        evidence.equal(key + ":snapshot-count-vector", [left["snapshot"], a["visit"], a["is_dataclass"], a["fields"], a["isfinite"], a["hex"]],
                       [right["snapshot"], b["visit"], b["is_dataclass"], b["fields"], b["isfinite"], b["hex"]], kind="oracles")
        saving = 6 * spec["history"] * spec["width"]
        evidence.equal(key + ":dispatch-reduction", a["visit_isinstance"] - b["visit_isinstance"], saving,
                       **({"kind": "relations", "family": "primitive-dispatch-work"} if saving else {}))
    for left, right in zip(before["readers"], after["readers"], strict=True):
        key = "paired:reader:" + left["id"]
        evidence.equal(key + ":full-values", normalized_reader(left), normalized_reader(right))
        evidence.equal(key + ":all-package-calls", package_calls(left["profiles"], roots["before"]), package_calls(right["profiles"], roots["after"]))
    for left, right in zip(before["jobs"], after["jobs"], strict=True):
        key = "paired:job:" + left["job"]["id"]
        evidence.equal(key + ":complete-job", left["job"], right["job"], kind="oracles")
        evidence.equal(key + ":all-package-calls", package_calls(left["profiles"], roots["before"]), package_calls(right["profiles"], roots["after"]))
    for group in ("semantics", "mutations", "helpers"):
        evidence.equal("paired:" + group, before[group], after[group])
    evidence.finish("paired-relations")
