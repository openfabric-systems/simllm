"""Derive a dense all-reduce-only capture without changing its timing block."""

import argparse
import hashlib
import json
from pathlib import Path


def transform(source, config):
    first = source.index("  // B1 peer bandwidth matrix.")
    last = source.index("  std::vector<size_t> sizes;", first)
    source = source[:first] + source[last:]
    first = source.index("    // B3 contention, run only at the full width.")
    last = source.index(
        "    for (int r = 0; r < width; ++r) {\n"
        "      CUDA_CHECK(cudaSetDevice(r));\n"
        "      CUDA_CHECK(cudaStreamDestroy(streams[r]));",
        first,
    )
    source = source[:first] + source[last:]
    grid = config["dense_grid"]
    replacements = {
        "  sizes.push_back(8);\n"
        "  for (int k = 10; k <= 30; ++k) sizes.push_back(1ull << k);": (f"  for (size_t b = {grid['start_bytes']}; b <= {grid['end_bytes']}; "
         f"b += {grid['step_bytes']}) sizes.push_back(b);"),
        "const Op ops[] = {Op::AllReduce, Op::AllGather, Op::ReduceScatter, Op::Broadcast};": "const Op ops[] = {Op::AllReduce};",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Inherited source changed: transformation is not unique")
        source = source.replace(old, new)
    body = source[source.index("#include") :]
    return (
        "// Dense compatibility capture derived from the hardware envelope.\n"
        "// Preserves its all-reduce buffers, warmups and per-rank event timing.\n"
        "// Peer bandwidth and compute contention experiments are omitted.\n\n" + body
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expectations", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    config = json.loads(args.expectations.read_text())
    output = transform(source, config)
    # The timing body must be byte-identical to the inherited implementation.
    start = "          for (int it = 0; it < kWarmup; ++it) issue();"
    end = "          if (op == Op::AllReduce) {"

    def block(text):
        return text[text.index(start) : text.index(end)]

    if block(source) != block(output):
        raise ValueError("Compatibility timing body changed")
    args.output.write_text(output)
    manifest = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "timing_block_sha256": hashlib.sha256(block(output).encode()).hexdigest(),
        "timing_block_identical": True,
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
