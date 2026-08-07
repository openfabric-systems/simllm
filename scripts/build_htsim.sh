#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/htsim"
run_tests=OFF
jobs="${HTSIM_JOBS:-$(nproc)}"

if [[ $# -gt 0 && "$1" != "--test" ]]; then
    build_dir="$1"
    shift
fi
if [[ $# -gt 0 && "$1" == "--test" ]]; then
    run_tests=ON
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [build-directory] [--test]" >&2
    exit 2
fi
if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo_root}/${build_dir}"
fi

cmake -S "${repo_root}/third_party/htsim/htsim/sim" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DENABLE_TESTS="${run_tests}" \
    -DHTSIM_CREATE_SOURCE_SYMLINKS=OFF
cmake --build "${build_dir}" --parallel "${jobs}"

if [[ "${run_tests}" == ON ]]; then
    ctest --test-dir "${build_dir}" --output-on-failure --parallel "${jobs}"
fi
