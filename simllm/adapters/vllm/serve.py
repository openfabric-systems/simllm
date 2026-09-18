"""Run the pinned API server with the executor class resolved before PP checks.

Usage: ``python -m simllm.adapters.vllm.serve --model MODEL
--pipeline-parallel-size 2 --enforce-eager --no-async-scheduling``.
"""

from __future__ import annotations


def main() -> None:
    import uvloop
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from simllm.adapters.vllm import SimExecutor

    cli_env_setup()
    parser = make_arg_parser(FlexibleArgumentParser(description="SimLLM vLLM API server"))
    args = parser.parse_args()
    if args.distributed_executor_backend not in (
        None, "simllm.adapters.vllm.SimExecutor",
    ):
        parser.error("this entry point selects SimExecutor")
    # EngineArgs reads supports_pp before resolving dotted backend strings.
    args.distributed_executor_backend = SimExecutor
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
