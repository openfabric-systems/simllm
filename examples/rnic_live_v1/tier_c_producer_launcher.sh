#!/bin/sh
set -eu

exec .venv/bin/python examples/rnic_live_v1/tier_c_producer.py "$@"
