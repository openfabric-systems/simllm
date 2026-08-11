#!/bin/sh
set -eu

exec .venv/bin/python examples/rnic_live_v1/tier_b_producer.py "$@"
