#!/bin/bash

# GPTSwarm Run Script
# Usage: ./run.sh [mode] [extra_args]
# Examples:
#   ./run.sh                                         -> OptimizedSwarm (default)
#   ./run.sh DirectAnswer --debug                    -> DirectAnswer in debug mode
#   ./run.sh DirectAnswer --debug --model_name mock  -> Mock LLM, no API key needed

cd "/Users/rmdn/Desktop/RUG/Deep Learning Project/GPTSwarm"

source swarm-env/bin/activate

MODE=${1:-OptimizedSwarm}
shift 2>/dev/null

PYTHONPATH=. python experiments/run_mmlu.py --mode "$MODE" "$@"