#!bin/bash

# usage in MAIN DIRECTORY USE THIS SCRIPT: bash scripts/run_jupyter.sh $port

export PYTHONPATH="$PWD/source"
jupyter lab --no-browser --port=$1
