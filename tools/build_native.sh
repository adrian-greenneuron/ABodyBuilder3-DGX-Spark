#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
c++ -O3 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) candidate/pdb_native.cpp -o "candidate/_pdb_native$(python3-config --extension-suffix)"
