#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
c++ -O3 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) src/abodybuilder3/pdb_native.cpp -o "src/abodybuilder3/_pdb_native$(python3-config --extension-suffix)"
