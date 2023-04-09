#!/usr/bin/bash

mkdir ff_network && cd "$_"
python -m venv .env
source .env/bin/activate
pip install maturin
maturin init --bindings pyo3
maturin develop