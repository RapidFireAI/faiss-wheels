#!/usr/bin/env bash

echo "To build wheel,"
echo "1. Edit pyproject.toml as desired (like version, name, python version, add requires-python = \"==3.12.*\" to project, add wheel.py-api = \"cp312\" to tool.scikit-build)"
echo "2. export FAISS_OPT_LEVELS=\"generic\""
echo "3. export FAISS_GPU_SUPPORT=CUDA"
echo "4. pipx run build --wheel -C cmake.args=\"-DCMAKE_BUILD_TYPE=Release\" -C cmake.args=\"-DCMAKE_CXX_FLAGS=-s\""
echo "5. auditwheel repair dist/rf_faiss_gpu*.whl -w dist/repaired/ --exclude libcublas.so --exclude libcudart.so --exclude libcublasLt.so"
echo "6. export TWINE_USERNAME=__token__"
echo "7. export TWINE_PASSWORD=\"<paste token>\""
echo "8. twine upload dist/repaired/*"
export FAISS_OPT_LEVELS="generic"
export FAISS_GPU_SUPPORT=CUDA
export TWINE_USERNAME=__token__
