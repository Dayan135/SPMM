#!/bin/bash
set -euo pipefail
BUILD=@@BUILD_DIR@@
SHA=@@SHA@@
TREE=@@SRC_HASH@@
start=$SECONDS
cd "$BUILD/torch_fast_spdmm"
python setup.py build_ext --inplace > "$BUILD/build.log" 2>&1 || { tail -40 "$BUILD/build.log" >&2; exit 1; }
so=$(find "$BUILD/torch_fast_spdmm" -maxdepth 1 -name 'torch_fast_spdmm_core*.so' -print -quit)
test -n "$so"
use "$BUILD" "{{SRC_DIR}}"
python -c 'import pathlib,torch_fast_spdmm,torch_fast_spdmm_core; b=pathlib.Path("@@BUILD_DIR@@").resolve(); assert pathlib.Path(torch_fast_spdmm.__file__).resolve().is_relative_to(b); assert pathlib.Path(torch_fast_spdmm_core.__file__).resolve().is_relative_to(b); print(torch_fast_spdmm.__file__,torch_fast_spdmm_core.__file__)'
export AGENT_BUILD="$BUILD" AGENT_SO="$so" AGENT_SHA="$SHA" AGENT_TREE="$TREE" AGENT_BUILD_S="$((SECONDS-start))"
python -c 'import json,os,hashlib,subprocess,re,pathlib,torch; p=pathlib.Path(os.environ["AGENT_BUILD"]); log=(p/"build.log").read_text(errors="replace"); out=dict(sha=os.environ["AGENT_SHA"],src_hash=os.environ["AGENT_TREE"],so_sha256=hashlib.sha256(pathlib.Path(os.environ["AGENT_SO"]).read_bytes()).hexdigest(),nvcc=subprocess.check_output(["nvcc","--version"],text=True).splitlines()[-1],torch=torch.__version__,build_s=float(os.environ["AGENT_BUILD_S"]),warnings=len(re.findall(r"warning",log,re.I))); (p/"BUILD_OK.json").write_text(json.dumps(out))'
