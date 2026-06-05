#!/usr/bin/env bash
# Run all sycl/test-e2e tests that use reqd_work_group_size.

# set_xmain_latest
# set_syclos_local
# export LD_LIBRARY_PATH=/iusers/$(whoami)/igc/build-igc/igc${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# set_env_igc_dump

# Discovers tests dynamically with grep, configures build-e2e on first run,
# then drives llvm-lit. Override SYCL_DEVICES / EXTRA_LIT_OPTS from the env:
#   SYCL_DEVICES=opencl:cpu ./run_reqd_wg_tests.sh
set -euo pipefail

REPO=/localdisk2/$(whoami)/llvm
LIT=$REPO/build/bin/llvm-lit
TEST_ROOT=$REPO/sycl/test-e2e
BUILD_E2E=$REPO/build-e2e

SYCL_DEVICES=${SYCL_DEVICES:-level_zero:gpu}
EXTRA_LIT_OPTS=${EXTRA_LIT_OPTS:-}

# Configure build-e2e on first run. lit.site.cfg.py is what `cmake -S sycl/test-e2e`
# generates; if it's missing, the tree hasn't been configured yet.
if [[ ! -f "$BUILD_E2E/lit.site.cfg.py" ]]; then
  echo "Configuring $BUILD_E2E (one-time setup)..."
  CXX=$(command -v clang++)
  if [[ -z "$CXX" ]]; then
    echo "clang++ not found on PATH — point PATH at your toolchain build first." >&2
    exit 1
  fi
  cmake -GNinja -B"$BUILD_E2E" -S"$TEST_ROOT" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DLLVM_LIT="$REPO/llvm/utils/lit/lit.py"
fi

mapfile -t TESTS < <(grep -rl --include='*.cpp' 'reqd_work_group_size' "$TEST_ROOT" | sort)

if [[ ${#TESTS[@]} -eq 0 ]]; then
  echo "No tests using reqd_work_group_size found under $TEST_ROOT" >&2
  exit 1
fi

echo "Running ${#TESTS[@]} test(s):"
printf '  %s\n' "${TESTS[@]}"

# Map source paths under sycl/test-e2e/ to lit invocation paths under build-e2e/.
LIT_TARGETS=()
for t in "${TESTS[@]}"; do
  LIT_TARGETS+=("${t/$TEST_ROOT/$BUILD_E2E}")
done

exec "$LIT" -a --no-progress-bar --time-tests \
  --param "dpcpp_compiler=$(which clang++)" \
  --param "sycl_devices=$SYCL_DEVICES" \
  $EXTRA_LIT_OPTS \
  "${LIT_TARGETS[@]}"
