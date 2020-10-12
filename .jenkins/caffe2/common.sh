set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/caffe2_tests"
gtest_reports_dir="${TEST_DIR}/cpp"
pytest_reports_dir="${TEST_DIR}/python"

# Figure out which Python to use
PYTHON="$(which python)"

# Figure out which Python to use for ROCm
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]] && [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
    PYTHON=$(which "python${BASH_REMATCH[1]}")
    alias python="$PYTHON"
    if which sccache > /dev/null; then
        # Save sccache logs to file
        sccache --stop-server || true
        rm ~/sccache_error.log || true
        SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server

        # Report sccache stats for easier debugging
        sccache --zero-stats
    fi
fi

# /usr/local/caffe2 is where the cpp bits are installed to in in cmake-only
# builds. In +python builds the cpp tests are copied to /usr/local/caffe2 so
# that the test code in .jenkins/test.sh is the same
INSTALL_PREFIX="/usr/local/caffe2"

if [[ "$BUILD_ENVIRONMENT" != *pytorch-win-* ]]; then
fi

mkdir -p "$gtest_reports_dir" || true
mkdir -p "$pytest_reports_dir" || true
mkdir -p "$INSTALL_PREFIX" || true
