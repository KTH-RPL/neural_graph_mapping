# This file defines task-independent environment variables
export PYTHONNOUSERSITE=1
export NGM_DATA_DIR="./datasets/"
export LDFLAGS="${LDFLAGS} -L${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib/stubs"
