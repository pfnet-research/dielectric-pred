#!/usr/bin/env bash
set -eEuo pipefail
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
script_abspath=$(realpath "${script_dir}")

function abort() {
    false
}

default_python_version="3.8"

function usage() {
    set +x
    echo ""
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [OPTION]..."
    echo ""
    echo "Options:"
    echo "  -h, --help           show this message"
    echo "  -r, --repodir        git repository directory (default: \$(git rev-parse --show-toplevel))"
    echo "      --build-prod-dir build directory for production (default: \"build\")"
    echo "      --build-dev-dir  build directory for development (default: \"build.dev\")"
    echo "      --python         python version (default: \"${default_python_version}\")"
    echo "  -m, --make-package   make a release package (default: false)"
    echo "      --python         python version (default: \"${default_python_version}\")"
    abort
}

opt=$(getopt -o h,r:,m: -l help,repodir:,build-dev-dir:,build-prod-dir:,make-package:,python: -- "$@")
if [ "$?" != 0 ]; then
    exit 1
fi
eval set -- "$opt"

while true
do
    case $1 in
        -h | --help)
            usage
            shift
            ;;
        -r | --repodir)
            opt_repo_dir="$2"
            shift 2
            ;;
        --build-dev-dir)
            opt_build_dev_dir="$2"
            shift 2
            ;;
        --build-prod-dir)
            opt_build_prod_dir="$2"
            shift 2
            ;;
        -m | --make-package)
            opt_make_package="$2"
            shift 2
            ;;
        --python)
            opt_python="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

set -x

repo_dir=${opt_repo_dir-$(git rev-parse --show-toplevel)}
build_dev_dir=${opt_build_dev_dir-build.dev}
build_prod_dir=${opt_build_prod_dir-build}
repo_abspath=$(realpath "${repo_dir}")
make_package=${opt_make_package-0}
python_version=${opt_python-${default_python_version}}

# NOTE: A workaround to avoid an error at get_best_invocation_for_this_pip
# cf. https://github.com/pypa/pip/issues/11309#issuecomment-1204963695
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "${repo_abspath}"
python${python_version} -m pip install pytest pytest-cov
for target in dev prod; do
    if [ "${target}" == "dev" ]; then
        make -C deploy/teanet DEPLOY_PYTHON_VERSION=${python_version} BUILD_DIR=${build_dev_dir} ${target}
        ADDITIONAL_PATH=${repo_abspath}/deploy/install/libtorch-dev/lib
        ln -sf deploy/teanet/target_onnx
    else
        make -C deploy/teanet DEPLOY_PYTHON_VERSION=${python_version} BUILD_DIR=${build_prod_dir} ${target}
        ADDITIONAL_PATH=""
        rm -rf target_onnx
    fi
    python${python_version} -m pip install ./release --no-deps --no-use-pep517
    env LD_LIBRARY_PATH="${ADDITIONAL_PATH-""}:${LD_LIBRARY_PATH-""}" python${python_version} -m pytest --cov=pfp tests/unit_tests/pfp -m 'not slow and gpu'
    env LD_LIBRARY_PATH="${ADDITIONAL_PATH-""}:${LD_LIBRARY_PATH-""}" python${python_version} -m pytest tests/integ_tests -m 'not slow and pfp'
done

if [ "${make_package}" -eq 1 ]; then
  cd "$(mktemp -d)"
  trap "cd - && rm -rf $(pwd)" EXIT
  mkdir -p deepmi
  ln -sf "${repo_abspath}"/release/{README.md,examples,pfp} deepmi
  tar -ch -I 'xz -3' --exclude "*.egg-info" --exclude "__pycache__" -f "${TMPDIR}"/deepmi.tar.xz deepmi
fi
