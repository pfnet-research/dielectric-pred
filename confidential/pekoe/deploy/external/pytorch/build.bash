#!/usr/bin/env bash
set -eEuo pipefail
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
script_abspath=$(realpath "${script_dir}")

function abort() {
    false
}

function usage() {
    set +x
    echo ""
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [OPTION]..."
    echo ""
    echo 'Options:'
    echo '  -h, --help       show this message'
    echo '      -j,--jobs    build jobs to run simultaneously. (default $(nproc))'
    echo '      -o,--outdir  output directory. (default .)'
    echo '      -w,--workdir work directory. (default .)'
    abort
}

opt=`getopt -o j:,o:,w:,h -l jobs:,outdir:,workdir:,help -- "$@"`
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
        -j | --jobs)
            opt_jobs="-j $2"
            shift 2
            ;;
        -o | --outdir)
            opt_outdir="$2"
            shift 2
            ;;
        -w | --workdir)
            opt_workdir="$2"
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
outdir=$(realpath ${opt_outdir-$(pwd)})
workdir=$(realpath ${opt_workdir-$(pwd)})
opt_jobs=${opt_jobs-""}

TORCH_VERSION=${TORCH_VERSION-1.5.0}
cd ${workdir}
if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch --depth 1 --branch v${TORCH_VERSION}
    cd pytorch
    git apply ${script_abspath}/pytorch.patch
    cd ..
fi

cd ${workdir}/pytorch
git submodule update --init --recursive \
    third_party/cpuinfo \
    third_party/cub \
    third_party/eigen \
    third_party/foxi \
    third_party/FP16 \
    third_party/FXdiv \
    third_party/onnx \
    third_party/protobuf \
    third_party/psimd \
    third_party/pthreadpool \
    third_party/sleef \
    third_party/XNNPACK
cd ..
mkdir -p build
cd build
cmake ../pytorch -G Ninja \
      -DBUILD_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DUSE_CUDA=ON \
      -DUSE_DISTRIBUTED=OFF \
      -DUSE_EIGEN_FOR_BLAS=OFF \
      -DUSE_FBGEMM=OFF \
      -DUSE_GLOO=OFF \
      -DUSE_MKLDNN=OFF \
      -DUSE_MPI=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_NUMA=OFF \
      -DUSE_NUMPY=OFF \
      -DUSE_OPENMP=OFF \
      -DUSE_PYTORCH_QNNPACK=OFF \
      -DUSE_QNNPACK=OFF \
      -DUSE_ROCM=OFF \
      -DPYTHON_EXECUTABLE=$(which python3.7) \
      -DBUILD_TEST=OFF \
      -DCUDA_NVCC_FLAGS="-DNDEBUG -Xfatbin -compress-all" \
      -DCUDA_TOOLKIT_ROOT_DIR=$(pkg-config --variable=cudaroot cuda-10.1) \
      -DTORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 7.5+PTX" \
      -DCMAKE_INSTALL_PREFIX=${workdir}/libtorch
ninja ${opt_jobs} install

cd ${workdir}
mkdir -p ${outdir}
zip ${outdir}/libtorch-${TORCH_VERSION}-cu101.zip -r libtorch
