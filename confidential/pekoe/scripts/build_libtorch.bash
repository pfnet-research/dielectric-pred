#!/bin/bash
set -eEuo pipefail

function abort() {
    false
}

function usage() {
    set +x
    echo ""
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [OPTION]..."
    echo ""
    echo 'Options:'
    echo '  -h, --help          show this message'
    echo '      --build-torch   build pytorch on k8s (default: false)'
    echo '  -p, --prefix        image prefix to build pytorch (default: "harbor.mnj.pfn.io/chem/pekoe")'
    abort
}

opt=`getopt -o p:,h -l build-torch,prefix:,help -- "$@"`
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
        --build-torch)
            opt_build_torch=1
            shift
            ;;
        -p | --prefix)
            opt_prefix="$2"
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
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
script_abspath=$(realpath "${script_dir}")

opt_build_torch=${opt_build_torch-0}
image_prefix=${opt_prefix-"harbor.mnj.pfn.io/chem/pekoe"}

JCNAME=${JCNAME-${USER}}
ssh_host=${JCNAME}@jm00z0cm07.mnj.pfn.io

if [ ${opt_build_torch} -eq 1 ] && ! which pfkube 2>&1 > /dev/null; then
    echo 'pfkube is needed'
    echo 'Please see this docment to set up pfkube and get ready to use MN-J.'
    echo 'https://go.pfn.io/cluster-setup'
    abort
fi

if ! ssh ${ssh_host} true; then
    echo 'Please set up to be able to ssh jm00z0cm07.mnj.pfn.io'
    abort
fi

image_builder=${image_prefix}-libtorch-builder
image_binary=${image_prefix}-libtorch-binary

nfs_dir="/mnt/vol21/${JCNAME}"
pekoe_dir=${nfs_dir}/.pekoe
ncpu=8 # 8 is maximum if you use priority-class is 'high'

if [ ${opt_build_torch} -eq 1 ]; then
    pfkube \
        --target-cluster=mnj run \
        --package=pod \
        --priority-class=high \
        -o activity-code=5170 \
        -o image="${image_builder}" \
        -o resource/cpu=${ncpu} \
        -o resource/memory=32Gi \
        -o auto-inject-https-proxy=enabled \
        -o auto-inject-http-proxy=enabled \
        -o nfs=enabled \
        -- bash -c "cd /tmp \
	&& env CCACHE_DIR=${pekoe_dir}/.cache \
	/tmp/pfp/pytorch/build.bash \
	-j ${ncpu} --outdir ${pekoe_dir}"
fi

if ! ssh ${ssh_host} test -f /home/${JCNAME}/.pekoe/'libtorch-*.zip' 2>&1 > /dev/null; then
    set +x
    echo ""
    echo "No libtorch binary found in ${ssh_host}:/home/${JCNAME}/.pekoe"
    echo "Try --build-torch option to build pytorch."
    abort
fi

scp ${script_abspath}/../deploy/external/pytorch/{Dockerfile,.dockerignore} ${ssh_host}:/home/${JCNAME}/.pekoe/
ssh -A ${ssh_host} -- "cd .pekoe && docker build -t ${image_binary} . && docker push ${image_binary}"
