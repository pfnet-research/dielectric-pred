#!/usr/bin/env bash
set -eEuo pipefail
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
script_abspath=$(realpath "${script_dir}")

function abort() {
    false
}

default_repo="git@github.pfidev.jp:chem/pekoe.git"
default_branch="dev"
default_prefix="asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe"
default_os="ubuntu20.04"

function usage() {
    set +x
    echo ""
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [OPTION]..."
    echo ""
    echo "Options:"
    echo "  -h, --help     show this message"
    echo "  -w, --workdir  working directory (default: tempory directory automatically will be created)"
    echo "  -r, --repo     git repository to be cloned (default: \"${default_repo}\")"
    echo "  -b, --branch   git branch where build images (default: \"${default_branch}\")"
    echo "  -p, --prefix   image prefix (default: \"${default_prefix}\")"
    echo "      --os       os (default: \"${default_os}\")"
    abort
}

function slack_notify() {
    set +x
    status=$1
    shift
    if [ $status -eq 0 ]; then
	icon=":flexci-notifier-success:"
	message="SUCCESS"
    else
	icon=":flexci-notifier-failed:"
	message="FAILED"
    fi
    if [ $# -gt 0 ];then
	branch=$1
	shift
    else
	branch=""
    fi
    
    echo "${icon} [ pekoe-build-images | ${branch} ] ${message}" |
        jq -sR '{ channel: "#prj-chem-pekoe-dev", text: . }' |
        curl \
            --http1.1 \
            -X POST \
            --data-urlencode payload@- \
            https://hooks.slack.com/services/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
}

opt=`getopt -o h,b:,r:,w:,p: -l help,branch:,repo:,workdir:,prefix:,os: -- "$@"`
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
        -b | --branch)
            opt_branch="$2"
            shift 2
            ;;
        -r | --repo)
            opt_repo="$2"
            shift 2
            ;;
        -w | --workdir)
            opt_workdir="$2"
            shift 2
            ;;
        -p | --prefix)
            opt_prefix="${opt_prefix} $2"
            shift 2
            ;;
        --os)
            opt_os="${opt_os} $2"
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

trap "slack_notify \$? \${branch-""}" EXIT
set -x

workdir=${opt_workdir-""}
repo=${opt_repo-${default_repo}}
branch=${opt_branch-${default_branch}}
prefix=${opt_prefix-${default_prefix}}
os=${opt_os-${default_os}}

if [ -z ${workdir} ]; then
    workdir="$(mktemp -d /tmp/build_images_XXXXXXX)"
    trap "slack_notify \$? \${branch-""}; rm -rf ${workdir}" EXIT
fi

cd ${workdir}
git clone ${repo} --depth 1 --branch ${branch} &&
cd $(basename ${repo%%.git})
git lfs pull

python3 -m virtualenv .venv
. .venv/bin/activate
pip3 install -U 'setuptools<50'
pip3 install -r requirements-style.txt

dt=$(TZ=Asia/Tokyo date +%Y%m%d)
for p in ${prefix}; do
    invoke docker.build --prefix="${p}" --os="${os}" --tag="${dt}" --push
done
