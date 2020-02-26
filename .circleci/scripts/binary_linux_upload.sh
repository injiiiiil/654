#!/bin/bash
# Do NOT set -x
source /home/circleci/project/env
set -eu -o pipefail
set +x
declare -x "AWS_ACCESS_KEY_ID=${PYTORCH_BINARY_AWS_ACCESS_KEY_ID}"
declare -x "AWS_SECRET_ACCESS_KEY=${PYTORCH_BINARY_AWS_SECRET_ACCESS_KEY}"

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# DO NOT TURN -x ON BEFORE THIS LINE
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
set -eux -o pipefail
export PATH="$MINICONDA_ROOT/bin:$PATH"

# Upload the package to the final location
pushd /home/circleci/project/final_pkgs
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda install -yq anaconda-client
  anaconda -t "${CONDA_PYTORCHBOT_TOKEN}" upload  "$(ls)" -u pytorch-nightly --label main --no-progress --force
elif [[ "$PACKAGE_TYPE" == libtorch ]]; then
  retry pip install -q awscli
  s3_dir="s3://pytorch/libtorch/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  for pkg in $(ls); do
    retry aws s3 cp "$pkg" "$s3_dir" --acl public-read
  done
else
  retry pip install -q awscli
  s3_dir="s3://pytorch/whl/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  retry aws s3 cp "$(ls)" "$s3_dir" --acl public-read
fi
