#!/bin/bash

function with_backoff {
  local max_attempts=${ATTEMPTS-5}
  local timeout=${TIMEOUT-1}
  local attempt=0
  local exitCode=0

  while [[ $attempt < $max_attempts ]]
  do
    "$@"
    exitCode=$?

    if [[ $exitCode == 0 ]]
    then
      break
    fi

    echo "Failure! Retrying in $timeout.." 1>&2
    sleep $timeout
    attempt=$(( attempt + 1 ))
    timeout=$(( timeout * 2 ))
  done

  if [[ $exitCode != 0 ]]
  then
    echo "You've failed me for the last time! ($@)" 1>&2
  fi

  return $exitCode
}

# conda pacakges (Only install necessary pacakges, the majority dependencies are installed by pip)
with_backoff conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# pip pacakges
with_backoff pip --proxy ${HTTP_PROXY} install pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python
with_backoff pip --proxy ${HTTP_PROXY} install open3d
# with_backoff pip --proxy ${HTTP_PROXY} install msgpack-rpc-python  airsim

# need "TORCH_CUDA_ARCH_LIST" flag to install torchsort
TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing;Ampere" pip install torchsort