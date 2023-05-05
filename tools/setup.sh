#!/bin/sh
# This is only to be used by instructors for installing required tools for PA2 extra credit

# GPU T4
# AWS instance g4dn.xlarge
# Ubuntu 20.04

# some commands may not execute as expected. Read all comments.
# uncomment everything below this line and run

# #chores
# sudo apt-get update
# sudo apt-get -y upgrade

# #required build tools
# sudo apt-get install -y gcc
# sudo apt-get install -y linux-headers-$(uname -r)


# # Download & install cuda
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# sudo apt install -y nvidia-cuda-toolkit
# sudo apt install -y nvidia-profiler

# sudo apt-get install -y libopenblas-dev

# # download the following file and install it
# # It may fail to download with wget. So download on local machine and upload to AWS. Then install it. This is required for ncu command.
# # also add path to ncu to the PATH variable
# wget https://developer.nvidia.com/rdp/assets/nsight-compute-2022_1_1_2-linux-installer
# chmod +x ../../nsight-compute-linux-2022.1.1.2-30914944.run
# sudo ../../nsight-compute-linux-2022.1.1.2-30914944.run
# echo $PATH
# # copy the following line in "sudo vim /etc/profile" file
# # export PATH=/usr/local/NVIDIA-Nsight-Compute-2022.1:$PATH