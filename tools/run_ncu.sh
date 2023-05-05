#!/bin/sh

##################################################################################
# For instructors
# to use ncu command install : ./nsight-compute-linux-2022.1.1.2-30914944.run
# update PATH env variable to include path to ncu binary - fix
##################################################################################

########## Profilers: FOR T4
# 1. nvprof (for system info)
# 2. ncu only (for gpu info)
# 3. ncu + Nsight-Compute (for gpu info with visualization)

##### 1. nvprof (for system info)
# Use the following command for measuring system frequencies and other details
nvprof --system-profiling on ./mmpy -v -n 256

##### 2. ncu only
# Use the following command for measuring all GPU related measurements
# Output is added to log file named "blort.log"
sudo /usr/local/NVIDIA-Nsight-Compute-2022.1/ncu -f --target-processes all --set full --log-file blort.log ./mmpy -v -n 256

# You may use "--section <section-name>" instead of "--set full". Available sections are listed at the end of this file.
sudo /usr/local/NVIDIA-Nsight-Compute-2022.1/ncu -f --target-processes all --section ComputeWorkloadAnalysis --log-file blort.log ./mmpy -v -n 256

##### 3. ncu + Nsight-Compute (for gpu info with visualization)
# The following commands are similar to "2. ncu only". These commands will produce a output file "<file-name>.ncu-rep", that can be downloaded to local machin and opened in Nsight-Compute software.
sudo /usr/local/NVIDIA-Nsight-Compute-2022.1/ncu -f -o profile --target-processes all --set full --log-file blort.log ./mmpy -v -n 256
sudo /usr/local/NVIDIA-Nsight-Compute-2022.1/ncu -f -o profile --target-processes all --section ComputeWorkloadAnalysis --log-file blort.log ./mmpy -v -n 256

# Install Nsight from the following link on you Local machine
# pick a configuration that fits your machine -> OS, CPU Architecture, etc...
# https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

##### Extra - may or may not be usefull
# nsys profile ./mmpy -v -D -n 256

##### old nvprof metrics (pre - spring-2022)
# # https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# nvprof --analysis-metrics -o  mmpy-analysis.nvprof ./mmpy -v -D -n 256
# nvprof --metrics gld_requested_throughput,gst_requested_throughput,gld_efficiency,gst_efficiency,shared_replay_overhead,shared_efficiency,shared_load_throughput,shared_load_transactions,alu_fu_utilization --log-file ../blort.log ../mmpy -v -D -n 256

##### ncu - Following are the sections available: <section-name>

# ComputeWorkloadAnalysis
# InstructionStatistics
# LaunchStatistics
# MemoryWorkloadAnalysis
# MemoryWorkloadAnalysis_Chart
# MemoryWorkloadAnalysis_Deprecated
# MemoryWorkloadAnalysis_Tables
# Nvlink
# Nvlink_Tables
# Nvlink_Topology
# Occupancy
# SchedulerStatistics
# SourceCounters
# SpeedOfLight
# SpeedOfLight_HierarchicalDoubleRooflineChart
# SpeedOfLight_HierarchicalHalfRooflineChart
# SpeedOfLight_HierarchicalSingleRooflineChart
# SpeedOfLight_HierarchicalTensorRooflineChart
# SpeedOfLight_RooflineChart
# WarpStateStatistics
