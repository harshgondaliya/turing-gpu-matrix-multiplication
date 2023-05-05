#!/bin/sh

# For K80

nvprof --metrics gld_requested_throughput,gst_requested_throughput,gld_efficiency,gst_efficiency,shared_replay_overhead,shared_efficiency,shared_load_throughput,shared_load_transactions,alu_fu_utilization --log-file blort.log ./mmpy -v -D -n 256
