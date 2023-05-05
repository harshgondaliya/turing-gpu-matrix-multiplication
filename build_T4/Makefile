#
# Makefile for building Matrix Multiplication under CUDA
# 
# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects
# make extra  # compile all binary for extra credit part


include ./arch.cuda.gnu

C++FLAGS += -DTARGET_T4
CFLAGS += -DTARGET_T4
NVCCFLAGS += -DTARGET_T4 

# Set cublastest=1 to build the variant that uses cuBLAS
# By default, build the code to use student written kernel
ifeq ($(cublastest), 1)
    C++FLAGS += -DCUBLAS_TEST
    CFLAGS += -DCUBLAS_TEST
    NVCCFLAGS += -DCUBLAS_TEST
endif  

# Set usecache=1 to build the variant that uses cache
# By default, build the code to use shared memory
ifeq ($(usecache), 1)
    C++FLAGS += -DUSE_CACHE
    CFLAGS += -DUSE_CACHE
    NVCCFLAGS += -DUSE_CACHE
endif   

# Set gprof=1 on make command line to compile for gprof
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        NVCCFLAGS += -G -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif

# Set ncu=1 on make command line to compile for nvidia compute profiler
ifeq ($(ncu), 1)
        CFLAGS += -g 
        NVCCFLAGS += -G -g 
        C++FLAGS += -g 
        LDFLAGS += -g 
endif


# Set debug=1 on make command line to keep symbol table info for gdb/cachegrind
ifeq ($(debug), 1)
        NVCCFLAGS += -g -G
	LDFLAGS += -g -G
#	NVCCFLAGS += -g
endif   

# If you want to compile for double precision,
# specify double=1 on the "make" command line
ifneq ($(double), 1)
else
    C++FLAGS += -D_DOUBLE
    CFLAGS += -D_DOUBLE
    NVCCFLAGS += -D_DOUBLE
endif

# Keep arround compiler output files, including the ptx assembler
ifeq ($(keep), 1)
	NVCCFLAGS	+= -keep
	NVCCFLAGS 	+= --ptx
endif

# If you want to use the  CUDA Timer
# specify cuda_timer=1 on the "make" command line
ifeq ($(cuda_timer), 1)
	NVCCFLAGS += -DCUDA_TIMER
endif

# Uncomment if you want to report resource requirements (registers etc)
NVCCFLAGS += --ptxas-options=-v
# NVCCFLAGS += --opencc-options -LIST:source=on

# You can set the thread block geometry by specifying bx= and by= on
# the make command line, e.g. make bx=16 by=32
# This feature is useful for the shared memory variant but
# not for the naive variant

# Set up for a default block size of 16 x 16
ifdef bx
	DIM_X = -DBLOCKDIM_X=$(bx)
else
	DIM_X = -DBLOCKDIM_X=8
endif

ifdef by
	DIM_Y = -DBLOCKDIM_Y=$(by)
else
	DIM_Y = -DBLOCKDIM_Y=8
endif

ifdef tm
	TILE_M = -DTILESCALE_M=$(tm)
else
	TILE_M = 
endif
ifdef tn
	TILE_N = -DTILESCALE_N=$(tn)
else
	TILE_N = 
endif
ifdef tk
	TILE_K = -DTILESCALE_K=$(tk)
else
	TILE_K = 
endif

ifdef naive
	NVCCFLAGS += "-DNAIVE"
	CFLAGS += "-DNAIVE"
endif

BLOCKING = $(DIM_X) $(DIM_Y) $(TILE_M) $(TILE_N) $(TILE_K)
NVCCFLAGS += $(BLOCKING)
CFLAGS += $(BLOCKING)
C++FLAGS += $(BLOCKING)

APP=mmpy

OBJECTS = ../src/mmpy.o  ../src/mmpy_host.o  ../src/genMatrix.o ../src/cmdLine.o ../src/Timer.o ../src/utils.o ../src/Report.o ../src_todo_T4/setGrid.o

all:
	$(NVCC) $(NVCCFLAGS) -c ../src/mmpy.cu -o ../src/mmpy.o
	$(C++) $(C++FLAGS) -c ../src/mmpy_host.cpp -o ../src/mmpy_host.o
	$(C++) $(C++FLAGS) -c ../src/genMatrix.cpp -o ../src/genMatrix.o
	$(C++) $(C++FLAGS) -c ../src/cmdLine.cpp -o ../src/cmdLine.o
	$(C++) $(C++FLAGS) -c ../src/Timer.cpp -o ../src/Timer.o
	$(NVCC) $(NVCCFLAGS) -c ../src/utils.cu -o ../src/utils.o
	$(C++) $(C++FLAGS) -c ../src/Report.cpp -o ../src/Report.o
	$(NVCC) $(NVCCFLAGS) -c ../src_todo_T4/setGrid.cu -o ../src_todo_T4/setGrid.o
	$(NVCC) $(NVCCFLAGS) -c ../src_todo_T4/mmpy_kernel.cu -o ../src_todo_T4/mmpy_kernel.o
	$(NVCC) -o ../$(APP) $(LDFLAGS) $(OBJECTS)  $(LDLIBS) -lcublas

clean:
	rm -rf ../src/*.o ../src_todo_T4/*.o ../$(APP)

