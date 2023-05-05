LIB_BLAS  = -lblas -lpthread -lm
LDLIBS   += $(LIB_BLAS)
INCLUDES += -I/usr/include/openblas
C++FLAGS += $(INCLUDES)

CUDA_INSTALL_PATH=/usr/local/cuda-11.6

GENCODE_SM75  :=-gencode arch=compute_75,code=sm_75

GENCODE_FLAGS :=$(GENCODE_SM75)
PTXFLAGS=-v
# PTXFLAGS=-dlcm=ca 
NVCCFLAGS= -O3 $(GENCODE_FLAGS) -c

# Compilers
NVCC            = $(shell which nvcc)
C++             = $(shell which g++)
C++LINK         = $(C++)
NVCCLINK        = $(NVCC)
CLINK           = $(CC)

.SUFFIXES:
.SUFFIXES: .cpp .c .cu .o

.cpp.o:
		$(C++) $(C++FLAGS) -c $<

.c.o:
		$(C++) $(C++FLAGS) -c $<

.cu.o:
	$(NVCC)  $(NVCCFLAGS) $<


