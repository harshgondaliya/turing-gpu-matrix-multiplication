// ;-*- mode: c;-*-
/*
 * Simplest matrix multiplication in CUDA
 *
 * Bryan Chin -University of California, San Diego
 * August 2022
 * All rights reserved.
 *
 * We compute C = A * B
 *
 * This code assumes that the  matrices are square though there
 * are hooks to facilitate extending the code to non-square matrices
 *
 */

// system includes
#include <stdio.h>
#include <assert.h>
#include <iostream>

//  include the kernel
#ifdef TARGET_T4
#include "../src_todo_T4/mmpy_kernel.cu"
#endif

#include "types.h"
#include "utils.h"
#include "cublas_v2.h"

// External function definitions
void genMatrix(_FTYPE_ *a, unsigned int m, unsigned int n);
void genMatrix_bt(_FTYPE_ *a, _FTYPE_ *b, unsigned int n);
void genMatrix_rand(_FTYPE_ *a, _FTYPE_ *b, unsigned int n);
void genMatrix_ISeq(_FTYPE_ *a, _FTYPE_ *b, unsigned int n);
void verify(_FTYPE_ *c, _FTYPE_ *a, _FTYPE_ *b, unsigned int m, unsigned int n, _FTYPE_ epsilon, const char *mesg);
void verify_bt(_FTYPE_ *c, unsigned int n, const char *mesg);
void verify(_FTYPE_ *c_d, _FTYPE_ *c_h, unsigned int m, unsigned int n, _FTYPE_ eps, const char *mesg);
void verify_bt(_FTYPE_ *c_d, _FTYPE_ *c_h, unsigned int n, const char *mesg);
void verify_bt(_FTYPE_ *c_d, _FTYPE_ *c_h, unsigned int m, unsigned int n, const char *mesg);
void verify_ISeq(_FTYPE_ *c, unsigned int m, unsigned int n);
void verify_rand(_FTYPE_ *a, _FTYPE_ *b, _FTYPE_ *c, unsigned int n, _FTYPE_ eps);

void printMatrix(_FTYPE_ *a, unsigned int m, unsigned int n);
void cmdLine(int argc, char *argv[], int &n, int &reps, int &ntx, int &nty, _FTYPE_ &eps, int &do_host, int &prefer_l1, int &use_rand, int &use_bt, int &use_seq, int &use_shm_double, int &verify_gpu);
void perfString(int n, int ntx, int nty, int reps, double t_h, double gflops_h, double t_d, double gflops_d, int do_host, int prefer_l1, int use_rand, int use_bt, int use_shm_double);

double getTime();
double gflops(int n, int niter, double time);

void matMulHost(_FTYPE_ *, const _FTYPE_ *, const _FTYPE_ *, unsigned int, unsigned int);
void setGrid(int n, dim3 &blockDim, dim3 &gridDim);

int main(int argc, char **argv)
{
    // To improve repeatabilty of measurements taken on the device,
    // we multiply the number of reps by this scale factor
    // Adjust as needed
  //    const int SCALE = 10;
    const int SCALE = 1;

#ifdef CUBLAS_TEST
    cublasHandle_t cublas_handle;
    const _FTYPE_ cublas_alpha = 1.0f;
    const _FTYPE_ cublas_beta = 0.0f;
    cublasCreate(&cublas_handle);
#endif

    // Read in the command line elements
    int n, reps, ntx, nty, do_host, prefer_l1, use_rand, use_bt, use_seq, use_shm_double, verify_gpu;
    _FTYPE_ eps;

    cmdLine(argc, argv, n, reps, ntx, nty, eps, do_host, prefer_l1, use_rand, use_bt, use_seq, use_shm_double, verify_gpu);

    // Total amount of storage for entries
    unsigned int n2 = n * n * sizeof(_FTYPE_);

    // Report on Device Characteristics
    int capability = ReportDevice();
    int major = capability / 100;
    int minor = capability % 100;
    printf(" capability %d.%d\n", major, minor);

    // setup execution configurations
    int _ntx, _nty;
#if (!defined(BLOCKDIM_X) && !defined(BLOCKDIM_Y))
    _ntx = ntx;
    _nty = nty;
#else
    _ntx = BLOCKDIM_X;
    _nty = BLOCKDIM_Y;
#endif

    dim3 threads(_ntx, _nty, 1);
    int numblocksX = n / _ntx;
    int numblocksY = n / _nty;


    if (n % _ntx != 0)
        numblocksX++;

    if (n % _nty != 0)
        numblocksY++;

    dim3 grid(numblocksX, numblocksY, 1);

    setGrid(n, threads, grid);

    // print configurations
    printf("n: %d, tx: %d, ty: %d, gridX: %d, gridY: %d, reps: %d, epsilon: %g\n\n", n, threads.x, threads.y, grid.x, grid.y, reps, eps);

#ifndef _DOUBLE
    printf("Using Single precision arithmetic\n\n");
#else
    printf("Using Double precision arithmetic\n\n");
#endif

    if (use_bt)
        printf("Using bidiagonal inputs\n");

    if (use_rand)
        printf("Using random inputs\n");

    if (do_host)
        printf("Doing host computation for comparison\n\n");

    printf("\n");

    // allocate an initialize host memory for A and B matrices
    _FTYPE_ *h_A = (_FTYPE_ *)malloc(n2);
    assert(h_A);
    _FTYPE_ *h_B = (_FTYPE_ *)malloc(n2);
    assert(h_B);
    if (use_bt)
    {
        genMatrix_bt(h_A, h_B, n);
    }
    else if (use_rand)
    {
        genMatrix_rand(h_A, h_B, n);
    }
    else if (use_seq)
    {
        genMatrix_ISeq(h_A, h_B, n); // I * Seq
    }
    else
    {
        genMatrix(h_A, n, n);
        genMatrix(h_B, n, n);
    }

    if (n <= 16)
    {
        cout << "\nA:\n";
        printMatrix(h_A, n, n);
        cout << "\nB:\n";
        printMatrix(h_B, n, n);
    }

    _FTYPE_ *hostC;
    double t_host = 0.0, gflops_h = 0.0;
    if (do_host)
    {
        // compute matrix product on the host
        hostC = (_FTYPE_ *)malloc(n2);
        t_host = -getTime();
        for (int r = 0; r < reps; r++)
            matMulHost(hostC, h_A, h_B, n, n);
        t_host += getTime();
        gflops_h = gflops(n, reps, t_host);
        printf("Host computation time: %f sec. [%f gflops]\n", t_host, gflops_h);

        // Verify host result
        if (use_bt)
            verify_bt(hostC, n, "Host result");
        else if (use_rand)
            cout << "Verfication of host result not supported for random matrices\n";
        else
            verify(hostC, h_A, h_B, n, n, eps, "Host result");

        if (n <= 8)
        {
            printf("\nC:\n");
            printMatrix(hostC, n, n);
        }
    }

    // allocate device memory
    _FTYPE_ *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n2);
    checkCUDAError("Error allocating device memory for matrix A");
    cudaMalloc((void **)&d_B, n2);
    checkCUDAError("Error allocating device memory for matrix B");
    cudaMalloc((void **)&d_C, n2);
    checkCUDAError("Error allocating device memory for matrix C");
    cudaMemset((void **)d_A, -99, n2);
    checkCUDAError("Error initializing device memory matrix A");
    cudaMemset((void **)d_B, -99, n2);
    checkCUDAError("Error initializing device memory matrix B");
    cudaMemset((void **)d_C, 0, n2);
    checkCUDAError("Error clearing device memory matrix C");

    // copy host memory to device
    cudaMemcpy(d_A, h_A, n2, cudaMemcpyHostToDevice);
    checkCUDAError("Error copying matrix A to device");
    cudaMemcpy(d_B, h_B, n2, cudaMemcpyHostToDevice);
    checkCUDAError("Error copying matrix B to device");

    // allocate host memory for the result
    _FTYPE_ *h_C = (_FTYPE_ *)malloc(n2);
    assert(h_C);

    // If we set the preference for L1 cache, rather than
    // shared memory, we may run slightly faster on devices that have the capability
    cudaFuncCache Preference;
    if (prefer_l1)
    {
        Preference = cudaFuncCachePreferL1;
    }
    else
    {
        Preference = cudaFuncCachePreferShared;
    }
    cudaFuncSetCacheConfig(matMul, Preference);

    #ifdef TARGET_T4
    cudaFuncSetAttribute(matMul, cudaFuncAttributePreferredSharedMemoryCarveout,
			 cudaSharedmemCarveoutMaxShared);
    checkCUDAError("Error seting shared memory to 64K carveout");
    cudaFuncSetAttribute(matMul, cudaFuncAttributeMaxDynamicSharedMemorySize,
			 1024 * 64);
    checkCUDAError("Error seting shared memory to 64K");
    #endif


// Start the timer
#ifdef CUDA_TIMER
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
#endif

#ifdef CUDA_TIMER
    cudaEventRecord(start_event, 0);
    float t_device;
#else
    cudaDeviceSynchronize();
    double t_device = -getTime();
#endif

// execute the kernel
#ifdef CUBLAS_TEST

#ifndef _DOUBLE
    for (int r = 0; r < SCALE * reps; r++)
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cublas_alpha, d_B, n, d_A, n, &cublas_beta, d_C, n);
#else
    for (int r = 0; r < SCALE * reps; r++)
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cublas_alpha, d_B, n, d_A, n, &cublas_beta, d_C, n);
#endif

#else // !CUBLAS_TEST
    for (int r = 0; r < SCALE * reps; r++)
        #ifdef TARGET_T4
            matMul<<<grid, threads, 64 *1024>>>(n, d_C, d_A, d_B);
        #endif
#endif

#ifdef CUDA_TIMER
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&t_device, start_event, stop_event);
    t_device /= 1000.0;
#else
    // block until the device has finished
    cudaDeviceSynchronize();
    // Stop the timer
    t_device += getTime();
#endif

    checkCUDAError("Error in matrixMul kernel");

    // copy result from device to host
    cudaMemcpy(h_C, d_C, n2, cudaMemcpyDeviceToHost);
    checkCUDAError("Unable to retrieve result from device");

    cudaDeviceSynchronize();

    double gflops_d = gflops(n, SCALE * reps, t_device);
    printf("Device computation time: %f sec. [%f gflops]\n", t_device, gflops_d);
    perfString(n, ntx, nty, reps, t_host, gflops_h, t_device, gflops_d, do_host, prefer_l1, use_rand, use_bt, use_shm_double);

    cudaMemset((void **)d_C, 0, n2);
    checkCUDAError("Error clearing device memory matrix C");
    #ifdef TARGET_T4
    matMul<<<grid, threads, 64 * 1024>>>(n, d_C, d_A, d_B);
    #endif
    cudaDeviceSynchronize();
    checkCUDAError("Error in matrixMul kernel");
    cudaMemcpy(h_C, d_C, n2, cudaMemcpyDeviceToHost);
    checkCUDAError("Unable to retrieve result from device");

    cudaDeviceSynchronize();

    if (n <= 16)
    {
        printf("\nC (device):\n");
        printMatrix(h_C, n, n);
    }
    // Verify the device result
    if (verify_gpu)
    {
        if (use_bt)
            verify_bt(h_C, n, "Device result");
        else if (use_rand)
            verify_rand(h_A, h_B, h_C, n, eps);
        else if (use_seq)
            verify_ISeq(h_C, n, n);
        else
            verify(h_C, h_A, h_B, n, n, eps, "Device result");
    }
    // But not for random matrices
    if (do_host)
        // Compare host and device results
        if (use_bt)
            verify_bt(h_C, hostC, n, "Device vs. host");
        else if (!use_rand)
            verify(h_C, hostC, n, n, eps, "Device vs. host");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    if (do_host)
        free(hostC);

    assert(cudaSuccess == cudaFree(d_A));
    assert(cudaSuccess == cudaFree(d_B));
    assert(cudaSuccess == cudaFree(d_C));

    cudaDeviceReset();
}
