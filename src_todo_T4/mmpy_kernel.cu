// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>
extern __shared__ float sharmem[];
#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
// bx=32,by=8,TILEDIM_M=128,TILEDIM_K=64,TILEDIM_N=64,32 ops per thread, interleaved access
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    _FTYPE_ * __restrict__ As = sharmem;
    _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    int I =  by*blockDim.y + ty; // rows
    int J =  bx*blockDim.x + tx; // cols   
    int count;
    if((I < N) && (J < N)){
        register _FTYPE_ _cij[32] = {0};
        #pragma unroll
        for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
            if(N%TILEDIM_M == 0){
                As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
                As[ty*TILEDIM_K + (tx+32)] = A[a + ty*N + (tx+32)];
                As[(ty+8)*TILEDIM_K + tx] = A[a + (ty+8)*N + tx];
                As[(ty+8)*TILEDIM_K + (tx+32)] = A[a + (ty+8)*N + (tx+32)];
                As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
                As[(ty+16)*TILEDIM_K + (tx+32)] = A[a + (ty+16)*N + (tx+32)];
                As[(ty+24)*TILEDIM_K + tx] = A[a + (ty+24)*N + tx];
                As[(ty+24)*TILEDIM_K + (tx+32)] = A[a + (ty+24)*N + (tx+32)];
                As[(ty+32)*TILEDIM_K + tx] = A[a + (ty+32)*N + tx];
                As[(ty+32)*TILEDIM_K + (tx+32)] = A[a + (ty+32)*N + (tx+32)];
                As[(ty+40)*TILEDIM_K + tx] = A[a + (ty+40)*N + tx];
                As[(ty+40)*TILEDIM_K + (tx+32)] = A[a + (ty+40)*N + (tx+32)];
                As[(ty+48)*TILEDIM_K + tx] = A[a + (ty+48)*N + tx];
                As[(ty+48)*TILEDIM_K + (tx+32)] = A[a + (ty+48)*N + (tx+32)];
                As[(ty+56)*TILEDIM_K + tx] = A[a + (ty+56)*N + tx];
                As[(ty+56)*TILEDIM_K + (tx+32)] = A[a + (ty+56)*N + (tx+32)];
                As[(ty+64)*TILEDIM_K + tx] = A[a + (ty+64)*N + tx];
                As[(ty+64)*TILEDIM_K + (tx+32)] = A[a + (ty+64)*N + (tx+32)];
                As[(ty+72)*TILEDIM_K + tx] = A[a + (ty+72)*N + tx];
                As[(ty+72)*TILEDIM_K + (tx+32)] = A[a + (ty+72)*N + (tx+32)];
                As[(ty+80)*TILEDIM_K + tx] = A[a + (ty+80)*N + tx];
                As[(ty+80)*TILEDIM_K + (tx+32)] = A[a + (ty+80)*N + (tx+32)]; 
                As[(ty+88)*TILEDIM_K + tx] = A[a + (ty+88)*N + tx];
                As[(ty+88)*TILEDIM_K + (tx+32)] = A[a + (ty+88)*N + (tx+32)];
                As[(ty+96)*TILEDIM_K + tx] = A[a + (ty+96)*N + tx];
                As[(ty+96)*TILEDIM_K + (tx+32)] = A[a + (ty+96)*N + (tx+32)]; 
                As[(ty+104)*TILEDIM_K + tx] = A[a + (ty+104)*N + tx];
                As[(ty+104)*TILEDIM_K + (tx+32)] = A[a + (ty+104)*N + (tx+32)];
                As[(ty+112)*TILEDIM_K + tx] = A[a + (ty+112)*N + tx];
                As[(ty+112)*TILEDIM_K + (tx+32)] = A[a + (ty+112)*N + (tx+32)];
                As[(ty+120)*TILEDIM_K + tx] = A[a + (ty+120)*N + tx];
                As[(ty+120)*TILEDIM_K + (tx+32)] = A[a + (ty+120)*N + (tx+32)];

                Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];
                Bs[ty*TILEDIM_N + (tx+32)] = B[b + ty*N + (tx+32)];
                Bs[(ty+8)*TILEDIM_N + tx] = B[b + (ty+8)*N + tx];
                Bs[(ty+8)*TILEDIM_N + (tx+32)] = B[b + (ty+8)*N + (tx+32)];
                Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
                Bs[(ty+16)*TILEDIM_N + (tx+32)] = B[b + (ty+16)*N + (tx+32)];
                Bs[(ty+24)*TILEDIM_N + tx] = B[b + (ty+24)*N + tx];
                Bs[(ty+24)*TILEDIM_N + (tx+32)] = B[b + (ty+24)*N + (tx+32)];                        
                Bs[(ty+32)*TILEDIM_N + tx] = B[b + (ty+32)*N + tx];
                Bs[(ty+32)*TILEDIM_N + (tx+32)] = B[b + (ty+32)*N + (tx+32)];
                Bs[(ty+40)*TILEDIM_N + tx] = B[b + (ty+40)*N + tx];
                Bs[(ty+40)*TILEDIM_N + (tx+32)] = B[b + (ty+40)*N + (tx+32)];
                Bs[(ty+48)*TILEDIM_N + tx] = B[b + (ty+48)*N + tx];
                Bs[(ty+48)*TILEDIM_N + (tx+32)] = B[b + (ty+48)*N + (tx+32)];
                Bs[(ty+56)*TILEDIM_N + tx] = B[b + (ty+56)*N + tx];
                Bs[(ty+56)*TILEDIM_N + (tx+32)] = B[b + (ty+56)*N + (tx+32)];
                Bs[(ty+64)*TILEDIM_N + tx] = B[b + (ty+64)*N + tx];
                Bs[(ty+64)*TILEDIM_N + (tx+32)] = B[b + (ty+64)*N + (tx+32)];
                Bs[(ty+72)*TILEDIM_N + tx] = B[b + (ty+72)*N + tx];
                Bs[(ty+72)*TILEDIM_N + (tx+32)] = B[b + (ty+72)*N + (tx+32)];
                Bs[(ty+80)*TILEDIM_N + tx] = B[b + (ty+80)*N + tx];
                Bs[(ty+80)*TILEDIM_N + (tx+32)] = B[b + (ty+80)*N + (tx+32)];
                Bs[(ty+88)*TILEDIM_N + tx] = B[b + (ty+88)*N + tx];
                Bs[(ty+88)*TILEDIM_N + (tx+32)] = B[b + (ty+88)*N + (tx+32)];
                Bs[(ty+96)*TILEDIM_N + tx] = B[b + (ty+96)*N + tx];
                Bs[(ty+96)*TILEDIM_N + (tx+32)] = B[b + (ty+96)*N + (tx+32)];             
                Bs[(ty+104)*TILEDIM_N + tx] = B[b + (ty+104)*N + tx];
                Bs[(ty+104)*TILEDIM_N + (tx+32)] = B[b + (ty+104)*N + (tx+32)];
                Bs[(ty+112)*TILEDIM_N + tx] = B[b + (ty+112)*N + tx];
                Bs[(ty+112)*TILEDIM_N + (tx+32)] = B[b + (ty+112)*N + (tx+32)];
                Bs[(ty+120)*TILEDIM_N + tx] = B[b + (ty+120)*N + tx];
                Bs[(ty+120)*TILEDIM_N + (tx+32)] = B[b + (ty+120)*N + (tx+32)];
            } else{
                for(int itx=tx; itx<64; itx+=32){
                    for(int ity=ty; ity<128; ity+=8){
                        if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
                            As[ity*TILEDIM_K + itx] = 0;
                        } else{
                            As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
                        }
                    }
                }
                for(int itx=tx; itx<64; itx+=32){
                    for(int ity=ty; ity<128; ity+=8){
                        if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
                            Bs[ity*TILEDIM_N + itx] = 0;
                        } else{
                            Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
                        }
                    }
                }
            }
             
            __syncthreads();        
            _FTYPE_ register bs_reg1, bs_reg2;
            for (int k = 0; k < TILEDIM_K; k++) {
                bs_reg1 = Bs[k*TILEDIM_N + tx];
                _cij[0] += As[ty*TILEDIM_K + k] * bs_reg1;
                _cij[1] += As[(ty+8)*TILEDIM_K + k] * bs_reg1;
                _cij[2] += As[(ty+16)*TILEDIM_K + k] * bs_reg1;
                _cij[3] += As[(ty+24)*TILEDIM_K + k] * bs_reg1;
                _cij[4] += As[(ty+32)*TILEDIM_K + k] * bs_reg1;
                _cij[5] += As[(ty+40)*TILEDIM_K + k] * bs_reg1;
                _cij[6] += As[(ty+48)*TILEDIM_K + k] * bs_reg1;
                _cij[7] += As[(ty+56)*TILEDIM_K + k] * bs_reg1;
                _cij[8] += As[(ty+64)*TILEDIM_K + k] * bs_reg1;
                _cij[9] += As[(ty+72)*TILEDIM_K + k] * bs_reg1;
                _cij[10] += As[(ty+80)*TILEDIM_K + k] * bs_reg1;
                _cij[11] += As[(ty+88)*TILEDIM_K + k] * bs_reg1;
                _cij[12] += As[(ty+96)*TILEDIM_K + k] * bs_reg1;
                _cij[13] += As[(ty+104)*TILEDIM_K + k] * bs_reg1;
                _cij[14] += As[(ty+112)*TILEDIM_K + k] * bs_reg1;
                _cij[15] += As[(ty+120)*TILEDIM_K + k] * bs_reg1;

                bs_reg2 = Bs[k*TILEDIM_N + (tx+32)];
                _cij[16] += As[(ty)*TILEDIM_K + k] * bs_reg2;
                _cij[17] += As[(ty+8)*TILEDIM_K + k] * bs_reg2;
                _cij[18] += As[(ty+16)*TILEDIM_K + k] * bs_reg2;
                _cij[19] += As[(ty+24)*TILEDIM_K + k] * bs_reg2;
                _cij[20] += As[(ty+32)*TILEDIM_K + k] * bs_reg2;
                _cij[21] += As[(ty+40)*TILEDIM_K + k] * bs_reg2;
                _cij[22] += As[(ty+48)*TILEDIM_K + k] * bs_reg2;
                _cij[23] += As[(ty+56)*TILEDIM_K + k] * bs_reg2;
                _cij[24] += As[(ty+64)*TILEDIM_K + k] * bs_reg2;
                _cij[25] += As[(ty+72)*TILEDIM_K + k] * bs_reg2;
                _cij[26] += As[(ty+80)*TILEDIM_K + k] * bs_reg2;
                _cij[27] += As[(ty+88)*TILEDIM_K + k] * bs_reg2;
                _cij[28] += As[(ty+96)*TILEDIM_K + k] * bs_reg2;
                _cij[29] += As[(ty+104)*TILEDIM_K + k] * bs_reg2;
                _cij[30] += As[(ty+112)*TILEDIM_K + k] * bs_reg2;
                _cij[31] += As[(ty+120)*TILEDIM_K + k] * bs_reg2;
                
            }                
            __syncthreads();
        }
        int c = N*TILEDIM_M*by + TILEDIM_N*bx;
        if(N%TILEDIM_M ==0){
            C[c + N * ty + tx] = _cij[0];
            C[c + N * ty + (tx+32)] = _cij[16];

            C[c + N * (ty+8) + tx] = _cij[1];
            C[c + N * (ty+8) + (tx+32)] = _cij[17];

            C[c + N * (ty+16) + tx] = _cij[2];
            C[c + N * (ty+16) + (tx+32)] = _cij[18];

            C[c + N * (ty+24) + tx] = _cij[3];
            C[c + N * (ty+24) + (tx+32)] = _cij[19];    

            C[c + N * (ty+32) + tx] = _cij[4];
            C[c + N * (ty+32) + (tx+32)] = _cij[20];    

            C[c + N * (ty+40) + tx] = _cij[5];
            C[c + N * (ty+40) + (tx+32)] = _cij[21];    

            C[c + N * (ty+48) + tx] = _cij[6];
            C[c + N * (ty+48) + (tx+32)] = _cij[22];    

            C[c + N * (ty+56) + tx] = _cij[7];
            C[c + N * (ty+56) + (tx+32)] = _cij[23];    

            C[c + N * (ty+64) + tx] = _cij[8];
            C[c + N * (ty+64) + (tx+32)] = _cij[24];    

            C[c + N * (ty+72) + tx] = _cij[9];
            C[c + N * (ty+72) + (tx+32)] = _cij[25];    

            C[c + N * (ty+80) + tx] = _cij[10];
            C[c + N * (ty+80) + (tx+32)] = _cij[26];    

            C[c + N * (ty+88) + tx] = _cij[11];
            C[c + N * (ty+88) + (tx+32)] = _cij[27];    

            C[c + N * (ty+96) + tx] = _cij[12];
            C[c + N * (ty+96) + (tx+32)] = _cij[28];    

            C[c + N * (ty+104) + tx] = _cij[13];
            C[c + N * (ty+104) + (tx+32)] = _cij[29];    

            C[c + N * (ty+112) + tx] = _cij[14];
            C[c + N * (ty+112) + (tx+32)] = _cij[30];    

            C[c + N * (ty+120) + tx] = _cij[15];
            C[c + N * (ty+120) + (tx+32)] = _cij[31];    
        } else{
            count = 0;
            for(int itx=tx; itx<64; itx+=32){
                for(int ity=ty; ity<128; ity+=8){
                    if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
                        C[c + N * ity + itx] = _cij[count];
                    }
                    count++;
                }
            }
        }
    }
}

// bx=32,by=8,TILEDIM_M=128,TILEDIM_K=64,TILEDIM_N=64,32 ops per thread, NOT interleaved access
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[32] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             if(N%TILEDIM_M == 0){
//                 As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//                 Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];

//                 As[(ty+8)*TILEDIM_K + tx] = A[a + (ty+8)*N + tx];
//                 Bs[(ty+8)*TILEDIM_N + tx] = B[b + (ty+8)*N + tx];

//                 As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
//                 Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
                
//                 As[(ty+24)*TILEDIM_K + tx] = A[a + (ty+24)*N + tx];
//                 Bs[(ty+24)*TILEDIM_N + tx] = B[b + (ty+24)*N + tx];
                
//                 As[(ty+32)*TILEDIM_K + tx] = A[a + (ty+32)*N + tx];
//                 Bs[(ty+32)*TILEDIM_N + tx] = B[b + (ty+32)*N + tx];
                
//                 As[(ty+40)*TILEDIM_K + tx] = A[a + (ty+40)*N + tx];
//                 Bs[(ty+40)*TILEDIM_N + tx] = B[b + (ty+40)*N + tx];
                
//                 As[(ty+48)*TILEDIM_K + tx] = A[a + (ty+48)*N + tx];
//                 Bs[(ty+48)*TILEDIM_N + tx] = B[b + (ty+48)*N + tx];
                
//                 As[(ty+56)*TILEDIM_K + tx] = A[a + (ty+56)*N + tx];
//                 Bs[(ty+56)*TILEDIM_N + tx] = B[b + (ty+56)*N + tx];
                
//                 As[(ty+64)*TILEDIM_K + tx] = A[a + (ty+64)*N + tx];
//                 Bs[(ty+64)*TILEDIM_N + tx] = B[b + (ty+64)*N + tx];
                
//                 As[(ty+72)*TILEDIM_K + tx] = A[a + (ty+72)*N + tx];
//                 Bs[(ty+72)*TILEDIM_N + tx] = B[b + (ty+72)*N + tx];
                
//                 As[(ty+80)*TILEDIM_K + tx] = A[a + (ty+80)*N + tx];
//                 Bs[(ty+80)*TILEDIM_N + tx] = B[b + (ty+80)*N + tx];
                
//                 As[(ty+88)*TILEDIM_K + tx] = A[a + (ty+88)*N + tx];
//                 Bs[(ty+88)*TILEDIM_N + tx] = B[b + (ty+88)*N + tx];
                
//                 As[(ty+96)*TILEDIM_K + tx] = A[a + (ty+96)*N + tx];
//                 Bs[(ty+96)*TILEDIM_N + tx] = B[b + (ty+96)*N + tx];
                
//                 As[(ty+104)*TILEDIM_K + tx] = A[a + (ty+104)*N + tx];
//                 Bs[(ty+104)*TILEDIM_N + tx] = B[b + (ty+104)*N + tx];
                
//                 As[(ty+112)*TILEDIM_K + tx] = A[a + (ty+112)*N + tx];
//                 Bs[(ty+112)*TILEDIM_N + tx] = B[b + (ty+112)*N + tx];
                
//                 As[(ty+120)*TILEDIM_K + tx] = A[a + (ty+120)*N + tx];
//                 Bs[(ty+120)*TILEDIM_N + tx] = B[b + (ty+120)*N + tx];
                
//                 As[ty*TILEDIM_K + (tx+32)] = A[a + ty*N + (tx+32)];
//                 Bs[ty*TILEDIM_N + (tx+32)] = B[b + ty*N + (tx+32)];
                
//                 As[(ty+8)*TILEDIM_K + (tx+32)] = A[a + (ty+8)*N + (tx+32)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+32)] = B[b + (ty+8)*N + (tx+32)];
                
//                 As[(ty+16)*TILEDIM_K + (tx+32)] = A[a + (ty+16)*N + (tx+32)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+32)] = B[b + (ty+16)*N + (tx+32)];
                
//                 As[(ty+24)*TILEDIM_K + (tx+32)] = A[a + (ty+24)*N + (tx+32)];
//                 Bs[(ty+24)*TILEDIM_N + (tx+32)] = B[b + (ty+24)*N + (tx+32)];                        
                
//                 As[(ty+32)*TILEDIM_K + (tx+32)] = A[a + (ty+32)*N + (tx+32)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+32)] = B[b + (ty+32)*N + (tx+32)];
                
//                 As[(ty+40)*TILEDIM_K + (tx+32)] = A[a + (ty+40)*N + (tx+32)];
//                 Bs[(ty+40)*TILEDIM_N + (tx+32)] = B[b + (ty+40)*N + (tx+32)];
                
//                 As[(ty+48)*TILEDIM_K + (tx+32)] = A[a + (ty+48)*N + (tx+32)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+32)] = B[b + (ty+48)*N + (tx+32)];
                
//                 As[(ty+56)*TILEDIM_K + (tx+32)] = A[a + (ty+56)*N + (tx+32)];
//                 Bs[(ty+56)*TILEDIM_N + (tx+32)] = B[b + (ty+56)*N + (tx+32)];
                
//                 As[(ty+64)*TILEDIM_K + (tx+32)] = A[a + (ty+64)*N + (tx+32)];
//                 Bs[(ty+64)*TILEDIM_N + (tx+32)] = B[b + (ty+64)*N + (tx+32)];
                
//                 As[(ty+72)*TILEDIM_K + (tx+32)] = A[a + (ty+72)*N + (tx+32)];
//                 Bs[(ty+72)*TILEDIM_N + (tx+32)] = B[b + (ty+72)*N + (tx+32)];
                
//                 As[(ty+80)*TILEDIM_K + (tx+32)] = A[a + (ty+80)*N + (tx+32)]; 
//                 Bs[(ty+80)*TILEDIM_N + (tx+32)] = B[b + (ty+80)*N + (tx+32)];
                
//                 As[(ty+88)*TILEDIM_K + (tx+32)] = A[a + (ty+88)*N + (tx+32)];
//                 Bs[(ty+88)*TILEDIM_N + (tx+32)] = B[b + (ty+88)*N + (tx+32)];
                
//                 As[(ty+96)*TILEDIM_K + (tx+32)] = A[a + (ty+96)*N + (tx+32)]; 
//                 Bs[(ty+96)*TILEDIM_N + (tx+32)] = B[b + (ty+96)*N + (tx+32)];             
                
//                 As[(ty+104)*TILEDIM_K + (tx+32)] = A[a + (ty+104)*N + (tx+32)];
//                 Bs[(ty+104)*TILEDIM_N + (tx+32)] = B[b + (ty+104)*N + (tx+32)];
                
//                 As[(ty+112)*TILEDIM_K + (tx+32)] = A[a + (ty+112)*N + (tx+32)];
//                 Bs[(ty+120)*TILEDIM_N + (tx+32)] = B[b + (ty+120)*N + (tx+32)];

//                 As[(ty+120)*TILEDIM_K + (tx+32)] = A[a + (ty+120)*N + (tx+32)];
//                 Bs[(ty+112)*TILEDIM_N + (tx+32)] = B[b + (ty+112)*N + (tx+32)];
//             } else{
//                 for(int itx=tx; itx<64; itx+=32){
//                     for(int ity=ty; ity<128; ity+=8){
//                         if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
//                             As[ity*TILEDIM_K + itx] = 0;
//                         } else{
//                             As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
//                         }
//                     }
//                 }
//                 for(int itx=tx; itx<64; itx+=32){
//                     for(int ity=ty; ity<128; ity+=8){
//                         if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
//                             Bs[ity*TILEDIM_N + itx] = 0;
//                         } else{
//                             Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
//                         }
//                     }
//                 }
//             }
             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij[0] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[1] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[2] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[3] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[4] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[5] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[6] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[7] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[8] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[9] += As[(ty+72)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[10] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[11] += As[(ty+88)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[12] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[13] += As[(ty+104)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[14] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[15] += As[(ty+120)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];

//                 _cij[16] += As[(ty)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[17] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[18] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[19] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[20] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[21] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[22] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[23] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[24] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[25] += As[(ty+72)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[26] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[27] += As[(ty+88)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[28] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[29] += As[(ty+104)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[30] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[31] += As[(ty+120)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
                
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         if(N%TILEDIM_M ==0){
//             C[c + N * ty + tx] = _cij[0];
//             C[c + N * (ty+8) + tx] = _cij[1];
//             C[c + N * (ty+16) + tx] = _cij[2];
//             C[c + N * (ty+24) + tx] = _cij[3];
//             C[c + N * (ty+32) + tx] = _cij[4];
//             C[c + N * (ty+40) + tx] = _cij[5];
//             C[c + N * (ty+48) + tx] = _cij[6];
//             C[c + N * (ty+56) + tx] = _cij[7];
//             C[c + N * (ty+64) + tx] = _cij[8];
//             C[c + N * (ty+72) + tx] = _cij[9];
//             C[c + N * (ty+80) + tx] = _cij[10];
//             C[c + N * (ty+88) + tx] = _cij[11];
//             C[c + N * (ty+96) + tx] = _cij[12];
//             C[c + N * (ty+104) + tx] = _cij[13];
//             C[c + N * (ty+112) + tx] = _cij[14];
//             C[c + N * (ty+120) + tx] = _cij[15];

//             C[c + N * ty + (tx+32)] = _cij[16];
//             C[c + N * (ty+8) + (tx+32)] = _cij[17];
//             C[c + N * (ty+16) + (tx+32)] = _cij[18];
//             C[c + N * (ty+24) + (tx+32)] = _cij[19];    
//             C[c + N * (ty+32) + (tx+32)] = _cij[20];    
//             C[c + N * (ty+40) + (tx+32)] = _cij[21];    
//             C[c + N * (ty+48) + (tx+32)] = _cij[22];    
//             C[c + N * (ty+56) + (tx+32)] = _cij[23];    
//             C[c + N * (ty+64) + (tx+32)] = _cij[24];    
//             C[c + N * (ty+72) + (tx+32)] = _cij[25];    
//             C[c + N * (ty+80) + (tx+32)] = _cij[26];    
//             C[c + N * (ty+88) + (tx+32)] = _cij[27];    
//             C[c + N * (ty+96) + (tx+32)] = _cij[28];    
//             C[c + N * (ty+104) + (tx+32)] = _cij[29];    
//             C[c + N * (ty+112) + (tx+32)] = _cij[30];    
//             C[c + N * (ty+120) + (tx+32)] = _cij[31];    
//         } else{
//             int count = 0;
//             for(int itx=tx; itx<64; itx+=32){
//                 for(int ity=ty; ity<128; ity+=8){
//                     if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
//                         C[c + N * ity + itx] = _cij[count];
//                     }
//                     count++;
//                 }
//             }
//         }
            
//     }
// }

// bx=16,by=8,TILEDIM_M=64,TILEDIM_K=64,TILEDIM_N=64,32 ops per thread, interleaved access
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[32] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             if(N%TILEDIM_M == 0){
//                 As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//                 As[ty*TILEDIM_K + (tx+16)] = A[a + ty*N + (tx+16)];
//                 As[ty*TILEDIM_K + (tx+32)] = A[a + ty*N + (tx+32)];
//                 As[ty*TILEDIM_K + (tx+48)] = A[a + ty*N + (tx+48)];

//                 As[(ty+8)*TILEDIM_K + tx] = A[a + (ty+8)*N + tx];
//                 As[(ty+8)*TILEDIM_K + (tx+16)] = A[a + (ty+8)*N + (tx+16)];
//                 As[(ty+8)*TILEDIM_K + (tx+32)] = A[a + (ty+8)*N + (tx+32)];
//                 As[(ty+8)*TILEDIM_K + (tx+48)] = A[a + (ty+8)*N + (tx+48)];

//                 As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
//                 As[(ty+16)*TILEDIM_K + (tx+16)] = A[a + (ty+16)*N + (tx+16)];
//                 As[(ty+16)*TILEDIM_K + (tx+32)] = A[a + (ty+16)*N + (tx+32)];
//                 As[(ty+16)*TILEDIM_K + (tx+48)] = A[a + (ty+16)*N + (tx+48)];

//                 As[(ty+24)*TILEDIM_K + tx] = A[a + (ty+24)*N + tx];
//                 As[(ty+24)*TILEDIM_K + (tx+16)] = A[a + (ty+24)*N + (tx+16)];
//                 As[(ty+24)*TILEDIM_K + (tx+32)] = A[a + (ty+24)*N + (tx+32)];
//                 As[(ty+24)*TILEDIM_K + (tx+48)] = A[a + (ty+24)*N + (tx+48)];

//                 As[(ty+32)*TILEDIM_K + tx] = A[a + (ty+32)*N + tx];
//                 As[(ty+32)*TILEDIM_K + (tx+16)] = A[a + (ty+32)*N + (tx+16)];
//                 As[(ty+32)*TILEDIM_K + (tx+32)] = A[a + (ty+32)*N + (tx+32)];
//                 As[(ty+32)*TILEDIM_K + (tx+48)] = A[a + (ty+32)*N + (tx+48)];

//                 As[(ty+40)*TILEDIM_K + tx] = A[a + (ty+40)*N + tx];
//                 As[(ty+40)*TILEDIM_K + (tx+16)] = A[a + (ty+40)*N + (tx+16)];
//                 As[(ty+40)*TILEDIM_K + (tx+32)] = A[a + (ty+40)*N + (tx+32)];
//                 As[(ty+40)*TILEDIM_K + (tx+48)] = A[a + (ty+40)*N + (tx+48)];
                
//                 As[(ty+48)*TILEDIM_K + tx] = A[a + (ty+48)*N + tx];
//                 As[(ty+48)*TILEDIM_K + (tx+16)] = A[a + (ty+48)*N + (tx+16)];
//                 As[(ty+48)*TILEDIM_K + (tx+32)] = A[a + (ty+48)*N + (tx+32)];
//                 As[(ty+48)*TILEDIM_K + (tx+48)] = A[a + (ty+48)*N + (tx+48)];
                
//                 As[(ty+56)*TILEDIM_K + tx] = A[a + (ty+56)*N + tx];
//                 As[(ty+56)*TILEDIM_K + (tx+16)] = A[a + (ty+56)*N + (tx+16)];
//                 As[(ty+56)*TILEDIM_K + (tx+32)] = A[a + (ty+56)*N + (tx+32)];
//                 As[(ty+56)*TILEDIM_K + (tx+48)] = A[a + (ty+56)*N + (tx+48)];
                
//                 Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];
//                 Bs[ty*TILEDIM_N + (tx+16)] = B[b + ty*N + (tx+16)];
//                 Bs[ty*TILEDIM_N + (tx+32)] = B[b + ty*N + (tx+32)];
//                 Bs[ty*TILEDIM_N + (tx+48)] = B[b + ty*N + (tx+48)];

//                 Bs[(ty+8)*TILEDIM_N + tx] = B[b + (ty+8)*N + tx];
//                 Bs[(ty+8)*TILEDIM_N + (tx+16)] = B[b + (ty+8)*N + (tx+16)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+32)] = B[b + (ty+8)*N + (tx+32)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+48)] = B[b + (ty+8)*N + (tx+48)];

//                 Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
//                 Bs[(ty+16)*TILEDIM_N + (tx+16)] = B[b + (ty+16)*N + (tx+16)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+32)] = B[b + (ty+16)*N + (tx+32)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+48)] = B[b + (ty+16)*N + (tx+48)];

//                 Bs[(ty+24)*TILEDIM_N + tx] = B[b + (ty+24)*N + tx];
//                 Bs[(ty+24)*TILEDIM_N + (tx+16)] = B[b + (ty+24)*N + (tx+16)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+32)] = B[b + (ty+24)*N + (tx+32)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+48)] = B[b + (ty+24)*N + (tx+48)];                        

//                 Bs[(ty+32)*TILEDIM_N + tx] = B[b + (ty+32)*N + tx];
//                 Bs[(ty+32)*TILEDIM_N + (tx+16)] = B[b + (ty+32)*N + (tx+16)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+32)] = B[b + (ty+32)*N + (tx+32)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+48)] = B[b + (ty+32)*N + (tx+48)];

//                 Bs[(ty+40)*TILEDIM_N + tx] = B[b + (ty+40)*N + tx];
//                 Bs[(ty+40)*TILEDIM_N + (tx+16)] = B[b + (ty+40)*N + (tx+16)];
//                 Bs[(ty+40)*TILEDIM_N + (tx+32)] = B[b + (ty+40)*N + (tx+32)];
//                 Bs[(ty+40)*TILEDIM_N + (tx+48)] = B[b + (ty+40)*N + (tx+48)];

//                 Bs[(ty+48)*TILEDIM_N + tx] = B[b + (ty+48)*N + tx];
//                 Bs[(ty+48)*TILEDIM_N + (tx+16)] = B[b + (ty+48)*N + (tx+16)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+32)] = B[b + (ty+48)*N + (tx+32)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+48)] = B[b + (ty+48)*N + (tx+48)];

//                 Bs[(ty+56)*TILEDIM_N + tx] = B[b + (ty+56)*N + tx];
//                 Bs[(ty+56)*TILEDIM_N + (tx+16)] = B[b + (ty+56)*N + (tx+16)];
//                 Bs[(ty+56)*TILEDIM_N + (tx+32)] = B[b + (ty+56)*N + (tx+32)];
//                 Bs[(ty+56)*TILEDIM_N + (tx+48)] = B[b + (ty+56)*N + (tx+48)];
//             } else{
//                 for(int itx=tx; itx<64; itx+=16){
//                     for(int ity=ty; ity<64; ity+=8){
//                         if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
//                             As[ity*TILEDIM_K + itx] = 0;
//                         } else{
//                             As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
//                         }
//                     }
//                 }
//                 for(int itx=tx; itx<64; itx+=16){
//                     for(int ity=ty; ity<64; ity+=8){
//                         if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
//                             Bs[ity*TILEDIM_N + itx] = 0;
//                         } else{
//                             Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
//                         }
//                     }
//                 }
//             }
             
//             __syncthreads();     
//             register _FTYPE_ bs_reg1, bs_reg2, bs_reg3, bs_reg4;   
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 bs_reg1 = Bs[k*TILEDIM_N + tx];
//                 _cij[0] += As[ty*TILEDIM_K + k] * bs_reg1;
//                 _cij[1] += As[(ty+8)*TILEDIM_K + k] * bs_reg1;
//                 _cij[2] += As[(ty+16)*TILEDIM_K + k] * bs_reg1;
//                 _cij[3] += As[(ty+24)*TILEDIM_K + k] * bs_reg1;
//                 _cij[4] += As[(ty+32)*TILEDIM_K + k] * bs_reg1;
//                 _cij[5] += As[(ty+40)*TILEDIM_K + k] * bs_reg1;
//                 _cij[6] += As[(ty+48)*TILEDIM_K + k] * bs_reg1;
//                 _cij[7] += As[(ty+56)*TILEDIM_K + k] * bs_reg1;

//                 bs_reg2 = Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[8] += As[ty*TILEDIM_K + k] * bs_reg2;
//                 _cij[9] += As[(ty+8)*TILEDIM_K + k] * bs_reg2;
//                 _cij[10] += As[(ty+16)*TILEDIM_K + k] * bs_reg2;
//                 _cij[11] += As[(ty+24)*TILEDIM_K + k] * bs_reg2;
//                 _cij[12] += As[(ty+32)*TILEDIM_K + k] * bs_reg2;
//                 _cij[13] += As[(ty+40)*TILEDIM_K + k] * bs_reg2;
//                 _cij[14] += As[(ty+48)*TILEDIM_K + k] * bs_reg2;
//                 _cij[15] += As[(ty+56)*TILEDIM_K + k] * bs_reg2;

//                 bs_reg3 = Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[16] += As[(ty)*TILEDIM_K + k] * bs_reg3;
//                 _cij[17] += As[(ty+8)*TILEDIM_K + k] * bs_reg3;
//                 _cij[18] += As[(ty+16)*TILEDIM_K + k] * bs_reg3;
//                 _cij[19] += As[(ty+24)*TILEDIM_K + k] * bs_reg3;
//                 _cij[20] += As[(ty+32)*TILEDIM_K + k] * bs_reg3;
//                 _cij[21] += As[(ty+40)*TILEDIM_K + k] * bs_reg3;
//                 _cij[22] += As[(ty+48)*TILEDIM_K + k] * bs_reg3;
//                 _cij[23] += As[(ty+56)*TILEDIM_K + k] * bs_reg3;
                
//                 bs_reg4 = Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[24] += As[(ty)*TILEDIM_K + k] * bs_reg4;
//                 _cij[25] += As[(ty+8)*TILEDIM_K + k] * bs_reg4;
//                 _cij[26] += As[(ty+16)*TILEDIM_K + k] * bs_reg4;
//                 _cij[27] += As[(ty+24)*TILEDIM_K + k] * bs_reg4;
//                 _cij[28] += As[(ty+32)*TILEDIM_K + k] * bs_reg4;
//                 _cij[29] += As[(ty+40)*TILEDIM_K + k] * bs_reg4;
//                 _cij[30] += As[(ty+48)*TILEDIM_K + k] * bs_reg4;
//                 _cij[31] += As[(ty+56)*TILEDIM_K + k] * bs_reg4;
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         if(N%TILEDIM_M ==0){
//             C[c + N * ty + tx] = _cij[0];
//             C[c + N * (ty+8) + tx] = _cij[1];
//             C[c + N * (ty+16) + tx] = _cij[2];
//             C[c + N * (ty+24) + tx] = _cij[3];
//             C[c + N * (ty+32) + tx] = _cij[4];
//             C[c + N * (ty+40) + tx] = _cij[5];
//             C[c + N * (ty+48) + tx] = _cij[6];
//             C[c + N * (ty+56) + tx] = _cij[7];

//             C[c + N * ty + (tx+16)] = _cij[8];
//             C[c + N * (ty+8) + (tx+16)] = _cij[9];
//             C[c + N * (ty+16) + (tx+16)] = _cij[10];
//             C[c + N * (ty+24) + (tx+16)] = _cij[11];    
//             C[c + N * (ty+32) + (tx+16)] = _cij[12];    
//             C[c + N * (ty+40) + (tx+16)] = _cij[13];    
//             C[c + N * (ty+48) + (tx+16)] = _cij[14];    
//             C[c + N * (ty+56) + (tx+16)] = _cij[15];    

//             C[c + N * ty + (tx+32)] = _cij[16];
//             C[c + N * (ty+8) + (tx+32)] = _cij[17];
//             C[c + N * (ty+16) + (tx+32)] = _cij[18];
//             C[c + N * (ty+24) + (tx+32)] = _cij[19];    
//             C[c + N * (ty+32) + (tx+32)] = _cij[20];    
//             C[c + N * (ty+40) + (tx+32)] = _cij[21];    
//             C[c + N * (ty+48) + (tx+32)] = _cij[22];    
//             C[c + N * (ty+56) + (tx+32)] = _cij[23];    

//             C[c + N * ty + (tx+48)] = _cij[24];
//             C[c + N * (ty+8) + (tx+48)] = _cij[25];
//             C[c + N * (ty+16) + (tx+48)] = _cij[26];
//             C[c + N * (ty+24) + (tx+48)] = _cij[27];    
//             C[c + N * (ty+32) + (tx+48)] = _cij[28];    
//             C[c + N * (ty+40) + (tx+48)] = _cij[29];    
//             C[c + N * (ty+48) + (tx+48)] = _cij[30];    
//             C[c + N * (ty+56) + (tx+48)] = _cij[31];    
//         } else{
//             int count = 0;
//             for(int itx=tx; itx<64; itx+=16){
//                 for(int ity=ty; ity<64; ity+=8){
//                     if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
//                         C[c + N * ity + itx] = _cij[count];
//                     }
//                     count++;
//                 }
//             }
//         }
            
//     }
// }

// bx=16,by=16,TILEDIM_M=128,TILEDIM_K=64,TILEDIM_N=64,32 ops per thread, interleaved access
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[32] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             if(N%TILEDIM_M == 0){
//                 As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//                 As[ty*TILEDIM_K + (tx+16)] = A[a + ty*N + (tx+16)];
//                 As[ty*TILEDIM_K + (tx+32)] = A[a + ty*N + (tx+32)];
//                 As[ty*TILEDIM_K + (tx+48)] = A[a + ty*N + (tx+48)];

//                 As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
//                 As[(ty+16)*TILEDIM_K + (tx+16)] = A[a + (ty+16)*N + (tx+16)];
//                 As[(ty+16)*TILEDIM_K + (tx+32)] = A[a + (ty+16)*N + (tx+32)];
//                 As[(ty+16)*TILEDIM_K + (tx+48)] = A[a + (ty+16)*N + (tx+48)];

//                 As[(ty+32)*TILEDIM_K + tx] = A[a + (ty+32)*N + tx];
//                 As[(ty+32)*TILEDIM_K + (tx+16)] = A[a + (ty+32)*N + (tx+16)];
//                 As[(ty+32)*TILEDIM_K + (tx+32)] = A[a + (ty+32)*N + (tx+32)];
//                 As[(ty+32)*TILEDIM_K + (tx+48)] = A[a + (ty+32)*N + (tx+48)];

//                 As[(ty+48)*TILEDIM_K + tx] = A[a + (ty+48)*N + tx];
//                 As[(ty+48)*TILEDIM_K + (tx+16)] = A[a + (ty+48)*N + (tx+16)];
//                 As[(ty+48)*TILEDIM_K + (tx+32)] = A[a + (ty+48)*N + (tx+32)];
//                 As[(ty+48)*TILEDIM_K + (tx+48)] = A[a + (ty+48)*N + (tx+48)];
                
//                 As[(ty+64)*TILEDIM_K + tx] = A[a + (ty+64)*N + tx];
//                 As[(ty+64)*TILEDIM_K + (tx+16)] = A[a + (ty+64)*N + (tx+16)];
//                 As[(ty+64)*TILEDIM_K + (tx+32)] = A[a + (ty+64)*N + (tx+32)];
//                 As[(ty+64)*TILEDIM_K + (tx+48)] = A[a + (ty+64)*N + (tx+48)];

//                 As[(ty+80)*TILEDIM_K + tx] = A[a + (ty+80)*N + tx];
//                 As[(ty+80)*TILEDIM_K + (tx+16)] = A[a + (ty+80)*N + (tx+16)];
//                 As[(ty+80)*TILEDIM_K + (tx+32)] = A[a + (ty+80)*N + (tx+32)];
//                 As[(ty+80)*TILEDIM_K + (tx+48)] = A[a + (ty+80)*N + (tx+48)];

//                 As[(ty+96)*TILEDIM_K + tx] = A[a + (ty+96)*N + tx];
//                 As[(ty+96)*TILEDIM_K + (tx+16)] = A[a + (ty+96)*N + (tx+16)];
//                 As[(ty+96)*TILEDIM_K + (tx+32)] = A[a + (ty+96)*N + (tx+32)];
//                 As[(ty+96)*TILEDIM_K + (tx+48)] = A[a + (ty+96)*N + (tx+48)];

//                 As[(ty+112)*TILEDIM_K + tx] = A[a + (ty+112)*N + tx];
//                 As[(ty+112)*TILEDIM_K + (tx+16)] = A[a + (ty+112)*N + (tx+16)];
//                 As[(ty+112)*TILEDIM_K + (tx+32)] = A[a + (ty+112)*N + (tx+32)];
//                 As[(ty+112)*TILEDIM_K + (tx+48)] = A[a + (ty+112)*N + (tx+48)];

//                 Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];
//                 Bs[ty*TILEDIM_N + (tx+16)] = B[b + ty*N + (tx+16)];
//                 Bs[ty*TILEDIM_N + (tx+32)] = B[b + ty*N + (tx+32)];
//                 Bs[ty*TILEDIM_N + (tx+48)] = B[b + ty*N + (tx+48)];

//                 Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
//                 Bs[(ty+16)*TILEDIM_N + (tx+16)] = B[b + (ty+16)*N + (tx+16)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+32)] = B[b + (ty+16)*N + (tx+32)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+48)] = B[b + (ty+16)*N + (tx+48)];

//                 Bs[(ty+32)*TILEDIM_N + tx] = B[b + (ty+32)*N + tx];
//                 Bs[(ty+32)*TILEDIM_N + (tx+16)] = B[b + (ty+32)*N + (tx+16)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+32)] = B[b + (ty+32)*N + (tx+32)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+48)] = B[b + (ty+32)*N + (tx+48)];

//                 Bs[(ty+48)*TILEDIM_N + tx] = B[b + (ty+48)*N + tx];
//                 Bs[(ty+48)*TILEDIM_N + (tx+16)] = B[b + (ty+48)*N + (tx+16)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+32)] = B[b + (ty+48)*N + (tx+32)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+48)] = B[b + (ty+48)*N + (tx+48)];

//                 Bs[(ty+64)*TILEDIM_N + tx] = B[b + (ty+64)*N + tx];
//                 Bs[(ty+64)*TILEDIM_N + (tx+16)] = B[b + (ty+64)*N + (tx+16)];
//                 Bs[(ty+64)*TILEDIM_N + (tx+32)] = B[b + (ty+64)*N + (tx+32)];
//                 Bs[(ty+64)*TILEDIM_N + (tx+48)] = B[b + (ty+64)*N + (tx+48)];

//                 Bs[(ty+80)*TILEDIM_N + tx] = B[b + (ty+80)*N + tx];
//                 Bs[(ty+80)*TILEDIM_N + (tx+16)] = B[b + (ty+80)*N + (tx+16)];
//                 Bs[(ty+80)*TILEDIM_N + (tx+32)] = B[b + (ty+80)*N + (tx+32)];
//                 Bs[(ty+80)*TILEDIM_N + (tx+48)] = B[b + (ty+80)*N + (tx+48)];

//                 Bs[(ty+96)*TILEDIM_N + tx] = B[b + (ty+96)*N + tx];
//                 Bs[(ty+96)*TILEDIM_N + (tx+16)] = B[b + (ty+96)*N + (tx+16)];
//                 Bs[(ty+96)*TILEDIM_N + (tx+32)] = B[b + (ty+96)*N + (tx+32)];
//                 Bs[(ty+96)*TILEDIM_N + (tx+48)] = B[b + (ty+96)*N + (tx+48)];

//                 Bs[(ty+112)*TILEDIM_N + tx] = B[b + (ty+112)*N + tx];
//                 Bs[(ty+112)*TILEDIM_N + (tx+16)] = B[b + (ty+112)*N + (tx+16)];
//                 Bs[(ty+112)*TILEDIM_N + (tx+32)] = B[b + (ty+112)*N + (tx+32)];
//                 Bs[(ty+112)*TILEDIM_N + (tx+48)] = B[b + (ty+112)*N + (tx+48)];
//             } else{
//                 for(int itx=tx; itx<64; itx+=16){
//                     for(int ity=ty; ity<128; ity+=16){
//                         if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
//                             As[ity*TILEDIM_K + itx] = 0;
//                         } else{
//                             As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
//                         }
//                     }
//                 }
//                 for(int itx=tx; itx<64; itx+=16){
//                     for(int ity=ty; ity<128; ity+=16){
//                         if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
//                             Bs[ity*TILEDIM_N + itx] = 0;
//                         } else{
//                             Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
//                         }
//                     }
//                 }
//             }
             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij[0] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[1] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[2] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[3] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[4] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[5] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[6] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[7] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];

//                 _cij[8] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[9] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[10] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[11] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[12] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[13] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[14] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[15] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];

//                 _cij[16] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[17] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[18] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[19] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[20] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[21] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[22] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];
//                 _cij[23] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+32)];

//                 _cij[24] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[25] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[26] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[27] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[28] += As[(ty+64)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[29] += As[(ty+80)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[30] += As[(ty+96)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//                 _cij[31] += As[(ty+112)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+48)];
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         if(N%TILEDIM_M ==0){
//             C[c + N * ty + tx] = _cij[0];
//             C[c + N * (ty+16) + tx] = _cij[1];
//             C[c + N * (ty+32) + tx] = _cij[2];
//             C[c + N * (ty+48) + tx] = _cij[3];
//             C[c + N * (ty+64) + tx] = _cij[4];
//             C[c + N * (ty+80) + tx] = _cij[5];
//             C[c + N * (ty+96) + tx] = _cij[6];
//             C[c + N * (ty+112) + tx] = _cij[7];

//             C[c + N * ty + (tx+16)] = _cij[8];
//             C[c + N * (ty+16) + (tx+16)] = _cij[9];
//             C[c + N * (ty+32) + (tx+16)] = _cij[10];    
//             C[c + N * (ty+48) + (tx+16)] = _cij[11];    
//             C[c + N * (ty+64) + (tx+16)] = _cij[12];    
//             C[c + N * (ty+80) + (tx+16)] = _cij[13];    
//             C[c + N * (ty+96) + (tx+16)] = _cij[14];    
//             C[c + N * (ty+112) + (tx+16)] = _cij[15];    

//             C[c + N * ty + (tx+32)] = _cij[16];
//             C[c + N * (ty+16) + (tx+32)] = _cij[17];
//             C[c + N * (ty+32) + (tx+32)] = _cij[18];    
//             C[c + N * (ty+48) + (tx+32)] = _cij[19];    
//             C[c + N * (ty+64) + (tx+32)] = _cij[20];    
//             C[c + N * (ty+80) + (tx+32)] = _cij[21];    
//             C[c + N * (ty+96) + (tx+32)] = _cij[22];    
//             C[c + N * (ty+112) + (tx+32)] = _cij[23];    

//             C[c + N * ty + (tx+48)] = _cij[24];
//             C[c + N * (ty+16) + (tx+48)] = _cij[25];
//             C[c + N * (ty+32) + (tx+48)] = _cij[26];    
//             C[c + N * (ty+48) + (tx+48)] = _cij[27];    
//             C[c + N * (ty+64) + (tx+48)] = _cij[28];    
//             C[c + N * (ty+80) + (tx+48)] = _cij[29];    
//             C[c + N * (ty+96) + (tx+48)] = _cij[30];    
//             C[c + N * (ty+112) + (tx+48)] = _cij[31];    

//         } else{
//             int count = 0;
//             for(int itx=tx; itx<64; itx+=16){
//                 for(int ity=ty; ity<128; ity+=16){
//                     if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
//                         C[c + N * ity + itx] = _cij[count];
//                     }
//                     count++;
//                 }
//             }
//         }
            
//     }
// }

// bx=8,by=8,TILEDIM_M=64,TILEDIM_K=32,TILEDIM_N=32,32 ops per thread, interleaved access
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[32] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             if(N%TILEDIM_M == 0){
//                 As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//                 As[ty*TILEDIM_K + (tx+8)] = A[a + ty*N + (tx+8)];
//                 As[ty*TILEDIM_K + (tx+16)] = A[a + ty*N + (tx+16)];
//                 As[ty*TILEDIM_K + (tx+24)] = A[a + ty*N + (tx+24)];

//                 As[(ty+8)*TILEDIM_K + tx] = A[a + (ty+8)*N + tx];
//                 As[(ty+8)*TILEDIM_K + (tx+8)] = A[a + (ty+8)*N + (tx+8)];
//                 As[(ty+8)*TILEDIM_K + (tx+16)] = A[a + (ty+8)*N + (tx+16)];
//                 As[(ty+8)*TILEDIM_K + (tx+24)] = A[a + (ty+8)*N + (tx+24)];

//                 As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
//                 As[(ty+16)*TILEDIM_K + (tx+8)] = A[a + (ty+16)*N + (tx+8)];
//                 As[(ty+16)*TILEDIM_K + (tx+16)] = A[a + (ty+16)*N + (tx+16)];
//                 As[(ty+16)*TILEDIM_K + (tx+24)] = A[a + (ty+16)*N + (tx+24)];

//                 As[(ty+24)*TILEDIM_K + tx] = A[a + (ty+24)*N + tx];
//                 As[(ty+24)*TILEDIM_K + (tx+8)] = A[a + (ty+24)*N + (tx+8)];
//                 As[(ty+24)*TILEDIM_K + (tx+16)] = A[a + (ty+24)*N + (tx+16)];
//                 As[(ty+24)*TILEDIM_K + (tx+24)] = A[a + (ty+24)*N + (tx+24)];

//                 As[(ty+32)*TILEDIM_K + tx] = A[a + (ty+32)*N + tx];
//                 As[(ty+32)*TILEDIM_K + (tx+8)] = A[a + (ty+32)*N + (tx+8)];
//                 As[(ty+32)*TILEDIM_K + (tx+16)] = A[a + (ty+32)*N + (tx+16)];
//                 As[(ty+32)*TILEDIM_K + (tx+24)] = A[a + (ty+32)*N + (tx+24)];

//                 As[(ty+40)*TILEDIM_K + tx] = A[a + (ty+40)*N + tx];
//                 As[(ty+40)*TILEDIM_K + (tx+8)] = A[a + (ty+40)*N + (tx+8)];
//                 As[(ty+40)*TILEDIM_K + (tx+16)] = A[a + (ty+40)*N + (tx+16)];
//                 As[(ty+40)*TILEDIM_K + (tx+24)] = A[a + (ty+40)*N + (tx+24)];
                
//                 As[(ty+48)*TILEDIM_K + tx] = A[a + (ty+48)*N + tx];
//                 As[(ty+48)*TILEDIM_K + (tx+8)] = A[a + (ty+48)*N + (tx+8)];
//                 As[(ty+48)*TILEDIM_K + (tx+16)] = A[a + (ty+48)*N + (tx+16)];
//                 As[(ty+48)*TILEDIM_K + (tx+24)] = A[a + (ty+48)*N + (tx+24)];
                
//                 As[(ty+56)*TILEDIM_K + tx] = A[a + (ty+56)*N + tx];
//                 As[(ty+56)*TILEDIM_K + (tx+8)] = A[a + (ty+56)*N + (tx+8)];
//                 As[(ty+56)*TILEDIM_K + (tx+16)] = A[a + (ty+56)*N + (tx+16)];
//                 As[(ty+56)*TILEDIM_K + (tx+24)] = A[a + (ty+56)*N + (tx+24)];
                
//                 Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];
//                 Bs[ty*TILEDIM_N + (tx+8)] = B[b + ty*N + (tx+8)];
//                 Bs[ty*TILEDIM_N + (tx+16)] = B[b + ty*N + (tx+16)];
//                 Bs[ty*TILEDIM_N + (tx+24)] = B[b + ty*N + (tx+24)];

//                 Bs[(ty+8)*TILEDIM_N + tx] = B[b + (ty+8)*N + tx];
//                 Bs[(ty+8)*TILEDIM_N + (tx+8)] = B[b + (ty+8)*N + (tx+8)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+16)] = B[b + (ty+8)*N + (tx+16)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+24)] = B[b + (ty+8)*N + (tx+24)];

//                 Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
//                 Bs[(ty+16)*TILEDIM_N + (tx+8)] = B[b + (ty+16)*N + (tx+8)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+16)] = B[b + (ty+16)*N + (tx+16)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+24)] = B[b + (ty+16)*N + (tx+24)];

//                 Bs[(ty+24)*TILEDIM_N + tx] = B[b + (ty+24)*N + tx];
//                 Bs[(ty+24)*TILEDIM_N + (tx+8)] = B[b + (ty+24)*N + (tx+8)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+16)] = B[b + (ty+24)*N + (tx+16)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+24)] = B[b + (ty+24)*N + (tx+24)];                        

//                 Bs[(ty+32)*TILEDIM_N + tx] = B[b + (ty+32)*N + tx];
//                 Bs[(ty+32)*TILEDIM_N + (tx+8)] = B[b + (ty+32)*N + (tx+8)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+16)] = B[b + (ty+32)*N + (tx+16)];
//                 Bs[(ty+32)*TILEDIM_N + (tx+24)] = B[b + (ty+32)*N + (tx+24)];

//                 Bs[(ty+40)*TILEDIM_N + tx] = B[b + (ty+40)*N + tx];
//                 Bs[(ty+40)*TILEDIM_N + (tx+8)] = B[b + (ty+40)*N + (tx+8)];
//                 Bs[(ty+40)*TILEDIM_N + (tx+16)] = B[b + (ty+40)*N + (tx+16)];
//                 Bs[(ty+40)*TILEDIM_N + (tx+24)] = B[b + (ty+40)*N + (tx+24)];

//                 Bs[(ty+48)*TILEDIM_N + tx] = B[b + (ty+48)*N + tx];
//                 Bs[(ty+48)*TILEDIM_N + (tx+8)] = B[b + (ty+48)*N + (tx+8)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+16)] = B[b + (ty+48)*N + (tx+16)];
//                 Bs[(ty+48)*TILEDIM_N + (tx+24)] = B[b + (ty+48)*N + (tx+24)];

//                 Bs[(ty+56)*TILEDIM_N + tx] = B[b + (ty+56)*N + tx];
//                 Bs[(ty+56)*TILEDIM_N + (tx+8)] = B[b + (ty+56)*N + (tx+8)];
//                 Bs[(ty+56)*TILEDIM_N + (tx+16)] = B[b + (ty+56)*N + (tx+16)];
//                 Bs[(ty+56)*TILEDIM_N + (tx+24)] = B[b + (ty+56)*N + (tx+24)];
//             } else{
//                 for(int itx=tx; itx<32; itx+=8){
//                     for(int ity=ty; ity<64; ity+=8){
//                         if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
//                             As[ity*TILEDIM_K + itx] = 0;
//                         } else{
//                             As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
//                         }
//                     }
//                 }
//                 for(int itx=tx; itx<32; itx+=8){
//                     for(int ity=ty; ity<64; ity+=8){
//                         if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
//                             Bs[ity*TILEDIM_N + itx] = 0;
//                         } else{
//                             Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
//                         }
//                     }
//                 }
//             }
             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij[0] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[1] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[2] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[3] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[4] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[5] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[6] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[7] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];

//                 _cij[8] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[9] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[10] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[11] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[12] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[13] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[14] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[15] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];

//                 _cij[16] += As[(ty)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[17] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[18] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[19] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[20] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[21] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[22] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[23] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
                
//                 _cij[24] += As[(ty)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[25] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[26] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[27] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[28] += As[(ty+32)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[29] += As[(ty+40)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[30] += As[(ty+48)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[31] += As[(ty+56)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         if(N%TILEDIM_M ==0){
//             C[c + N * ty + tx] = _cij[0];
//             C[c + N * (ty+8) + tx] = _cij[1];
//             C[c + N * (ty+16) + tx] = _cij[2];
//             C[c + N * (ty+24) + tx] = _cij[3];
//             C[c + N * (ty+32) + tx] = _cij[4];
//             C[c + N * (ty+40) + tx] = _cij[5];
//             C[c + N * (ty+48) + tx] = _cij[6];
//             C[c + N * (ty+56) + tx] = _cij[7];

//             C[c + N * ty + (tx+8)] = _cij[8];
//             C[c + N * (ty+8) + (tx+8)] = _cij[9];
//             C[c + N * (ty+16) + (tx+8)] = _cij[10];
//             C[c + N * (ty+24) + (tx+8)] = _cij[11];    
//             C[c + N * (ty+32) + (tx+8)] = _cij[12];    
//             C[c + N * (ty+40) + (tx+8)] = _cij[13];    
//             C[c + N * (ty+48) + (tx+8)] = _cij[14];    
//             C[c + N * (ty+56) + (tx+8)] = _cij[15];    

//             C[c + N * ty + (tx+16)] = _cij[16];
//             C[c + N * (ty+8) + (tx+16)] = _cij[17];
//             C[c + N * (ty+16) + (tx+16)] = _cij[18];
//             C[c + N * (ty+24) + (tx+16)] = _cij[19];    
//             C[c + N * (ty+32) + (tx+16)] = _cij[20];    
//             C[c + N * (ty+40) + (tx+16)] = _cij[21];    
//             C[c + N * (ty+48) + (tx+16)] = _cij[22];    
//             C[c + N * (ty+56) + (tx+16)] = _cij[23];    

//             C[c + N * ty + (tx+24)] = _cij[24];
//             C[c + N * (ty+8) + (tx+24)] = _cij[25];
//             C[c + N * (ty+16) + (tx+24)] = _cij[26];
//             C[c + N * (ty+24) + (tx+24)] = _cij[27];    
//             C[c + N * (ty+32) + (tx+24)] = _cij[28];    
//             C[c + N * (ty+40) + (tx+24)] = _cij[29];    
//             C[c + N * (ty+48) + (tx+24)] = _cij[30];    
//             C[c + N * (ty+56) + (tx+24)] = _cij[31];    
//         } else{
//             int count = 0;
//             for(int itx=tx; itx<32; itx+=8){
//                 for(int ity=ty; ity<64; ity+=8){
//                     if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
//                         C[c + N * ity + itx] = _cij[count];
//                     }
//                     count++;
//                 }
//             }
//         }
            
//     }
// }

// bx=8,by=8,TILEDIM_M=32,TILEDIM_K=32,TILEDIM_N=32,16 ops per thread, interleaved access, [ty][tx] indexing
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[16] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             if(N%TILEDIM_M == 0){
//                 As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//                 As[ty*TILEDIM_K + (tx+8)] = A[a + ty*N + (tx+8)];
//                 As[ty*TILEDIM_K + (tx+16)] = A[a + ty*N + (tx+16)];
//                 As[ty*TILEDIM_K + (tx+24)] = A[a + ty*N + (tx+24)];

//                 As[(ty+8)*TILEDIM_K + tx] = A[a + (ty+8)*N + tx];
//                 As[(ty+8)*TILEDIM_K + (tx+8)] = A[a + (ty+8)*N + (tx+8)];
//                 As[(ty+8)*TILEDIM_K + (tx+16)] = A[a + (ty+8)*N + (tx+16)];
//                 As[(ty+8)*TILEDIM_K + (tx+24)] = A[a + (ty+8)*N + (tx+24)];

//                 As[(ty+16)*TILEDIM_K + tx] = A[a + (ty+16)*N + tx];
//                 As[(ty+16)*TILEDIM_K + (tx+8)] = A[a + (ty+16)*N + (tx+8)];
//                 As[(ty+16)*TILEDIM_K + (tx+16)] = A[a + (ty+16)*N + (tx+16)];
//                 As[(ty+16)*TILEDIM_K + (tx+24)] = A[a + (ty+16)*N + (tx+24)];

//                 As[(ty+24)*TILEDIM_K + tx] = A[a + (ty+24)*N + tx];
//                 As[(ty+24)*TILEDIM_K + (tx+8)] = A[a + (ty+24)*N + (tx+8)];
//                 As[(ty+24)*TILEDIM_K + (tx+16)] = A[a + (ty+24)*N + (tx+16)];
//                 As[(ty+24)*TILEDIM_K + (tx+24)] = A[a + (ty+24)*N + (tx+24)];
                
//                 Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];
//                 Bs[ty*TILEDIM_N + (tx+8)] = B[b + ty*N + (tx+8)];
//                 Bs[ty*TILEDIM_N + (tx+16)] = B[b + ty*N + (tx+16)];
//                 Bs[ty*TILEDIM_N + (tx+24)] = B[b + ty*N + (tx+24)];

//                 Bs[(ty+8)*TILEDIM_N + tx] = B[b + (ty+8)*N + tx];
//                 Bs[(ty+8)*TILEDIM_N + (tx+8)] = B[b + (ty+8)*N + (tx+8)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+16)] = B[b + (ty+8)*N + (tx+16)];
//                 Bs[(ty+8)*TILEDIM_N + (tx+24)] = B[b + (ty+8)*N + (tx+24)];

//                 Bs[(ty+16)*TILEDIM_N + tx] = B[b + (ty+16)*N + tx];
//                 Bs[(ty+16)*TILEDIM_N + (tx+8)] = B[b + (ty+16)*N + (tx+8)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+16)] = B[b + (ty+16)*N + (tx+16)];
//                 Bs[(ty+16)*TILEDIM_N + (tx+24)] = B[b + (ty+16)*N + (tx+24)];

//                 Bs[(ty+24)*TILEDIM_N + tx] = B[b + (ty+24)*N + tx];
//                 Bs[(ty+24)*TILEDIM_N + (tx+8)] = B[b + (ty+24)*N + (tx+8)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+16)] = B[b + (ty+24)*N + (tx+16)];                        
//                 Bs[(ty+24)*TILEDIM_N + (tx+24)] = B[b + (ty+24)*N + (tx+24)];                        
//             } else{
//                 for(int itx=tx; itx<32; itx+=8){
//                     for(int ity=ty; ity<32; ity+=8){
//                         if((a + ity*N >= N*N) || (a + itx >= (by*TILEDIM_M + 1)*N)){
//                             As[ity*TILEDIM_K + itx] = 0;
//                         } else{
//                             As[ity*TILEDIM_K + itx] = A[a + ity*N + itx];
//                         }
//                     }
//                 }
//                 for(int itx=tx; itx<32; itx+=8){
//                     for(int ity=ty; ity<32; ity+=8){
//                         if((b + ity*N >= N*N) || (bx*TILEDIM_N + itx >= N)){
//                             Bs[ity*TILEDIM_N + itx] = 0;
//                         } else{
//                             Bs[ity*TILEDIM_N + itx] = B[b + ity*N + itx];
//                         }
//                     }
//                 }
//             }
             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij[0] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[1] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[2] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//                 _cij[3] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];

//                 _cij[4] += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[5] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[6] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];
//                 _cij[7] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+8)];

//                 _cij[8] += As[(ty)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[9] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[10] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
//                 _cij[11] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+16)];
                
//                 _cij[12] += As[(ty)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[13] += As[(ty+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[14] += As[(ty+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//                 _cij[15] += As[(ty+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (tx+24)];
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         if(N%TILEDIM_M ==0){
//             C[c + N * ty + tx] = _cij[0];
//             C[c + N * (ty+8) + tx] = _cij[1];
//             C[c + N * (ty+16) + tx] = _cij[2];
//             C[c + N * (ty+24) + tx] = _cij[3];

//             C[c + N * ty + (tx+8)] = _cij[4];
//             C[c + N * (ty+8) + (tx+8)] = _cij[5];
//             C[c + N * (ty+16) + (tx+8)] = _cij[6];
//             C[c + N * (ty+24) + (tx+8)] = _cij[7];    

//             C[c + N * ty + (tx+16)] = _cij[8];
//             C[c + N * (ty+8) + (tx+16)] = _cij[9];
//             C[c + N * (ty+16) + (tx+16)] = _cij[10];
//             C[c + N * (ty+24) + (tx+16)] = _cij[11];    

//             C[c + N * ty + (tx+24)] = _cij[12];
//             C[c + N * (ty+8) + (tx+24)] = _cij[13];
//             C[c + N * (ty+16) + (tx+24)] = _cij[14];
//             C[c + N * (ty+24) + (tx+24)] = _cij[15];    
//         } else{
//             int count = 0;
//             for(int itx=tx; itx<32; itx+=8){
//                 for(int ity=ty; ity<32; ity+=8){
//                     if((c + N * ity + itx < N*N) && (c + itx < N*(TILEDIM_M*by+1))){
//                         C[c + N * ity + itx] = _cij[count];
//                     }
//                     count++;
//                 }
//             }
//         }
            
//     }
// }

// bx=8,by=8,TILEDIM_M=32,TILEDIM_K=32,TILEDIM_N=32,16 ops per thread, interleaved access, [tx][ty] indexing
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij[16] = {0};
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//                 As[tx*TILEDIM_K + ty] = A[a + tx*N + ty];
//                 As[tx*TILEDIM_K + (ty+8)] = A[a + tx*N + (ty+8)];
//                 As[tx*TILEDIM_K + (ty+16)] = A[a + tx*N + (ty+16)];
//                 As[tx*TILEDIM_K + (ty+24)] = A[a + tx*N + (ty+24)];

//                 As[(tx+8)*TILEDIM_K + ty] = A[a + (tx+8)*N + ty];
//                 As[(tx+8)*TILEDIM_K + (ty+8)] = A[a + (tx+8)*N + (ty+8)];
//                 As[(tx+8)*TILEDIM_K + (ty+16)] = A[a + (tx+8)*N + (ty+16)];
//                 As[(tx+8)*TILEDIM_K + (ty+24)] = A[a + (tx+8)*N + (ty+24)];

//                 As[(tx+16)*TILEDIM_K + ty] = A[a + (tx+16)*N + ty];
//                 As[(tx+16)*TILEDIM_K + (ty+8)] = A[a + (tx+16)*N + (ty+8)];
//                 As[(tx+16)*TILEDIM_K + (ty+16)] = A[a + (tx+16)*N + (ty+16)];
//                 As[(tx+16)*TILEDIM_K + (ty+24)] = A[a + (tx+16)*N + (ty+24)];

//                 As[(tx+24)*TILEDIM_K + ty] = A[a + (tx+24)*N + ty];
//                 As[(tx+24)*TILEDIM_K + (ty+8)] = A[a + (tx+24)*N + (ty+8)];
//                 As[(tx+24)*TILEDIM_K + (ty+16)] = A[a + (tx+24)*N + (ty+16)];
//                 As[(tx+24)*TILEDIM_K + (ty+24)] = A[a + (tx+24)*N + (ty+24)];
                
//                 Bs[tx*TILEDIM_N + ty] = B[b + tx*N + ty];
//                 Bs[tx*TILEDIM_N + (ty+8)] = B[b + tx*N + (ty+8)];
//                 Bs[tx*TILEDIM_N + (ty+16)] = B[b + tx*N + (ty+16)];
//                 Bs[tx*TILEDIM_N + (ty+24)] = B[b + tx*N + (ty+24)];

//                 Bs[(tx+8)*TILEDIM_N + ty] = B[b + (tx+8)*N + ty];
//                 Bs[(tx+8)*TILEDIM_N + (ty+8)] = B[b + (tx+8)*N + (ty+8)];
//                 Bs[(tx+8)*TILEDIM_N + (ty+16)] = B[b + (tx+8)*N + (ty+16)];
//                 Bs[(tx+8)*TILEDIM_N + (ty+24)] = B[b + (tx+8)*N + (ty+24)];

//                 Bs[(tx+16)*TILEDIM_N + ty] = B[b + (tx+16)*N + ty];
//                 Bs[(tx+16)*TILEDIM_N + (ty+8)] = B[b + (tx+16)*N + (ty+8)];
//                 Bs[(tx+16)*TILEDIM_N + (ty+16)] = B[b + (tx+16)*N + (ty+16)];
//                 Bs[(tx+16)*TILEDIM_N + (ty+24)] = B[b + (tx+16)*N + (ty+24)];

//                 Bs[(tx+24)*TILEDIM_N + ty] = B[b + (tx+24)*N + ty];
//                 Bs[(tx+24)*TILEDIM_N + (ty+8)] = B[b + (tx+24)*N + (ty+8)];                        
//                 Bs[(tx+24)*TILEDIM_N + (ty+16)] = B[b + (tx+24)*N + (ty+16)];                        
//                 Bs[(tx+24)*TILEDIM_N + (ty+24)] = B[b + (tx+24)*N + (ty+24)];                        
             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij[0] += As[tx*TILEDIM_K + k] * Bs[k*TILEDIM_N + ty];
//                 _cij[1] += As[(tx+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + ty];
//                 _cij[2] += As[(tx+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + ty];
//                 _cij[3] += As[(tx+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + ty];

//                 _cij[4] += As[tx*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+8)];
//                 _cij[5] += As[(tx+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+8)];
//                 _cij[6] += As[(tx+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+8)];
//                 _cij[7] += As[(tx+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+8)];

//                 _cij[8] += As[(tx)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+16)];
//                 _cij[9] += As[(tx+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+16)];
//                 _cij[10] += As[(tx+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+16)];
//                 _cij[11] += As[(tx+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+16)];
                
//                 _cij[12] += As[(tx)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+24)];
//                 _cij[13] += As[(tx+8)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+24)];
//                 _cij[14] += As[(tx+16)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+24)];
//                 _cij[15] += As[(tx+24)*TILEDIM_K + k] * Bs[k*TILEDIM_N + (ty+24)];
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//             C[c + N * tx + ty] = _cij[0];
//             C[c + N * (tx+8) + ty] = _cij[1];
//             C[c + N * (tx+16) + ty] = _cij[2];
//             C[c + N * (tx+24) + ty] = _cij[3];

//             C[c + N * tx + (ty+8)] = _cij[4];
//             C[c + N * (tx+8) + (ty+8)] = _cij[5];
//             C[c + N * (tx+16) + (ty+8)] = _cij[6];
//             C[c + N * (tx+24) + (ty+8)] = _cij[7];    

//             C[c + N * tx + (ty+16)] = _cij[8];
//             C[c + N * (tx+8) + (ty+16)] = _cij[9];
//             C[c + N * (tx+16) + (ty+16)] = _cij[10];
//             C[c + N * (tx+24) + (ty+16)] = _cij[11];    

//             C[c + N * tx + (ty+24)] = _cij[12];
//             C[c + N * (tx+8) + (ty+24)] = _cij[13];
//             C[c + N * (tx+16) + (ty+24)] = _cij[14];
//             C[c + N * (tx+24) + (ty+24)] = _cij[15];    
//     }
// }

// bx=8,by=8,TILEDIM_M=8,TILEDIM_K=8,TILEDIM_N=8,1 ops per thread
// __global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

//     _FTYPE_ * __restrict__ As = sharmem;
//     _FTYPE_ * __restrict__ Bs = &sharmem[TILEDIM_M * TILEDIM_K];

//     int ty = threadIdx.y, tx = threadIdx.x;
//     int by = blockIdx.y, bx = blockIdx.x;

//     int I =  by*blockDim.y + ty; // rows
//     int J =  bx*blockDim.x + tx; // cols   

//     if((I < N) && (J < N)){
//         register _FTYPE_ _cij = 0;
//         #pragma unroll
//         for (int a = by*TILEDIM_M*N, b = bx*TILEDIM_N; a < N + by*TILEDIM_M*N; a += TILEDIM_K, b += TILEDIM_K*N) {
//             As[ty*TILEDIM_K + tx] = A[a + ty*N + tx];
//             Bs[ty*TILEDIM_N + tx] = B[b + ty*N + tx];             
//             __syncthreads();        
//             for (int k = 0; k < TILEDIM_K; k++) {
//                 _cij += As[ty*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx];
//             }                
//             __syncthreads();
//         }
//         int c = N*TILEDIM_M*by + TILEDIM_N*bx;
//         C[c + N * ty + tx] = _cij;            
//     }
// }

#endif
