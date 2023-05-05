#include "types.h"
#include "cblas.h"

void matMulHost(_FTYPE_ *C, const _FTYPE_ *A, const _FTYPE_ *B, unsigned int M, unsigned int N)
{
    const _FTYPE_ Beta = 0.0;
    const _FTYPE_ Alpha = 1.0;
    const int K = N;
    const int LDA = N, LDB = N, LDC = N;
    const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
#ifndef _DOUBLE
    cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, Alpha, A, LDA, B, LDB, Beta, C, LDC);
#else
    cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, Alpha, A, LDA, B, LDB, Beta, C, LDC);
#endif
}

void reference_dgemm(unsigned int N, _FTYPE_ Alpha, _FTYPE_ *A, _FTYPE_ *B, _FTYPE_ *C)
{
    const _FTYPE_ Beta = 1.0;
    const int M = N, K = N;
    const int LDA = N, LDB = N, LDC = N;
    const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
#ifndef _DOUBLE
    cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, Alpha, A, LDA, B, LDB, Beta, C, LDC);
#else
    cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, Alpha, A, LDA, B, LDB, Beta, C, LDC);
#endif
}
