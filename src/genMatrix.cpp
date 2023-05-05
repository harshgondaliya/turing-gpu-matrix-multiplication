/* Generates three different kinds of input matrices
 * and provides corresponding verifiers
 * Hilbert Matrix
 * Bidiagonal
 * Random
 *
 * Hilbert Matrix H(i,j)
 * H(i,j) = 1/(i+j+1),   0 < i,j < n
 * It's easy to check if the multiplication is correct;
 * entry (i,j) of H * H is
 * Sum(k) { 1.0/(i+k+1)*(k+j+1) }
 */

#include <stdlib.h>
#include <stdio.h> // For: perror
#include <assert.h>
#include <iostream>
#include <float.h> // For: DBL_EPSILON
#include <math.h>  // For: fabs

#include "types.h"

using namespace std;

#define MAX_ERRORS 20
#define A(i, j) (a[(i)*n + (j)])
#define B(i, j) (b[(i)*n + (j)])
#define C(i, j) (c[(i)*n + (j)])
#define fabs(x) ((x) < 0 ? -(x) : (x))

void matMulHost(_FTYPE_ *c, const _FTYPE_ *a, const _FTYPE_ *b, unsigned int m, unsigned int n);
void reference_dgemm(unsigned int n, _FTYPE_ Alpha, _FTYPE_ *a, _FTYPE_ *b, _FTYPE_ *c);

void absolute_value(_FTYPE_ *p, int n)
{
    for (int i = 0; i < n; ++i)
    {
        p[i] = fabs(p[i]);
    }
}

void genMatrix(_FTYPE_ *a, unsigned int m, unsigned int n)
{
    unsigned int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            A(i, j) = 1.0 / (_FTYPE_)(i + j + 1);
        }
    }
}

void genMatrix_bt(_FTYPE_ *a, _FTYPE_ *b, unsigned int n)
{
    unsigned int i;
    for (i = 0; i < n; i++)
    {
        A(i, i) = 1.0;
        B(i, i) = 1.0;
    }
    for (i = 1; i < n; i++)
    {
        A(i, i - 1) = 1.0;
        B(i - 1, i) = 1.0;
    }
}

void genMatrix_rand(_FTYPE_ *a, _FTYPE_ *b, unsigned int n)
{
    long int Rmax = RAND_MAX; // Typical value of RAND_MAX: 2^31 - 1
    long int Rmax_2 = Rmax >> 1;
    long int RM = Rmax_2 + 1;

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            long int r = random();            // Uniformly distributed ints over [0,RAND_MAX]
            long int R = r - RM;              // Uniformly distributed over [-1, 1]
            A(i, j) = (double)R / (double)RM; // Uniformly distributed over [-1, 1]

            long int r2 = random();            // Uniformly distributed ints over [0,RAND_MAX]
            long int R2 = r2 - RM;             // Uniformly distributed over [-1, 1]
            B(i, j) = (double)R2 / (double)RM; // Uniformly distributed over [-1, 1]
        }
    }
}

void genMatrix_ISeq(_FTYPE_ *a, _FTYPE_ *b, unsigned int n)
{
    _FTYPE_ count = 1.0;
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            if (i == j)
            {
                A(i, j) = 1.0;
            }
            else
            {
                A(i, j) = 0.0;
            }
            B(i, j) = count;
            count = count + 1.0;
        }
    }
}

void verify_ISeq(_FTYPE_ *c, unsigned int m, unsigned int n)
{
    _FTYPE_ x = 1.0;
    _FTYPE_ *cptr = c;
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            if (*cptr != x)
            {
                printf("C[%d, %d] = %4.2f but expected %4.2f\n", i, j, *cptr, x);
            }
            x = x + 1.0;
            cptr++;
        }
    }
}

// Verify against exact answer
void verify(_FTYPE_ *c, _FTYPE_ *a, _FTYPE_ *b, unsigned int m, unsigned int n, _FTYPE_ epsilon, const char *mesg)
{
    _FTYPE_ error = 0.0;
    int ierror = 0;

    _FTYPE_ *C_exact = new _FTYPE_[m * n];
    matMulHost(C_exact, a, b, m, n);

    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            _FTYPE_ delta = fabs(C(i, j) - C_exact[i * n + j]);
            if (delta > epsilon)
            {
                ierror++;
                error += delta;
                if (ierror == 1)
                    cout << "Error report for " << mesg << ":" << endl;
                if (ierror <= MAX_ERRORS)
                    cout << "C[" << i << ", " << j << "] is " << C(i, j) << ", should be: " << C_exact[i * n + j] << endl;
            }
        }
    }

    // Normalize the error
    error /= (_FTYPE_)(n * n);

    if (ierror)
    {
        cout << "  *** A total of " << ierror << " differences, error = " << error;
    }
    else
    {
        cout << endl
             << mesg << ": ";
        cout << "answers matched to within " << epsilon;
    }
    cout << endl
         << endl;
    delete[] C_exact;
}

// Verify against exact answer
void verify_rand(_FTYPE_ *a, _FTYPE_ *b, _FTYPE_ *c, unsigned int n, _FTYPE_ eps)
{
    int ierror = 0;
    /* Do not explicitly check that A and B were unmodified on square_dgemm exit
     * If they were, the following will most likely detect it:
     * C := C - A * B, computed with reference_dgemm */
    reference_dgemm(n, -1., a, b, c);

    /* A := |A|, B := |B|, C := |C| */
    absolute_value(a, n * n);
    absolute_value(b, n * n);
    absolute_value(c, n * n);

    /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */
    reference_dgemm(n, -3. * eps * n, a, b, c);

    /* If any element in C is positive, then something went wrong in square_dgemm */
    for (unsigned int i = 0; i < n * n; ++i)
    {
        if (c[i] > 0)
        {
            ierror++;
            if (ierror <= MAX_ERRORS)
            {
                cout << "*** Error in matrix multiply exceeds componentwise error bounds @ i=" << i << ": " << c[i] << endl;
            }
        }
    }
    absolute_value(a, n * n);
    absolute_value(b, n * n);
    absolute_value(c, n * n);

    if (ierror)
    {
        cout << "  *** A total of " << ierror << " differences" << endl;
    }
    else
    {
        cout << "*** Answers verified" << endl;
    }
    cout << endl
         << endl;
}

#define ASSERT(i, j, z)                                                                               \
    if (C((i), (j)) != (z))                                                                           \
    {                                                                                                 \
        if (ierror == 1)                                                                              \
            cout << "Error report for " << mesg << ":" << endl;                                       \
        if (ierror <= MAX_ERRORS)                                                                     \
            cout << "C[" << i << ", " << j << "] is " << C((i), (j)) << ", should be: " << z << endl; \
        ierror++;                                                                                     \
    }

void verify_bt(_FTYPE_ *c, unsigned int n, const char *mesg)
{
    _FTYPE_ error = 0.0;
    int ierror = 0;

    ASSERT(0, 0, 1.0);
    for (unsigned int i = 1; i < n; i++)
    {
        ASSERT(i, i, 2.0);
        ASSERT(i, i - 1, 1.0);
        ASSERT(i - 1, i, 1.0);
    }

    if (ierror)
    {
        cout << "  *** A total of " << ierror << " differences, error = " << error;
    }
    else
    {
        cout << endl
             << mesg << ": ";
        cout << "answers matched" << endl;
    }
    cout << endl
         << endl;
}

#define C_h(i, j) (c_h[i * n + j])
#define C_d(i, j) (c_d[i * n + j])

// Verify host result against device result
void verify_bt(_FTYPE_ *c_d, _FTYPE_ *c_h, unsigned int n, const char *mesg)
{
    verify_bt(c_d, n, mesg);
}

// Verify host result against device result
void verify(_FTYPE_ *c_d, _FTYPE_ *c_h, unsigned int m, unsigned int n, _FTYPE_ epsilon, const char *mesg)
{
    _FTYPE_ error = 0.0;
    int ierror = 0;
    unsigned int mn = m * n;

    for (unsigned int ij = 0; ij < mn; ij++)
    {
        _FTYPE_ diff = fabs(c_h[ij] - c_d[ij]);
        if (diff > epsilon)
        {
            ierror++;
            error += diff;
            if (ierror == 1)
                cout << "Error report for " << mesg << ":" << endl;
            if (ierror <= 10)
            {
                int i = ij / n;
                int j = ij % n;
                cout << "C_d[" << i << ", " << j << "] == " << C_d(i, j);
                cout << ", C_h[" << i << ", " << j << "] == " << C_h(i, j) << endl;
            }
        }
    }

    /* Normalize the error */
    error /= (_FTYPE_)(n * n);

    if (ierror)
    {
        cout << "  *** A total of " << ierror << " differences, error = " << error;
    }
    else
    {
        cout << endl
             << mesg << ": ";
        cout << "answers matched to within " << epsilon;
    }
    cout << endl
         << endl;
}

void printMatrix(_FTYPE_ *a, unsigned int m, unsigned int n)
{
    unsigned int i, j;
    cout.precision(4);
    cout.width(8);
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            cout << A(i, j) << " ";
        cout << endl;
    }
}
