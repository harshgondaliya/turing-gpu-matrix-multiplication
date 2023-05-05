#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <float.h> // For: FLT_EPSILON

#include "types.h"

using namespace std;

void cmdLine(
    int argc,
    char *argv[],
    int &n,
    int &reps,
    int &ntx,
    int &nty,
    _FTYPE_ &eps,
    int &do_host,
    int &prefer_l1,
    int &use_rand,
    int &use_bt,
    int &use_seq,
    int &use_shm_double,
    int &verify_gpu)
{

    // Command line arguments, default settings
    n = 8;
    reps = 10;
    eps = -1;           // Threshold for comparison
    do_host = 0;        // We don't do the computation on the host.
    prefer_l1 = 0;      // We prefer Shared memory by default
    use_rand = 0;       // Use biadiagonal A & B for initial inputs
    use_bt = 0;         // Use random initial A & B
    use_seq = 0;        // Use Identify matrix for A and sequential matrix for B
    use_shm_double = 0; // default to 4B interleaved shared memory
    verify_gpu = 1;

// Ntx and Nty will be overriden by statically specified values
// from the Make command line but are only useful when optimizing
// for shared memory
#ifdef BLOCKDIM_X
    ntx = BLOCKDIM_X;
#else
    ntx = 8;
#endif

#ifdef BLOCKDIM_Y
    nty = BLOCKDIM_Y;
#else
    nty = 8;
#endif

    // Default value of the domain sizes
    static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"r", required_argument, 0, 'r'},
        {"ntx", required_argument, 0, 'x'},
        {"nty", required_argument, 0, 'y'},
        {"do_host", no_argument, 0, 'h'},
        {"eps", required_argument, 0, 'e'},
        {"l1", no_argument, 0, 'l'},
        {"bt", no_argument, 0, 't'},
        {"rand", no_argument, 0, 'q'},
        {"seq", no_argument, 0, 's'},
        {"shared_mem_double", no_argument, 0, 'D'},
        {"shared_mem_single", no_argument, 0, 'S'},
        {"n0_verify", no_argument, 0, 'v'},
    };

    // Process command line arguments
    int ac;
    for (ac = 1; ac < argc; ac++)
    {
        int c;
        while ((c = getopt_long(argc, argv, "n:r:x:y:he:lbRDSsv", long_options, NULL)) != -1)
        {
            switch (c)
            {

            case 'n': // Size of the computational box
                n = atoi(optarg);
                break;

            case 'r': // Number of repititions
                reps = atoi(optarg);
                break;

            case 'x': // X thread block geometry
#ifdef BLOCKDIM_X
                cout << " *** The thread block size is statically compiled.\n     Ignoring the X thread geometry command-line setting\n";
#else
                ntx = atoi(optarg);
#endif
                break;

            case 'y': // Y thread block geometry
#ifdef BLOCKDIM_Y
                cout << " *** The thread block size is statically compiled.\n      Ignoring the Y thread geometry command-line setting\n";
#else
                nty = atoi(optarg);
#endif
                break;

            case 'h': // Run on the host (default: don't run on the host)
                do_host = 1;
                break;

            case 'e': // comparison tolerance
#ifdef _DOUBLE
                sscanf(optarg, "%lf", &eps);
#else
                sscanf(optarg, "%f", &eps);
#endif
                break;

            case 'l': // Favor L1 cache (48 KB), else favor Shared memory
                prefer_l1 = 1;
                break;

            case 'b': // Use bidiagonal matrices as inputs
                use_bt = 1;
                break;

            case 'D': // set shared memory config for this kernel to 8B
                use_shm_double = 1;
                break;

            case 'R': // Use random matrices as inputs
                use_rand = 1;
                break;

            case 'S':
                use_shm_double = 0;
                break;

            case 's':
                use_seq = 1;
                break;

            case 'v':
                verify_gpu = 0;
                break;

            default: // Error
                printf("Usage: mm [-n <domain size>] [-r <reps>] [-x <x thread geometry> [-y <y thread geometry] [-e <epsilon>] [-h {do_host}] [-l  <prefer l1>] [-b <use bt>] [-R <use rand>]\n");
                exit(-1);
            }
        }
    }
    if (eps == -1)
    {
#ifdef TARGET_T4
        eps = FLT_EPSILON * 3 * n;
#endif
    }
    if (use_rand && use_bt)
    {
        cout << "You asked to use a random, bidiagonal matrix. This option is not supported.\n";
        exit(0);
    }
}
