
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   gridDim.x = n / TILEDIM_N;
   gridDim.y = n / TILEDIM_M;

   // you can overwrite blockDim here if you like.
   if (n % TILEDIM_N != 0)
      gridDim.x++;
   if (n % TILEDIM_M != 0)
      gridDim.y++;
}
