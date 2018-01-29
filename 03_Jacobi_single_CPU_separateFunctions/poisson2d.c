/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "common.h"

#define NY 4096
#define NX 4096

void set_rhs(real rhs[NY][NX]){

	for (int iy = 1; iy < NY-1; iy++)
    {
        for( int ix = 1; ix < NX-1; ix++ )
        {
            const real x = -1.0 + (2.0*ix/(NX-1));
            const real y = -1.0 + (2.0*iy/(NY-1));
            rhs[iy][ix] = expr(-10.0*(x*x + y*y));
        }
    }

}

void calculate_jacobi_iteration(real A[NY][NX], real Anew[NY][NX], real rhs[NY][NX], real *error){

	for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));

                *error = fmaxr(*error, fabsr(Anew[iy][ix]-A[iy][ix]));
            }
        }

}

void Anew_to_A(real A[NY][NX], real Anew[NY][NX]){

	for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }

}

void periodic_BC(real A[NY][NX]){

	for( int ix = 1; ix < NX-1; ix++ )
        {
                A[0][ix]      = A[(NY-2)][ix];
                A[(NY-1)][ix] = A[1][ix];
        }
        for (int iy = 1; iy < NY-1; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }

}

int main(int argc, char** argv)
{
    int iter_max = 1000;
    const real tol = 1.0e-5;

    real (*A)[NY] = malloc(sizeof(real[NY][NX]));
    real (*Anew)[NY] = malloc(sizeof(real[NY][NX]));
    real (*rhs)[NY] = malloc(sizeof(real[NY][NX]));
    
    memset(A, 0, NY * NX * sizeof(real));

		set_rhs(rhs);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);

    StartTimer();
    int iter  = 0;
    real error = 1.0;
    
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

				calculate_jacobi_iteration(A, Anew, rhs, &error);
				Anew_to_A(A, Anew);
				periodic_BC(A);

        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();

    printf( "%dx%d: 1 CPU: %8.4f s\n", NY,NX, runtime/ 1000.0 );

		free(A);
		free(Anew);
		free(rhs);

    return 0;
}
