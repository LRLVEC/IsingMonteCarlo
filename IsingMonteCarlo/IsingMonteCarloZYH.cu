#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define gridDim 256
#define Tc 2.268

__global__ void initRandom(curandState* state, unsigned int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, state + id);
}
__global__ void clear(float* grid)
{
	for (int c0(0); c0 < gridDim; ++c0)
		grid[gridDim*threadIdx.x+c0] = -1;
}
__global__ void ising(float* grid, float H, float T, int dx, curandState* state) {
	int y = blockIdx.x;
	int x = 2 * threadIdx.x + (y + dx) % 2;
	float S0 = grid[x + y * gridDim];
	float Sn =
		grid[(x + 1) % gridDim + y * gridDim] +
		grid[(x + gridDim - 1) % gridDim + y * gridDim] +
		grid[x + ((y + 1) % gridDim) * gridDim] +
		grid[x + ((y + gridDim - 1) % gridDim) * gridDim];
	float dE = 2 * S0 * (H + Sn);
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (dE < 0 || curand_uniform(state + id) < expf(-dE / T))
		grid[x + y * gridDim] = -S0;
}
__global__ void calcM(float* grid, float* M) {
	float m = 0;
	for (int c0(0); c0 < gridDim * gridDim; ++c0)
		m += grid[c0];
	*M = m;
}

int main(int argc, char* argv[]) {
	if (argc != 8) {
		::printf("%s [H1] [H2] [nH] [T1] [T2] [nT] [Cycle]\n", argv[0]);
		return 0;
	}

	float H1 = atof(argv[1]);
	float H2 = atof(argv[2]);
	int   nH = atoi(argv[3]);
	float T, T1 = atof(argv[4]) * 2.268;
	float T2 = atof(argv[5]) * 2.268;
	int   nT = atoi(argv[6]);
	int cycles = atoi(argv[7]);
	H2 -= H1;
	H2 /= nH;
	T2 -= T1;
	T2 /= nT;

	unsigned int memSize = sizeof(float) * gridDim * gridDim;
	float* originGrid((float*)malloc(memSize));
	float* grid;
	curandState* state;
	float* Md;
	std::mt19937 mt(0);
	cudaMalloc(&Md, 4);
	cudaMalloc(&grid, memSize);
	cudaMalloc(&state, sizeof(curandState) * gridDim * gridDim / 2);

	// init lattice
	/*for (int c2(0); c2 < gridDim * gridDim; ++c2) {
		originGrid[c2] = -1;
	}*/
	std::uniform_int_distribution<int> rd;
	initRandom<<<gridDim, gridDim / 2 >>>(state, rd(mt));
	for (int c0(0); c0 <= nH; ++c0) {
		T = T1;
		for (int c1(0); c1 <= nT; ++c1) {
			clear << <1, gridDim >> > (grid);
			//cudaMemcpy(grid, originGrid, memSize, cudaMemcpyHostToDevice);
			for (int c2(0); c2 < cycles; ++c2) {
				ising << < gridDim, gridDim / 2 >> > (grid, H1, T, 0, state);
				ising << < gridDim, gridDim / 2 >> > (grid, H1, T, 1, state);
			}
			float M;
			calcM << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (grid, Md);
			cudaMemcpy(&M, Md, 4, cudaMemcpyDeviceToHost);
			M /= (gridDim * gridDim);
			::printf("%.8f %.8f %.8f\n", H1, T, M);
			fflush(stdout);
			T += T2;
		}
		H1 += H2;
	}
	cudaFree(grid);
	cudaFree(Md);
	cudaFree(state);
	::free(originGrid);
}