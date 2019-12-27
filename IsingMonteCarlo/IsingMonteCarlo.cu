#include "IsingMonteCarlo.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <random>
#include <string>
#include <_Time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <curand_kernel.h>

#define gridDim 256
#define Tc 2.268

__device__ __inline__ float random(float s, float t)
{
	float a(fabsf(sin(s * 12.9898f + t * 78.233f)) * 43758.5453123f);
	return a - floorf(a);
}
const dim3 IsingBlocks[8] =
{
	{1,1,1},//8
	{1,1,1},//16
	{1,1,1},//32
	{1,1,1},//64
	{32,1,1},//128
	{64,1,1},//256
	{128,1,1},//512
	{256,1,1}//1024
};
const dim3 IsingThreads[8] =
{
	{1,1,1},//8
	{1,1,1},//16
	{1,1,1},//32
	{1,1,1},//64
	{16,4,1},//128
	{32,4,1},//256
	{64,4,1},//512
	{128,4,1}//1024
};
const dim3 ReduceThreads[8] =
{
	{1,1,1},//8
	{1,1,1},//16
	{1,1,1},//32
	{1,1,1},//64
	{16,1,1},//128
	{32,1,1},//256
	{64,1,1},//512
	{128,1,1}//1024
};



//__global__ void ising(float* grid, float H, float T, int dx, int seed)
//{
//	int y = blockIdx.x;
//	int x = 2 * threadIdx.x + (y + dx) % 2;
//	float S0 = grid[x + y * gridDim];
//	float Sn =
//		grid[(x + 1) % gridDim + y * gridDim] +
//		grid[(x + gridDim - 1) % gridDim + y * gridDim] +
//		grid[x + ((y + 1) % gridDim) * gridDim] +
//		grid[x + ((y + gridDim - 1) % gridDim) * gridDim];
//	float dE = 2 * S0 * (H + Sn);
//	if (dE < 0 || random(float(x - seed), float(y + seed)) < expf(-dE / T))
//		grid[x + y * gridDim] = -S0;
//}
//__global__ void ising_optimize0(float* grid, float H, float T, int step, int seed)//step: 0, 1
//{
//	__shared__ float subGrid[66][66];
//	int t0 = (blockIdx.y * 64 - 1) * gridDim + 64 * blockIdx.x + threadIdx.x;
//	int t1 = t0 + 32;
//	for (int c0(1); c0 < 65; ++c0)
//	{
//		subGrid[c0][threadIdx.x + 1] = grid[t0 + c0 * gridDim];
//		subGrid[c0][threadIdx.x + 33] = grid[t1 + c0 * gridDim];
//	}
//	subGrid[0][threadIdx.x * 2 + step + 1] = grid[((blockIdx.y * 64 + gridDim - 1) % gridDim) * gridDim + 64 * blockIdx.x + threadIdx.x * 2 + step];
//	subGrid[65][threadIdx.x * 2 + !step + 1] = grid[((blockIdx.y * 64 + 1) % gridDim) * gridDim + 64 * blockIdx.x + threadIdx.x * 2 + !step];
//	subGrid[threadIdx.x * 2 + step + 1][0] = grid[(blockIdx.y * 64 + threadIdx.x * 2 + step) * gridDim + (blockIdx.x * 64 + gridDim - 1) % gridDim];
//	subGrid[threadIdx.x * 2 + !step + 1][65] = grid[(blockIdx.y * 64 + threadIdx.x * 2 + !step) * gridDim + (blockIdx.x * 64 + 1) % gridDim];
//	__syncthreads();
//	int y0 = 2 * threadIdx.x + 1;
//	int y1 = y0 + !step;
//	y0 += step;
//	for (int c0(1 + step); c0 < 65; ++c0)
//	{
//		float ds = subGrid[y0][c0];
//		float dE = 2 * ds * (H + subGrid[y0 - 1][c0] + subGrid[y0 + 1][c0] + subGrid[y0][c0 - 1] + subGrid[y0][c0 + 1]);
//		if (dE < 0 || random(float(y0 + c0 - seed), float(t0 + seed)) < expf(-dE / T))subGrid[y0][c0] = -ds;
//		ds = subGrid[y1][++c0];
//		dE = 2 * ds * (H + subGrid[y1 - 1][c0] + subGrid[y1 + 1][c0] + subGrid[y1][c0 - 1] + subGrid[y1][c0 + 1]);
//		if (dE < 0 || random(float(y0 + c0 - seed), float(t0 + seed)) < expf(-dE / T))subGrid[y1][c0] = -ds;
//	}
//	__syncthreads();
//	for (int c0(1); c0 < 65; ++c0)
//	{
//		grid[t0 + c0 * gridDim] = subGrid[c0][threadIdx.x + 1];
//		grid[t1 + c0 * gridDim] = subGrid[c0][threadIdx.x + 33];
//	}
//}
//__global__ void ising_optimize1(float* grid, float H, float T, int step, int seed)
//{
//	__shared__ float subGrid[8][8];
//	int dx = (threadIdx.x & 3) << 1;
//	int dy = threadIdx.x >> 2;
//	int x = blockIdx.x * 8 + dx;
//	int y = blockIdx.y * 8 + dy;
//	subGrid[dy][dx] = grid[y * gridDim + x];
//	subGrid[dy][dx + 1] = grid[y * gridDim + x + 1];
//	int k = (step ^ (dy & 1));
//	dx += k;
//	x += k;
//	float s0 = subGrid[dy][dx];
//	float s1 = dx < 7 ? subGrid[dy][dx + 1] : grid[y * gridDim + (x + 1) % gridDim];
//	s1 += dx ? subGrid[dy][dx - 1] : grid[y * gridDim + (x + gridDim - 1) % gridDim];
//	s1 += dy < 7 ? subGrid[dy + 1][dx] : grid[((y + 1) % gridDim) * gridDim + x];
//	s1 += dy ? subGrid[dy - 1][dx] : grid[((y + gridDim - 1) % gridDim) * gridDim + x];
//	s0 *= 2 * (H + s1);
//	if (s0 < 0 || random(float(x - seed), float(y + seed)) < expf(-s0 / T))
//		grid[y * gridDim + x] = -subGrid[dy][dx];
//}
//__global__ void ising_optimize1_1(float* grid, float H, float T, int step, int seed)
//{
//	//gd: {128, 1, 1}, threads: {32, 1, 1}
//	//call time per M: 4552000000
//	//run time per M: 2323000000
//	__shared__ float subGrid[4][gridDim];
//	int y = blockIdx.x * 2;
//	for (int c0(2 * threadIdx.x + !step); c0 < gridDim; c0 += 64)
//		subGrid[0][c0] = grid[((y + gridDim - 1) % gridDim) * gridDim + c0];
//	for (int c0(threadIdx.x); c0 < gridDim; c0 += 32)
//		subGrid[1][c0] = grid[y * gridDim + c0];
//	for (int c0(threadIdx.x); c0 < gridDim; c0 += 32)
//		subGrid[2][c0] = grid[((y + 1) % gridDim) * gridDim + c0];
//	for (int c0(2 * threadIdx.x + !step); c0 < gridDim; c0 += 64)
//		subGrid[3][c0] = grid[((y + 2) % gridDim) * gridDim + c0];
//	int y0(y + (threadIdx.x + step) % 2);
//	y = (y == y0) ? 1 : 2;
//	for (int c0(0); c0 < gridDim; c0 += 32)
//	{
//		int x(c0 + threadIdx.x);
//		float s1 =
//			subGrid[y - 1][x] +
//			subGrid[y + 1][x] +
//			subGrid[y][(x + gridDim - 1) % gridDim] +
//			subGrid[y][(x + 1) % gridDim];
//		float s0(subGrid[y][x] * 2 * (H + s1));
//		if (s0 < 0 || random(float(x - seed), float(y0 + seed)) < expf(-s0 / T))
//			grid[y0 * gridDim + x] = -subGrid[y][x];
//	}
//}
//__global__ void ising_optimize2(float* M, float H, float T, float seed0, float seed1, int cycle)
//{
//	__shared__ unsigned int grid[256][8];
//	__shared__ unsigned int sum[32];
//#define get(a, b) (!(grid[b][a>>5]&(1<<(a&31))))
//#define set(a, b) (grid[b][a>>5]^=(1<<(a&31)))
//	const int y0 = threadIdx.x << 3;
//	float p0(seed0 + threadIdx.x - blockIdx.x);
//	float p1(seed1 - threadIdx.x + blockIdx.x);
//	//init
//	for (int c0(0); c0 < 256; c0 += 4)
//		grid[(threadIdx.x >> 3) + c0][threadIdx.x & 7] = 0;
//	for (int n = 0; n < (cycle << 1); ++n)
//	{
//		for (int y(y0); y < y0 + 8; ++y)
//		{
//			for (int x(n & 1); x < 256; x += 2)
//			{
//				unsigned char qq;
//				int s0 = 2 * get(x, y) - 1;
//				qq = x - 1;
//				int s1 = get(qq, y);
//				qq = x + 1;
//				s1 += get(qq, y);
//				qq = y - 1;
//				s1 += get(x, qq);
//				qq = y + 1;
//				s1 += get(x, qq);
//				s1 = s1 * 2 - 4;
//				float de = s0 * 2 * (H + s1);
//				if (de < 0)set(x, y);
//				else if (random(p0 - x, p1 + y) < expf(-de / T))set(x, y);
//			}
//		}
//		p0 = random(p0, p1);
//		p1 = random(p0, p1);
//	}
//	unsigned char* cg = 256 * threadIdx.x + (unsigned char*)grid;
//	int m = 0;
//	for (int c0(0); c0 < 256; ++c0)m += counter(cg[c0]);
//	sum[threadIdx.x] = m;
//	if (threadIdx.x % 2 == 0)sum[threadIdx.x] += sum[threadIdx.x + 1];
//	if (threadIdx.x % 4 == 0)sum[threadIdx.x] += sum[threadIdx.x + 2];
//	if (threadIdx.x % 8 == 0)sum[threadIdx.x] += sum[threadIdx.x + 4];
//	if (threadIdx.x % 16 == 0)sum[threadIdx.x] += sum[threadIdx.x + 8];
//	if (threadIdx.x == 0)*M = float(sum[0] + sum[16]) / 65536;
//#undef get
//#undef set
//}
//__global__ void ising_optimize3(unsigned long long* grid, float H, float T, int step, float seed)
//{
//	//gd: {256, 1, 1}, threads: {32, 1, 1}
//	//call time per M: 4552000000
//	//run time per M: 2323000000
//	__shared__ unsigned long long subGrid[3][4];
//	if (threadIdx.x < 12)
//		*((unsigned long long*)(subGrid) + threadIdx.x) =
//		grid[((blockIdx.x + gridDim - 1 + threadIdx.x / 4) % gridDim) * 4 + threadIdx.x % 4];
//	unsigned char* p((unsigned char*)(subGrid[1]));
//#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
//#define set(ff) ((p[threadIdx.x]^=(1<<(ff&7))))
//	int _step(step ^ (blockIdx.x % 2));
//	int d0(threadIdx.x * 8 + _step);
//	for (int c0(d0); c0 < d0 + 8; c0 += 2)
//	{
//		int s1 = get(0, c0) + get(2, c0);
//		int dx((c0 + gridDim - 1) % gridDim);
//		s1 += get(1, dx);
//		dx = (c0 + 1) % gridDim;
//		s1 += get(1, dx);
//		s1 = s1 * 2 - 4;
//		float s0 = 2 * int(get(1, c0)) - 1;
//		s0 *= H + s1;
//		if (s0 <= 0 || random(c0 - seed, blockIdx.x + seed) < expf(-s0 / T))
//			set(c0);
//	}
//	unsigned char* s = (unsigned char*)&grid[blockIdx.x * 4];
//	s[threadIdx.x] = p[threadIdx.x];
//#undef get
//#undef set
//}
//__device__ __inline__ int countBit(unsigned long long a)
//{
//	a = (a & 0x5555555555555555) + ((a >> 1) & 0x5555555555555555);
//	a = (a & 0x3333333333333333) + ((a >> 2) & 0x3333333333333333);
//	a = (a & 0x0F0F0F0F0F0F0F0F) + ((a >> 4) & 0x0F0F0F0F0F0F0F0F);
//	a = (a & 0x00FF00FF00FF00FF) + ((a >> 8) & 0x00FF00FF00FF00FF);
//	a = (a & 0x0000FFFF0000FFFF) + ((a >> 16) & 0x0000FFFF0000FFFF);
//	a = (a & 0x00000000FFFFFFFF) + ((a >> 32) & 0x00000000FFFFFFFF);
//	return a;
//}
//__global__ void calcBarM_optimize(unsigned long long* grid, float* M)
//{
//	//gd: {1, 1, 1}, threads: {1, 1, 1}
//	int N(0);
//	for (int c0(0); c0 < 4 * gridDim; ++c0)
//		N += countBit(grid[c0]);
//	*M = 2.0f * N / (gridDim * gridDim) - 1.0f;
//}

__global__ void initRandom(curandState* state, unsigned int seed)
{
	int id = (threadIdx.y + blockIdx.x * blockDim.y) * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, state + id);
}
void initRandom(curandState* state, int seed, unsigned int id)
{
	initRandom << <IsingBlocks[id], IsingThreads[id], 0, 0 >> > (state, seed);
}

//This is just used to reach balance.
template<unsigned int Dim>__global__ void ising(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	//For Dim == 256, 512, 1024
	//gd: {Dim/4, 1, 1}, threads: {Dim/8, 4, 1}
	//call time per M: 4552000000
	//run time per M: 2210000000
	constexpr unsigned int XX = Dim / 64;
	__shared__ unsigned long long subGrid[6][XX];
	unsigned int yy(blockIdx.x * 4 + threadIdx.y);
	int tid(threadIdx.y * blockDim.x + threadIdx.x);
	if (tid < 6 * XX)
	{
		unsigned int gy(tid / XX);
		unsigned int gx(tid % XX);
		subGrid[gy][gx] = grid[((blockIdx.x * 4 + gy + Dim - 1) % Dim) * XX + gx];
	}
	__syncthreads();
	unsigned char* p((unsigned char*)(subGrid[1]));
#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
#define set(ff) ((p[tid]^=(1<<(ff&7))))
	int _step(step ^ (yy % 2));
	int d0(threadIdx.x * 8 + _step);
	for (int c0(d0); c0 < d0 + 8; c0 += 2)
	{
		int s1 = get(threadIdx.y, c0) + get(2 + threadIdx.y, c0);
		int dx((c0 + Dim - 1) % Dim);
		s1 += get(1 + threadIdx.y, dx);
		dx = (c0 + 1) % Dim;
		s1 += get(1 + threadIdx.y, dx);
		s1 = s1 * 2 - 4;
		int ss = get(1 + threadIdx.y, c0);
		float s0 = 2 * ss - 1;
		s0 *= H + s1;
		if (s0 <= 0 || curand_uniform(state + tid + blockIdx.x * blockDim.x * blockDim.y)/* random(c0 - seed, yy + seed) */ < expf(-2 * s0 / T))set(c0);
	}
	unsigned char* s = (unsigned char*)&grid[yy * XX];
	s[threadIdx.x] = p[tid];
#undef get
#undef set
}
//This is used to create new ensembles and count MList, EList (still need to reduce), used for many points.
template<unsigned int Dim>__global__ void ising(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	constexpr unsigned int XX = Dim / 64;
	__shared__ unsigned long long subGrid[6][XX];
	__shared__ int dataM[Dim / 2];
	__shared__ float dataE[Dim / 2];
	unsigned int yy(blockIdx.x * 4 + threadIdx.y);
	int tid(threadIdx.y * blockDim.x + threadIdx.x);
	if (tid < 6 * XX)
	{
		unsigned int gy(tid / XX);
		unsigned int gx(tid % XX);
		subGrid[gy][gx] = grid[((blockIdx.x * 4 + gy + Dim - 1) % Dim) * XX + gx];
	}
	__syncthreads();
	int M(0);
	float E(0);
	unsigned char* p((unsigned char*)(subGrid[1]));
#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
#define set(ff) ((p[tid]^=(1<<(ff&7))))
	int _step(step ^ (yy % 2));
	int d0(threadIdx.x * 8 + _step);
	for (int c0(d0); c0 < d0 + 8; c0 += 2)
	{
		int dx((c0 + Dim - 1) % Dim);
		int s1 = get(1 + threadIdx.y, dx); M += s1;
		int ss = get(1 + threadIdx.y, c0); M += ss;
		dx = (c0 + 1) % Dim;
		s1 += get(1 + threadIdx.y, dx);
		s1 += get(threadIdx.y, c0);
		s1 += get(2 + threadIdx.y, c0);
		s1 = s1 * 2 - 4;
		float s0 = ss * 2 - 1;
		E -= s0 * s1;
		s0 *= H + s1;
		if (s0 <= 0 || curand_uniform(state + tid + blockIdx.x * blockDim.x * blockDim.y)/* random(c0 - seed, yy + seed) */ < expf(-2 * s0 / T))set(c0);
	}
	//E -= H * (2 * M - 8);
	unsigned char* s = (unsigned char*)&grid[yy * XX];
	s[threadIdx.x] = p[tid];
	dataM[tid] = M; dataE[tid] = E;
	__syncthreads();
	if (Dim > 512)if (tid < 256) { dataM[tid] += dataM[tid + 256]; dataE[tid] += dataE[tid + 256]; __syncthreads(); }
	if (Dim > 256)if (tid < 128) { dataM[tid] += dataM[tid + 128]; dataE[tid] += dataE[tid + 128]; __syncthreads(); }
	if (Dim > 128)if (tid < 64) { dataM[tid] += dataM[tid + 64]; dataE[tid] += dataE[tid + 64]; __syncthreads(); }
	if (tid < 32)
	{
		dataM[tid] += dataM[tid + 32];
		dataE[tid] += dataE[tid + 32]; __syncthreads();
		dataM[tid] += dataM[tid + 16];
		dataE[tid] += dataE[tid + 16]; __syncthreads();
		dataM[tid] += dataM[tid + 8];
		dataE[tid] += dataE[tid + 8]; __syncthreads();
		dataM[tid] += dataM[tid + 4];
		dataE[tid] += dataE[tid + 4]; __syncthreads();
		dataM[tid] += dataM[tid + 2];
		dataE[tid] += dataE[tid + 2]; __syncthreads();
		dataM[tid] += dataM[tid + 1];
		dataE[tid] += dataE[tid + 1]; __syncthreads();
	}
	if (tid == 0)
	{
		MList[blockIdx.x] = dataM[0];
		EList[blockIdx.x] = dataE[0];
	}
	//as a matter of fact, we can add M, E here
#undef get
#undef set
}
//This is used to create new ensembles and count MList, EList (still need to reduce) and Table, used for one point.
template<unsigned int Dim>__global__ void ising(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	constexpr unsigned int XX = Dim / 64;
	__shared__ unsigned long long subGrid[6][XX];
	__shared__ int dataM[Dim / 2];
	__shared__ float dataE[Dim / 2];
	unsigned int yy(blockIdx.x * 4 + threadIdx.y);
	int tid(threadIdx.y * blockDim.x + threadIdx.x);
	if (tid < 6 * XX)
	{
		unsigned int gy(tid / XX);
		unsigned int gx(tid % XX);
		subGrid[gy][gx] = grid[((blockIdx.x * 4 + gy + Dim - 1) % Dim) * XX + gx];
	}
	__syncthreads();
	int M(0);
	float E(0);
	unsigned char* p((unsigned char*)(subGrid[1]));
#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
#define set(ff) ((p[tid]^=(1<<(ff&7))))
	int _step(step ^ (yy % 2));
	int d0(threadIdx.x * 8 + _step);
	for (int c0(d0); c0 < d0 + 8; c0 += 2)
	{
		int dx((c0 + Dim - 1) % Dim);
		int s1 = get(1 + threadIdx.y, dx);
		table[yy * Dim + dx] += s1; M += s1;
		int ss = get(1 + threadIdx.y, c0);
		table[yy * Dim + c0] += ss; M += ss;
		dx = (c0 + 1) % Dim;
		s1 += get(1 + threadIdx.y, dx);
		s1 += get(threadIdx.y, c0);
		s1 += get(2 + threadIdx.y, c0);
		s1 = s1 * 2 - 4;
		float s0 = ss * 2 - 1;
		E -= s0 * s1;
		s0 *= H + s1;
		if (s0 <= 0 || curand_uniform(state + tid + blockIdx.x * blockDim.x * blockDim.y)/* random(c0 - seed, yy + seed) */ < expf(-2 * s0 / T))set(c0);
	}
	//E -= H * (2 * M - 8);
	unsigned char* s = (unsigned char*)&grid[yy * XX];
	s[threadIdx.x] = p[tid];
	dataM[tid] = M;
	dataE[tid] = E;
	__syncthreads();
	if (Dim > 512)if (tid < 256) { dataM[tid] += dataM[tid + 256]; dataE[tid] += dataE[tid + 256]; __syncthreads(); }
	if (Dim > 256)if (tid < 128) { dataM[tid] += dataM[tid + 128]; dataE[tid] += dataE[tid + 128]; __syncthreads(); }
	if (Dim > 128)if (tid < 64) { dataM[tid] += dataM[tid + 64]; dataE[tid] += dataE[tid + 64]; __syncthreads(); }
	if (tid < 32)
	{
		dataM[tid] += dataM[tid + 32];
		dataE[tid] += dataE[tid + 32]; __syncthreads();
		dataM[tid] += dataM[tid + 16];
		dataE[tid] += dataE[tid + 16]; __syncthreads();
		dataM[tid] += dataM[tid + 8];
		dataE[tid] += dataE[tid + 8]; __syncthreads();
		dataM[tid] += dataM[tid + 4];
		dataE[tid] += dataE[tid + 4]; __syncthreads();
		dataM[tid] += dataM[tid + 2];
		dataE[tid] += dataE[tid + 2]; __syncthreads();
		dataM[tid] += dataM[tid + 1];
		dataE[tid] += dataE[tid + 1]; __syncthreads();
	}
	if (tid == 0)
	{
		MList[blockIdx.x] = dataM[0];
		EList[blockIdx.x] = dataE[0];
	}
	//as a matter of fact, we can add M, E here
#undef get
#undef set
}
//This is used to reduce MList and EList.
template<unsigned int Dim>__global__ void reduce(int* MList, float* EList, int* M, float* E, float H)
{
	constexpr unsigned int sz = Dim / 8;
	__shared__ int dataM[sz];
	__shared__ float dataE[sz];
	int tid = threadIdx.x;
	dataM[tid] = MList[tid] + MList[tid + sz];
	dataE[tid] = EList[tid] + EList[tid + sz];
	__syncthreads();
	if (Dim > 512)if (tid < 64) { dataM[tid] += dataM[tid + 64]; dataE[tid] += dataE[tid + 64]; __syncthreads(); }
	if (Dim > 256)if (tid < 32) { dataM[tid] += dataM[tid + 32]; dataE[tid] += dataE[tid + 32]; __syncthreads(); }
	if (Dim > 128)if (tid < 16) { dataM[tid] += dataM[tid + 16]; dataE[tid] += dataE[tid + 16]; __syncthreads(); }
	if (tid < 8)
	{
		dataM[tid] += dataM[tid + 8];
		dataE[tid] += dataE[tid + 8]; __syncthreads();
		dataM[tid] += dataM[tid + 4];
		dataE[tid] += dataE[tid + 4]; __syncthreads();
		dataM[tid] += dataM[tid + 2];
		dataE[tid] += dataE[tid + 2]; __syncthreads();
		dataM[tid] += dataM[tid + 1];
		dataE[tid] += dataE[tid + 1]; __syncthreads();
	}
	if (tid == 0)
	{
		int MM = dataM[0];
		float EE = dataE[0];
		float gg = 2 * MM;
		gg -= Dim * Dim;
		EE -= H * gg;
		*M = MM;
		*E = EE;
	}
}


//template<unsigned int Dim>__global__ void calcBarM(unsigned long long* grid, float* M);
//template<unsigned int Dim>__global__ void addM(unsigned long long* grid, float* table);
//template<>__global__ void calcBarM<256u>(unsigned long long* grid, float* M)
//{
//	//gd: {1, 1, 1}, threads: {32, 1, 1}
//	__shared__ volatile int ahh[32];
//	int N(0);
//	grid += threadIdx.x * 32;
//	for (int c0(0); c0 < 32; ++c0)
//	{
//		unsigned long long gg = grid[c0];
//		unsigned int n;
//		asm("popc.b64 %0, %1;": "=r"(n) : "l"(gg));
//		N += n;
//	}
//	ahh[threadIdx.x] = N;
//	if (threadIdx.x < 16)ahh[threadIdx.x] += ahh[threadIdx.x + 16];
//	if (threadIdx.x < 8)ahh[threadIdx.x] += ahh[threadIdx.x + 8];
//	if (threadIdx.x < 4)ahh[threadIdx.x] += ahh[threadIdx.x + 4];
//	if (threadIdx.x < 2)ahh[threadIdx.x] += ahh[threadIdx.x + 2];
//	if (!threadIdx.x)
//	{
//		N = ahh[0] + ahh[1];
//		*M = 2 * float(N) / 65536 - 1;
//	}
//}
//template<>__global__ void addM<256u>(unsigned long long* grid, float* table)
//{
//	//gd: {256, 1, 1}, threads: {32, 1, 1}
//	unsigned char a(((unsigned char*)(grid + 4 * blockIdx.x))[threadIdx.x]);
//	table += 256 * blockIdx.x + 8 * threadIdx.x;
//	for (int c0(0); c0 < 8; ++c0)
//		table[c0] += (a >> c0) & 1;
//}
//void printGrid(unsigned long long* grid)
//{
//	char t[5];
//	std::string ahh;
//	for (int c0(0); c0 < 4 * gridDim; ++c0)
//	{
//		unsigned long long ft(grid[c0]);
//		for (int c1(0); c1 < 64; ++c1)
//		{
//			::sprintf(t, "%2d ", int((ft >> c1) & 1) * 2 - 1);
//			ahh += t;
//		}
//		if (c0 % 4 == 3)
//		{
//			::sprintf(t, "\n");
//			ahh += t;
//		}
//	}
//	FILE* temp(::fopen("./grid.txt", "w+"));
//	::fprintf(temp, "%s", ahh.c_str());
//	::fclose(temp);
//}

void ising128(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	ising<128> << < IsingBlocks[4], IsingThreads[4], 0, 0 >> > (grid, H, T, step, state);
}
void ising256(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	ising<256> << < IsingBlocks[5], IsingThreads[5], 0, 0 >> > (grid, H, T, step, state);
}
void ising512(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	ising<512> << < IsingBlocks[6], IsingThreads[6], 0, 0 >> > (grid, H, T, step, state);
}
void ising1024(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	ising<1024> << < IsingBlocks[7], IsingThreads[7], 0, 0 >> > (grid, H, T, step, state);
}
void ising128(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	ising<128> << < IsingBlocks[4], IsingThreads[4], 0, 0 >> > (grid, H, T, MList, EList, step, state);
}
void ising256(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	ising<256> << < IsingBlocks[5], IsingThreads[5], 0, 0 >> > (grid, H, T, MList, EList, step, state);
}
void ising512(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	ising<512> << < IsingBlocks[6], IsingThreads[6], 0, 0 >> > (grid, H, T, MList, EList, step, state);
}
void ising1024(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	ising<1024> << < IsingBlocks[7], IsingThreads[7], 0, 0 >> > (grid, H, T, MList, EList, step, state);
}
void ising128(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	ising<128> << < IsingBlocks[4], IsingThreads[4], 0, 0 >> > (grid, H, T, MList, EList, table, step, state);
}
void ising256(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	ising<256> << < IsingBlocks[5], IsingThreads[5], 0, 0 >> > (grid, H, T, MList, EList, table, step, state);
}
void ising512(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	ising<512> << < IsingBlocks[6], IsingThreads[6], 0, 0 >> > (grid, H, T, MList, EList, table, step, state);
}
void ising1024(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	ising<1024> << < IsingBlocks[7], IsingThreads[7], 0, 0 >> > (grid, H, T, MList, EList, table, step, state);
}
void reduce128(int* MList, float* EList, int* M, float* E, float H)
{
	reduce<128> << < dim3(1, 1, 1), ReduceThreads[4], 0, 0 >> > (MList, EList, M, E, H);
}
void reduce256(int* MList, float* EList, int* M, float* E, float H)
{
	reduce<256> << < dim3(1, 1, 1), ReduceThreads[5], 0, 0 >> > (MList, EList, M, E, H);
}
void reduce512(int* MList, float* EList, int* M, float* E, float H)
{
	reduce<512> << < dim3(1, 1, 1), ReduceThreads[6], 0, 0 >> > (MList, EList, M, E, H);
}
void reduce1024(int* MList, float* EList, int* M, float* E, float H)
{
	reduce<1024> << < dim3(1, 1, 1), ReduceThreads[7], 0, 0 >> > (MList, EList, M, E, H);
}
