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
template<unsigned int Dim>__global__ void ising(unsigned long long* grid, float H, float T, int step, float seed);
template<unsigned int Dim>__global__ void calcBarM(unsigned long long* grid, float* M);
template<unsigned int Dim>__global__ void addM(unsigned long long* grid, float* table);
template<>__global__ void ising<256u>(unsigned long long* grid, float H, float T, int step, float seed)
{
	//gd: {64, 1, 1}, threads: {32, 4, 1}
	//call time per M: 4552000000
	//run time per M: 2210000000
	__shared__ unsigned long long subGrid[6][4];
	unsigned int yy(blockIdx.x * 4 + threadIdx.y);
	if (threadIdx.x < 6)
		subGrid[threadIdx.x][threadIdx.y] =
		grid[((blockIdx.x * 4 + threadIdx.x + 255) % 256) * 4 + threadIdx.y];
	__syncthreads();
	unsigned char* p((unsigned char*)(subGrid[1]));
#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
#define set(ff) ((p[threadIdx.y*32+threadIdx.x]^=(1<<(ff&7))))
	int _step(step ^ (yy % 2));
	int d0(threadIdx.x * 8 + _step);
	for (int c0(d0); c0 < d0 + 8; c0 += 2)
	{
		int s1 = get(threadIdx.y, c0) + get(2 + threadIdx.y, c0);
		int dx((c0 + 255) % 256);
		s1 += get(1 + threadIdx.y, dx);
		dx = (c0 + 1) % 256;
		s1 += get(1 + threadIdx.y, dx);
		s1 = s1 * 2 - 4;
		float s0 = 2 * int(get(1 + threadIdx.y, c0)) - 1;
		s0 *= H + s1;
		if (s0 <= 0 || random(c0 - seed, yy + seed) < expf(-s0 / T))
			set(c0);
	}
	unsigned char* s = (unsigned char*)&grid[yy * 4];
	s[threadIdx.x] = p[threadIdx.y * 32 + threadIdx.x];
#undef get
#undef set
}
template<>__global__ void calcBarM<256u>(unsigned long long* grid, float* M)
{
	//gd: {1, 1, 1}, threads: {32, 1, 1}
	__shared__ volatile int ahh[32];
	int N(0);
	grid += threadIdx.x * 32;
	for (int c0(0); c0 < 32; ++c0)
	{
		unsigned long long gg = grid[c0];
		unsigned int n;
		asm("popc.b64 %0, %1;": "=r"(n) : "l"(gg));
		N += n;
	}
	ahh[threadIdx.x] = N;
	if (threadIdx.x < 16)ahh[threadIdx.x] += ahh[threadIdx.x + 16];
	if (threadIdx.x < 8)ahh[threadIdx.x] += ahh[threadIdx.x + 8];
	if (threadIdx.x < 4)ahh[threadIdx.x] += ahh[threadIdx.x + 4];
	if (threadIdx.x < 2)ahh[threadIdx.x] += ahh[threadIdx.x + 2];
	if (!threadIdx.x)
	{
		N = ahh[0] + ahh[1];
		*M = 2 * float(N) / 65536 - 1;
	}
}
template<>__global__ void addM<256u>(unsigned long long* grid, float* table)
{
	//gd: {256, 1, 1}, threads: {32, 1, 1}
	unsigned char a(((unsigned char*)(grid + 4 * blockIdx.x))[threadIdx.x]);
	table += 256 * blockIdx.x + 8 * threadIdx.x;
	for (int c0(0); c0 < 8; ++c0)
		table[c0] += (a >> c0) & 1;
}

void printTable(float* table, int num)
{
	char t[50];
	std::string ahh;
	for (int c0(0); c0 < gridDim * gridDim; ++c0)
	{
		::sprintf(t, "%2.8f ", (table[c0] * 2) / num - 1);
		ahh += t;
		if (c0 % gridDim == 255)
		{
			::sprintf(t, "\n");
			ahh += t;
		}
	}
	FILE* temp(::fopen("./gridAverage.txt", "w+"));
	::fprintf(temp, "%s", ahh.c_str());
	::fclose(temp);
}
void printGrid(unsigned long long* grid)
{
	char t[5];
	std::string ahh;
	for (int c0(0); c0 < 4 * gridDim; ++c0)
	{
		unsigned long long ft(grid[c0]);
		for (int c1(0); c1 < 64; ++c1)
		{
			::sprintf(t, "%2d ", int((ft >> c1) & 1) * 2 - 1);
			ahh += t;
		}
		if (c0 % 4 == 3)
		{
			::sprintf(t, "\n");
			ahh += t;
		}
	}
	FILE* temp(::fopen("./grid.txt", "w+"));
	::fprintf(temp, "%s", ahh.c_str());
	::fclose(temp);
}

void calcOnePoint(float H, float T, unsigned int cycles, unsigned int num,
	int seed, unsigned long long* grid, unsigned long long* gridHost)
{
}
void calcOnePoint(float H, float T, unsigned int cycles, unsigned int num,
	int seed, unsigned long long* grid, unsigned long long* gridHost, float* table, float* tableHost)
{

}


int main(int argc, char** argv)
{
	//unsigned int memSize = sizeof(float) * gridDim * gridDim;
	//float* originGrid((float*)malloc(memSize));
	//float* grid;

	//[Book1]Sheet2!1[1]:256[256]
	unsigned int gridSize = sizeof(unsigned long long) * 4 * gridDim;
	unsigned int tableSize = sizeof(float) * gridDim * gridDim;
	unsigned long long* gridHost((unsigned long long*)malloc(gridSize));
	float* tableHost((float*)::malloc(tableSize));

	unsigned long long* grid;
	float* table;
	float* Md;
	std::mt19937 mt(0);
	cudaMalloc(&Md, 4);
	cudaMalloc(&grid, gridSize);
	cudaMalloc(&table, tableSize);
	dim3  gd(64, 1, 1);
	dim3  threads(32, 4, 1);
	float H1 = 0.02941, H2 = 0;
	float T, T1 = 1.5, T2 = 2 * 2.268;
	int nH = 0;
	int nT = 0;
	int cycles = 10000;
	int num = 1000;
	::scanf("%d", &cycles);
	::scanf("%d", &num);
	H2 -= H1;
	H2 /= nH ? nH : 1;
	T2 -= T1;
	T2 /= nT ? nT : 1;
	char t[100];
	std::string answer;
	//for (int c2(0); c2 < gridDim * gridDim; ++c2)
		//originGrid[c2] =/* rd(mt) ? 1 :*/ -1;

	std::uniform_int_distribution<unsigned long long> gg;
	std::uniform_real_distribution<float> rd(0, 1.0f);
	for (int c0(0); c0 < gridDim; ++c0)
	{
		unsigned long long s;
		if (c0 % 2)s = 0xaaaaaaaaaaaaaaaa;
		else s = 0x5555555555555555;
		for (int c1(0); c1 < 4; ++c1)
			for (int c2(0); c2 < 64; ++c2)
			{
				gridHost[c0 * 4 + c1] = gg(mt)/*s*/;
			}
	}
	memset(tableHost, 0, tableSize);


	for (int c0(0); c0 <= nH; ++c0)
	{
		if (abs(H1) < 1e-5)H1 = 0;
		T = T1;
		for (int c1(0); c1 <= nT; ++c1)
		{
			//cudaDeviceSynchronize();
			cudaMemcpy(grid, gridHost, gridSize, cudaMemcpyHostToDevice);
			for (int c2(0); c2 < cycles; ++c2)
			{
				ising << < gd, threads, 0, 0 >> > (grid, H1, T, 0, rd(mt));
				ising << < gd, threads, 0, 0 >> > (grid, H1, T, 1, rd(mt));
			}

			cudaMemcpy(table, tableHost, tableSize, cudaMemcpyHostToDevice);
			for (int c2(0); c2 < num; ++c2)
			{
				ising << < gd, threads, 0, 0 >> > (grid, H1, T, 0, rd(mt));
				ising << < gd, threads, 0, 0 >> > (grid, H1, T, 1, rd(mt));
				addM << <dim3(256, 1, 1), dim3(32, 1, 1) >> > (grid, table);
				//float m;
				//calcBarM_optimize_1 << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (grid, Md);
				//cudaMemcpy(&m, Md, 4, cudaMemcpyDeviceToHost);
				//M += m;
				//cudaMemcpy(gridHost, grid, gridSize, cudaMemcpyDeviceToHost);
			}
			//cudaEventSynchronize(start);
			//cudaEventSynchronize(stop);
			//cudaEventElapsedTime(&elapsedTime, start, stop);
			//::printf("%02d %02d\n", c0, c1);
			::printf("%.8f %.8f %.8f\t", H1, T, M);
			::sprintf(t, "%.8f %.8f %.8f\n", H1, T, M);
			answer += t;
			T += T2;
			//printf("%f ms\n", elapsedTime);
		}
		H1 += H2;
	}
	cudaMemcpy(tableHost, table, tableSize, cudaMemcpyDeviceToHost);
	printTable(tableHost, num);
	//cudaMemcpy(originGrid, grid, memSize, cudaMemcpyDeviceToHost);
	//printGrid(originGrid);
	//::printf(answer.c_str());
	FILE* temp(::fopen("./answer.txt", "w+"));
	::fprintf(temp, "%s", answer.c_str());
	::fclose(temp);
	cudaFree(grid);
	cudaFree(Md);
	cudaFree(table);
	free(gridHost);
	free(tableHost);
}
