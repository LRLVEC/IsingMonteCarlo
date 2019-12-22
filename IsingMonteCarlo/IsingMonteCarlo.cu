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
__device__ unsigned int counter(unsigned char a)
{
#define BIT2(n) n, n+1, n+1, n+2
#define BIT4(n) BIT2(n), BIT2(n+1), BIT2(n+1), BIT2(n+2)
#define BIT6(n) BIT4(n), BIT4(n+1), BIT4(n+1), BIT4(n+2)
#define BIT8(n) BIT6(n), BIT6(n+1), BIT6(n+1), BIT6(n+2)
	const unsigned char table[256] = { BIT8(0) };
	return table[a];
}
__global__ void ising(float* grid, float H, float T, int dx, int seed)
{
	int y = blockIdx.x;
	int x = 2 * threadIdx.x + (y + dx) % 2;
	float S0 = grid[x + y * gridDim];
	float Sn =
		grid[(x + 1) % gridDim + y * gridDim] +
		grid[(x + gridDim - 1) % gridDim + y * gridDim] +
		grid[x + ((y + 1) % gridDim) * gridDim] +
		grid[x + ((y + gridDim - 1) % gridDim) * gridDim];
	float dE = 2 * S0 * (H + Sn);
	if (dE < 0 || random(float(x - seed), float(y + seed)) < expf(-dE / T))
		grid[x + y * gridDim] = -S0;
}
__global__ void ising_optimize0(float* grid, float H, float T, int step, int seed)//step: 0, 1
{
	__shared__ float subGrid[66][66];
	int t0 = (blockIdx.y * 64 - 1) * gridDim + 64 * blockIdx.x + threadIdx.x;
	int t1 = t0 + 32;
	for (int c0(1); c0 < 65; ++c0)
	{
		subGrid[c0][threadIdx.x + 1] = grid[t0 + c0 * gridDim];
		subGrid[c0][threadIdx.x + 33] = grid[t1 + c0 * gridDim];
	}
	subGrid[0][threadIdx.x * 2 + step + 1] = grid[((blockIdx.y * 64 + gridDim - 1) % gridDim) * gridDim + 64 * blockIdx.x + threadIdx.x * 2 + step];
	subGrid[65][threadIdx.x * 2 + !step + 1] = grid[((blockIdx.y * 64 + 1) % gridDim) * gridDim + 64 * blockIdx.x + threadIdx.x * 2 + !step];
	subGrid[threadIdx.x * 2 + step + 1][0] = grid[(blockIdx.y * 64 + threadIdx.x * 2 + step) * gridDim + (blockIdx.x * 64 + gridDim - 1) % gridDim];
	subGrid[threadIdx.x * 2 + !step + 1][65] = grid[(blockIdx.y * 64 + threadIdx.x * 2 + !step) * gridDim + (blockIdx.x * 64 + 1) % gridDim];
	__syncthreads();
	int y0 = 2 * threadIdx.x + 1;
	int y1 = y0 + !step;
	y0 += step;
	for (int c0(1 + step); c0 < 65; ++c0)
	{
		float ds = subGrid[y0][c0];
		float dE = 2 * ds * (H + subGrid[y0 - 1][c0] + subGrid[y0 + 1][c0] + subGrid[y0][c0 - 1] + subGrid[y0][c0 + 1]);
		if (dE < 0 || random(float(y0 + c0 - seed), float(t0 + seed)) < expf(-dE / T))subGrid[y0][c0] = -ds;
		ds = subGrid[y1][++c0];
		dE = 2 * ds * (H + subGrid[y1 - 1][c0] + subGrid[y1 + 1][c0] + subGrid[y1][c0 - 1] + subGrid[y1][c0 + 1]);
		if (dE < 0 || random(float(y0 + c0 - seed), float(t0 + seed)) < expf(-dE / T))subGrid[y1][c0] = -ds;
	}
	__syncthreads();
	for (int c0(1); c0 < 65; ++c0)
	{
		grid[t0 + c0 * gridDim] = subGrid[c0][threadIdx.x + 1];
		grid[t1 + c0 * gridDim] = subGrid[c0][threadIdx.x + 33];
	}
}
__global__ void ising_optimize1(float* grid, float H, float T, int step, int seed)
{
	__shared__ float subGrid[8][8];
	int dx = (threadIdx.x & 3) << 1;
	int dy = threadIdx.x >> 2;
	int x = blockIdx.x * 8 + dx;
	int y = blockIdx.y * 8 + dy;
	subGrid[dy][dx] = grid[y * gridDim + x];
	subGrid[dy][dx + 1] = grid[y * gridDim + x + 1];
	int k = (step ^ (dy & 1));
	dx += k;
	x += k;
	float s0 = subGrid[dy][dx];
	float s1 = dx < 7 ? subGrid[dy][dx + 1] : grid[y * gridDim + (x + 1) % gridDim];
	s1 += dx ? subGrid[dy][dx - 1] : grid[y * gridDim + (x + gridDim - 1) % gridDim];
	s1 += dy < 7 ? subGrid[dy + 1][dx] : grid[((y + 1) % gridDim) * gridDim + x];
	s1 += dy ? subGrid[dy - 1][dx] : grid[((y + gridDim - 1) % gridDim) * gridDim + x];
	s0 *= 2 * (H + s1);
	if (s0 < 0 || random(float(x - seed), float(y + seed)) < expf(-s0 / T))
		grid[y * gridDim + x] = -subGrid[dy][dx];
}
__global__ void ising_optimize1_1(float* grid, float H, float T, int step, int seed)
{
	//gd: {128, 1, 1}, threads: {32, 1, 1}
	__shared__ float subGrid[4][gridDim];
	int y = blockIdx.x * 2;
	for (int c0(2 * threadIdx.x + !step); c0 < gridDim; c0 += 64)
		subGrid[0][c0] = grid[((y + gridDim - 1) % gridDim) * gridDim + c0];
	for (int c0(threadIdx.x); c0 < gridDim; c0 += 32)
		subGrid[1][c0] = grid[y * gridDim + c0];
	for (int c0(threadIdx.x); c0 < gridDim; c0 += 32)
		subGrid[2][c0] = grid[((y + 1) % gridDim) * gridDim + c0];
	for (int c0(2 * threadIdx.x + !step); c0 < gridDim; c0 += 64)
		subGrid[3][c0] = grid[((y + 2) % gridDim) * gridDim + c0];
	int y0(y + (threadIdx.x + step) % 2);
	y = (y == y0) ? 1 : 2;
	for (int c0(0); c0 < gridDim; c0 += 32)
	{
		int x(c0 + threadIdx.x);
		float s1 =
			subGrid[y - 1][x] +
			subGrid[y + 1][x] +
			subGrid[y][(x + gridDim - 1) % gridDim] +
			subGrid[y][(x + 1) % gridDim];
		float s0(subGrid[y][x] * 2 * (H + s1));
		if (s0 < 0 || random(float(x - seed), float(y0 + seed)) < expf(-s0 / T))
			grid[y0 * gridDim + x] = -subGrid[y][x];
	}
}
__global__ void ising_optimize2(float* M, float H, float T, float seed0, float seed1, int cycle)
{
	__shared__ unsigned int grid[256][8];
	__shared__ unsigned int sum[32];
#define get(a, b) (!(grid[b][a>>5]&(1<<(a&31))))
#define set(a, b) (grid[b][a>>5]^=(1<<(a&31)))
	const int y0 = threadIdx.x << 3;
	float p0(seed0 + threadIdx.x - blockIdx.x);
	float p1(seed1 - threadIdx.x + blockIdx.x);
	//init
	for (int c0(0); c0 < 256; c0 += 4)
		grid[(threadIdx.x >> 3) + c0][threadIdx.x & 7] = 0;
	for (int n = 0; n < (cycle << 1); ++n)
	{
		for (int y(y0); y < y0 + 8; ++y)
		{
			for (int x(n & 1); x < 256; x += 2)
			{
				unsigned char qq;
				int s0 = 2 * get(x, y) - 1;
				qq = x - 1;
				int s1 = get(qq, y);
				qq = x + 1;
				s1 += get(qq, y);
				qq = y - 1;
				s1 += get(x, qq);
				qq = y + 1;
				s1 += get(x, qq);
				s1 = s1 * 2 - 4;
				float de = s0 * 2 * (H + s1);
				if (de < 0)set(x, y);
				else if (random(p0 - x, p1 + y) < expf(-de / T))set(x, y);
			}
		}
		p0 = random(p0, p1);
		p1 = random(p0, p1);
	}
	unsigned char* cg = 256 * threadIdx.x + (unsigned char*)grid;
	int m = 0;
	for (int c0(0); c0 < 256; ++c0)m += counter(cg[c0]);
	sum[threadIdx.x] = m;
	if (threadIdx.x % 2 == 0)sum[threadIdx.x] += sum[threadIdx.x + 1];
	if (threadIdx.x % 4 == 0)sum[threadIdx.x] += sum[threadIdx.x + 2];
	if (threadIdx.x % 8 == 0)sum[threadIdx.x] += sum[threadIdx.x + 4];
	if (threadIdx.x % 16 == 0)sum[threadIdx.x] += sum[threadIdx.x + 8];
	if (threadIdx.x == 0)*M = float(sum[0] + sum[16]) / 65536;
#undef get
#undef set
}
//__inline__ __device__ void shiftLeftByOne4(unsigned long long* a, unsigned long long* b)
//{
//	b[3] = (a[3] << 1) | (a[2] >> 63);
//	b[2] = (a[2] << 1) | (a[1] >> 63);
//	b[1] = (a[1] << 1) | (a[0] >> 63);
//	b[0] = (a[0] << 1) | (a[3] >> 63);
//}
//__inline__ __device__ void shiftRightByOne4(unsigned long long* a, unsigned long long* b)
//{
//	b[0] = (a[0] >> 1) | (a[1] << 63);
//	b[1] = (a[1] >> 1) | (a[2] << 63);
//	b[2] = (a[2] >> 1) | (a[3] << 63);
//	b[3] = (a[3] >> 1) | (a[0] << 63);
//}
//__inline__ __device__ void shiftLeftByTwo4(unsigned long long* a, unsigned long long* b)
//{
//	b[3] = (a[3] << 2) | (a[2] >> 62);
//	b[2] = (a[2] << 2) | (a[1] >> 62);
//	b[1] = (a[1] << 2) | (a[0] >> 62);
//	b[0] = (a[0] << 2) | (a[3] >> 62);
//}
//__inline__ __device__ void shiftRightByTwo4(unsigned long long* a, unsigned long long* b)
//{
//	b[0] = (a[0] >> 2) | (a[1] << 62);
//	b[1] = (a[1] >> 2) | (a[2] << 62);
//	b[2] = (a[2] >> 2) | (a[3] << 62);
//	b[3] = (a[3] >> 2) | (a[0] << 62);
//}
//__inline__ __device__ void and4(unsigned long long* a, unsigned long long* b, unsigned long long* c)
//{
//	c[0] = a[0] & b[0];
//	c[1] = a[1] & b[1];
//	c[2] = a[2] & b[2];
//	c[3] = a[3] & b[3];
//}
//__inline__ __device__ void or4(unsigned long long* a, unsigned long long* b, unsigned long long* c)
//{
//	c[0] = a[0] | b[0];
//	c[1] = a[1] | b[1];
//	c[2] = a[2] | b[2];
//	c[3] = a[3] | b[3];
//}
//__inline__ __device__ void xor4(unsigned long long* a, unsigned long long* b, unsigned long long* c)
//{
//	c[0] = a[0] ^ b[0];
//	c[1] = a[1] ^ b[1];
//	c[2] = a[2] ^ b[2];
//	c[3] = a[3] ^ b[3];
//}
//__inline__ __device__ void not4(unsigned long long* a, unsigned long long* b)
//{
//	b[0] = ~a[0];
//	b[1] = ~a[1];
//	b[2] = ~a[2];
//	b[3] = ~a[3];
//}
//__inline__ __device__ void chooseHigh(unsigned long long* a)
//{
//	a[0] &= 0xAAAAAAAAAAAAAAAA;
//	a[1] &= 0xAAAAAAAAAAAAAAAA;
//	a[2] &= 0xAAAAAAAAAAAAAAAA;
//	a[3] &= 0xAAAAAAAAAAAAAAAA;
//}
//__inline__ __device__ void chooseLow(unsigned long long* a)
//{
//	a[0] &= 0x5555555555555555;
//	a[1] &= 0x5555555555555555;
//	a[2] &= 0x5555555555555555;
//	a[3] &= 0x5555555555555555;
//}
__global__ void ising_optimize3(unsigned long long* grid, float H, float T, int step, int seed)
{
	//gd: {256, 1, 1}, threads: {32, 1, 1}
	__shared__ unsigned long long subGrid[3][4];
	if (threadIdx.x < 12)
		*((unsigned long long*)(subGrid) + threadIdx.x) =
		grid[((blockIdx.x + gridDim - 1) % gridDim) * 4 + threadIdx.x];
	unsigned char* p((unsigned char*)(subGrid[1]));
#define get(y, x) ((subGrid[y][x>>6]>>(x&63))&1)
#define set(ff) ((p[threadIdx.x]^=(1<<(ff&7))))
	step ^= blockIdx.x % 2;
	int d0(threadIdx.x * 8 + step);
	for (int c0(d0); c0 < d0 + 8; c0 += 2)
	{
		int s1 = get(0, c0) + get(2, c0);
		int dx((c0 + gridDim - 1) % gridDim);
		s1 += get(1, dx);
		dx = (c0 + 1) % gridDim;
		s1 += get(1, dx);
		s1 = s1 * 2 - 4;
		float s0 = 2 * get(1, c0);
		s0 *= H + s1;
		if (s0 < 0 || random(float(c0 - seed), float(blockIdx.x + seed)) < expf(-s0 / T))
			set(c0);//bug here
	}
	unsigned char* s = (unsigned char*)&grid[blockIdx.x * 4];
	s[threadIdx.x] = p[threadIdx.x];
#undef get
#undef set
}
__global__ void ising_optimize4(unsigned long long* grid, float H, float T, int step, int seed)
{
	__shared__ unsigned int subGrid[3][6];

}
__global__ void calcM(float* grid, float* M)
{
	float m = 0;
	for (int c0(0); c0 < gridDim * gridDim; ++c0)
		m += grid[c0];
	*M = m / (gridDim * gridDim);
}
__device__ __inline__ int countBit(unsigned long long a)
{
	a = (a & 0x5555555555555555) + ((a >> 1) & 0x5555555555555555);
	a = (a & 0x3333333333333333) + ((a >> 2) & 0x3333333333333333);
	a = (a & 0x0F0F0F0F0F0F0F0F) + ((a >> 4) & 0x0F0F0F0F0F0F0F0F);
	a = (a & 0x00FF00FF00FF00FF) + ((a >> 8) & 0x00FF00FF00FF00FF);
	a = (a & 0x0000FFFF0000FFFF) + ((a >> 16) & 0x0000FFFF0000FFFF);
	a = (a & 0x00000000FFFFFFFF) + ((a >> 32) & 0x00000000FFFFFFFF);
	return a;
}
__global__ void calcM_optimize(unsigned long long* grid, float* M)
{
	float m = 0;
	for (int c0(0); c0 < 4 * gridDim; ++c0)
		m += countBit(grid[c0]);
	*M = (2 * m / (gridDim * gridDim)) - 1;
}
int main(int argc, char** argv)
{
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//float elapsedTime;
	//unsigned int memSize = sizeof(float) * gridDim * gridDim;
	//float* originGrid((float*)malloc(memSize));
	//float* grid;
	unsigned int memSize = sizeof(unsigned long long) * 4 * gridDim;
	unsigned long long* originGrid((unsigned long long*)malloc(memSize));
	unsigned long long* grid;
	float* Md;
	std::mt19937 mt(0);
	cudaMalloc(&Md, 4);
	cudaMalloc(&grid, memSize);
	dim3  gd(256, 1, 1);
	dim3  threads(32, 1, 1);
	float H1 = -0.1f, H2 = 0.1f;
	float T, T1 = 0.1 * 2.268, T2 = 1.9 * 2.268;
	int nH = 2;
	int nT = 5;
	int cycles = 1;
	//::scanf("%f%f%f%f%d%d", &H, &T1, &T2, &n, &cycles);
	//::printf("%f%f%f%f%d%d", H, J, T1, T2, n, cycles);
	H2 -= H1;
	H2 /= nH;
	T2 -= T1;
	T2 /= nT;
	char t[100];
	std::string answer;
	//for (int c2(0); c2 < gridDim * gridDim; ++c2)
		//originGrid[c2] =/* rd(mt) ? 1 :*/ -1;
	for (int c0(0); c0 < gridDim; ++c0)
	{
		unsigned long long s;
		if (c0 % 2)s = 0xaaaaaaaaaaaaaaaa;
		else s = 0x5555555555555555;
		for (int c1(0); c1 < 4; ++c1)
			originGrid[c0 * 4 + c1] = s;
	}
	std::uniform_real_distribution<float> rd(0, 1.0f);
	Timer timer;
	for (int c0(0); c0 <= nH; ++c0)
	{
		T = T1;
		for (int c1(0); c1 <= nT; ++c1)
		{
			timer.begin();
			cudaMemcpy(grid, originGrid, memSize, cudaMemcpyHostToDevice);

			//cudaEventRecord(start, 0);
			for (int c2(0); c2 < cycles; ++c2)
			{
				//ising_optimize2 << < gd, threads, 0, 0 >> > (Md, H1, T, rd(mt), rd(mt), cycles);
				//ising_optimize3 << < gd, threads, 0, 0 >> > (grid, H1, T, 0, rd(mt));
				ising_optimize3 << < gd, threads, 0, 0 >> > (grid, H1, T, 1, rd(mt));
			}
			//cudaEventRecord(stop, 0);
			float M;
			calcM_optimize << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (grid, Md);
			cudaMemcpy(&M, Md, 4, cudaMemcpyDeviceToHost);
			timer.end();
			//cudaEventSynchronize(start);
			//cudaEventSynchronize(stop);
			//cudaEventElapsedTime(&elapsedTime, start, stop);
			//::printf("%02d %02d\n", c0, c1);
			::printf("%.8f %.8f %.8f\t", H1, T, M);
			timer.print();
			::sprintf(t, "%.8f %.8f %.8f\n", H1, T, M);
			answer += t;
			T += T2;
			//printf("%f ms\n", elapsedTime);
		}
		H1 += H2;
	}
	//::printf(answer.c_str());
	FILE* temp(::fopen("./answer_0.205_0.4.txt", "w+"));
	::fprintf(temp, "%s", answer.c_str());
	::fclose(temp);
	//for (int c0(0); c0 < testNum; ++c0)
	//	printf("%f ms\n", elapsedTime[c0]);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//cudaFree(grid);
	cudaFree(Md);
	//::free(originGrid);
}
