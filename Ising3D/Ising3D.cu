#include <_Time.h>
#include <_File.h>
#include <_String.h>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <curand_kernel.h>


constexpr unsigned int blockHeight = 4;

//sizes: 128, 256, 512, 1024
const dim3 Ising3DBlockSize[4]
{
	{1, 32,1},
	{1, 64,1},
	{4, 128,1},
	{16, 256,1},
};

const dim3 Ising3DThreadSize[4]
{
	{8, 128, 1},
	{4, 256, 1},
	{8, 128, 1},
	{16, 64, 1},
};

const dim3 RandomBlockSize[4]
{
	{1, 1, 1},
	{2, 1, 1},
	{16, 1, 1},
	{128, 1, 1},
};

//test for size of 1024
__global__ void initRandom(curandState* states, unsigned int seed)
{
	int id((blockIdx.x * blockDim.x + threadIdx.x) * 32);
	for (unsigned int c0(0); c0 < 32; ++c0)
	{
		int idd(id + c0);
		curand_init(seed, idd, 0, states + idd);
	}
}
__global__ void ising3d(unsigned long long* grid, float H, float T, int step, curandState* states)
{
	constexpr unsigned int dim = 1024;
	constexpr unsigned int dimM1 = 1023;
	__shared__ unsigned long long subGrid[3][66][16];
	//unsigned long long* g(grid+);
	unsigned int c0(blockIdx.y * blockHeight);
	unsigned int c0e(c0 + blockHeight);
	unsigned int idx(0);
	int randIdx(blockIdx.y * 16384 + (threadIdx.y + blockIdx.x * 64) * 16 + threadIdx.x);
	curandState state(states[randIdx]);
	//gather the extra rows
	if (threadIdx.y == 0)
		subGrid[1][0][threadIdx.x] = grid[c0 * 16384 + ((blockIdx.x * 64 + dimM1) & dimM1) * 16 + threadIdx.x];
	else if (threadIdx.y == 1)
		subGrid[1][65][threadIdx.x] = grid[c0 * 16384 + (((blockIdx.x + 1) * 64) & dimM1) * 16 + threadIdx.x];
	//gather the main part
	subGrid[0][threadIdx.y + 1][threadIdx.x] = grid[((c0 + dimM1) & dimM1) * 16384 + (blockIdx.x * 64 + threadIdx.y) * 16 + threadIdx.x];
	subGrid[1][threadIdx.y + 1][threadIdx.x] = grid[c0 * 16384 + (blockIdx.x * 64 + threadIdx.y) * 16 + threadIdx.x];
	for (; c0 < c0e; ++c0)
	{
		idx = (idx + 1) % 3;
		if (c0 < c0e - 1)
		{
			if (threadIdx.y == 0)
				subGrid[(idx + 1) % 3][0][threadIdx.x] = grid[((c0 + 1) & dimM1) * 16384 + ((blockIdx.x * 64 + dimM1) & dimM1) * 16 + threadIdx.x];
			else if (threadIdx.y == 1)
				subGrid[(idx + 1) % 3][65][threadIdx.x] = grid[((c0 + 1) & dimM1) * 16384 + (((blockIdx.x + 1) * 64) & dimM1) * 16 + threadIdx.x];
		}
		subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x] = grid[((c0 + 1) & dimM1) * 16384 + (blockIdx.x * 64 + threadIdx.y) * 16 + threadIdx.x];
		__syncthreads();
		unsigned long long center;
		unsigned long long nearby[4];
		int s0, s1;
		nearby[0] = subGrid[(idx + 2) % 3][threadIdx.y + 1][threadIdx.x];
		nearby[1] = subGrid[idx][threadIdx.y][threadIdx.x];
		center = subGrid[idx][threadIdx.y + 1][threadIdx.x];
		nearby[2] = subGrid[idx][threadIdx.y + 2][threadIdx.x];
		nearby[3] = subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x];
#define get(x, y) (((x) >> (y)) & 1)
#define set(x) (center ^= (1 << x))
		if (step == 0)
			s0 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + 15) & 15], 63);
		for (int c1(step); c1 < 64; c1 += 2)
		{
			if (c1 < 63)
				s1 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + 1) & 15], 0);
			else
				s1 = get(center, c1 + 1);
			int s(s0 + s1);
			for (int c2(0); c2 < 4; ++c2)
				s += get(nearby[c2], c1);
			s = s * 2 - 6;
			int ss(get(center, c1));
			float sf(2 * ss - 1);
			sf *= H + s;
			if (sf <= 0 || curand_uniform(&state) < expf(-2 * sf / T))
				set(c1);
			s0 = s1;
		}
		grid[c0 * 16384 + (blockIdx.x * 64 + threadIdx.y) * 16 + threadIdx.x] = center;
	}
	states[randIdx] = state;
#undef get
#undef set
}

int main()
{
	File file("./");
	float T(1.9);
	float H(1);
	Timer timer;
	std::mt19937 mt(time(0));
	std::uniform_int_distribution<unsigned long long>rd;
	File& sf(file.findInThis("states.bin"));
	bool statesNeeded(false);
	if (&sf == nullptr)statesNeeded = true;
	size_t gridSize(1024 * 1024 * 1024 / 8);
	size_t statesSize(sizeof(curandState) * 1024 * RandomBlockSize[3].x * 32);
	void* states(::malloc(statesSize));
	curandState* statesDevice;
	unsigned long long* grid((unsigned long long*)::malloc(gridSize));
	unsigned long long* gridDevice;
	for (unsigned long long c0(0); c0 < gridSize / sizeof(unsigned long long); ++c0)grid[c0] = rd(mt);
	cudaMalloc(&gridDevice, gridSize);
	cudaMalloc(&statesDevice, statesSize);
	cudaMemcpy(gridDevice, grid, gridSize, cudaMemcpyHostToDevice);

	if (statesNeeded)
	{
		cudaDeviceSynchronize();
		timer.begin();
		initRandom << <RandomBlockSize[3], 1024, 0, 0 >> > (statesDevice, rd(mt));
		cudaDeviceSynchronize();
		timer.end();
		timer.print("Generate states: ");
	}
	else
	{
		timer.begin();
		Vector<unsigned char>ss(sf.readBinary());
		cudaMemcpy(statesDevice, ss.data, statesSize, cudaMemcpyHostToDevice);
		timer.end();
		timer.print("Read states: ");
	}

	cudaDeviceSynchronize();
	timer.begin();
	for (unsigned long long c0(0); c0 < 100; ++c0)
	{
		ising3d << <Ising3DBlockSize[3], Ising3DThreadSize[3], 0, 0 >> > (gridDevice, H, T, 0, statesDevice);
		ising3d << <Ising3DBlockSize[3], Ising3DThreadSize[3], 0, 0 >> > (gridDevice, H, T, 1, statesDevice);
	}
	cudaDeviceSynchronize();
	timer.end();
	timer.print("Calculate Ising3D: ");

	timer.begin();
	cudaMemcpy(states, statesDevice, statesSize, cudaMemcpyDeviceToHost);
	file.createBinary("states.bin", states, statesSize);
	timer.end();
	timer.print("Store states: ");

	cudaFree(gridDevice);
	cudaFree(statesDevice);
	free(grid);
	free(states);
}