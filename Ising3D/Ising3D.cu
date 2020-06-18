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
	{1, 8,1},
	{1, 64,1},//blockHeight == 4
	{4, 32,1},
	{16, 64,1},
};

const dim3 Ising3DThreadSize[4]
{
	{8, 128, 1},
	{4, 256, 1},
	{8, 128, 1},
	{16, 64, 1},
};

const dim3 InitGridBlockSize[4]
{
	{2, 1, 1},
	{64, 1, 1},//blockHeight == 4
	{128, 1, 1},//blockHeight == 16
	{1024, 1, 1},//blockHeight == 16
};

const unsigned int RandomSize[4]{ 2048,16384,131072,1048576 };

const dim3 RandomBlockSize[4]
{
	{2, 1, 1},
	{64, 1, 1},//blockHeight == 4
	{128, 1, 1},//blockHeight == 16
	{256, 1, 1},//blockHeight == 16
};

constexpr unsigned int ReductionBlockSize[4]{ 32,256, 1024, 1024 };

constexpr __device__ unsigned int getSubWidth(unsigned int dim)
{
	if (dim > 256)return 256 / (dim / 256) + 2;
	else return dim;
}

//test for size of 1024
__global__ void initRandom(curandState* states, unsigned int seed, unsigned long long nStates)
{
	int id(blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int gridSize(blockDim.x * gridDim.x);
	while (id < nStates)
	{
		curand_init(seed, id, 0, states + id);
		id += gridSize;
	}
}
__global__ void initGrid(unsigned long long* grid, curandState* states, unsigned long long N)
{
	unsigned long long stateIdx(threadIdx.x + 1024 * blockIdx.x);
	unsigned long long idx(stateIdx);
	unsigned long long gridSize(gridDim.x * 1024);
	curandState state(states[stateIdx]);
	while (idx < N)
	{
		unsigned long long a(curand(&state));
		a <<= 32;
		a |= curand(&state);
		grid[idx] = a;
		idx += gridSize;
	}
	states[stateIdx] = state;
}
template<unsigned long long dim>__global__ void ising3d(unsigned long long* grid, float H, float T, int step, curandState* states)
{
	constexpr unsigned int dimM1(dim - 1);
	constexpr unsigned int bandWidth(256 / (dim / 256));
	constexpr unsigned int subGridBandWidth(getSubWidth(dim));
	constexpr unsigned int rowLength(dim / 64);
	constexpr unsigned long long layerSize(dim * dim / 64);
	__shared__ unsigned long long subGrid[3][subGridBandWidth][rowLength];
	__shared__ float table[2][7];
	unsigned int c0(blockIdx.y * blockHeight);
	unsigned int c0e(c0 + blockHeight);
	unsigned int idx(0);
	int randIdx(blockIdx.y * layerSize + (threadIdx.y + blockIdx.x * bandWidth) * rowLength + threadIdx.x);
	curandState state(states[randIdx]);
	//build table
	if (threadIdx.x < 2 && threadIdx.y < 7)
		table[threadIdx.x][threadIdx.y] = expf((2 - 4 * int(threadIdx.x)) * (H + float(2 * threadIdx.y) - 6) / T);
	//gather the extra rows, for dim == 1024, 512
	if constexpr (dim > 256)
	{
		if (threadIdx.y == 0)
			subGrid[1][0][threadIdx.x] = grid[c0 * layerSize + ((blockIdx.x * bandWidth + dimM1) & dimM1) * rowLength + threadIdx.x];
		else if (threadIdx.y == 1)
			subGrid[1][bandWidth + 1][threadIdx.x] = grid[c0 * layerSize + (((blockIdx.x + 1) * bandWidth) & dimM1) * rowLength + threadIdx.x];
		//gather the main part
		subGrid[0][threadIdx.y + 1][threadIdx.x] = grid[((c0 + dimM1) & dimM1) * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
		subGrid[1][threadIdx.y + 1][threadIdx.x] = grid[c0 * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
	}
	else
	{
		//gather the main part
		subGrid[0][threadIdx.y][threadIdx.x] = grid[((c0 + dimM1) & dimM1) * layerSize + threadIdx.y * rowLength + threadIdx.x];
		subGrid[1][threadIdx.y][threadIdx.x] = grid[c0 * layerSize + threadIdx.y * rowLength + threadIdx.x];
	}
	for (; c0 < c0e; ++c0)
	{
		idx = (idx + 1) % 3;
		if constexpr (dim > 256)
		{
			if (c0 < c0e - 1)
			{
				if (threadIdx.y == 0)
					subGrid[(idx + 1) % 3][0][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + ((blockIdx.x * bandWidth + dimM1) & dimM1) * rowLength + threadIdx.x];
				else if (threadIdx.y == 1)
					subGrid[(idx + 1) % 3][bandWidth + 1][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + (((blockIdx.x + 1) * bandWidth) & dimM1) * rowLength + threadIdx.x];
			}
		}
		if constexpr (dim > 256)
			subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
		else
			subGrid[(idx + 1) % 3][threadIdx.y][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + threadIdx.y * rowLength + threadIdx.x];
		__syncthreads();
		unsigned long long center;
		unsigned long long nearby[4];
		int s0, s1;
		if constexpr (dim > 256)
		{
			nearby[0] = subGrid[(idx + 2) % 3][threadIdx.y + 1][threadIdx.x];
			nearby[1] = subGrid[idx][threadIdx.y][threadIdx.x];
			center = subGrid[idx][threadIdx.y + 1][threadIdx.x];
			nearby[2] = subGrid[idx][threadIdx.y + 2][threadIdx.x];
			nearby[3] = subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x];
		}
		else
		{
			constexpr unsigned int bdwM1(bandWidth - 1);
			nearby[0] = subGrid[(idx + 2) % 3][threadIdx.y][threadIdx.x];
			nearby[1] = subGrid[idx][(threadIdx.y + bdwM1) & bdwM1][threadIdx.x];
			center = subGrid[idx][threadIdx.y][threadIdx.x];
			nearby[2] = subGrid[idx][(threadIdx.y + 1) & bdwM1][threadIdx.x];
			nearby[3] = subGrid[(idx + 1) % 3][threadIdx.y][threadIdx.x];
		}
#define get(x, y) (((x) >> (y)) & 1)
#define set(x) (center ^= (1llu << (x)))
		int stepNow((step + c0 + threadIdx.y) & 1);
		constexpr unsigned int rlM1(rowLength - 1);
		if (stepNow == 0)
			s0 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + rlM1) & rlM1], 63);
		else
			s0 = center & 1;
		for (int c1(stepNow); c1 < 64; c1 += 2)
		{
			if (c1 < 63)
				s1 = get(center, c1 + 1);
			else
				s1 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + 1) & rlM1], 0);
			s0 += s1;
			for (int c2(0); c2 < 4; ++c2)
				s0 += get(nearby[c2], c1);
			int ss(get(center, c1));
			if (curand_uniform(&state) < table[ss][s0])
				set(c1);
			s0 = s1;
		}
		if constexpr (dim > 256)
			grid[c0 * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x] = center;
		else
			grid[c0 * layerSize + threadIdx.y * rowLength + threadIdx.x] = center;
	}
	states[randIdx] = state;
#undef get
#undef set
}
//template<unsigned long long dim>__global__ void ising3d(unsigned long long* grid, float H, float T, int step, curandState* states, float* MList, float* EList)
//{
//	constexpr unsigned int dimM1(dim - 1);
//	constexpr unsigned int bandWidth(256 / (dim / 256));
//	constexpr unsigned int subGridBandWidth(getSubWidth(dim));
//	constexpr unsigned int rowLength(dim / 64);
//	constexpr unsigned long long layerSize(dim * dim / 64);
//	__shared__ unsigned long long subGrid[3][subGridBandWidth][rowLength];
//	__shared__ float table[2][7];
//	unsigned int c0(blockIdx.y * blockHeight);
//	unsigned int c0e(c0 + blockHeight);
//	unsigned int idx(0);
//	int randIdx(blockIdx.y * layerSize + (threadIdx.y + blockIdx.x * bandWidth) * rowLength + threadIdx.x);
//	curandState state(states[randIdx]);
//	//build table
//	if (threadIdx.x < 2 && threadIdx.y < 7)
//		table[threadIdx.x][threadIdx.y] = expf((2 - 4 * int(threadIdx.x)) * (H + float(2 * threadIdx.y) - 6) / T);
//	//gather the extra rows, for dim == 1024, 512
//	if constexpr (dim > 256)
//	{
//		if (threadIdx.y == 0)
//			subGrid[1][0][threadIdx.x] = grid[c0 * layerSize + ((blockIdx.x * bandWidth + dimM1) & dimM1) * rowLength + threadIdx.x];
//		else if (threadIdx.y == 1)
//			subGrid[1][bandWidth + 1][threadIdx.x] = grid[c0 * layerSize + (((blockIdx.x + 1) * bandWidth) & dimM1) * rowLength + threadIdx.x];
//		//gather the main part
//		subGrid[0][threadIdx.y + 1][threadIdx.x] = grid[((c0 + dimM1) & dimM1) * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
//		subGrid[1][threadIdx.y + 1][threadIdx.x] = grid[c0 * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
//	}
//	else
//	{
//		//gather the main part
//		subGrid[0][threadIdx.y][threadIdx.x] = grid[((c0 + dimM1) & dimM1) * layerSize + threadIdx.y * rowLength + threadIdx.x];
//		subGrid[1][threadIdx.y][threadIdx.x] = grid[c0 * layerSize + threadIdx.y * rowLength + threadIdx.x];
//	}
//	for (; c0 < c0e; ++c0)
//	{
//		idx = (idx + 1) % 3;
//		if constexpr (dim > 256)
//		{
//			if (c0 < c0e - 1)
//			{
//				if (threadIdx.y == 0)
//					subGrid[(idx + 1) % 3][0][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + ((blockIdx.x * bandWidth + dimM1) & dimM1) * rowLength + threadIdx.x];
//				else if (threadIdx.y == 1)
//					subGrid[(idx + 1) % 3][bandWidth + 1][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + (((blockIdx.x + 1) * bandWidth) & dimM1) * rowLength + threadIdx.x];
//			}
//		}
//		if constexpr (dim > 256)
//			subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x];
//		else
//			subGrid[(idx + 1) % 3][threadIdx.y][threadIdx.x] = grid[((c0 + 1) & dimM1) * layerSize + threadIdx.y * rowLength + threadIdx.x];
//		__syncthreads();
//		unsigned long long center;
//		unsigned long long nearby[4];
//		int s0, s1;
//		if constexpr (dim > 256)
//		{
//			nearby[0] = subGrid[(idx + 2) % 3][threadIdx.y + 1][threadIdx.x];
//			nearby[1] = subGrid[idx][threadIdx.y][threadIdx.x];
//			center = subGrid[idx][threadIdx.y + 1][threadIdx.x];
//			nearby[2] = subGrid[idx][threadIdx.y + 2][threadIdx.x];
//			nearby[3] = subGrid[(idx + 1) % 3][threadIdx.y + 1][threadIdx.x];
//		}
//		else
//		{
//			constexpr unsigned int bdwM1(bandWidth - 1);
//			nearby[0] = subGrid[(idx + 2) % 3][threadIdx.y][threadIdx.x];
//			nearby[1] = subGrid[idx][(threadIdx.y + bdwM1) & bdwM1][threadIdx.x];
//			center = subGrid[idx][threadIdx.y][threadIdx.x];
//			nearby[2] = subGrid[idx][(threadIdx.y + 1) & bdwM1][threadIdx.x];
//			nearby[3] = subGrid[(idx + 1) % 3][threadIdx.y][threadIdx.x];
//		}
//#define get(x, y) (((x) >> (y)) & 1)
//#define set(x) (center ^= (1llu << (x)))
//		int stepNow((step + c0 + threadIdx.y) & 1);
//		constexpr unsigned int rlM1(rowLength - 1);
//		if (stepNow == 0)
//			s0 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + rlM1) & rlM1], 63);
//		else
//			s0 = center & 1;
//		for (int c1(stepNow); c1 < 64; c1 += 2)
//		{
//			if (c1 < 63)
//				s1 = get(center, c1 + 1);
//			else
//				s1 = get(subGrid[idx][threadIdx.y + 1][(threadIdx.x + 1) & rlM1], 0);
//			s0 += s1;
//			for (int c2(0); c2 < 4; ++c2)
//				s0 += get(nearby[c2], c1);
//			int ss(get(center, c1));
//			if (curand_uniform(&state) < table[ss][s0])
//				set(c1);
//			s0 = s1;
//		}
//		if constexpr (dim > 256)
//			grid[c0 * layerSize + (blockIdx.x * bandWidth + threadIdx.y) * rowLength + threadIdx.x] = center;
//		else
//			grid[c0 * layerSize + threadIdx.y * rowLength + threadIdx.x] = center;
//	}
//	states[randIdx] = state;
//#undef get
//#undef set
//}
template<unsigned long long blockSize>__device__ void warpReduce(volatile unsigned long long* sdata, unsigned int tid)
{
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template<unsigned long long blockSize>__global__ void isingReduction(unsigned long long* a, unsigned long long* b, unsigned long long N)
{
	unsigned int tid(threadIdx.x);
	unsigned long long i(blockIdx.x * blockSize + tid);
	unsigned long long gridSize(blockSize * gridDim.x);
	unsigned long long ans(0);
	while (i < N)
	{
		unsigned long long gg = a[i];
		unsigned int n;
		asm("popc.b64 %0, %1;": "=r"(n) : "l"(gg));
		ans += n;
		i += gridSize;
	}
	__shared__ unsigned long long sdata[blockSize];//must fix it size and don't use extern!!!
	sdata[tid] = ans;
	__syncthreads();
	if constexpr (blockSize == 1024)
	{
		if (tid < 512)
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if constexpr (blockSize >= 512)
	{
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if constexpr (blockSize >= 256)
	{
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if constexpr (blockSize >= 128)
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if (tid < 32)warpReduce<blockSize>(sdata, tid);
	if (tid == 0)b[blockIdx.x] = sdata[0];
}
template<unsigned long long blockSize>__global__ void reduction(unsigned long long* a, unsigned long long* b, unsigned long long N)
{
	unsigned int tid(threadIdx.x);
	unsigned long long i(blockIdx.x * blockSize + tid);
	unsigned long long gridSize(blockSize * gridDim.x);
	unsigned long long ans(0);
	while (i < N)
	{
		ans += a[i];
		i += gridSize;
	}
	__shared__ unsigned long long sdata[blockSize];//must fix it size and don't use extern!!!
	sdata[tid] = ans;
	__syncthreads();
	if constexpr (blockSize == 1024)
	{
		if (tid < 512)
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if constexpr (blockSize >= 512)
	{
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if constexpr (blockSize >= 256)
	{
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if constexpr (blockSize >= 128)
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if (tid < 32)warpReduce<blockSize>(sdata, tid);
	if (tid == 0)b[blockIdx.x] = sdata[0];
}

constexpr unsigned int chooseIdx(unsigned long long dim)
{
	if (dim == 1024)return 3;
	if (dim == 512)return 2;
	if (dim == 256)return 1;
	if (dim == 128)return 0;
}

int main()
{
	File file("./");
	float T(1);
	float H(0);
	Timer timer;
	std::mt19937 mt(time(0));
	std::uniform_int_distribution<unsigned long long>rd;
	File& sf(file.findInThis("states.bin"));
	bool statesNeeded(false);
	if (&sf == nullptr)statesNeeded = true;
	constexpr size_t dim(256);
	constexpr size_t spinNum(dim * dim * dim);
	constexpr size_t gridSize(spinNum / 8);
	constexpr size_t gridNum(spinNum / 64);
	constexpr size_t layerSize(dim * dim / 8);
	constexpr size_t layerNum(dim * dim / 64);
	constexpr unsigned int idx(chooseIdx(dim));
	size_t statesSize(sizeof(curandState) * (layerNum * dim) / blockHeight);
	::printf("stateSize: %llu\n", statesSize);
	size_t sumSize(ReductionBlockSize[idx] * sizeof(unsigned long long));
	void* states(::malloc(statesSize));
	unsigned long long* debugBuffer((unsigned long long*)::malloc(layerSize));
	curandState* statesDevice;
	//unsigned long long* grid((unsigned long long*)::malloc(gridSize));
	unsigned long long* gridDevice;
	unsigned long long* sumDevice;
	unsigned long long* sumSumDevice;
	//for (unsigned long long c0(0); c0 < gridSize / sizeof(unsigned long long); ++c0)grid[c0] = rd(mt);
	cudaMalloc(&gridDevice, gridSize);
	cudaMalloc(&statesDevice, statesSize);
	cudaMalloc(&sumDevice, sumSize);
	cudaMalloc(&sumSumDevice, sizeof(unsigned long long));

	if (statesNeeded)
	{
		cudaDeviceSynchronize();
		timer.begin();
		initRandom << <RandomBlockSize[idx], 1024, 0, 0 >> > (statesDevice, rd(mt), RandomSize[idx]);
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

	for (unsigned long long c0(0); c0 < 1; ++c0)
	{
		//cudaMemcpy(gridDevice, grid, gridSize, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		timer.begin();
		initGrid << <InitGridBlockSize[idx], 1024 >> > (gridDevice, statesDevice, gridNum);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("Init Grid: ");

		timer.begin();
		for (unsigned long long c1(0); c1 < 100000; ++c1)
		{
			ising3d<dim> << <Ising3DBlockSize[idx], Ising3DThreadSize[idx] >> > (gridDevice, H, T, 0, statesDevice);
			ising3d<dim> << <Ising3DBlockSize[idx], Ising3DThreadSize[idx] >> > (gridDevice, H, T, 1, statesDevice);
		}
		cudaDeviceSynchronize();
		timer.end();
		timer.print("Ising3D: ");

		timer.begin();
		isingReduction <1024> << <ReductionBlockSize[idx], 1024 >> > (gridDevice, sumDevice, spinNum / 64);
		reduction<ReductionBlockSize[idx]> << <1, ReductionBlockSize[idx] >> > (sumDevice, sumSumDevice, ReductionBlockSize[idx]);
		/*for (unsigned long long cc(0); cc < 1; ++cc)
		{
			cudaMemcpy(debugBuffer, gridDevice + (cc & (dim - 1)) * layerNum, layerSize, cudaMemcpyDeviceToHost);
			for (unsigned long long c1(0); c1 < dim; ++c1)
			{
				for (unsigned long long c2(0); c2 < dim / 64; ++c2)
					for (unsigned long long c3(0); c3 < 64; ++c3)
						::printf("%d", (debugBuffer[dim * c1 / 64 + c2] >> c3) & 1);
				::printf("\n");
			}
			::printf("\n");
			::printf("\n");
		}*/
		unsigned long long sumAnswer;
		cudaMemcpy(&sumAnswer, sumSumDevice, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		timer.end();
		timer.print("Reduce: ");
		::printf("Average M:\t%lf\n", double(2 * sumAnswer) / spinNum - 1);
	}

	timer.begin();
	cudaMemcpy(states, statesDevice, statesSize, cudaMemcpyDeviceToHost);
	file.createBinary("states.bin", states, statesSize);
	timer.end();
	timer.print("Store states: ");

	cudaFree(gridDevice);
	cudaFree(statesDevice);
	cudaFree(sumDevice);
	cudaFree(sumSumDevice);
	//free(grid);
	free(states);
	free(debugBuffer);
}