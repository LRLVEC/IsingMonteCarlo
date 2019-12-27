#include <random>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <_String.h>
#include <_File.h>

constexpr unsigned int listSize[] =
{
	0, 0, 0, 0, 32, 64, 128, 256
};


struct IsingBase
{
	virtual void run() = 0;
	virtual void freeMemory() = 0;
	virtual void printTableText(File& folder, String<char>const& name) = 0;
	virtual void printAnswerText(File& folder, String<char>const& name) = 0;
	virtual void printTableBin(File& folder, String<char>const& name) = 0;
	virtual void printAnswerBin(File& folder, String<char>const& name) = 0;
};
template<unsigned int Dim>struct Ising :IsingBase
{
	static constexpr unsigned int Dim = Dim;
	static constexpr unsigned int SpinNum = Dim * Dim;
	static constexpr unsigned int gridSize = (Dim * Dim) >> 3;
	static constexpr unsigned int tableSize = Dim * Dim * sizeof(float);
	unsigned int id = log2f(Dim) - 3;
	unsigned int Mode;
	unsigned int Cycles;
	unsigned int Num;
	unsigned int nH;
	unsigned int nT;
	float H0;
	float H1;
	float T0;
	float T1;
	bool calc;
	unsigned long long* grid;
	unsigned long long* gridHost;
	float* table;
	float* tableHost;
	int* MList;
	float* EList;
	int* M;
	float* E;
	float AveM, AveE;//One point
	float* Answer;//Many points (in syntax: H T M E\n)
	curandState* state;
	std::mt19937 mt;
	Ising() = delete;
	Ising(int _Mode, unsigned int _Cycles, unsigned int _Num, unsigned int _nH, unsigned int _nT,
		float _H0, float _H1, float _T0, float _T1)
		:
		Mode(_Mode),
		Cycles(_Cycles),
		Num(_Num),
		nH(_nH),
		nT(_nT),
		H0(_H0),
		H1(_H1),
		T0(_T0),
		T1(_T1),
		calc(false),
		grid(nullptr),
		gridHost(nullptr),
		table(nullptr),
		tableHost(nullptr),
		MList(nullptr),
		EList(nullptr),
		M(nullptr),
		E(nullptr),
		mt(time(0))
	{
		//std::uniform_int_distribution<int> gg;
		//::printf("%d\n", mt);
		//::printf("%d\n", mt);
		//::printf("%d\n", gg(mt));
		//::printf("%d\n", mt);
		//::printf("%d\n", gg(mt));
		char flag('n');
		if (Mode)::printf("Dim: %u\nCycles: %u\nNum: %u\nH0: %f\nH1: %f\nnH: %u\n\
T0: %f\nT1: %f\nnT: %u\nIs that right?[n: No/y: Yes]\n", Dim, Cycles, Num, H0, H1, nH, T0, T1, nT);
		else ::printf("Dim: %u\nCycles: %u\nNum: %u\nH: %f\nT: %f\n\
Is that right?[n: No/y: Yes]\n", Dim, Cycles, Num, H0, T0);
		::scanf("%c", &flag);
		if (flag == 'y')
		{
			calc = true;
			mallocMemory();
		}
		else exit(-1);
	}
	void mallocMemory();
	void freeMemory();
	void reachBalance(float H, float T);
	void countEnsembles(float H, float T, unsigned int an);
	void phaseTransition(float T, unsigned int an);
	void printTableText(File& folder, String<char>const& name);
	void printAnswerText(File& folder, String<char>const& name);
	void printTableBin(File& folder, String<char>const& name);
	void printAnswerBin(File& folder, String<char>const& name);
	void run();
	void hysteresisCurve(float H, unsigned int an);
};

void ising128(unsigned long long* grid, float H, float T, int step, curandState* state);
void ising256(unsigned long long* grid, float H, float T, int step, curandState* state);
void ising512(unsigned long long* grid, float H, float T, int step, curandState* state);
void ising1024(unsigned long long* grid, float H, float T, int step, curandState* state);
void ising128(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state);
void ising256(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state);
void ising512(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state);
void ising1024(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state);
void ising128(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state);
void ising256(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state);
void ising512(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state);
void ising1024(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state);
void reduce128(int* MList, float* EList, int* M, float* E, float H);
void reduce256(int* MList, float* EList, int* M, float* E, float H);
void reduce512(int* MList, float* EList, int* M, float* E, float H);
void reduce1024(int* MList, float* EList, int* M, float* E, float H);
void initRandom(curandState* state, int seed, unsigned int id);
template<unsigned int Dim>void isingHost(unsigned long long* grid, float H, float T, int step, curandState* state)
{
	if constexpr (Dim == 8);
	else if constexpr (Dim == 16);
	else if constexpr (Dim == 32);
	else if constexpr (Dim == 64);
	else if constexpr (Dim == 128)ising128(grid, H, T, step, state);
	else if constexpr (Dim == 256)ising256(grid, H, T, step, state);
	else if constexpr (Dim == 512)ising512(grid, H, T, step, state);
	else if constexpr (Dim == 1024)ising1024(grid, H, T, step, state);
}
template<unsigned int Dim>void isingHost(unsigned long long* grid, float H, float T, int* MList, float* EList, int step, curandState* state)
{
	if constexpr (Dim == 8);
	else if constexpr (Dim == 16);
	else if constexpr (Dim == 32);
	else if constexpr (Dim == 64);
	else if constexpr (Dim == 128)ising128(grid, H, T, MList, EList, step, state);
	else if constexpr (Dim == 256)ising256(grid, H, T, MList, EList, step, state);
	else if constexpr (Dim == 512)ising512(grid, H, T, MList, EList, step, state);
	else if constexpr (Dim == 1024)ising1024(grid, H, T, MList, EList, step, state);
}
template<unsigned int Dim>void isingHost(unsigned long long* grid, float H, float T, int* MList, float* EList, float* table, int step, curandState* state)
{
	if constexpr (Dim == 8);
	else if constexpr (Dim == 16);
	else if constexpr (Dim == 32);
	else if constexpr (Dim == 64);
	else if constexpr (Dim == 128)ising128(grid, H, T, MList, EList, table, step, state);
	else if constexpr (Dim == 256)ising256(grid, H, T, MList, EList, table, step, state);
	else if constexpr (Dim == 512)ising512(grid, H, T, MList, EList, table, step, state);
	else if constexpr (Dim == 1024)ising1024(grid, H, T, MList, EList, table, step, state);
}
template<unsigned int Dim>void reduceHost(int* MList, float* EList, int* M, float* E, float H)
{
	if constexpr (Dim == 8);
	else if constexpr (Dim == 16);
	else if constexpr (Dim == 32);
	else if constexpr (Dim == 64);
	else if constexpr (Dim == 128)reduce128(MList, EList, M, E, H);
	else if constexpr (Dim == 256)reduce256(MList, EList, M, E, H);
	else if constexpr (Dim == 512)reduce512(MList, EList, M, E, H);
	else if constexpr (Dim == 1024)reduce1024(MList, EList, M, E, H);
}

template<unsigned int Dim>void Ising<Dim>::mallocMemory()
{
	gridHost = (unsigned long long*)::malloc(gridSize);
	cudaMalloc(&grid, gridSize);
	cudaMalloc(&state, sizeof(curandState) * gridSize);
	std::uniform_int_distribution<int> gg;
	initRandom(state, gg(mt), id);
	if constexpr (Dim >= 128)
	{
		cudaMalloc(&MList, sizeof(int) * listSize[id]);
		cudaMalloc(&EList, sizeof(float) * listSize[id]);
	}
	cudaMalloc(&M, sizeof(int));
	cudaMalloc(&E, sizeof(float));
	if (Mode == 0)
	{
		tableHost = (float*)::malloc(tableSize);
		cudaMalloc(&table, tableSize);
	}
	else if (Mode == 1)
	{
		Answer = (float*)::malloc((nH + 1) * (nT + 1) * 4 * sizeof(float));
	}
	else if (Mode == 2)
	{
		Answer = (float*)::malloc((2 * nH + 1) * 4 * sizeof(float));
	}
	else if (Mode == 3)
	{
		Answer = (float*)::malloc((nT + 1) * 4 * sizeof(float));
	}
}
template<unsigned int Dim>void Ising<Dim>::freeMemory()
{
	free(gridHost);
	cudaFree(grid);
	cudaFree(state);
	if constexpr (Dim >= 128)
	{
		cudaFree(MList);
		cudaFree(EList);
	}
	cudaFree(M);
	cudaFree(E);
	if (Mode == 0)
	{
		free(tableHost);
		cudaFree(&table);
	}
	else
	{
		free(Answer);
	}
}
template<unsigned int Dim>void Ising<Dim>::reachBalance(float H, float T)
{
	std::uniform_int_distribution<unsigned long long> gg;
	std::uniform_real_distribution<float> rd(0, 1.0f);
	for (int c0(0); c0 < gridSize / 8; ++c0)gridHost[c0] = gg(mt);
	cudaMemcpy(grid, gridHost, gridSize, cudaMemcpyHostToDevice);
	for (int c0(0); c0 < Cycles; ++c0)
	{
		isingHost<Dim>(grid, H, T, 0, state);
		isingHost<Dim>(grid, H, T, 1, state);
	}
}
template<unsigned int Dim>void Ising<Dim>::countEnsembles(float H, float T, unsigned int an)
{
	AveE = 0;
	if (Mode == 0)
	{
		cudaMemset(table, 0, tableSize);
		long long LM(0);
		double DE(0);
		for (int c0(0); c0 < Num; ++c0)
		{
			isingHost<Dim>(grid, H, T, 0, state);
			isingHost<Dim>(grid, H, T, MList, EList, table, 1, state);
			reduceHost<Dim>(MList, EList, M, E, H);
			int dM;
			float dE;
			cudaMemcpy(&dM, M, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&dE, E, sizeof(float), cudaMemcpyDeviceToHost);
			LM += dM;
			DE += dE;
			//Table? Should we record the table?
			//	Yes, if we want to calculate the fluctuation between the ensembles.
			//	So the table size is ... that's a lot of memory...
			//	We will do it later.
			//And how about the fluctuation of M and E?
			//	For girds whose Dim is less than 128, we can launch the kernel once...
			//	So a list is needed to record M and E in the cycles, instead of just get the average.
		}
		double DM = LM;
		DM /= SpinNum;
		DM /= Num;
		AveM = DM * 2 - 1;
		DE /= SpinNum;
		DE /= Num;
		AveE = DE;
		cudaMemcpy(tableHost, table, tableSize, cudaMemcpyDeviceToHost);
	}
	else
	{
		long long LM(0);
		double DE(0);
		for (int c0(0); c0 < Num; ++c0)
		{
			isingHost<Dim>(grid, H, T, 0, state);
			isingHost<Dim>(grid, H, T, MList, EList, 1, state);
			reduceHost<Dim>(MList, EList, M, E, H);
			int dM;
			float dE;
			cudaMemcpy(&dM, M, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&dE, E, sizeof(float), cudaMemcpyDeviceToHost);
			LM += dM;
			DE += dE;
		}
		double DM = LM;
		DM /= SpinNum;
		DM /= Num;
		DM = DM * 2 - 1;
		DE /= SpinNum;
		DE /= Num;
		Answer[4 * an] = H;
		Answer[4 * an + 1] = T;
		Answer[4 * an + 2] = DM;
		Answer[4 * an + 3] = DE;
		::printf("%.8f %.8f %.8f %.8f\n", H, T, DM, DE);
	}
}
template<unsigned int Dim>void Ising<Dim>::phaseTransition(float T, unsigned int an)
{
	long long LM(0);
	double DE(0);
	for (int c0(0); c0 < nH; ++c0)
	{
		reachBalance(H0, T);
		for (int c1(0); c1 < Num; ++c1)
		{
			isingHost<Dim>(grid, H0, T, 0, state);
			isingHost<Dim>(grid, H0, T, MList, EList, 1, state);
			reduceHost<Dim>(MList, EList, M, E, H0);
			int dM;
			float dE;
			cudaMemcpy(&dM, M, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&dE, E, sizeof(float), cudaMemcpyDeviceToHost);
			LM += (dM - int(SpinNum / 2) >= 0) ? dM : SpinNum - dM;
			DE += dE;
		}
	}
	double DM = LM;
	DM /= SpinNum;
	DM /= Num;
	DM /= nH;
	DM = DM * 2 - 1;
	DE /= SpinNum;
	DE /= Num;
	DE /= nH;
	Answer[4 * an] = H0;
	Answer[4 * an + 1] = T;
	Answer[4 * an + 2] = DM;
	Answer[4 * an + 3] = DE;
	::printf("%.8f %.8f %.8f %.8f\n", H0, T, DM, DE);
}
template<unsigned int Dim>void Ising<Dim>::run()
{
	if (!calc)return;
	if (Mode == 0)
	{
		reachBalance(H0, T0);
		countEnsembles(H0, T0, 0);
	}
	else if (Mode == 1)
	{
		float H, T;
		unsigned int nn(0);
		for (int c0(0); c0 <= nH; ++c0)
		{
			if (nH)H = (H1 * c0 + H0 * (nH - c0)) / nH;
			else H = H0;
			if (abs(H) < 1e-5f)H = 0;
			for (int c1(0); c1 <= nT; ++c1)
			{
				if (nT)T = (T1 * c1 + T0 * (nT - c1)) / nT;
				else T = T0;
				reachBalance(H, T);
				countEnsembles(H, T, nn++);
			}
		}
	}
	else if (Mode == 2)
	{
		reachBalance(H0, T0);
		float H;
		unsigned int nn(0);
		for (int c0(0); c0 <= nH; ++c0)
		{
			H = (H1 * c0 + H0 * (nH - c0)) / nH;
			if (abs(H) < 1e-5f)H = 0;
			hysteresisCurve(H, nn++);
		}
		for (int c0(nH - 1); c0 >= 0; --c0)
		{
			H = (H1 * c0 + H0 * (nH - c0)) / nH;
			if (abs(H) < 1e-5f)H = 0;
			hysteresisCurve(H, nn++);
		}
	}
	else if (Mode == 3)
	{
		float T;
		for (int c0(0); c0 <= nT; ++c0)
		{
			if (nT)T = (T1 * c0 + T0 * (nT - c0)) / nT;
			else T = T0;
			phaseTransition(T, c0);
		}
	}
}
template<unsigned int Dim>void Ising<Dim>::hysteresisCurve(float H, unsigned int an)
{
	long long LM(0);
	double DE(0);
	for (int c0(0); c0 < 100; ++c0)
	{
		isingHost<Dim>(grid, H, T0, 0, state);
		isingHost<Dim>(grid, H, T0, 1, state);
	}
	for (int c0(0); c0 < Num; ++c0)
	{
		isingHost<Dim>(grid, H, T0, 0, state);
		isingHost<Dim>(grid, H, T0, MList, EList, 1, state);
		reduceHost<Dim>(MList, EList, M, E, H);
		int dM;
		float dE;
		cudaMemcpy(&dM, M, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&dE, E, sizeof(float), cudaMemcpyDeviceToHost);
		LM += dM;
		DE += dE;
	}
	double DM = LM;
	DM /= SpinNum;
	DM /= Num;
	DM = DM * 2 - 1;
	DE /= SpinNum;
	DE /= Num;
	Answer[4 * an] = H;
	Answer[4 * an + 1] = T0;
	Answer[4 * an + 2] = DM;
	Answer[4 * an + 3] = DE;
	::printf("%.8f %.8f %.8f %.8f\n", H, T0, DM, DE);
}