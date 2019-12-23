#include <cstdio>
#include <thread>
#include <random>
#include <cstring>
#include <_Time.h>
#define gridDim 256


inline float random(float s, float t)
{
	float a(fabsf(sin(s * 12.9898f + t * 78.233f)) * 43758.5453123f);
	return a - floorf(a);
}
void kernel(unsigned long long* grid, float H, float T, int step, float seed, int idx)
{
	for (int blockIdx(0); blockIdx < gridDim; ++blockIdx)
	{
		unsigned char* p((unsigned char*)(&grid[blockIdx * 4llu]));
#define get(y, x) ((grid[4*y+(x>>6)]>>(x&63))&1)
#define set(ff) ((p[idx]^=(unsigned char(1)<<(ff&7))))
		int _step = step ^ (blockIdx % 2);
		int d0(idx * 8 + _step);
		for (int c0(d0); c0 < d0 + 8; c0 += 2)
		{
			int y((blockIdx + gridDim - 1) % gridDim);
			int s1 = get(y, c0);
			y = (blockIdx + 1) % gridDim;
			s1 += get(y, c0);
			int dx((c0 + gridDim - 1) % gridDim);
			s1 += get(blockIdx, dx);
			dx = (c0 + 1) % gridDim;
			s1 += get(blockIdx, dx);
			s1 = s1 * 2 - 4;
			float s0 = 2 * int(get(blockIdx, c0)) - 1;
			s0 *= H + s1;
			if (s0 <= 0 || random(c0 - seed, blockIdx + seed) < expf(-s0 / T))
				set(c0);
		}
	}
#undef get
#undef set
}
inline int countBit(unsigned long long a)
{
	a = (a & 0x5555555555555555) + ((a >> 1) & 0x5555555555555555);
	a = (a & 0x3333333333333333) + ((a >> 2) & 0x3333333333333333);
	a = (a & 0x0F0F0F0F0F0F0F0F) + ((a >> 4) & 0x0F0F0F0F0F0F0F0F);
	a = (a & 0x00FF00FF00FF00FF) + ((a >> 8) & 0x00FF00FF00FF00FF);
	a = (a & 0x0000FFFF0000FFFF) + ((a >> 16) & 0x0000FFFF0000FFFF);
	a = (a & 0x00000000FFFFFFFF) + ((a >> 32) & 0x00000000FFFFFFFF);
	return a;
}
float calcM(unsigned long long* grid)
{
	float m = 0;
	for (int c0(0); c0 < 4 * gridDim; ++c0)
		m += countBit(grid[c0]);
	return 2 * m / (gridDim * gridDim) - 1;
}

int main()
{
	//::printf("%u\n", std::this_thread::get_id());
	unsigned int memSize = sizeof(unsigned long long) * 4 * gridDim;
	unsigned long long* grid((unsigned long long*)malloc(memSize));

	float H1 = 0.3, H2 = 0.5f;
	float T, T1 = 1 * 2.268, T2 = 1.9 * 2.268;
	int nH = 0;
	int nT = 0;
	int cycles = 100;
	::scanf("%d", &cycles);
	H2 -= H1;
	H2 /= nH ? nH : 1;
	T2 -= T1;
	T2 /= nT ? nT : 1;
	char t[100];
	std::string answer;
	std::mt19937 mt(0);
	std::uniform_real_distribution<float> rd(0, 1.0f);
	Timer timer;
	for (int c0(0); c0 <= nH; ++c0)
	{
		T = T1;
		for (int c1(0); c1 <= nT; ++c1)
		{
			for (int c0(0); c0 < gridDim; ++c0)
			{
				unsigned long long s;
				if (c0 % 2)s = 0xaaaaaaaaaaaaaaaa;
				else s = 0x5555555555555555;
				for (int c1(0); c1 < 4; ++c1)
					grid[c0 * 4 + c1] = 0/*s*/;
			}
			timer.begin();
			for (int c2(0); c2 < cycles; ++c2)
			{
				std::thread threads[32];
				float seed(rd(mt));
				for (int c3(0); c3 < 32; ++c3)threads[c3] = std::thread(kernel, grid, H1, T, 0, seed, c3);
				for (int c3(0); c3 < 32; ++c3)threads[c3].join();
				seed = rd(mt);
				for (int c3(0); c3 < 32; ++c3)threads[c3] = std::thread(kernel, grid, H1, T, 1, seed, c3);
				for (int c3(0); c3 < 32; ++c3)threads[c3].join();
			}
			float M(calcM(grid));
			timer.end();
			::printf("%.8f %.8f %.8f\t", H1, T, M);
			timer.print();
			::sprintf(t, "%.8f %.8f %.8f\n", H1, T, M);
			answer += t;
			T += T2;
		}
		H1 += H2;
	}
	FILE* temp(::fopen("./answer.txt", "w+"));
	::fprintf(temp, "%s", answer.c_str());
	::fclose(temp);
}