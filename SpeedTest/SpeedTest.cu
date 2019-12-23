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

__global__ void add(int* ahh, int a, int b, int c)
{
	int d(a + b), e(b + c), f(c + a), g(a + c), h(a + b + c);
	for (int c0(0); c0 < 100000000; ++c0)
	{
		asm("add.s32 %0, %0, %1;\n\t"
			"add.s32 %2, %2, %3;\n\t"
			"add.s32 %4, %4, %5;\n\t"
			"add.s32 %6, %6, %7;\n\t"
			"add.s32 %1, %0, %1;\n\t"
			"add.s32 %3, %2, %3;\n\t"
			"add.s32 %5, %4, %5;\n\t"
			"add.s32 %7, %6, %7;\n\t"
			: "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e), "+r"(f), "+r"(g), "+r"(h));
	}
	*ahh = a + c + e + g;
}
__global__ void popc(int* ahh, unsigned int a)
{
	int N(0);
	unsigned int b(a * a), c(a * b), d(a * c), e(a * d), f(a * e), g(a * f), h(a * g);
	unsigned int i(a * h), j(a * i), k(a * j), l(a * k), m(a * l), n(a * m), o(a * n), p(a * o);
	for (int c0(0); c0 < 10000000; ++c0)
	{
		asm("popc.b32 %0, %1;\n\t"
			"popc.b32 %2, %3;\n\t"
			"popc.b32 %4, %5;\n\t"
			"popc.b32 %1, %0;\n\t"
			"popc.b32 %3, %2;\n\t"
			"popc.b32 %5, %4;\n\t"
			"popc.b32 %6, %7;\n\t"
			"popc.b32 %7, %6;\n\t"
			/*"popc.b32 %8, %9;\n\t"
			"popc.b32 %9, %8;\n\t"
			"popc.b32 %10, %11;\n\t"
			"popc.b32 %11, %10;\n\t"
			"popc.b32 %12, %13;\n\t"
			"popc.b32 %13, %12;\n\t"
			"popc.b32 %14, %15;\n\t"
			"popc.b32 %15, %14;\n\t"*/
			: "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e), "+r"(f), "+r"(g), "+r"(h)
			//, "+r"(i), "+r"(j), "+r"(k), "+r"(l), "+r"(m), "+r"(n), "+r"(o), "+r"(p)
			);
	}
	*ahh = a + c + e + g/* + i + k + m + o*/;
}
void testAdd()
{
	Timer timer;
	int* ahh;
	cudaMalloc(&ahh, 4);
	for (int c0(0); c0 < 10; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		add << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (ahh, 1, 2, 3);
		cudaDeviceSynchronize();
		timer.end();
		timer.print();
	}
	cudaFree(ahh);
}
void testPopc()
{
	Timer timer;
	int* ahh;
	cudaMalloc(&ahh, 4);
	for (int c0(0); c0 < 10; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		popc << <dim3(1, 1, 1), dim3(1, 1, 1) >> > (ahh, 13925u);
		cudaDeviceSynchronize();
		timer.end();
		timer.print();
	}
	cudaFree(ahh);
}
int main()
{
	//Conclusion: Level 4 Assembly Line.??
	//testAdd();
	testPopc();
}