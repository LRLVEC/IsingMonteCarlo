#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <random>
#include <string>
#include <_Time.h>
#include <_File.h>
#include <_String.h>
#include "IsingMonteCarlo.cuh"

//Things to achieve (in 32*32, 64*64, ..., 1024*1024 or more sizes):
//	1. <M>(H, T), <E>(H, T), <Cv>(H, T), <S>(H, T)
//	2. Heat graph of grids at some special points (average, bar)
//	3. Fluctuation?

//Some acknowledgements:
//	1. "Dim" is the length of grid side.
//	2. "Cycles" is the cycles before we begin to count the ensembles.
//	3. "Num" is the number of all ensembles that we count.
//	4. "Average" means ensemble average, while "Bar" means grid average.
//	5. All simulations will start with a random generated grid because we found some
//		issues when we try to initialize the grid with single toward spin.


//Program functions:
//	1. Input is a text file named "sdf.txt" (simulation description file).

template<unsigned int Dim>void printMAverage(float* table, int Num, String<char>const& name, File const& folder)
{

}
template<unsigned int Dim>void printGrid(unsigned long long* grid, String<char>const& name, File const& folder)
{

}






//Input will be:
//	1. First choose between calculating one point or many points;
//	2. Second enter the parameters including H, T, Dim, Num, Divisions and so on;
//	3. Then wait with a cup of coffee to get the result.
int main()
{
	File folder("./");
	String<char> sdf(folder.findInThis("sdf.txt").readText());
	int mode(1);
	unsigned int Dim, Cycles, Num;
	float H0, H1, T0, T1;
	unsigned int nH, nT;

	int dn, n(0);
	::sscanf(sdf.data, "Mode:%d\n%n", &mode, &dn); n += dn;
	::sscanf(sdf.data + n, "Dim:%u\n%n", &Dim, &dn); n += dn;
	::sscanf(sdf.data + n, "Cycles:%u\n%n", &Cycles, &dn); n += dn;
	::sscanf(sdf.data + n, "Num:%u\n%n", &Num, &dn); n += dn;
	::sscanf(sdf.data + n, "H0:%f\n%n", &H0, &dn); n += dn;
	::sscanf(sdf.data + n, "H1:%f\n%n", &H1, &dn); n += dn;
	::sscanf(sdf.data + n, "nH:%u\n%n", &nH, &dn); n += dn;
	::sscanf(sdf.data + n, "T0:%f\n%n", &T0, &dn); n += dn;
	::sscanf(sdf.data + n, "T1:%f\n%n", &T1, &dn); n += dn;
	::sscanf(sdf.data + n, "nT:%u", &nT);
	switch (mode)
	{
		case 0:
		{
			::printf("Dim: %u\nCycles: %u\nNum: %u\nH: %f\nT: %f\nIs that right?[n: No/y: Yes]\n",
				Dim, Cycles, Num, H0, T0);
			char flag('n');
			::scanf("%c", &flag);
			if (flag == 'y')
			{

			}
			break;
		}
		case 1:
		{
			::printf("Dim: %u\nCycles: %u\nNum: %u\nH0: %f\nH1: %f\nnH: %u\nT0: %f\nT1: %f\nnT: %u\nIs that right?[n: No/y: Yes]\n",
				Dim, Cycles, Num, H0, H1, nH, T0, T1, nT);
			char flag('n');
			::scanf("%c", &flag);
			if (flag == 'y')
			{

			}
			break;
		}
	}
	return 0;
}