#include <_Time.h>
#include <_File.h>
#include <_String.h>
#include <cuda_runtime.h>
#include "IsingMonteCarlo.cuh"

//Things to achieve (in 16*16, 32*32, 64*64, ..., 1024*1024 or more sizes):
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



template<unsigned int Dim>void Ising<Dim>::printTableText(File& folder, String<char> const& name)
{
	char t[100];
	String<char>tp;
	for (int c0(0); c0 < SpinNum; ++c0)
	{
		::sprintf(t, "%.8f ", (tableHost[c0] / Num) * 2 - 1);
		tp += t;
		if ((c0 + 1) % Dim == 0)
		{
			::sprintf(t, "\n");
			tp += t;
		}
	}
	folder.createText(name, tp);
}
template<unsigned int Dim>void Ising<Dim>::printAnswerText(File& folder, String<char> const& name)
{
	char t[100];
	String<char>tp;
	if (Mode == 0)
	{
		::printf("H: %.8f\nT: %.8f\n<M>: %.8f\n<E>: %.8f\n", H0, T0, AveM, AveE);
		::sprintf(t, "%.8f %.8f %.8f %.8f\n", H0, T0, AveM, AveE);
		tp += t;
	}
	else if (Mode == 1)
	{
		for (int c0(0); c0 < (nH + 1) * (nT + 1); ++c0)
		{
			::sprintf(t, "%.8f %.8f %.8f %.8f\n", Answer[c0 * 4], Answer[c0 * 4 + 1],
				Answer[c0 * 4 + 2], Answer[c0 * 4 + 3]);
			tp += t;
		}
	}
	else if (Mode == 2)
	{
		for (int c0(0); c0 < (2 * nH + 1); ++c0)
		{
			::sprintf(t, "%.8f %.8f %.8f %.8f\n", Answer[c0 * 4], Answer[c0 * 4 + 1],
				Answer[c0 * 4 + 2], Answer[c0 * 4 + 3]);
			tp += t;
		}
	}
	else if (Mode == 3)
	{
		for (int c0(0); c0 < (nT + 1); ++c0)
		{
			::sprintf(t, "%.8f %.8f %.8f %.8f\n", Answer[c0 * 4], Answer[c0 * 4 + 1],
				Answer[c0 * 4 + 2], Answer[c0 * 4 + 3]);
			tp += t;
		}
	}
	folder.createText(name, tp);
}
template<unsigned int Dim>void Ising<Dim>::printTableBin(File& folder, String<char> const& name)
{

}
template<unsigned int Dim>void Ising<Dim>::printAnswerBin(File& folder, String<char> const& name)
{
}
//Program functions:
//	1. Input is a text file named "sdf.txt" (simulation description file).


//Input will be:
//	1. First choose between calculating one point or many points;
//		For many points we can get <M>, <E>... while for one point we can
//		also get the distribution of spin in the whole grid.
//	2. Second enter the parameters including H, T, Dim, Num, Divisions and so on;
//	3. Then wait with a cup of coffee to get the result.
int main()
{
	File folder("./");
	String<char> sdf(folder.findInThis("sdf.txt").readText());

	unsigned int Dim;

	int Mode, Cycles, Num;
	float H0, H1, T0, T1;
	unsigned int nH, nT;

	int dn, n(0);
	::sscanf(sdf.data, "Dim:%u\n%n", &Dim, &dn); n += dn;
	::sscanf(sdf.data + n, "Mode:%d\n%n", &Mode, &dn); n += dn;
	::sscanf(sdf.data + n, "Cycles:%u\n%n", &Cycles, &dn); n += dn;
	::sscanf(sdf.data + n, "Num:%u\n%n", &Num, &dn); n += dn;
	::sscanf(sdf.data + n, "H0:%f\n%n", &H0, &dn); n += dn;
	::sscanf(sdf.data + n, "H1:%f\n%n", &H1, &dn); n += dn;
	::sscanf(sdf.data + n, "nH:%u\n%n", &nH, &dn); n += dn;
	::sscanf(sdf.data + n, "T0:%f\n%n", &T0, &dn); n += dn;
	::sscanf(sdf.data + n, "T1:%f\n%n", &T1, &dn); n += dn;
	::sscanf(sdf.data + n, "nT:%u", &nT);
	IsingBase* ising((IsingBase*)malloc(sizeof(Ising<8>)));
	switch (Dim)
	{
		//case 8:new(ising)Ising<8>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1, time(0)); break;
		//case 16:new(ising)Ising<16>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1, time(0)); break;
		//case 32:new(ising)Ising<32>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1, time(0)); break;
		//case 64:new(ising)Ising<64>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1, time(0)); break;
		case 128:new(ising)Ising<128>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1); break;
		case 256:new(ising)Ising<256>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1); break;
		case 512:new(ising)Ising<512>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1); break;
		case 1024:new(ising)Ising<1024>(Mode, Cycles, Num, nH, nT, H0, H1, T0, T1); break;
		default: ::printf("Wrong Dim! Try 2^n in [8, 1024] next time."); ::free(ising); return -1;
	}
	ising->run();
	char t[100];
	String<char>dir("Dim_");
	sprintf(t, "%u_", Dim);
	dir += t;
	if (Mode == 0)sprintf(t, "OnePoint");
	else sprintf(t, "ManyPoints");
	dir += t;
	folder.createDirectory(dir);
	File& subDir(folder.findInThis(dir));
	if (Mode == 0)ising->printTableText(subDir, "table.txt");
	ising->printAnswerText(subDir, "/answer.txt");
	subDir.createText("sdf.txt", sdf);
	ising->freeMemory();
	::free(ising);
	return 0;
}