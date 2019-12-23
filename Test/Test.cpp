#include <cstdio>
#include <cmath>

int main()
{
	unsigned char a = 0;
	int x[256] = { 0 };
	x[255] = 1;
	::printf("%d", x[(int)(unsigned char)(a - 1)]);
}