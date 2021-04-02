#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define MAT_SIZE 840

void matrixMultiply(int n, int *a, int *b, int *c)
{
	int i, j, k;

	for (i = 0; i < n; i++)
	{
		for (k = 0; k < n; k++)
		{
			for (j = 0; j < n; j++) 
			{
				c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

void cannonAlgorithm(int n, int *a, int *b, int *c, int *globalC, 
	MPI_Comm comm)
{
	int i;
	int nlocal;
	int nrProcesses, dims[2], periods[2];
	int myrank, my2drank, mycoords[2];
	int upRank, downRank, leftRank, rightRank, coords[2];
	int shiftsource, shiftdest;
	MPI_Status status;
	MPI_Comm comm_2d;

	// Obtinere numar de procese si rank-ul in intregul world
	MPI_Comm_size(comm, &nrProcesses);
	MPI_Comm_rank(comm, &myrank);

	// Factorul de divizare este radical(nr_procese)
	dims[0] = dims[1] = sqrt(nrProcesses);

	// Specificam ca gridul este periodic, sub forma unui TOR
	//   task-urile se distribuie prin wraparound
	periods[0] = periods[1] = 1;

	// Creare topologie
	MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

	// Obtinem rank-ul si coordonatele procesului
	//	in topologia sub forma de tor creata mai sus
	MPI_Comm_rank(comm_2d, &my2drank);
	MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

	// Algoritmul Cannon shifteaza liniile si coloanele matricii
	MPI_Cart_shift(comm_2d, 1, -1, &rightRank, &leftRank);
	MPI_Cart_shift(comm_2d, 0, -1, &downRank, &upRank);

	// Fiecare proces se ocupa de o submatrice n/sqrt(nr_proc) x n/sqrt(nr_proc)
	nlocal = n / dims[0];

	// Shiftare initiala A si B
	MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_INT, shiftdest,
		1, shiftsource, 1, comm_2d, &status);

	MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_INT,
		shiftdest, 1, shiftsource, 1, comm_2d, &status);

	// De sqrt(nr_procese) ori
	for (i = 0; i < dims[0]; i++) 
	{
		// fiecare proces inmulteste submatricile alocate
		matrixMultiply(nlocal, a, b, c);

		// shiftare matrice A la stanga (deci shiftam linia)
		MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_INT,
			leftRank, 1, rightRank, 1, comm_2d, &status);

		// shiftare coloana pt matricea B in sus
		MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_INT,
			upRank, 1, downRank, 1, comm_2d, &status);

	}

	// Readuce matricile A si B la distributia initiala a elementelor
	MPI_Cart_shift(comm_2d, 1, +mycoords[0], &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_INT,
		shiftdest, 1, shiftsource, 1, comm_2d, &status);

	MPI_Cart_shift(comm_2d, 0, +mycoords[1], &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_INT,
		shiftdest, 1, shiftsource, 1, comm_2d, &status);

	MPI_Comm_free(&comm_2d);

	printf("proc with rank %d\n", myrank);

	// Fiecare proces pune in matricea rezultat globala numerele calculate
	//		de el, care vor trebui reunite cu rezultatele celorlate procese
	for (int i = 0; i < nlocal; i++)
	{
		for (int j = 0; j < nlocal; j++)
		{
			int global_i = mycoords[0] * nlocal + i;
			int global_j = mycoords[1] * nlocal + j;
			globalC[global_i * n + global_j] = c[i * nlocal + j];
		}
	}

	// Acele rezultate ale fiecarui proces se insumeaza intr-o singura valoare finala
	//  (colectare) p1_res + p2_res + p3_res + ... = c_final
	//		si se pune in globalC (matricea globala rezultat)
	if (myrank != 0)
	{
		MPI_Reduce(globalC, globalC, n * n, MPI_INT, MPI_SUM, 0, comm);
	}
	else
	{
		MPI_Reduce(MPI_IN_PLACE, globalC, n * n, MPI_INT, MPI_SUM, 0, comm);
	}
}

int isCorrectResult(int *a, int *b, int *c)
{
	int count = MAT_SIZE * MAT_SIZE;
	int *correct = (int*)malloc(sizeof(int) * count);
	for (int i = 0; i < count; i++)
	{
		correct[i] = 0;
	}
	matrixMultiply(MAT_SIZE, a, b, correct);

	for (int i = 0; i < count; i++)
	{
		if (c[i] != correct[i])
		{
			free(correct);
			return 0;
		}
	}

	free(correct);
	return 1;
}

int main(int argc, char* argv[]) {

	int *a = (int*)malloc(MAT_SIZE * MAT_SIZE * sizeof(int));
	int *b = (int*)malloc(MAT_SIZE * MAT_SIZE * sizeof(int));
	int *c = (int*)malloc(MAT_SIZE * MAT_SIZE * sizeof(int));
	srand(time(NULL));
	for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++)
	{
		a[i] = rand() % 101;
		b[i] = rand() % 101;
		c[i] = 0;
	}

	MPI_Init(&argc, &argv);

	int nrProcese;
	MPI_Comm_size(MPI_COMM_WORLD, &nrProcese);

	// Dimensiune bloc = n/sqrt(nr_proc) x n/sqrt(nr_proc)
	int blockSize = MAT_SIZE / sqrt(nrProcese);
	
	// Stocarea blocurilor pentru a le putea trimite fiecarui proces
	int **blocksOfA = new int*[nrProcese];
	int **blocksOfC = new int*[nrProcese];
	int **blocksOfB = new int*[nrProcese];
	for (int i = 0; i < nrProcese; i++)
	{
		blocksOfA[i] = new int[blockSize * blockSize];
		blocksOfB[i] = new int[blockSize * blockSize];
		blocksOfC[i] = new int[blockSize * blockSize];
	}

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int noBlocks = MAT_SIZE / blockSize;

	int columnBlockIndex = my_rank % noBlocks;
	int rowBlockIndex = (my_rank - columnBlockIndex) / noBlocks;

	int startRow = rowBlockIndex * blockSize;
	int stopRow = startRow + blockSize;

	int startCol = columnBlockIndex * blockSize;
	int stopCol = startCol + blockSize;

	int blockIndex = 0;

	// Popularea blocurilor dedicate fiecarui proces cu
	//	valorile din matricile initiale A si B
	for (int rowBlock = startRow; rowBlock < stopRow; rowBlock++)
	{
		for (int colBlock = startCol; colBlock < stopCol; colBlock++)
		{
			blocksOfA[my_rank][blockIndex] = a[rowBlock * MAT_SIZE + colBlock];
			blocksOfB[my_rank][blockIndex] = b[rowBlock * MAT_SIZE + colBlock];
			blocksOfC[my_rank][blockIndex] = 0;
			blockIndex++;
		}
	}

	// Asteptam ca toate procesele sa-si initializeze blocurile...
	MPI_Barrier(MPI_COMM_WORLD);

	double start = MPI_Wtime();
	// Pornim algoritmul Cannon
	cannonAlgorithm(MAT_SIZE, blocksOfA[my_rank], blocksOfB[my_rank],
		blocksOfC[my_rank], &c[0], MPI_COMM_WORLD);

	// Asteptam ca toate procesele sa termine algoritmul..
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	
	if (my_rank == 0)
	{
		/*printf("\n");
		for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++)
		{
			printf("%d ", a[i]);
			if ((i + 1) % MAT_SIZE == 0)
			{
				printf("\n");
			}
		}

		printf("\n");
		for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++)
		{
			printf("%d ", b[i]);
			if ((i + 1) % MAT_SIZE == 0)
			{
				printf("\n");
			}
		}

		printf("\n");
		for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++)
		{
			printf("%d ", c[i]);
			if ((i + 1) % MAT_SIZE == 0)
			{
				printf("\n");
			}
		}*/

		printf("Timp scurs: %lf secunde\n", end - start);
		printf("\n");
		/*printf("Rezultat corect?\n");
		if (isCorrectResult(a, b, c))
		{
			printf("DA\n");
		}
		else
		{
			printf("NU, nu e bine.\n");
		}*/
	}

	MPI_Finalize();

	return 0;
}