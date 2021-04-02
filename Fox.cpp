#include<stdlib.h>
#include <ctime>
#include<math.h>
#include"mpi.h"
#include <iostream>

#define MATRIX_SIZE 840

int first_matrix[MATRIX_SIZE][MATRIX_SIZE];
int second_matrix[MATRIX_SIZE][MATRIX_SIZE];

typedef struct {
	int proc_count;
	int dim;
	int row;
	int col;
	int rank;
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
} GridInfo;

void Setup_grid(GridInfo *grid)
{
	int dimensions[2];
	int wrap_around[2];
	int coordinates[2];
	int free_coords[2];
	int world_rank;

	// Returneaza numarul de procese folosite in total
	MPI_Comm_size(MPI_COMM_WORLD, &(grid->proc_count));

	// Returneaza rank-ul procesului de care apartine grid-ul
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Numarul de procese trebuie sa fie patrat perfect
	//   grid->dim este numarul de blocuri pe orizontala si pe verticala 
	//		=> matricele sunt patratice si numarul de blocuri e acelasi pe cele doua directii
	grid->dim = (int)sqrt((double)grid->proc_count);
	dimensions[0] = dimensions[1] = grid->dim;

	// Dispunerea circulara a task-urilor, sub forma de tor
	wrap_around[0] = wrap_around[1] = 1;

	// Creeaza un nou communicator cu informatie de topologie (dimensiuni si rotatie)
	//  practic, construim communicators pentru linii si coloane
	// Acesti communicators sunt creati pentru ca procesele sa comunice intre ele
	//		si sa isi trimita blocurile pe care le prelucreaza celorlalte procese
	//		de pe aceeasi linie/coloana
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->grid_comm));
	MPI_Comm_rank(grid->grid_comm, &(grid->rank));
	// Determinare coordonate proces in topologia carteziana
	MPI_Cart_coords(grid->grid_comm, grid->rank, 2, coordinates);
	grid->row = coordinates[0];
	grid->col = coordinates[1];

	// Communicators pe linii
	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

	// Communicators pe coloane
	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));
}

void MultiplyLocal(int **a, int **b, int **c, int size) 
{
	// O imbunatatire a inmultirii, prin schimbarea ordinii for-urilor
		// i -> k -> j   in loc de   i -> j -> k
	for (int i = 0; i < size; i++)
	{
		for (int k = 0; k < size; k++)
		{
			for (int j = 0; j < size; j++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

// Transforma vectorul buff intr-o matrice
void ConvertVectorToMatrix(int *buff, int **a, int size)
{
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			a[i][j] = buff[k];
			k++;
		}
	}
}

// Transforma matricea a intr-un vector
void ConvertMatrixToVector(int *buff, int **a, int size) 
{
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			buff[k] = a[i][j];
			k++;
		}
	}
}
void PrintMatrix(int **matrix, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << "|";
		for (int j = 0; j < size; j++) {
			int el = matrix[i][j];
			if (el < 10)
				std::cout << " ";
			std::cout << el;
			std::cout << "|";
		}
		std::cout << std::endl;
	}
}

void GenerateMatrices(int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			first_matrix[i][j] = rand() % 100;
			second_matrix[i][j] = rand() % 100;
		}
	}
}

void FoxMultiply(int n, GridInfo *grid, int **a, int **b, int **c)
{
	int **temp_a, *buff, stage, root, submat_dim, src, dst;
	MPI_Status status;

	submat_dim = n / grid->dim;

	// Folosita pentru broadcast-ul blocului A
	temp_a = (int**)malloc(submat_dim * sizeof(int*));
	for (int i = 0; i < submat_dim; ++i)
	{
		temp_a[i] = (int*)malloc(submat_dim * sizeof(int));
	}

	for (int i = 0; i < submat_dim; i++)
	{
		for (int j = 0; j < submat_dim; j++)
		{
			temp_a[i][j] = 0;
		}
	}

	int submatTotalElem = submat_dim * submat_dim;
	buff = (int*)malloc(submatTotalElem * sizeof(int));
	for (int i = 0; i < submatTotalElem; i++)
	{
		buff[i] = 0;
	}

	// Sursa si destinatie pentru rotirea circulara a matricei B
	src = (grid->row + 1) % grid->dim;
	dst = (grid->row + grid->dim - 1) % grid->dim;

	for (stage = 0; stage < grid->dim; stage++) 
	{
		// Procesul cu root = grid_col va trimite blocul din A proceselor de pe aceeasi linie (row)
		//	  Celelalte procese (care prelucreaza acea linie) vor astepta dupa blocul din A
		//		(care va fi pus intr-un vector buff dupa ce root o trimite)


		root = (grid->row + stage) % grid->dim;
		// Trimite (sub)matricea A (prin intermediul buff) catre celelalte procese
		if (root == grid->col) 
		{
			ConvertMatrixToVector(buff, a, submat_dim);
			MPI_Bcast(buff, submatTotalElem, MPI_INT, root, grid->row_comm);
			ConvertVectorToMatrix(buff, a, submat_dim);
			MultiplyLocal(a, b, c, submat_dim);
		}
		// Primeste (sub)matricea A
		else
		{
			ConvertMatrixToVector(buff, temp_a, submat_dim);
			MPI_Bcast(buff, submatTotalElem, MPI_INT, root, grid->row_comm);
			ConvertVectorToMatrix(buff, temp_a, submat_dim);
			MultiplyLocal(temp_a, b, c, submat_dim);
		}

		// Rotirea matricei B (task-urile sunt impartite dupa un TOR)
		// Asta se face prin trimiterea ei catre taskul urmator care va procesa
		//	 coloana curenta (prin intermediul communicatorului de coloana)
		ConvertMatrixToVector(buff, b, submat_dim);
		MPI_Sendrecv_replace(buff, submatTotalElem, MPI_INT, dst, 0, src, 0, grid->col_comm, &status);
		ConvertVectorToMatrix(buff, b, submat_dim);
	}

	free(temp_a);
	free(buff);
}

int** sequentialMultiplication()
{
	int **test_result, **test_a, **test_b;
	test_a = new int*[MATRIX_SIZE];
	test_b = new int*[MATRIX_SIZE];
	test_result = new int*[MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		test_a[i] = new int[MATRIX_SIZE];
		test_b[i] = new int[MATRIX_SIZE];
		test_result[i] = new int[MATRIX_SIZE];
	}
	for (int i = 0; i < MATRIX_SIZE; i++)
		for (int j = 0; j < MATRIX_SIZE; j++) {
			test_a[i][j] = first_matrix[i][j];
			test_b[i][j] = second_matrix[i][j];
			test_result[i][j] = 0;
		}
	MultiplyLocal(test_a, test_b, test_result, MATRIX_SIZE);

	//std::cout << "Local result:" << std::endl;
	//PrintMatrix(test_result, MATRIX_SIZE);
	return test_result;
}

void initLocalMatrices(int ***local_a, int ***local_b, int ***local_c, GridInfo grid, int blockSize)
{
	// Fiecare proces isi calculeaza linia si coloana
	//	de la care incepe sa prelucreze matricile
	int startRow = grid.row * blockSize;
	int startCol = grid.col * blockSize;

	*local_a = (int**)malloc(MATRIX_SIZE * sizeof(int*));
	*local_b = (int**)malloc(MATRIX_SIZE * sizeof(int*));
	*local_c = (int**)malloc(MATRIX_SIZE * sizeof(int*));
	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		(*local_a)[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
		(*local_b)[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
		(*local_c)[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
	}

	// Fiecare proces isi pune elementele de care are nevoie
	//	(submatrici) in copii (A si B sunt matricile ce se inmultesc, 
	//    C e matricea rezultat)
	for (int i = startRow; i < startRow + blockSize; i++)
	{
		for (int j = startCol; j < startCol + blockSize; j++)
		{
			(*local_a)[i - startRow][j - startCol] = first_matrix[i][j];
			(*local_b)[i - startRow][j - startCol] = second_matrix[i][j];
			(*local_c)[i - startRow][j - startCol] = 0;
		}
	}
}

int main(int argc, char *argv[]) {

	int block_size;
	int **local_a, **local_b, **local_c;
	clock_t start, end;
	MPI_Init(&argc, &argv);

	GridInfo grid;
	Setup_grid(&grid);

	printf("Procesul %d ready...\n", grid.rank);
	MPI_Barrier(MPI_COMM_WORLD);

	// Punand srand(time(NULL)) pentru fiecare proces, ne asiguram ca vom avea
	//		aceleasi numere generate pentru toate procesele!
	// srand cu acest tip de seed functioneaza asa
	srand(time(NULL));
	GenerateMatrices(MATRIX_SIZE);


	// Fiecare proces prelucreaza o submatrice
	//   patratica (bloc) de dimensiune MATRIX_SIZE / sqrt(nr_procese)
	// Ar trebui, asadar, sa luam MATRIX_SIZE asa incat impartirea sa se faca exact
	block_size = MATRIX_SIZE / grid.dim;

	initLocalMatrices(&local_a, &local_b, &local_c, grid, block_size);

	if (grid.rank == 0)
	{
		std::cout << "Incepere inmultire..." << std::endl;
	}
	
	// Apel ce blocheaza procesul curent
	//	pana cand toate celelalte procese asociate grid-ului
	//	ajung aici
	MPI_Barrier(grid.grid_comm);
	if (grid.rank == 0)
	{
		start = clock();
	}

	// Fiecare proces face alg Fox pe blocul sau
	FoxMultiply(MATRIX_SIZE, &grid, local_a, local_b, local_c);

	// ... si se asteapta aici ca toate sa termine
	MPI_Barrier(grid.grid_comm);

	if (grid.rank == 0)
	{
		end = clock();
		clock_t result_time = end - start;
		std::cout << "Timp scurs: " << double(result_time) / CLOCKS_PER_SEC << " sec"
			<< std::endl;
	}

	int *result_buffer = new int[MATRIX_SIZE * MATRIX_SIZE];
	int *local_buffer = new int[block_size * block_size];
	// Se pune submatricea (blocul) calculatata de procesul curent intr-un vector
	ConvertMatrixToVector(local_buffer, local_c, block_size);

	// Procesele copil trimit blocul rezultat calculat (local_buff)
	//  Procesul parinte colecteaza aceste blocuri si le asambleaza
	//   in matricea finala (in result_buff)
	MPI_Gather(local_buffer, block_size * block_size, MPI_INT, result_buffer, block_size * block_size, MPI_INT, 0, grid.grid_comm);
	
	// asteptam sa se fi trimis si sa se fi colectat tot
	//   inainte de a merge mai departe cu orice proces
	MPI_Barrier(grid.grid_comm);

	// Procesul parinte face verificarea corectitudinii
	if (grid.rank == 0) 
	{
		int k = 0;
		std::cout << "Rezultat bun? ";
		int **res = sequentialMultiplication();

		for (int blockIndexRow = 0; blockIndexRow < grid.dim; blockIndexRow++)
		{
			for (int blockIndexCol = 0; blockIndexCol < grid.dim; blockIndexCol++)
			{
				for (int i = blockIndexRow * block_size; i < blockIndexRow * block_size + block_size; i++)
				{
					for (int j = blockIndexCol * block_size; j < blockIndexCol * block_size + block_size; j++)
					{
						if (res[i][j] != result_buffer[k])
						{
							std::cout << "Eroare!" << std::endl;
							return 1;
						}
						k++;
					}
				}
			}
		}

		std::cout << "Da" << std::endl;
	}

	free(local_a);
	free(local_b);
	free(local_c);

	MPI_Finalize();
	return 0;
}