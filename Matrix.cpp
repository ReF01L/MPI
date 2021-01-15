#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include "Matrix.h"


#define MASTER 0
#define _MASTER 1
#define _WORKER 2

double Matrix::MatrixDeterminant(int nDim, double *pfMatrix) {
    double fDet = 1.;
    double fMaxElem;
    double fAcc;
    int i, j, k, m;

    for (k = 0; k < (nDim - 1); k++) {
        fMaxElem = fabs(pfMatrix[k * nDim + k]);
        m = k;
        for (i = k + 1; i < nDim; i++)
            if (fMaxElem < fabs(pfMatrix[i * nDim + k])) {
                fMaxElem = pfMatrix[i * nDim + k];
                m = i;
            }

        if (m != k) {
            for (i = k; i < nDim; i++) {
                fAcc = pfMatrix[k * nDim + i];
                pfMatrix[k * nDim + i] = pfMatrix[m * nDim + i];
                pfMatrix[m * nDim + i] = fAcc;
            }
            fDet *= (-1.);
        }

        if (pfMatrix[k * nDim + k] == 0.)
            return 0.0;

        for (j = (k + 1); j < nDim; j++) {
            fAcc = -pfMatrix[j * nDim + k] / pfMatrix[k * nDim + k];
            for (i = k; i < nDim; i++)
                pfMatrix[j * nDim + i] = pfMatrix[j * nDim + i] + fAcc * pfMatrix[k * nDim + i];
        }
    }

    for (i = 0; i < nDim; i++)
        fDet *= pfMatrix[i * nDim + i];

    return fDet;
}

double Matrix::Partition(double **a, int s, int end, int n) {
    int i, j, j1, j2;
    double det = 0;

    for (j1 = s; j1 < end; j1++) {
        auto **m = (double **) malloc((n - 1) * sizeof(double *));

        for (i = 0; i < n - 1; i++)
            m[i] = (double *) malloc((n - 1) * sizeof(double));

        for (i = 1; i < n; i++) {
            j2 = 0;
            for (j = 0; j < n; j++) {
                if (j == j1)
                    continue;
                m[i - 1][j2] = a[i][j];
                j2++;
            }
        }
        int dim = n - 1;
        auto *fMatrix = new double[dim * dim];
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                fMatrix[i * dim + j] = m[i][j];

        det += pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * MatrixDeterminant(dim, fMatrix);

        for (i = 0; i < n - 1; i++)
            free(m[i]);
        free(m);

    }

    return (det);
}

void Matrix::FindDeterminant(int argc, char **argv) {
    srand(time(nullptr));
    int NumberProcesses, rank, NumberWorkers, source, destination, message_type, offset, i, j, k, rc, len, n;
    double det, StartTime, EndTime, read_StartTime, read_EndTime, print_StartTime, print_EndTime, **matrix, *buffer, determinant_of_matrix;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &NumberProcesses);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);
    StartTime = MPI_Wtime();
    NumberWorkers = NumberProcesses - 1;

    if (rank) {
        message_type = _MASTER;
        MPI_Recv(&n, 1, MPI_INT, MASTER, message_type, MPI_COMM_WORLD, &status);
        buffer = (double *) malloc(sizeof(double) * n * n);
        MPI_Recv(buffer, n * n, MPI_DOUBLE, MASTER, message_type, MPI_COMM_WORLD, &status);

        offset = (n / NumberProcesses) + 0.5;

        int end;
        int start = (rank) * offset;
        if ((rank) == NumberWorkers)
            end = n;
        else
            end = (start + offset);

        matrix = (double **) malloc((n) * 8 * n);
        for (k = 0; k < n; ++k)
            matrix[k] = (double *) malloc((n) * sizeof(double));

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                matrix[i][j] = buffer[i * n + j];


        det = Partition(matrix, start, end, n);
        int h = 0;
        for (h = 0; h < n; ++h)
            free(matrix[h]);
        free(matrix);
        free(buffer);
        message_type = _WORKER;
        MPI_Send(&det, 1, MPI_DOUBLE, MASTER, message_type, MPI_COMM_WORLD);
    }

    if (!rank) {
        read_StartTime = MPI_Wtime();
        printf("Enter the n: ");
        fflush(stdout);
        scanf("%d", &n);
        read_EndTime = MPI_Wtime();

        buffer = (double *) malloc(sizeof(double) * n * n);
        printf("Number of  tasks = %d\n", NumberProcesses);
        print_StartTime = MPI_Wtime();
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                buffer[i * n + j] = (double) (rand() % 5) + 1;;
        print_EndTime = MPI_Wtime();

        offset = (n / NumberProcesses) + 0.5;

        message_type = _MASTER;
        for (destination = 1; destination <= NumberWorkers; destination++) {
            MPI_Send(&n, 1, MPI_INT, destination, message_type, MPI_COMM_WORLD);
            MPI_Send(buffer, n * n, MPI_DOUBLE, destination, message_type, MPI_COMM_WORLD);
        }
        matrix = (double **) malloc((n) * 8 * n);
        for (k = 0; k < n; ++k)
            matrix[k] = (double *) malloc((n) * sizeof(double));

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                matrix[i][j] = buffer[i * n + j];


        determinant_of_matrix = Partition(matrix, 0, offset, n);
        printf("%s calculate it's part with determinant=%3.2lf\n", hostname, determinant_of_matrix);

        free(buffer);

        message_type = _WORKER;

        for (i = 1; i <= NumberWorkers; i++) {
            source = i;
            MPI_Recv(&det, 1, MPI_DOUBLE, source, message_type, MPI_COMM_WORLD, &status);
            determinant_of_matrix += det;
        }

        EndTime = MPI_Wtime();
        printf("Elapsed time is: %f\n",
               ((EndTime - StartTime) - (print_EndTime - print_StartTime) - (read_EndTime - read_StartTime)));
        printf("Determinant of matrix is: %3.2lf\n", determinant_of_matrix);
    }

    MPI_Finalize();
}
