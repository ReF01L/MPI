#include "Merge.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

void Merge::merge(int *a, int *b, int l, int m, int r) {
    int h = l;
    int i = l;
    int j = m + 1;

    while ((h <= m) && (j <= r)) {
        if (a[h] <= a[j]) {
            b[i] = a[h];
            h++;
        } else {
            b[i] = a[j];
            j++;
        }
        i++;
    }

    if (m < h)
        for (int k = j; k <= r; k++) {
            b[i] = a[k];
            i++;
        }
    else
        for (int k = h; k <= m; k++) {
            b[i] = a[k];
            i++;
        }
    for (int k = l; k <= r; k++)
        a[k] = b[k];
}

void Merge::mergeSort(int *a, int *b, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}

void Merge::run(int argc, char **argv) {
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    size_t n = 50000000;
    int *original_array = new int[n];
    double t1 = 0, t2, dt;
    if (world_rank == 0) {
        t1 = MPI_Wtime();

        srand(time(nullptr));

        for (int c = 0; c < n; c++)
            original_array[c] = rand() % n;
        printf("\n");
        printf("\n");
    }

    int size = n / world_size;

    int *sub_array = new int[size];
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);

    int *tmp_array = new int[size];
    mergeSort(sub_array, tmp_array, 0, (size - 1));

    int *sorted = nullptr;
    if (world_rank == 0)
        sorted = new int[n];

    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int *other_array = new int[n];
        mergeSort(sorted, other_array, 0, n - 1);

        printf("\n");
        printf("\n");

        t2 = MPI_Wtime();
        dt = t2 - t1;
        printf("Time of execution: %lf\n", dt);

        delete[](sorted);
        delete[](other_array);

    }

    delete[](original_array);
    delete[](sub_array);
    delete[](tmp_array);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
