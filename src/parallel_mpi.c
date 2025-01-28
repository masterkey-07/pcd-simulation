#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define tam 100

int main(int argc, char *argv[])
{

    float *vec, max, min, *maxTot, *minTot;
    int i, pri, qtde, tamLocal, ierr, numProc, esteProc, iproc;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &esteProc);

    maxTot = (float *)malloc(numProc * sizeof(float));
    if (!maxTot)
    {
        printf("Impossivel alocar MaxTot\n");
        return (0);
    }
    minTot = (float *)malloc(numProc * sizeof(float));
    if (!minTot)
    {
        printf("Impossivel alocar MinTot\n");
        return (0);
    }

    tamLocal = tam / numProc;
    pri = esteProc * tamLocal;

    if (esteProc == numProc - 1)
    {
        qtde = tam - pri;
    }
    else
        qtde = floor((float)tam / numProc);

    vec = (float *)malloc(qtde * sizeof(float));
    if (!vec)
    {
        printf("Impossivel alocar\n");
        return (0);
    }

    printf("no %d/%d: pri=%d, qtde=%d, ultimo=%d\n",
           esteProc, numProc, pri, qtde, (pri + qtde));

    for (i = 0; i < qtde; i++)
    {
        vec[i] = ((float)(i + pri) - (float)tam / 2.0);
        vec[i] *= vec[i];
    }
    for (i = 0; i < qtde; i++)
    {
        vec[i] = sqrt(vec[i]);
    }

    maxTot[esteProc] = vec[0];
    minTot[esteProc] = vec[0];
    for (i = 0; i < qtde; i++)
    {
        if (vec[i] > maxTot[esteProc])
            maxTot[esteProc] = vec[i];
        if (vec[i] < minTot[esteProc])
            minTot[esteProc] = vec[i];
    }

    printf("no %d/%d: min= %f, max=%f\n", esteProc, numProc, minTot[esteProc], maxTot[esteProc]);
    fflush(stdout);

    if (esteProc != 0)
    { /* Processos escravos */
        MPI_Send(&maxTot[esteProc], 1, MPI_FLOAT, 0, 12, MPI_COMM_WORLD);
        MPI_Send(&minTot[esteProc], 1, MPI_FLOAT, 0, 13, MPI_COMM_WORLD);
    }
    else
    { /* Processo Mestre */
        for (iproc = 1; iproc < numProc; iproc++)
        {
            MPI_Recv(&maxTot[iproc], 1, MPI_FLOAT, iproc, 12, MPI_COMM_WORLD, &status);
            MPI_Recv(&minTot[iproc], 1, MPI_FLOAT, iproc, 13, MPI_COMM_WORLD, &status);
        }
    }

    if (esteProc == 0)
    { /* Processo Mestre */
        max = maxTot[0];
        min = minTot[0];
        for (i = 0; i < numProc; i++)
        {
            printf("MESTRE: i=%d, min= %f, max=%f\n", i, minTot[i], maxTot[i]);
            fflush(stdout);
            if (maxTot[i] > max)
                max = maxTot[i];
            if (minTot[i] < min)
                min = minTot[i];
        }
        printf("Max=%f, Min=%f\n", max, min);
    }
    MPI_Finalize();
    return (0);
}