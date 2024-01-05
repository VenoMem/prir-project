#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "Helper.h"

#define MAX_LINE_LENGTH 64

void simpleLinearRegression(double **data, int numRows, int numCols, double *slope, double *intercept, double *rSquared, double *mse, double *mae);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *filename = "prir_data.csv";

    double **data;
    int numRows, numCols;

    if (rank == 0)
    {
        // Only rank 0 loads the data
        loadCSV(filename, &data, &numRows, &numCols);

        // Broadcasting numRows and numCols to all processes
        MPI_Bcast(&numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcasting the data to all processes
        for (int i = 0; i < numRows; i++)
        {
            MPI_Bcast(data[i], numCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Receiving numRows and numCols
        MPI_Bcast(&numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocating memory for data
        data = (double **)malloc(numRows * sizeof(double *));
        for (int i = 0; i < numRows; i++)
        {
            data[i] = (double *)malloc(numCols * sizeof(double));
        }

        // Receiving the data
        for (int i = 0; i < numRows; i++)
        {
            MPI_Bcast(data[i], numCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    double slope, intercept, rSquared, mse, mae;

    // Fitting model
    double startTime = MPI_Wtime();
    simpleLinearRegression(data, numRows, numCols, &slope, &intercept, &rSquared, &mse, &mae);
    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        // Display the results on each process
        printf("Rank %d - Slope: %.4f\n", rank, slope);
        printf("Rank %d - Intercept: %.4f\n", rank, intercept);
        printf("Rank %d - R^2: %.4f\n", rank, rSquared);
        printf("Rank %d - MSE(Mean Square Error): %.4f\n", rank, mse);
        printf("Rank %d - MAE(Mean Absolute Error): %.4f\n", rank, mae);

        // Display execution time on each process
        printf("Rank %d - Execution Time: %.4f seconds\n", rank, endTime - startTime);
    }

    // Deallocating the memory
    for (int i = 0; i < numRows; i++)
    {
        free(data[i]);
    }
    free(data);
    MPI_Finalize();
    return 0;
}

void simpleLinearRegression(double **data, int numRows, int numCols, double *slope, double *intercept, double *rSquared, double *mse, double *mae)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProcess = numRows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? numRows : startRow + rowsPerProcess;

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;

    for (int i = startRow; i < endRow; i++)
    {
        sumX += data[i][1];
        sumY += data[i][0];
        sumXY += data[i][1] * data[i][0];
        sumX2 += data[i][1] * data[i][1];
    }

    double globalSumX, globalSumY, globalSumXY, globalSumX2;
    MPI_Allreduce(&sumX, &globalSumX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sumY, &globalSumY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sumXY, &globalSumXY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sumX2, &globalSumX2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double n = (double)numRows;

    *slope = (n * globalSumXY - globalSumX * globalSumY) / (n * globalSumX2 - globalSumX * globalSumX);
    *intercept = (globalSumY - *slope * globalSumX) / n;

    // Calculate R^2
    double yMean = globalSumY / n;
    double ssr = 0.0; // Regression sum of squares
    double sst = 0.0; // Total sum of squares

    for (int i = startRow; i < endRow; i++)
    {
        double yPredicted = *intercept + *slope * data[i][1];
        ssr += (yPredicted - yMean) * (yPredicted - yMean);
        sst += (data[i][0] - yMean) * (data[i][0] - yMean);
    }

    double globalSsr, globalSst;
    MPI_Allreduce(&ssr, &globalSsr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sst, &globalSst, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    *rSquared = globalSsr / globalSst;

    // Calculate MSE and MAE
    double locMSE = 0.0;
    double locMAE = 0.0;

    for (int i = startRow; i < endRow; i++)
    {
        double yPredicted = *intercept + *slope * data[i][1];
        locMSE += pow(data[i][0] - yPredicted, 2);
        locMAE += fabs(data[i][0] - yPredicted);
    }

    double globalMSE, globalMAE;
    MPI_Allreduce(&locMSE, &globalMSE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&locMAE, &globalMAE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    *mse = globalMSE / n;
    *mae = globalMAE / n;
}
