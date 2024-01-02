#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Helper.h"
#include <time.h>
#include <math.h>
#include <omp.h>


void simpleLinearRegression(double **data, int numRows, int numCols, double *slope, double *intercept, double *rSquared, double *mse, double *mae);

int main() {
    const char *filename = "prir_data.csv";
    
    double **data;
    int numRows, numCols;

    // Loading data from CSV
    loadCSV(filename, &data, &numRows, &numCols);
    
    // Checking the data
    printf("Number of rows: %d\n", numRows);
    printf("Number of columns: %d\n", numCols);
    printf("First 5 rows:\n");
    printData(data, 5, numCols);


    double slope, intercept, rSquared, mse, mae;

    // Fitting model
    double startTime = omp_get_wtime();
    simpleLinearRegression(data, numRows, numCols, &slope, &intercept, &rSquared, &mse, &mae);
    double endTime = omp_get_wtime();


    // Display the results
    printf("\n\nResults\n");
    printf("Slope: %.4f\n", slope);
    printf("Intercept: %.4f\n", intercept);
    printf("R^2: %.4f\n", rSquared);
    printf("MSE(Mean Square Error): %.4f\n", mse);
    printf("MAE(Mean Absolute Error): %.4f\n", mae);

    // Display execution time
    printf("Execution Time: %.4f seconds\n", endTime - startTime);

    // Deallocating the memory
    for (int i = 0; i < numRows; i++) { free(data[i]); }
    free(data);
    return 0;
}


void simpleLinearRegression(double **data, int numRows, int numCols, double *slope, double *intercept, double *rSquared,  double *mse, double *mae) {
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;

    #pragma omp parallel for reduction(+:sumX,sumY,sumXY,sumX2)
    for (int i = 0; i < numRows; i++) {
        sumX += data[i][1];
        sumY += data[i][0];
        sumXY += data[i][1] * data[i][0];
        sumX2 += data[i][1] * data[i][1];
    }

    double n = (double)numRows;

    *slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    *intercept = (sumY - *slope * sumX) / n;

    // Calculate R^2
    double yMean = sumY / n;
    double ssr = 0.0; // Regression sum of squares
    double sst = 0.0; // Total sum of squares

    #pragma omp parallel for reduction(+:ssr,sst)
    for (int i = 0; i < numRows; i++) {
        double yPredicted = *intercept + *slope * data[i][1];
        ssr += (yPredicted - yMean) * (yPredicted - yMean);
        sst += (data[i][0] - yMean) * (data[i][0] - yMean);
    }

    *rSquared = ssr / sst;

    // Calculate MSE and MAE
    double locMSE = 0.0;
    double locMAE = 0.0;

    #pragma omp parallel for reduction(+:locMSE,locMAE)
    for (int i = 0; i < numRows; i++) {
        double yPredicted = *intercept + *slope * data[i][1];
        locMSE += pow(data[i][0] - yPredicted, 2);
        locMAE += fabs(data[i][0] - yPredicted);
    }

    *mse = locMSE  / n;
    *mae = locMAE / n;
}