#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Helper.h"
#include <time.h>
#include <math.h>


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
    clock_t start = clock();
    simpleLinearRegression(data, numRows, numCols, &slope, &intercept, &rSquared, &mse, &mae);
    clock_t end = clock();


    // Display the results
    printf("\n\nResults\n");
    printf("Slope: %.4f\n", slope);
    printf("Intercept: %.4f\n", intercept);
    printf("R^2: %.4f\n", rSquared);
    printf("MSE(Mean Square Error): %.4f\n", mse);
    printf("MAE(Mean Absolute Error): %.4f\n", mae);

    // Display execution time
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution Time: %.4f seconds\n", cpu_time_used);

    // Deallocating the memory
    for (int i = 0; i < numRows; i++) { free(data[i]); }
    free(data);
    return 0;
}


void simpleLinearRegression(double **data, int numRows, int numCols, double *slope, double *intercept, double *rSquared,  double *mse, double *mae) {
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;

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

    for (int i = 0; i < numRows; i++) {
        double yPredicted = *intercept + *slope * data[i][1];
        ssr += (yPredicted - yMean) * (yPredicted - yMean);
        sst += (data[i][0] - yMean) * (data[i][0] - yMean);
    }

    *rSquared = ssr / sst;

    // Calculate MSE and MAE
    *mse = 0.0;
    *mae = 0.0;

    for (int i = 0; i < numRows; i++) {
        double yPredicted = *intercept + *slope * data[i][1];
        *mse += pow(data[i][0] - yPredicted, 2);
        *mae += fabs(data[i][0] - yPredicted);
    }

    *mse /= n;
    *mae /= n;
}