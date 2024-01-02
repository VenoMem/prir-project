#include "Helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 64


void loadCSV(const char *filename, double ***data, int *numRows, int *numCols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int rows = 0;
    int cols = 0;


    while (!feof(file)) {
        char line[MAX_LINE_LENGTH];
        fgets(line, MAX_LINE_LENGTH, file);
        if (rows == 0) {
            char *token = strtok(line, ",");
            while (token != NULL) {
                token = strtok(NULL, ",");
                cols++;
            }
        }
        rows++;
    }
    
    *data = (double **)calloc(rows, sizeof(double *));
    if (*data == NULL) { exit(EXIT_FAILURE); }

    for (int i = 0; i < rows; i++) {
        (*data)[i] = (double *)calloc(cols, sizeof(double));
        if ((*data)[i] == NULL) { exit(EXIT_FAILURE); }
    }

    rewind(file);

    for (int i = 0; i < rows; i++) {
        char line[MAX_LINE_LENGTH];
        fgets(line, MAX_LINE_LENGTH, file);
        char *token = strtok(line, ",");
        for (int j = 0; j < cols; j++) {
            (*data)[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(file);

    *numRows = rows;
    *numCols = cols;
}

void printData(double **data, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%f\t", data[i][j]);
        }
        printf("\n");
    }
}