#ifndef HELPER_H
#define HELPER_H


void loadCSV(const char *filename, double ***data, int *numRows, int *numCols);
void printData(double **data, int numRows, int numCols);

#endif