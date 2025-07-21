Naive Matrix Multiplier
Using the GPU with OpenCL

Functions:
populateInput() - takes in a reference to a 2D vector (matrix), number of rows, and number of columns of the matrix. matrix holds the holds the populated matrix with specified number of rows and columns.

flatArr() - takes in a two dimensional matrix as reference to a 2D vector (matrix) and reference to a vector (out) to store the matrix in one dimension. Out contains the flattened matrix in row major. 

computeOutputOnCpu - takes in two references to two dimensional vectors (data1 and data2) that represents two dimensional matrices, and a reference to a vector (output). data1 must have dimensions m * k and data2 have dimensions k*n. This function multiplies data1 and data2 and stores the resulting m*n matrix in ouput as a one dimensional vector.

verifyOutput() - takes in two references to vectors (cpuOutput and gpuOutput). Compares values of both flattened matrix and returns a boolean value for whether the two are equivalent. 

main() - takes in three arguments (m, k, n) which are the dimensions of the matrices. 

*values in matrices are integers.

Kernel0 - takes in matrix dimensions (mDims[m,k,n]), two input arrays (in1 and in2), and a output array (out), these arrays are flattened matrices.

Description: Each value in the ouput matrix is a work-item. Each work-item multiplies and adds its corresponding row in data1 and column in data2 using a for-loop.

Benchmark (m = 2048, k = 2048, n = 2048, 100 iterations):
Avg GPU runtime: 2.380930 ms