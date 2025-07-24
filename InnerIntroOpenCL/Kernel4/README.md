Matrix Multiplier with Tiling and Local Memory
Using the GPU with OpenCL

Functions:
populateInput() - takes in a reference to a 2D vector (matrix), number of rows, and number of columns of the matrix. matrix holds the holds the populated matrix with specified number of rows and columns.

flatArr() - takes in a two dimensional matrix as reference to a 2D vector (matrix) and reference to a vector (out) to store the matrix in one dimension. Out contains the flattened matrix in row major. 

computeOutputOnCpu - takes in two references to two dimensional vectors (data1 and data2) that represents two dimensional matrices, and a reference to a vector (output). data1 must have dimensions m * k and data2 have dimensions k*n. This function multiplies data1 and data2 and stores the resulting m*n matrix in ouput as a one dimensional vector.

verifyOutput() - takes in two references to vectors (cpuOutput and gpuOutput). Compares values of both flattened matrix and returns a boolean value for whether the two are equivalent. 

main() - takes in three arguments (m, k, n) which are the dimensions of the matrices. 

Previous:
*Changed values of matrices to be floats instead of ints.
*Added tile size TS = 128. 
*Found local work size (1, TS) = 128 to have the lowest average latency. 

Current:
**Changed TS to be 16, Local work size is (TS, TS) = 256.

Kernel2 - takes in two input arrays (in1 and in2), and a output array (out), these arrays are flattened matrices. Also takes in matrix dimensions as integers (m, k, n).

Description: Output matrix is divided into local workgroups of TS x TS (or a tile). Each fiber within the tile stores RTS = TS/WPT corresponding values from data1 and data2 into local memory. Then each fiber perform multiply and add within the tile for WPT number of values in output. 

Theoritical performance gain: Significantly reducing the number of loads from data1 when performing the computations. 

Previous:
*Caching tiles gives faster access to memory that is used by work-items in the same local work group.

Previous:
*replaced mDims[m, k, n] with constants m, k, n.

Benchmark (m = 2048, k = 2048, n = 2048, 100 iterations):
Avg GPU runtime: 0.312229 ms
% Decrease in latency compared to the previous kernel (0.499088 ms): 37.4%
% Decrease in latency compared to kernel0 (2.380930 ms): 86.9%