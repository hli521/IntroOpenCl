__kernel void mult_matrix_kernel(const __global int* mDim, const __global int* in1, const __global int* in2, __global int* out){
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // printf("row = %d, col = %d\n", row, col);
    float sum = 0.0;
    for (int i = 0; i < mDim[1]; i++){
        sum += in1[row * mDim[1] + i] * in2[col + mDim[2] * i];
    }

    out[row * mDim[2] + col] = sum;
}