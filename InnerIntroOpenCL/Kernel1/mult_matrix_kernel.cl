__kernel void mult_matrix_kernel(const __global float* in1, const __global float* in2, __global float* out, const uint m, const uint k, const uint n){
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // printf("row = %d, col = %d\n", row, col);
    float sum = 0.0;
    for (int i = 0; i < k; i++){
        sum += in1[row * k + i] * in2[col + n * i];
    }

    out[row * n + col] = sum;
}