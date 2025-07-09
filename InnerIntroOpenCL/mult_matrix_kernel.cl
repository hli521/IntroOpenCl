kernel void mult_matrix(global uint* mDims, global int* in1, global int* in2, global int* out){
    int row = get_global_id(0);
    int col = get_global_id(1);

    for (int i = 0; i < mDims[1]; i++){
        out[row * mDims[2] + col] = in1[row * mDims[1] + i] * in2[col + mDims[2] * i];
    }
}