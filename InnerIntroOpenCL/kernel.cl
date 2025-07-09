kernel void multiply_array(global int *in1, global int *in2, global int *out){
    int id = get_global_id(0);
    out[id] = in1[id] * in2[id];
}