#include <stdio.h>
#include <stdlib.h> 
#include <stdint.h>
#include <OpenCL/opencl.h>
#include <sys/time.h>
#include <vector>

// void flatArr(std::vector<std::vector<int>> &data, std::vector<int> &in, const uint *mDims){
//     printf("hi\n");
//     for(int i = 0; i < data.size(); i++){
//         for(int j = 0; j < data.at(i).size(); j++){
//             in.push_back(data.at(i).at(j));
//         }
//     }
//     printf("hi2\n");
// }

int main(int argc, char** argv){
    
    cl_device_id gpu;
    cl_context context;
    cl_int result;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    std::vector<std::vector<int>> data1 = {{1,2}, {1,2}};
    std::vector<std::vector<int>> data2 = {{1,2}, {1,2}};
    std::vector<std::vector<int>> output = {{0,0}, {0,0}};

    const uint m = data1.size();
    const uint k = data2.size();
    const uint n = data2.at(0).size();
    const size_t globalSize[2] = {m,n};
    const uint matrixDims[3] = {m, k, n};
    
    //std::vector<int> in1;
    //std::vector<int> in2;
    //std::vector<int> out;
    int in1[4] = {1,2,1,2};
    int in2[4] = {1,2,1,2};
    int out[4] = {};

    //flatArr(data1, in1, matrixDims);
    // printf("hi342\n");
    // for(int i : in1){
    //     printf("%d\n", i);
    // }
    //flatArr(data2, in2, matrixDims);
    // for(int i : in2){
    //     printf("%d\n", i);
    // }
    

    cl_mem buffer_in1, buffer_in2, buffer_out, buffer_mDims;

    FILE *fp;
    char kernel_code[1024];

    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &gpu, NULL);

    context = clCreateContext(NULL, 1, &gpu, NULL, NULL, &result);

    if (result != CL_SUCCESS){
        printf("Couldn't create context\n");
        return 1;
    }

    queue = clCreateCommandQueue(context, gpu, 0, &result);

    if (result != CL_SUCCESS){
        printf("Couldn't create command queue\n");
        return 2;
    }

    fp = fopen("mult_matrix_kernel.cl", "r");
    fread(kernel_code, 1, 1024, fp);
    fclose(fp);
    printf("%s\n", kernel_code);

    program = clCreateProgramWithSource(context, 1, (const char *[]) {kernel_code}, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't create program\n");
        return 3;
    }

    result = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't build program\n");
        return 4;
    }

    buffer_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint) * m * k, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint) * k * n, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint) * m * n, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize output buffer\n");
        return 6;
    }

    buffer_mDims = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 3, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize output buffer\n");
        return 6;
    }

    result = clEnqueueWriteBuffer(queue, buffer_in1, CL_TRUE, 0, sizeof(int) * m * k, in1, 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 1\n");
        return 7;
    }

    result = clEnqueueWriteBuffer(queue, buffer_in2, CL_TRUE, 0, sizeof(int) * k * n, in2, 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 2\n");
        return 8;
    }

    result = clEnqueueWriteBuffer(queue, buffer_mDims, CL_TRUE, 0, sizeof(int) * 3, matrixDims, 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 3\n");
        return 8;
    }

    kernel = clCreateKernel(program, "mult_matrix_kernel", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_mDims);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_in1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_in2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_out);

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, sizeof(int) * m * n, out, 0, NULL, NULL);
    //printf("hdfs\n");
    for(int i = 0; i < m*n; i++){
        //printf("gfs\n");
        printf("%d\n", out[i]);
        //printf("gfs2\n");
    }
}