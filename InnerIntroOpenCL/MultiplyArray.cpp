#include <stdio.h>
#include <stdlib.h> 
#include <stdint.h>
#include <OpenCL/opencl.h>
#include <sys/time.h>

int main(int argc, char **argv){
    cl_device_id gpu;
    cl_context context;
    cl_int result;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    size_t count = 3;
    int i;
    const int arrlen = 3;

    int in1[arrlen] = {9, 25, 64};
    int in2[arrlen] = {9, 25, 64};
    int out[arrlen] = {0,0,0};
    cl_mem buffer_in1, buffer_in2, buffer_out;

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

    fp = fopen("kernel.cl", "r");
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

    buffer_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * arrlen, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * arrlen, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * arrlen, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize output buffer\n");
        return 6;
    }

    result = clEnqueueWriteBuffer(queue, buffer_in1, CL_TRUE, 0, sizeof(int) * arrlen, in1, 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 1\n");
        return 7;
    }

    result = clEnqueueWriteBuffer(queue, buffer_in2, CL_TRUE, 0, sizeof(int) * arrlen, in2, 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 2\n");
        return 8;
    }

    kernel = clCreateKernel(program, "multiply_array", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_in2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_out);

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL);

    clFinish(queue);

    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, sizeof(int) * arrlen, out, 0, NULL, NULL);

    for (i = 0; i < arrlen; i++){
        printf("%d\n", out[i]);
    }
}

////How to read device info.
// int main(int argc, char **argv) {
//     cl_device_id my_device_id;
//     int result;
//     cl_uint n_compute_units;
//     cl_ulong mem_size;

//     result = clGetDeviceIDs(
//         NULL, 
//         CL_DEVICE_TYPE_GPU, 1, &my_device_id, NULL
//     );
//     if (result != CL_SUCCESS){
//         printf("Something went wrong\n");
//         return 1;
//     }
//     printf("good\n");

//     clGetDeviceInfo(
//         my_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 
//         sizeof(cl_uint), &n_compute_units, NULL
//     );
//     printf("Max compute units: %d\n", n_compute_units);

//     clGetDeviceInfo(
//         my_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 
//         sizeof(cl_ulong), &mem_size, NULL
//     );
//     printf("Global mem size: %llu GB\n", mem_size / 1000000000L);
// }

// void randomizeArray(int size, uint32_t* arr);
// void multiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3);
// void timeMultiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3);
// void printArray(int size, uint32_t* arr);

// int main(int argc, char **argv) {

//     int size = atoi(argv[1]);

//     uint32_t* arr1 = (uint32_t*)malloc(size * sizeof(uint32_t));
//     uint32_t* arr2 = (uint32_t*)malloc(size * sizeof(uint32_t));
//     uint32_t* arr3 = (uint32_t*)malloc(size * sizeof(uint32_t));

//     randomizeArray(size, arr1);
//     randomizeArray(size, arr2);
    
//     timeMultiplyArray(size, arr1, arr2, arr3);
//     //printArray(size, arr3);

//     free(arr1);
//     free(arr2);
//     free(arr3);

//     return 0;
// }

// void randomizeArray(int size, uint32_t* arr){
//     for (int i = 0; i < size; i++){
//         arr[i] = i;
//     }
// }

// void multiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3){
    
//     for (int i = 0; i < size; i++){
//         arr3[i] = arr1[i] * arr2[i];
//     }
// }

// void timeMultiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3){
//     struct timeval start, end;
//     double seconds;

//     gettimeofday(&start, NULL);
//     multiplyArray(size, arr1, arr2, arr3);
//     gettimeofday(&end, NULL);

//     seconds = ((((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - start.tv_usec) * 1e-6;

//     printf("Time taken: %f seconds\n", seconds);
// }


// void printArray(int size, uint32_t* arr){
//     for (int i = 0; i < size; i++){
//         printf("%u, ", arr[i]);
//     }
//     printf("\n");
// }