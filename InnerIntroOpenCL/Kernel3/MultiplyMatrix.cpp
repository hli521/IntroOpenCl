// clang++ -framework OpenCL Kernel0/MultiplyMatrix.cpp && ./a.out 1000 1000 1000

#include <chrono>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <sys/time.h>
#include <vector>

#include <OpenCL/opencl.h>

#define BUILD
//#define CPU
// #define DEBUG
#define FILE_PATH "./Kernel3/mult_matrix_kernel.cl"
// #define TEST
#define NUM_ITERATIONS 3

long getFileSize(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);
    return size;
}

void flatArr(std::vector<std::vector<float>>& data, std::vector<float>& in){
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data.at(i).size(); j++){
            in.push_back(data.at(i).at(j));
        }
    }
}

void populateInput(std::vector<std::vector<float>>& v, size_t num_rows, size_t num_cols) {
    for (int i = 0; i < num_rows; i++) {
        std::vector<float> row;
        for (int j = 0; j < num_cols; j++) {
            row.push_back(static_cast <float> (std::rand()) / (static_cast <float> (RAND_MAX/32768)));
        }
        v.push_back(row);
    }
}


// compute the result of data1 x data2 and store in output as a flattened vector
void computeOutputOnCpu(std::vector<std::vector<float>>& data1, std::vector<std::vector<float>>& data2, std::vector<float>& output) {
    int m = data1.size();
    int k = data1.at(0).size();
    int n = data2.at(0).size();
    for (int row = 0; row < m; row++){
        for (int col = 0; col < n; col++){
            for (int i = 0; i < k; i++){
                output[row * n + col] += data1.at(row).at(i) * data2.at(i).at(col);
            }
        }
    }
}

// take in the output from the CPU and GPU and verify that the values matches and return True only if they are equal
bool verifyOutput(std::vector<float>& cpuOutput, std::vector<float>& gpuOutput) {    
    return cpuOutput == gpuOutput;
}

int main(int argc, char** argv){
    
    cl_device_id gpu;
    cl_context context;
    cl_int result;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // const uint m = data1.size();
    // const uint k = data2.size();
    // const uint n = data2.at(0).size();
    // size_t globalSize[2] = {m,n};
    // const uint matrixDims[3] = {m, k, n};
    

#ifdef TEST
    std::vector<std::vector<float>> data1 = {{0.1,2}, {34,22},{213,3}};
    std::vector<std::vector<float>> data2 = {{21,32,43,12}, {-2,-21,12,2}};
    const uint m = data1.size();
    const uint k = data2.size();
    const uint n = data2.at(0).size();
#else
    const uint m = atoi(argv[1]);
    const uint k = atoi(argv[2]);
    const uint n = atoi(argv[3]);
    const uint TS = 16;
    const uint WPT = 8;
    const size_t local[2] = { TS / WPT, TS };
    std::vector<std::vector<float>> data1;// = {{1,2}, {34,22},{213,3}};
    std::vector<std::vector<float>> data2;// = {{21,32,43,12}, {-2,-21,12,2}};
    populateInput(data1, m, k);
    populateInput(data2, k, n);
#endif
    const size_t globalSize[2] = {m/WPT,n};

    std::vector<float> in1;
    std::vector<float> in2;
    std::vector<float> out(m*n);

    // populateInput(data1, m, k);
    // populateInput(data2, k, n);

    flatArr(data1, in1);
    flatArr(data2, in2);
#ifdef DEBUG    
    for (int i = 0; i < 10; i++) {
        printf("in1 %d - %d\n", i, in1[i]);
        printf("in2 %d - %d\n", i, in2[i]);
    }
#endif


    cl_mem buffer_in1, buffer_in2, buffer_out, buffer_mDims;

    FILE *fp;
    const size_t file_size = getFileSize(FILE_PATH);
    char kernel_code[file_size];

    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &gpu, NULL);

    context = clCreateContext(NULL, 1, &gpu, NULL, NULL, &result);

    if (result != CL_SUCCESS){
        printf("Couldn't create context\n");
        return 1;
    }

    queue = clCreateCommandQueue(context, gpu, CL_QUEUE_PROFILING_ENABLE, &result);

    if (result != CL_SUCCESS){
        printf("Couldn't create command queue\n");
        return 2;
    }
    
    fp = fopen(FILE_PATH, "r");
    size_t readSize = fread(kernel_code, 1, file_size, fp);
    fclose(fp);
    kernel_code[readSize] = '\0';
    printf("%s\n", kernel_code);

    program = clCreateProgramWithSource(context, 1, (const char *[]) {kernel_code}, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't create program\n");
        return 3;
    }

    std::string build_options;
#ifdef BUILD
    build_options += "-DBUILD";
    build_options += " -DTS=" + std::to_string(TS);
    build_options += " -DWPT=" + std::to_string(WPT);
    build_options += " -DRTS=" + std::to_string(TS/WPT);
#endif
    result = clBuildProgram(program, 0, NULL, build_options.c_str(), NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't build program\n");
        return 4;
    }
    
    buffer_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * m * k, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * k * n, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize input buffer\n");
        return 5;
    }

    buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m * n, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't initialize output buffer\n");
        return 6;
    }

    // buffer_mDims = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint) * 3, NULL, &result);
    // if (result != CL_SUCCESS){
    //     printf("Couldn't initialize output buffer\n");
    //     return 6;
    // }
    
    result = clEnqueueWriteBuffer(queue, buffer_in1, CL_TRUE, 0, sizeof(float) * m * k, in1.data(), 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 1\n");
        printf("Res: %d\n", result);
        printf("size: %lu, %lu\n", sizeof(int) * m * k, in1.size());
        for (int i = 0; i < 10; i++) {
            printf("in1 %d - %0.3f\n", i, in1[i]);
        }
        return 7;
    }
    
    result = clEnqueueWriteBuffer(queue, buffer_in2, CL_TRUE, 0, sizeof(float) * k * n, in2.data(), 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 2\n");
        return 8;
    }

    // result = clEnqueueWriteBuffer(queue, buffer_mDims, CL_TRUE, 0, sizeof(int) * 3, matrixDims, 0, NULL, NULL);
    // if (result != CL_SUCCESS){
    //     printf("Couldn't enqueue write buffer 3\n");
    //     return 8;
    // }
    
    kernel = clCreateKernel(program, "mult_matrix_kernel", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_in2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 3, sizeof(uint), &m);
    clSetKernelArg(kernel, 4, sizeof(uint), &k);
    clSetKernelArg(kernel, 5, sizeof(uint), &n);

    size_t maxWGS;
    clGetKernelWorkGroupInfo(kernel, gpu, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWGS, NULL);
    printf("Work Group size: %lu\n", maxWGS);

    cl_event event;
    double elapsed_time_ms = 0;
    cl_ulong time_start, time_end;
    for (int i = 0; i < NUM_ITERATIONS; i++) {

#ifdef TEST
    const size_t localTest[2] = { 1, 1 };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localTest, 0, NULL, &event); 
#else
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, local, 0, NULL, &event); 
#endif

    clFinish(queue);
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, sizeof(int) * m * n, out.data(), 0, NULL, NULL);

#ifdef TEST
    for(int i = 0; i < m*n; i++){
        printf("%f\n", out[i]);
    }
#endif

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    elapsed_time_ms += (time_end - time_start) * 1e-6; // convert ns to ms
    
    // Added dummy if statement to prevent compiler from reordering code messing up elapsed time calc
    // if (elapsed_time_ms == 0) 
        //printf("i: %d | ns: %llu | ms: %f | elapsed: %f\n", i, time_end - time_start, (time_end - time_start) * 1e-6, elapsed_time_ms);
    
    }
    elapsed_time_ms /= NUM_ITERATIONS;
    std::vector<float> cpuOutput(m*n);
    auto start = std::chrono::high_resolution_clock::now();
    
#ifdef CPU
    computeOutputOnCpu(data1, data2, cpuOutput);
#endif
    auto end = std::chrono::high_resolution_clock::now();
    printf("Verification %s\n", (verifyOutput(cpuOutput, out) ? "PASSED" : "FAILED"));

    std::chrono::duration<double, std::milli> duration_ms = end - start;
    printf("CPU Runtime: %.6f ms\n", duration_ms.count());

    printf("GPU Runtime: %.6f ms\n", elapsed_time_ms);

    // char device_name[1024];
    // clGetDeviceInfo(
    //     gpu, CL_DEVICE_NAME, 
    //     sizeof(char)*1024, &device_name, NULL
    // );
    // printf("DeviceName: %s\n", device_name);

    double throughput = 1e-9 * m * n * k * (1000 / elapsed_time_ms);
    printf("Throughput: %.3f GFLOPs\n", throughput);

    double efficiency = 100 * throughput / 6800;
    printf("Efficiency: %.2f%%\n", efficiency);

    uint n_compute_units;
    clGetDeviceInfo(
        gpu, CL_DEVICE_MAX_COMPUTE_UNITS, 
        sizeof(cl_uint), &n_compute_units, NULL
    );
    printf("Max compute units: %d\n", n_compute_units);
}

// 6800 GFlops