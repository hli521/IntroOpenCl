// clang++ -framework OpenCL Kernel4/MultiplyMatrix.cpp && ./a.out 2048 2048 2048
// clang++ -std=c++20 "-L/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -IOpenCL -lOpenCL Kernel4/MultiplyMatrix.cpp && ./a.exe 2048 2048 2048 

#define _CRT_SECURE_NO_WARNINGS

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
// #include <sys/time.h>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#define BUILD
// #define CPU
// #define DEBUG
#define FILE_PATH "./Kernel4/mult_matrix_kernel.cl"
// #define TEST
#define NUM_ITERATIONS 10
#define TS 32
#define WPT 8
#define WIDTH 4

#define RANDOM_INPUT

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
#ifdef RANDOM_INPUT
            float randomNum = static_cast<float>(std::rand()) ;
            float denom = (static_cast<float> (RAND_MAX));
            row.push_back((randomNum / denom));
            // printf("%10.4f\n", randomNum);
            // printf("%10.4f\n", denom);
#else
            row.push_back(static_cast<float>(i*num_cols + j));
#endif
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
    // return cpuOutput == gpuOutput;

    if (cpuOutput.size() != gpuOutput.size()) {
        return false;
    }

    for (size_t i = 0; i < cpuOutput.size(); i++) {
        if (fabs(cpuOutput[i] - gpuOutput[i]) > 0.001) {
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv){
    
    cl_device_id gpu;
    cl_context context;
    cl_int result;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // const uint32_t m = data1.size();
    // const uint32_t k = data2.size();
    // const uint32_t n = data2.at(0).size();
    // size_t globalSize[2] = {m,n};
    // const uint32_t matrixDims[3] = {m, k, n};
    

#ifdef TEST
    std::vector<std::vector<float>> data1 = {{1,2,3,6}, {34,22,5,4}, {32,22,5,4}, {34,3,5,4}};
    std::vector<std::vector<float>> data2 = {{65,4,3,4}, {4,2,55,44}, {3,22,5,4}, {33,21,5,42}};
    const uint32_t m = data1.size();
    const uint32_t k = data2.size();
    const uint32_t n = data2.at(0).size();
#else
    const uint32_t m = atoi(argv[1]);
    const uint32_t k = atoi(argv[2]);
    const uint32_t n = atoi(argv[3]);
    std::vector<std::vector<float>> data1;// = {{1,2}, {34,22},{213,3}};
    std::vector<std::vector<float>> data2;// = {{21,32,43,12}, {-2,-21,12,2}};
    populateInput(data1, m, k);
    populateInput(data2, k, n);
#endif
    const size_t local[2] = { TS / WPT, TS / WIDTH };
    const size_t globalSize[2] = {m / WPT, n / WIDTH};

    std::vector<float> in1;
    std::vector<float> in2;
    std::vector<float> out(m*n);

    // populateInput(data1, m, k);
    // populateInput(data2, k, n);

    flatArr(data1, in1);
    flatArr(data2, in2);
#ifdef DEBUG    
    // for (int i = 0; i < 10; i++) {
    //     printf("in1 %d - %d\n", i, in1[i]);
    //     printf("in2 %d - %d\n", i, in2[i]);
    // }

    printf("Matrix A\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%10.4f ", in1[i*n + j]);
        }
        printf("\n");
    }

    printf("Matrix B\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.4f ", in2[i*n + j]);
        }
        printf("\n");
    }
#endif

    cl_mem buffer_in1, buffer_in2, buffer_out, buffer_mDims;

    FILE *fp;
    const size_t file_size = getFileSize(FILE_PATH);
    // char kernel_code[file_size];
    std::vector<char> kernel_code(file_size);

    cl_uint num_platforms;
    cl_platform_id platform;

    result = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (result != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id nvidia_platform = nullptr;

    for (auto platform : platforms) {
        char name[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        if (strstr(name, "NVIDIA")) {
            nvidia_platform = platform;
            std::cout << "Found NVIDIA platform: " << name << "\n";
            break;
        }
    }

    if (!nvidia_platform) {
        std::cerr << "NVIDIA OpenCL platform not found.\n";
        return 1;
    }

    result = clGetDeviceIDs(nvidia_platform, CL_DEVICE_TYPE_GPU, 1, &gpu, NULL);

    if (result != CL_SUCCESS) {
        std::cerr << "Failed to get GPU device: " << result << std::endl;
        return 0;
    }

    char device_name[1024];
    clGetDeviceInfo(
        gpu, CL_DEVICE_NAME, 
        sizeof(char)*1024, &device_name, NULL
    );
    printf("DeviceName: %s\n", device_name);

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
    size_t readSize = fread(kernel_code.data(), 1, file_size, fp);
    fclose(fp);
    kernel_code[readSize] = '\0';
    //printf("%s\n", kernel_code);
    
    program = clCreateProgramWithSource(context, 1, (const char *[]) {kernel_code.data()}, NULL, &result);
    if (result != CL_SUCCESS){
        printf("Couldn't create program\n");
        return 3;
    }
    
    std::string build_options;
#ifdef BUILD
    build_options += "-DBUILD";
    build_options += " -DTILE_SIZE=" + std::to_string(TS);
    build_options += " -DWORK_PER_THREAD=" + std::to_string(WPT);
    build_options += " -DWIDTH=" + std::to_string(WIDTH);
    build_options += " -DTS_OVER_WPT=" + std::to_string(TS/WPT);
    build_options += " -DTS_OVER_WIDTH=" + std::to_string(TS/WIDTH);
    build_options += " -DK_OVER_WIDTH=" + std::to_string(k/WIDTH);
    build_options += " -DN_OVER_WIDTH=" + std::to_string(n/WIDTH);
#endif
    result = clBuildProgram(program, 0, NULL, build_options.c_str(), NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't build program\n");
        // Get the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate space for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("Build log:\n%s\n", log);
        free(log);
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

    // buffer_mDims = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * 3, NULL, &result);
    // if (result != CL_SUCCESS){
    //     printf("Couldn't initialize output buffer\n");
    //     return 6;
    // }
    
    result = clEnqueueWriteBuffer(queue, buffer_in1, CL_TRUE, 0, sizeof(float) * m * k, in1.data(), 0, NULL, NULL);
    if (result != CL_SUCCESS){
        printf("Couldn't enqueue write buffer 1\n");
        printf("Res: %d\n", result);
        printf("size: %llu, %zu\n", sizeof(int) * m * k, in1.size());
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
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &m);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &k);
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &n);

    size_t maxWGS;
    clGetKernelWorkGroupInfo(kernel, gpu, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWGS, NULL);
    printf("Work Group size: %zu\n", maxWGS);

    cl_event event = nullptr;
    double elapsed_time_ms = 0;
    cl_ulong time_start, time_end;
    for (int i = 0; i < NUM_ITERATIONS; i++) {

#ifdef TEST
        const size_t localTest[2] = { 1, 1 };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, local, 0, NULL, &event); 
#else
        result = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, local, 0, NULL, &event); 
#endif
        if (result != CL_SUCCESS) {
            printf("Kernel Enqueue Failure %d\n", result);
        }
        
        result = clFinish(queue);

        if (result != CL_SUCCESS) {
            printf("clFinish Failure %d\n", result);
        }

        result = clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, sizeof(float) * m * n, out.data(), 0, NULL, NULL);
        if (result != CL_SUCCESS) {
            printf("read buffer Failure %d\n", result);
        }

#ifdef TEST
        // for(int i = 0; i < m*n; i++){
        //     printf("%f\n", out[i]);
        // }
        
        printf("GPU Output\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%10.4f ", out[i*n + j]);
            }
            printf("\n");
        }
#endif
        if (event == nullptr) {
            printf("event: %p\n", event);
            return 0;
        }
        cl_ulong queued, submit;

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);

        result = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
        if (result != CL_SUCCESS) {
            printf("getevent %d\n", result);
            return 0;
        }
        
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
    // printf("===============\n");
    // printf("CPU Output\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%10.4f ", cpuOutput[i*n + j]);
    //     }
    //     printf("\n");
    // }
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

    uint32_t n_compute_units;
    clGetDeviceInfo(
        gpu, CL_DEVICE_MAX_COMPUTE_UNITS, 
        sizeof(cl_uint), &n_compute_units, NULL
    );
    printf("Max compute units: %d\n", n_compute_units);
}

// 6800 GFlops