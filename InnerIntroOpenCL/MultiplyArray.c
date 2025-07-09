#include <stdio.h>
#include <stdlib.h> 
#include <stdint.h>
#include <sys/time.h>

void randomizeArray(int size, uint32_t* arr);
void multiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3);
void timeMultiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3);
void printArray(int size, uint32_t* arr);

int main(int argc, char **argv) {

    int size = atoi(argv[1]);

    uint32_t* arr1 = (uint32_t*)malloc(size * sizeof(uint32_t));
    uint32_t* arr2 = (uint32_t*)malloc(size * sizeof(uint32_t));
    uint32_t* arr3 = (uint32_t*)malloc(size * sizeof(uint32_t));

    randomizeArray(size, arr1);
    randomizeArray(size, arr2);
    
    timeMultiplyArray(size, arr1, arr2, arr3);
    //printArray(size, arr3);

    free(arr1);
    free(arr2);
    free(arr3);

    return 0;
}

void randomizeArray(int size, uint32_t* arr){
    for (int i = 0; i < size; i++){
        arr[i] = i;
    }
}

void multiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3){
    
    for (int i = 0; i < size; i++){
        arr3[i] = arr1[i] * arr2[i];
    }
}

void timeMultiplyArray(int size, uint32_t* arr1, uint32_t* arr2, uint32_t* arr3){
    struct timeval start, end;
    double seconds;

    gettimeofday(&start, NULL);
    multiplyArray(size, arr1, arr2, arr3);
    gettimeofday(&end, NULL);

    seconds = ((((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - start.tv_usec) * 1e-6;

    printf("Time taken: %f seconds\n", seconds);
}


void printArray(int size, uint32_t* arr){
    for (int i = 0; i < size; i++){
        printf("%u, ", arr[i]);
    }
    printf("\n");
}