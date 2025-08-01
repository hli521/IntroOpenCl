// Increasing work per thread (one thread multiple localColumns of output)
// #define WIDTH 1

#ifndef BUILD
    #define WIDTH 8
#endif

#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

typedef union{
    float index[WIDTH];
    floatX s;
} vector;

__kernel void mult_matrix_kernel(const __global floatX* matrixA, const __global floatX* matrixB, __global floatX* outputMatrix, const uint M, const uint K, const uint N){
    #ifndef BUILD
        const int TILE_SIZE = 16;
        const int WORK_PER_THREAD = 4;
        const int TS_OVER_WPT = TILE_SIZE / WORK_PER_THREAD;
        const int TS_OVER_WIDTH = TILE_SIZE / WIDTH;
        const int K_OVER_WIDTH = K / WIDTH;
        const int N_OVER_WIDTH = N / WIDTH;
    #endif
    // Thread identifiers
    const int localRow = get_local_id(0); //0..TS/WPT
    const int localCol = get_local_id(1); //0..TS/WIDTH
    const int globalRow = TILE_SIZE * get_group_id(0) + localRow; //0..m/WPT
    const int globalCol = (TILE_SIZE/WIDTH) * get_group_id(1) + localCol; //0..n/WIDTH

    // Local memory to fit a tile of TS*TS elements of matrixA and matrixB
    __local floatX localMatrixA[TILE_SIZE][TS_OVER_WIDTH];
    __local floatX localMatrixB[TILE_SIZE][TS_OVER_WIDTH];

    //printf("row = %d, localCol = %d\n", localRow, localCol);
    //printf("globalRow = %d, globalCol = %d\n", globalRow, globalCol);
    // Initialise the accumulation registers

    #if WIDTH == 1
        floatX sum[WORK_PER_THREAD];
        for(int i = 0; i < WORK_PER_THREAD; i++){
            floatX s = 0.0f;
            sum[i] = s;
        }
    #elif WIDTH == 2
        floatX sum[WORK_PER_THREAD];
        for(int i = 0; i < WORK_PER_THREAD; i++){
            floatX s = {0.0f, 0.0f};
            sum[i] = s;
        }
    #elif WIDTH == 4
        floatX sum[WORK_PER_THREAD];
        for(int i = 0; i < WORK_PER_THREAD; i++){
            floatX s = {0.0f, 0.0f, 0.0f, 0.0f};
            sum[i] = s;
        }
    #elif WIDTH == 8
        floatX sum[WORK_PER_THREAD];
        for(int i = 0; i < WORK_PER_THREAD; i++){
            floatX s = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            sum[i] = s;
        }
    #endif

    
    const int numTiles = K / TILE_SIZE;

    for(int t = 0; t < numTiles; t++){

        // Load one tile of matrixA and matrixB into local memory
        const int tiledRow = TILE_SIZE * t + localRow; //Get global row for this tile.
        const int tiledCol = (TS_OVER_WIDTH) * t + localCol;

        for (int w = 0; w < WORK_PER_THREAD; w++){
            //int a = (globalRow + w * TS_OVER_WPT) * K_OVER_WIDTH + tiledCol;
            //int b = (tiledRow + w * TS_OVER_WPT) * N_OVER_WIDTH + globalCol;
            localMatrixA[localRow + w * TS_OVER_WPT][localCol] = matrixA[(globalRow + w * TS_OVER_WPT) * K_OVER_WIDTH + tiledCol];
            localMatrixB[localRow + w * TS_OVER_WPT][localCol] = matrixB[(tiledRow + w * TS_OVER_WPT) * N_OVER_WIDTH + globalCol];
            // if (globalRow == 1 && globalCol == 0){
                //printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, matrixA.data = %f, matrixB.data = %f\n", w, localRow, localCol, globalRow, globalCol, (globalRow + w * WPT) * (K_OVER_WIDTH) + tiledCol, (tiledRow + w * WPT) * (N_OVER_WIDTH) + globalCol, localMatrixA[localRow + w * WPT][localCol], localMatrixB[localRow + w * WPT][localCol].s1);
                // printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, matrixA.data = %f, matrixB.data = %f\n", w, localRow, localCol, globalRow, globalCol, a, b, localMatrixA[localRow + w * WORK_PER_THREAD][localCol].s0, localMatrixB[localRow + w * WORK_PER_THREAD][localCol].s1);
            // }
            //printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, x = %d\n", w, localRow, localCol, globalRow, globalCol, a, b, localRow + w * WPT);
        }

        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // if (globalRow == 0 && globalCol == 1){
        //     printf("\nLocal Matrix A: \n");
        //     for (int i = 0; i < TILE_SIZE; i++) {
        //         for (int j = 0; j < TS_OVER_WIDTH; j++) {
        //             printf("%10.4f %10.4f ", localMatrixA[i][j].s0, localMatrixA[i][j].s1);
        //         }
        //         printf("\n");
        //     }

        //     printf("\nLocal Matrix B: \n");
        //     for (int i = 0; i < TILE_SIZE; i++) {
        //         for (int j = 0; j < TS_OVER_WIDTH; j++) {
        //             printf("%10.4f %10.4f ", localMatrixB[i][j].s0, localMatrixB[i][j].s1);
        //         }
        //         printf("\n");
        //     }
        // }
        // Perform the computation for a single tile
        vector vec1, vec2;
        for (int k = 0; k < TS_OVER_WIDTH; k++){
            for (int wpt = 0; wpt < WORK_PER_THREAD; wpt++){
                vec1.s = localMatrixA[localRow + wpt * TS_OVER_WPT][k];
                for (int w = 0; w < WIDTH; w++){
                    //vec2.s = localMatrixB[WIDTH * k + w][localCol];
                    sum[wpt] += vec1.index[w] * localMatrixB[WIDTH * k + w][localCol];
                }
            }
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Store result into output

    for (int w = 0; w < WORK_PER_THREAD; w++){
        //int a = ((globalRow + w * TS_OVER_WPT) * (N_OVER_WIDTH)) + globalCol;
        outputMatrix[((globalRow + w * TS_OVER_WPT) * (N_OVER_WIDTH)) + globalCol] = sum[w];
        // printf("w = %d, a = %d, sum1 = %f, sum2 = %f \n", w, a, sum[w].s0, sum[w].s1);
    }
}