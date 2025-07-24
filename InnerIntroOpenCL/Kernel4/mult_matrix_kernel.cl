// Increasing work per thread (one thread multiple localColumns of output)
#define WIDTH 1

#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

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
            int a = (globalRow + w * TS_OVER_WPT) * K_OVER_WIDTH + tiledCol;
            int b = (tiledRow  + w * TS_OVER_WPT) * N_OVER_WIDTH + globalCol;
            localMatrixA[localRow + w * TS_OVER_WPT][localCol] = matrixA[a];
            localMatrixB[localRow + w * TS_OVER_WPT][localCol] = matrixB[b];
            // if (globalRow == 1 && globalCol == 0){
                //printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, matrixA.data = %f, matrixB.data = %f\n", w, localRow, localCol, globalRow, globalCol, (globalRow + w * WPT) * (K_OVER_WIDTH) + tiledCol, (tiledRow + w * WPT) * (N_OVER_WIDTH) + globalCol, localMatrixA[localRow + w * WPT][localCol], localMatrixB[localRow + w * WPT][localCol].s1);
                // printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, matrixA.data = %f, matrixB.data = %f\n", w, localRow, localCol, globalRow, globalCol, a, b, localMatrixA[localRow + w * WORK_PER_THREAD][localCol].s0, localMatrixB[localRow + w * WORK_PER_THREAD][localCol].s1);
            // }
            //printf("w = %d, row = %d, localCol = %d, globalRow = %d, globalCol = %d, matrixA = %d, matrixB = %d, x = %d\n", w, localRow, localCol, globalRow, globalCol, a, b, localRow + w * WPT);
        }

        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vec1, vec2;
        float val1;
        for (int k = 0; k < TS_OVER_WIDTH; k++){
            for (int wpt = 0; wpt < WORK_PER_THREAD; wpt++){
                vec1 = localMatrixA[localRow + wpt * TS_OVER_WPT][k];
                for (int w = 0; w < WIDTH; w++){
                    vec2 = localMatrixB[WIDTH * k + w][localCol];
                    val1 = vec1;
                    sum[w] = val1 * vec2;
                }
            }
        }



        // for (int w = 0; w < WORK_PER_THREAD; w++){ // Loop over tile rows (for number of WPT)
        //     vec1 = localMatrixA[localRow + w * TS_OVER_WPT][localCol];
        //     for (int j = 0; j < WIDTH; j++){
        //         vec2 = localMatrixB[localCol][WIDTH * w + j];
        //         #if WIDTH == 1
        //                 val1 = vec1;
        //         #elif WIDTH == 2
        //             switch (j){
        //                 case 0: 
        //                     val1 = vec1.s0;
        //                     break;
        //                 case 1: 
        //                     val1 = vec1.s1;
        //                     break;
        //             }

        //             for (int numTimes = 0; numTimes < TILE_SIZE/WIDTH; numTimes++){
        //                 sum[w].s0 += val1 * vec2.s0;
        //                 sum[w].s1 += val1 * vec2.s1;
        //             }
        //             if (localRow == 0 && localCol == 0){
        //                 printf("row = %d, localCol = %d, w = %d, j = %d, val1 = %f, s0 = %f, s1 = %f\n", localRow, localCol, w, j, val1, vec2.s0, vec2.s1);
        //             }

        //         #elif WIDTH == 4
        //             switch (j){
        //                 case 0: val1 = vec1.s0;
        //                     break;
        //                 case 1: val1 = vec1.s1;
        //                     break;
        //                 case 2: val1 = vec1.s2;
        //                     break;
        //                 case 3: val1 = vec1.s3;
        //                     break;
        //             }
        //             sum[w].s0 += val1 * vec2.s0;
        //             sum[w].s1 += val1 * vec2.s1;
        //             sum[w].s2 += val1 * vec2.s2;
        //             sum[w].s3 += val1 * vec2.s3;
        //         #elif WIDTH == 8
        //             switch (j){
        //                 case 0: val1 = vec1.s0;
        //                     break;
        //                 case 1: val1 = vec1.s1;
        //                     break;
        //                 case 2: val1 = vec1.s2;
        //                     break;
        //                 case 3: val1 = vec1.s3;
        //                     break;
        //                 case 4: val1 = vec1.s4;
        //                     break;
        //                 case 5: val1 = vec1.s5;
        //                     break;
        //                 case 6: val1 = vec1.s6;
        //                     break;
        //                 case 7: val1 = vec1.s7;
        //                     break;
        //             }
        //             sum[w].s0 += val1 * vec2.s0;
        //             sum[w].s1 += val1 * vec2.s1;
        //             sum[w].s2 += val1 * vec2.s2;
        //             sum[w].s3 += val1 * vec2.s3;
        //             sum[w].s4 += val1 * vec2.s4;
        //             sum[w].s5 += val1 * vec2.s5;
        //             sum[w].s6 += val1 * vec2.s6;
        //             sum[w].s7 += val1 * vec2.s7;
        //         #endif
        //     }
        // }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Store result into output
    for (int w = 0; w < WORK_PER_THREAD; w++){
        int a = ((globalRow + w * TS_OVER_WPT) * (N_OVER_WIDTH)) + globalCol;
        outputMatrix[a] = sum[w];
        // printf("w = %d, a = %d, sum1 = %f, sum2 = %f \n", w, a, sum[w].s0, sum[w].s1);
    }
}