// Increasing work per thread (one thread multiple columns of output)
// #define TS 16
// #define WPT 4
// #define RTS 4

__kernel void mult_matrix_kernel(const __global float* in1, const __global float* in2, __global float* out, const uint m, const uint k, const uint n){
#ifndef BUILD
    const int TS = 16;
    const int WPT = 4;
    const int RTS = TS/WPT;
#endif

    // Thread identifiers
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    // Local memory to fit a tile of TS*TS elements of in1 and in2
    __local float in1Sub[TS][TS];
    __local float in2Sub[TS][TS];

    //printf("row = %d, col = %d\n", row, col);
    //printf("globalRow = %d, globalCol = %d\n", globalRow, globalCol);
    float sum[WPT];
    for(int i = 0; i < WPT; i++){
        sum[i] = 0.0;
    }

    const int numTiles = k/TS;

    for(int t = 0; t < numTiles; t++){

        // Load one tile of in1 and in2 into local memory
        for(int w = 0; w < WPT; w++){
            const int tiledRow = TS * t + row; //Get global row for this tile.
            const int tiledCol = TS * t + col;
            in1Sub[row + w * RTS][col] = in1[(globalRow + w * RTS) * k + tiledCol];
            in2Sub[row + w * RTS][col] = in2[(tiledRow + w * RTS) * n + globalCol];
            // if (globalRow == 1 && globalCol == 3){
            //     printf("globalRow = %d, globalCol = %d, in1 = %d, in2 = %d\n", globalRow, globalCol, globalRow * k + tiledCol, tiledRow * k + globalCol);
            // }
        }

        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for(int i=0; i<TS; i++) {
            for (int w = 0; w < WPT; w++){
                sum[w] += in1Sub[row + w * RTS][i] * in2Sub[i][col];
            }
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Store result into output
    for (int w = 0; w < WPT; w++){
        out[(globalRow + w * RTS) * n + globalCol] = sum[w];
    }
}