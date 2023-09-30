#define TILE_SIZE 32

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i0 = i - local_i;
    int j0 = j - local_j;

    if (i < k && j < m) {
        tile[local_j * TILE_SIZE + local_i] = a[j * k + i];
    //} else {
    //    tile[local_j * TILE_SIZE + local_i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float value = tile[local_i * TILE_SIZE + local_j];
    barrier(CLK_LOCAL_MEM_FENCE);
    tile[local_j * TILE_SIZE + local_i] = value;

    if (j0 + local_i < m && i0 + local_j < k)
        at[(i0 + local_j) * m + j0 + local_i] = tile[local_j * TILE_SIZE + local_i];
}