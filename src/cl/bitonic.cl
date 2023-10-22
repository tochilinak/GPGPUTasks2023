__kernel void bitonic(__global float *as, int n, int block_size, int period) {
    int id = get_global_id(0);
    int items_in_block = block_size / 2;
    int block_id = id / items_in_block;
    int i = block_id * block_size + (id % items_in_block);
    if (i >= n)
        return;
    float a  = as[i];
    int j = i + block_size / 2;
    float b = as[j];
    int direction = block_id / period % 2;
    if (direction == 0 && a > b || direction == 1 && a < b) {
        as[i] = b;
        as[j] = a;
    }
}
