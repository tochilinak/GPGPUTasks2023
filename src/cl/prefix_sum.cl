__kernel void calc_res(const __global unsigned int *as, 
                       __global unsigned int *res,
                       unsigned int mask, unsigned int n) {
    unsigned int i = get_global_id(0) + 1;
    if (i > n)
        return;
    if (i & mask) {
        res[i - 1] += as[i - (i & (mask - 1)) - 1];
    }
    /*if (i == 64) {
        printf("in kernel: %d %d\n", res[i - 1], as[i - i % mask - 1]);
    }*/
}

__kernel void reduce_a(__global unsigned int *as, 
                       unsigned int n,
                       unsigned int step) {
    int i = get_global_id(0) * step * 2;
    if (i + step * 2 > n)
        return;
    as[i + step * 2 - 1] += as[i + step - 1];
}