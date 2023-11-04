#define DIGITS 4


__kernel void radix(const __global unsigned int *as,
                    __global unsigned int *res,
                    const __global unsigned int *p,
                    const __global unsigned int *cnt,
                    int n,
                    unsigned int shift) {
    int i = get_global_id(0);
    int gid = get_group_id(0);
    int number_of_groups = n / get_local_size(0);
    unsigned int value = as[i];
    unsigned int digit = (value & (((1 << DIGITS) - 1) << shift)) >> shift;
    int new_idx = get_local_id(0);
    //for (unsigned int i = 0; i < digit; ++i) {
    //    new_idx -= cnt[gid * (1 << DIGITS) + i];
    //}
    if (digit)
        new_idx -= cnt[gid * (1 << DIGITS) + digit - 1];
    if (gid || digit) {
        new_idx += p[digit * number_of_groups + gid - 1];
    }
    res[new_idx] = value;
}

__kernel void prefix_inside_group(__global unsigned int *cnt) {
    int gid = get_group_id(0);
    int i = get_local_id(0);
    __global unsigned int *ptr = cnt + gid * (1 << DIGITS);
    unsigned int result = 0;
    int start = 0;
    int border = 1 << (DIGITS - 1);
    if (i >= border)
        start = border;
    for (int j = start; j <= i; j++)
        result += ptr[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    ptr[i] = result;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i >= border)
        ptr[i] += ptr[border - 1];
}

__kernel void set_to_zero(__global unsigned int *as) {
    int i = get_global_id(0);
    as[i] = 0;
}


__kernel void copy(const __global unsigned int *src, __global unsigned int *dst) {
    int i = get_global_id(0);
    dst[i] = src[i];
}


__kernel void count(const __global unsigned int *as, 
                    __global unsigned int *result,
                    unsigned int shift) {
    int idx = get_global_id(0);
    int gid = get_group_id(0);
    int start = gid * (1 << DIGITS);
    unsigned int mask = ((1 << DIGITS) - 1) << shift;
    unsigned int value = (as[idx] & mask) >> shift;
    atomic_add(result + (start + value), 1);
}

#define TILE_SIZE 16

__kernel void matrix_transpose(__global unsigned int *a, __global unsigned int *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local unsigned int tile[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i0 = i - local_i;
    int j0 = j - local_j;

    if (i < k && j < m) {
        tile[local_j * TILE_SIZE + (local_i + local_j) % TILE_SIZE] = a[j * k + i];
        a[j * k + i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int value = tile[local_i * TILE_SIZE + (local_j + local_i) % TILE_SIZE];

    if (j0 + local_i < m && i0 + local_j < k)
        at[(i0 + local_j) * m + j0 + local_i] = value;
}

__kernel void calc_prefix(const __global unsigned int *as, 
                         __global unsigned int *res,
                         unsigned int mask, unsigned int n) {
    unsigned int i = get_global_id(0);
    i = ((i & (~(mask - 1))) << 1) + mask + (i & (mask - 1));
    if (i > n)
        return;
    res[i - 1] += as[i - (i & (mask - 1)) - 1];
}

__kernel void reduce_a(__global unsigned int *as, 
                       unsigned int n,
                       unsigned int step) {
    int i = get_global_id(0) * step * 2;
    if (i + step * 2 > n)
        return;
    as[i + step * 2 - 1] += as[i + step - 1];
}

__kernel void small_merge_sort(__global unsigned int *a,
                               __global unsigned int *b,
                               unsigned int shift) {
    int id = get_local_id(0);
    int sz = get_local_size(0);
    int gid = get_global_id(0);
    int start = gid - gid % sz;
    __global unsigned int *src = a + start;
    __global unsigned int *dst = b + start;
    int cnt = 0;
    int mask = ((1 << DIGITS) - 1) << shift;
    for (int block = 1; block <= sz / 2; block <<= 1, ++cnt) {
        unsigned int value = src[id];
        int k = id / (2 * block);
        int i = id % (2 * block);
        int j;
        if (i < block) {
            int l0 = k * (2 * block) + block - 1;
            int l = l0;
            int r = (k + 1) * (2 * block);
            if (l > sz)
                l = sz;
            if (r > sz)
                r = sz;
            while (r - l > 1) {
                int m = (l + r) / 2;
                if ((src[m] & mask) < (value & mask)) {
                    l = m;
                } else {
                    r = m;
                }
            }
            j = i + (l - l0);
        } else {
            int l0 = k * (2 * block) - 1;
            int l = l0;
            int r = k * (2 * block) + block;
            while (r - l > 1) {
                int m = (l + r) / 2;
                if ((src[m] & mask) <= (value & mask)) {
                    l = m;
                } else {
                    r = m;
                }
            }
            j = (i - block) + (l - l0);
        }
        dst[k * (2 * block) + j] = value;
        __global unsigned int *tmp = src;
        src = dst;
        dst = tmp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (cnt % 2 == 0) {
        b[id] = a[id];
    }
}