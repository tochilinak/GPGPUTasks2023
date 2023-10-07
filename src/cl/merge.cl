__kernel void merge(const __global float *a,
                    __global float *r,
                    unsigned int n, unsigned int b)
{
    int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    float value = a[global_id];
    int k = global_id / (2 * b);
    int i = global_id % (2 * b);
    int j;
    if (i < b) {
        int l0 = k * (2 * b) + b - 1;
        int l = l0;
        int r = (k + 1) * (2 * b);
        if (l > n)
            l = n;
        if (r > n)
            r = n;
        while (r - l > 1) {
            int m = (l + r) / 2;
            if (a[m] < value) {
                l = m;
            } else {
                r = m;
            }
        }
        j = i + (l - l0);
    } else {
        int l0 = k * (2 * b) - 1;
        int l = l0;
        int r = k * (2 * b) + b;
        while (r - l > 1) {
            int m = (l + r) / 2;
            if (a[m] <= value) {
                l = m;
            } else {
                r = m;
            }
        }
        j = (i - b) + (l - l0);
    }
    r[k * (2 * b) + j] = value;
}