#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_atomic(__global const unsigned int *arr,
                         __global unsigned int *sum,
                         unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, arr[index]);
}

__kernel void sum_cycle(__global const unsigned int *arr,
                        __global unsigned int *sum,
                        unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    int local_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = gid * VALUES_PER_WORKITEM + i;
        if (index < n) {
            local_sum += arr[index];
        }
    }

    atomic_add(sum, local_sum);
}

__kernel void sum_cycle_coalesced(__global const unsigned int *arr,
                                  __global unsigned int *sum,
                                  unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    int local_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = (wid * VALUES_PER_WORKITEM + i) * grs + lid;
        if (index < n) {
            local_sum += arr[index];
        }
    }

    atomic_add(sum, local_sum);
}

__kernel void sum_local_memory(__global const unsigned int *arr,
                               __global unsigned int *sum,
                               unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    if (gid < n)
        buf[lid] = arr[gid];
    else
        buf[lid] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int local_sum = 0;
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            local_sum += buf[i];
        }

        atomic_add(sum, local_sum);
    }
}

__kernel void sum_tree(__global const unsigned int *arr,
                       __global unsigned int *sum,
                       unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    if (gid < n)
        buf[lid] = arr[gid];
    else
        buf[lid] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n = WORKGROUP_SIZE; n > 1; n /= 2) {
        if (2 * lid < n) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + n / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}