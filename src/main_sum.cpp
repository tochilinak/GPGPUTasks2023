#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"
#include "cl/sum_cl_defs.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned int zero[] = {0};
        char defines[1000];
        sprintf(defines, "-D VALUES_PER_WORKITEM=%d -D WORKGROUP_SIZE=%d", VALUES_PER_WORKITEM, WORKGROUP_SIZE);
        {
            ocl::Kernel sumAtomic(sum_kernel, sum_kernel_length, "sum_atomic", defines);
            sumAtomic.compile();
            unsigned int global_work_size = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero, 1);
                sumAtomic.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                               as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU sum_atomic result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU sum_atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU sum_atomic: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {   
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_cycle", defines);
            sum.compile();
            unsigned int groups = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
            unsigned int global_work_size = (groups + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                               as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU sum_cycle result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU sum_cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU sum_cycle: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {   
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_cycle_coalesced", defines);
            sum.compile();
            unsigned int groups = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
            unsigned int global_work_size = (groups + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                               as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU sum_cycle_coalesced result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU sum_cycle_coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU sum_cycle_coalesced: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {   
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_local_memory", defines);
            sum.compile();
            unsigned int global_work_size = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                               as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU sum_local_memory result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU sum_local_memory: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU sum_local_memory: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {   
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_tree", defines);
            sum.compile();
            unsigned int global_work_size = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                               as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU sum_tree result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU sum_tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU sum_tree: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
