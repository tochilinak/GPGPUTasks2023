#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

#define WORK_GROUP_SIZE 128
#define DIGITS 4

gpu::WorkSize calculate_work_size(int items) {
    unsigned int workGroupSize = WORK_GROUP_SIZE;
    unsigned int global_work_size = (items + workGroupSize - 1) / workGroupSize * workGroupSize;
    return gpu::WorkSize(workGroupSize, global_work_size);
}

gpu::WorkSize calculate_work_size_2_dim(int items_1, int items_2) {
    unsigned int workGroupSizeX = 16;
    unsigned int workGroupSizeY = 16;
    unsigned int workSizeX = (items_1 + workGroupSizeX - 1) / workGroupSizeX * workGroupSizeX;
    unsigned int workSizeY = (items_2 + workGroupSizeY - 1) / workGroupSizeY * workGroupSizeY;
    return gpu::WorkSize(workGroupSizeX, workGroupSizeY, workSizeX, workSizeY);
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;
    gpu::gpu_mem_32u cs_gpu;
    gpu::gpu_mem_32u ds_gpu;
    gpu::gpu_mem_32u es_gpu;
    as_gpu.resizeN(n);
    unsigned int count_sz = (1 << DIGITS) * n / WORK_GROUP_SIZE;
    bs_gpu.resizeN(count_sz);
    std::vector<unsigned int> bs(count_sz);
    cs_gpu.resizeN(count_sz);
    es_gpu.resizeN(count_sz);
    ds_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        transpose.compile();
        ocl::Kernel calc_prefix(radix_kernel, radix_kernel_length, "calc_prefix");
        calc_prefix.compile();
        ocl::Kernel reduce_a(radix_kernel, radix_kernel_length, "reduce_a");
        reduce_a.compile();
        ocl::Kernel small_merge_sort(radix_kernel, radix_kernel_length, "small_merge_sort");
        small_merge_sort.compile();
        ocl::Kernel set_to_zero(radix_kernel, radix_kernel_length, "set_to_zero");
        set_to_zero.compile();
        ocl::Kernel copy(radix_kernel, radix_kernel_length, "copy");
        copy.compile();
        ocl::Kernel prefix_inside_group(radix_kernel, radix_kernel_length, "prefix_inside_group");
        prefix_inside_group.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(bs.data(), count_sz);

            t.restart();  // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int shift = 0; shift < 32; shift += DIGITS) {
                small_merge_sort.exec(calculate_work_size(n), as_gpu, ds_gpu, shift);
                count.exec(calculate_work_size(n), ds_gpu, bs_gpu, shift);
                copy.exec(calculate_work_size(count_sz), bs_gpu, es_gpu);
                transpose.exec(calculate_work_size_2_dim(1 << DIGITS, n / WORK_GROUP_SIZE), bs_gpu, cs_gpu, n / WORK_GROUP_SIZE, 1 << DIGITS);
                for (unsigned int step = 1; step <= count_sz / 2; step <<= 1) {
                    unsigned int groups = (((count_sz & (~step)) - (count_sz & (step - 1))) >> 1) + (count_sz & (step - 1));
                    calc_prefix.exec(calculate_work_size(groups), cs_gpu, bs_gpu, step, count_sz);
                    if (step < count_sz / 2) {
                        reduce_a.exec(calculate_work_size(count_sz / step / 2), cs_gpu, count_sz, step);
                    }
                }
                prefix_inside_group.exec(gpu::WorkSize(1 << DIGITS, (1 << DIGITS) * n / WORK_GROUP_SIZE), es_gpu);
                radix.exec(calculate_work_size(n), ds_gpu, as_gpu, bs_gpu, es_gpu, n, shift);
                set_to_zero.exec(calculate_work_size(count_sz), bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
