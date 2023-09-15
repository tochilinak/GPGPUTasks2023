#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


std::vector<unsigned char> get_device_name(cl_device_id device) {
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    return deviceName;
}

std::vector<cl_device_id> get_device_list(cl_platform_id platform) {
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    return devices;
}

cl_device_type get_device_type(cl_device_id device) {
    cl_device_type deviceType;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    return deviceType;
}

cl_platform_id get_device_platform(cl_device_id device) {
    cl_platform_id platform;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr));
    return platform;
}

cl_device_id choose_device() {
    cl_device_id device_gpu = nullptr, device_cpu = nullptr;
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        auto devices = get_device_list(platform);
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];
            cl_device_type deviceType = get_device_type(device);
            /*auto name = get_device_name(device);
            std::string s(name.begin(), name.end());
            if (strcmp(s.c_str(), "Intel(R) UHD Graphics 620") != 0)
                continue;*/
            if (deviceType == CL_DEVICE_TYPE_CPU && device_cpu == nullptr) {
                device_cpu = device;
            } else if (deviceType == CL_DEVICE_TYPE_GPU) {
                device_gpu = device;
                break;
            }
        }
        if (device_gpu != nullptr)
            break;
    }
    if (device_gpu == nullptr && device_cpu == nullptr) {
        return nullptr;
    }
    return device_gpu != nullptr ? device_gpu : device_cpu;
}

#define CLEAN(context, queue, as_buffer, bs_buffer, cs_buffer, program, kernel) \
    if (context != nullptr) \
        OCL_SAFE_CALL(clReleaseContext(context)); \
    if (queue != nullptr) \
        OCL_SAFE_CALL(clReleaseCommandQueue(queue)); \
    if (as_buffer != nullptr) \
        OCL_SAFE_CALL(clReleaseMemObject(as_buffer)); \
    if (bs_buffer != nullptr) \
        OCL_SAFE_CALL(clReleaseMemObject(bs_buffer)); \
    if (cs_buffer != nullptr) \
        OCL_SAFE_CALL(clReleaseMemObject(cs_buffer)); \
    if (program != nullptr) \
        OCL_SAFE_CALL(clReleaseProgram(program)); \
    if (kernel != nullptr) \
        OCL_SAFE_CALL(clReleaseKernel(kernel));


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");
    cl_int errcode;

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    
    cl_device_id device = choose_device();
    if (device == nullptr) {
        std::cout << "No device found!" << std::endl;
        return 1;
    }
    std::cout << "Chosen device: " << get_device_name(device).data() << std::endl;

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_device_id devices[] = {device};
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_command_queue queue = clCreateCommandQueue(context, device, 0L, &errcode);
    OCL_SAFE_CALL(errcode);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem as_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, as.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem bs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, bs.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem cs_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, cs.data(), &errcode);
    OCL_SAFE_CALL(errcode);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    const char *source_ptr = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }

    cl_int buildResult = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);
    size_t logSize = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
    std::vector<unsigned char> log(logSize, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), 0));
    if (logSize > 2) {
        std::cout << "Log: " << std::endl;
        std::cout << log.data() << std::endl;
    }
    if (buildResult != CL_SUCCESS) {
        std::cout << "Build failed." << std::endl;
        CLEAN(context, queue, as_buffer, bs_buffer, cs_buffer, program, nullptr);
        return 1;
    }
    std::cout << "Build successful" << std::endl;

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode);
    OCL_SAFE_CALL(errcode);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_buffer);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_buffer);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_buffer);
        clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            size_t global_work_sizes[] = {global_work_size};
            cl_event kernel_started;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_sizes, nullptr, 0, nullptr, &kernel_started));
            cl_event events[] = {kernel_started};
            OCL_SAFE_CALL(clWaitForEvents(1, events));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << (double) 3 * n * sizeof(float) / (1 << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            cl_event read_event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cs_buffer, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, &read_event));
            cl_event events[] = {read_event};
            OCL_SAFE_CALL(clWaitForEvents(1, events));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double) n * sizeof(float) / (1 << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    CLEAN(context, queue, as_buffer, bs_buffer, cs_buffer, program, kernel);
    return 0;
}
