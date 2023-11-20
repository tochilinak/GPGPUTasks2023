В этом репозитории предложены задания для курса по вычислениям на видеокартах 2023

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2023/).

# Задание 9. Просто космос




https://github.com/GPGPUCourse/GPGPUTasks2023/assets/22657075/9b2b4065-5492-4751-a587-f4f5c6df36ec





[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2023/actions/workflows/cmake.yml/badge.svg?branch=task09&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2023/actions/workflows/cmake.yml)

0. Сделать fork проекта
1. Выполнить задания 9.0, 9.1, 9.2, 9.3
2. Отправить **Pull-request** с названием ```Task09 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: 23:59 10 декабря.


Задание 9.0. GPU N-body
=========

Запустите и проверьте, что работает тест `(LBVH, Nbody)`, если закомментировать все варианты реализации кроме первого. 
Так же можно поэкспериментировать и освоиться с настройками, перечисленными в начале файла, позапускав тест `(LBVH, Nbody_meditation)` с наивной CPU реализацией и включенным GUI.


Задание 9.1. GPU N-body 
=========

Реализуйте кернел `nbody_calculate_force_global` и запустите тест `(LBVH, Nbody)` без последних двух вариантов.


Задание 9.2. CPU LBVH
=========

Реализуйте TODO в файле ```src/main_lbvh.cpp```, чтобы начал проходить тест `(LBVH, CPU)` и тест `(LBVH, Nbody)` без последнего варианта.


Задание 9.3. GPU LBVH
=========

Реализуйте оставшиеся TODO в файлах ```src/main_lbvh.cpp``` и ```src/cl/lbvh.cl```, чтобы начал проходить тест `(LBVH, GPU)` и тест `(LBVH, Nbody)` полностью.
