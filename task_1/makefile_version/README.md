SINSUM_PROJECT:

Этот проект вычисляет сумму значений синуса для одного периода на 10⁷ элементов. 
Он поддерживает два типа данных для массива: float и double. 
Выбор типа массива производится на этапе сборки.

Проект имеет две версии:

    makefile_version — [сборка с помощью Makefile]
    cmake_version — [сборка с помощью CMake]

-> Сборка с Makefile

    - Перейди в папку makefile_version:
        cd makefile_version

    - Для сборки с типом данных float (по умолчанию) просто используй команду:
        make ARRAY_TYPE=float
        ./float_version

    - Для сборки с типом данных double, используй:
        make ARRAY_TYPE=double
        ./double_version