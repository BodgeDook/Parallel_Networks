cmake_minimum_required(VERSION 3.10)
project(SinSum)

# compiler we need:
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# adding an option of choosing the value's type (default: float):
option(USE_DOUBLE "Use double instead of float" OFF)

# compilation flag:
if(USE_DOUBLE)
    add_compile_definitions(USE_DOUBLE)
endif()

# executable file making:
add_executable(sin_sum main.cpp)

# optimization flag:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")