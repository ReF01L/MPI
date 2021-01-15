cmake_minimum_required(VERSION 3.17)
project(MPI)

set(CMAKE_CXX_STANDARD 20)
include_directories(/usr/include/x86_64-linux-gnu/mpi)

set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(MPI main.cpp Matrix.cpp Matrix.h Floyd.cpp Floyd.h Merge.cpp Merge.h)