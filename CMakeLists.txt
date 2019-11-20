cmake_minimum_required(VERSION 2.6)

project(tidal_field)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC")

# Uses pkgconfig to find genericio
pkg_check_modules(GENERICIO genericio)

include_directories(${GENERICIO_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
link_directories(${GENERICIO_LIBRARY_DIRS})

add_executable(tidal_field src/main.cpp)
target_link_libraries(tidal_field ${GENERICIO_LIBRARIES}  ${MPI_CXX_LIBRARIES})
