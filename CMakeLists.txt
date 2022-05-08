cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 20)
project(NNFS VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(NNFS main.cpp)

find_package(Eigen3 3.4 NO_MODULE)
target_link_libraries(NNFS Eigen3::Eigen)
list(VCPKG_MANIFEST_DIR)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
