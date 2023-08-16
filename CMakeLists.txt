cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(PkgConfig REQUIRED)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
find_package(GLPK REQUIRED)

BISON_TARGET (citip_parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cxx
              COMPILE_FLAGS --defines=${CMAKE_CURRENT_BINARY_DIR}/parser.hxx)
FLEX_TARGET  (citip_scanner scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cxx
              COMPILE_FLAGS --header-file=${CMAKE_CURRENT_BINARY_DIR}/scanner.hxx)
ADD_FLEX_BISON_DEPENDENCY(citip_scanner citip_parser)

pkg_check_modules(clp REQUIRED IMPORTED_TARGET clp)
pkg_check_modules(osi-clp REQUIRED IMPORTED_TARGET osi-clp)
pkg_check_modules(cbc REQUIRED IMPORTED_TARGET cbc)


include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(Citip
    main.cpp
    citip.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/parser.cxx
    ${CMAKE_CURRENT_BINARY_DIR}/scanner.cxx
)
set_property(TARGET Citip PROPERTY CXX_STANDARD 20)
target_link_libraries(Citip glpk PkgConfig::clp PkgConfig::osi-clp PkgConfig::cbc)
