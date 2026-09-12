cmake_minimum_required(VERSION 3.14)
project(AuraPulseCore VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(AURAPULSE_SOURCES
    src/filters.cpp
    src/pos.cpp
    src/chrom.cpp
    src/aurapulse.cpp
)

add_library(aurapulse_core STATIC ${AURAPULSE_SOURCES})
target_include_directories(aurapulse_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(aurapulse_shared SHARED ${AURAPULSE_SOURCES})
target_include_directories(aurapulse_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(EMSCRIPTEN)
    add_executable(aurapulse_wasm ${AURAPULSE_SOURCES})
    set_target_properties(aurapulse_wasm PROPERTIES
        OUTPUT_NAME "aurapulse"
        SUFFIX ".js"
        LINK_FLAGS "-O3 -s WASM=1 -s EXPORTED_RUNTIME_METHODS=['cwrap','ccall'] -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME='AuraPulseModule'"
    )
endif()
