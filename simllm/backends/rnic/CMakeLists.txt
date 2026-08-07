cmake_minimum_required(VERSION 3.20)

project(simllm_rnic LANGUAGES CXX)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(SIMLLM_RNIC_DEV_DEFAULT ON)
else()
    set(SIMLLM_RNIC_DEV_DEFAULT OFF)
endif()

option(SIMLLM_RNIC_BUILD_TESTS
       "Build dependency-free RNIC native tests"
       ${SIMLLM_RNIC_DEV_DEFAULT})
option(SIMLLM_RNIC_BUILD_TOOLS
       "Build RNIC validation tools"
       ${SIMLLM_RNIC_DEV_DEFAULT})
option(SIMLLM_RNIC_WARNINGS_AS_ERRORS
       "Treat RNIC compiler warnings as errors"
       OFF)

add_library(simllm_rnic STATIC
    src/work_queue.cpp
)
add_library(simllm::rnic ALIAS simllm_rnic)

target_include_directories(simllm_rnic
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(simllm_rnic PUBLIC cxx_std_17)
set_target_properties(simllm_rnic PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simllm_rnic PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
        target_compile_options(simllm_rnic PRIVATE -Werror)
    endif()
endif()

if(SIMLLM_RNIC_BUILD_TESTS)
    enable_testing()
    add_executable(simllm_rnic_work_queue_test
        tests/work_queue_test.cpp
    )
    target_link_libraries(simllm_rnic_work_queue_test PRIVATE simllm::rnic)
    target_include_directories(simllm_rnic_work_queue_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_work_queue_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_work_queue_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_work_queue_test
             COMMAND simllm_rnic_work_queue_test)
endif()

if(SIMLLM_RNIC_BUILD_TOOLS)
    add_executable(simllm_rnic_wq_probe
        tools/wq_probe.cpp
    )
    target_link_libraries(simllm_rnic_wq_probe PRIVATE simllm::rnic)
    target_include_directories(simllm_rnic_wq_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_wq_probe PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_wq_probe PRIVATE -Werror)
        endif()
    endif()
endif()
