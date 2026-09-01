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
    src/gpu_device.cpp
    src/host_memory.cpp
    src/pcie_fabric.cpp
    src/rnic_anomaly_table.cpp
    src/rnic_cmodel_c.cpp
    src/rnic_device.cpp
    src/rnic_hw_profile.cpp
    src/rnic_tx_pipeline.cpp
    src/session_record.cpp
    src/submission.cpp
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
    add_executable(simllm_rnic_pcie_fabric_test
        tests/pcie_fabric_test.cpp
    )
    target_link_libraries(simllm_rnic_pcie_fabric_test PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_pcie_fabric_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_pcie_fabric_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_pcie_fabric_test
             COMMAND simllm_rnic_pcie_fabric_test)

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

    add_executable(simllm_rnic_host_memory_test
        tests/host_memory_test.cpp
    )
    target_link_libraries(simllm_rnic_host_memory_test PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_host_memory_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_host_memory_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_host_memory_test
             COMMAND simllm_rnic_host_memory_test)

    add_executable(simllm_rnic_submission_test
        tests/submission_test.cpp
    )
    target_link_libraries(simllm_rnic_submission_test PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_submission_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_submission_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_submission_test
             COMMAND simllm_rnic_submission_test)

    add_executable(simllm_rnic_gpu_device_test
        tests/gpu_device_test.cpp
    )
    target_link_libraries(simllm_rnic_gpu_device_test PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_gpu_device_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_gpu_device_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_gpu_device_test
             COMMAND simllm_rnic_gpu_device_test)

    add_executable(simllm_rnic_device_test
        tests/rnic_device_test.cpp
        tests/session_record_test.cpp
    )
    target_link_libraries(simllm_rnic_device_test PRIVATE simllm::rnic)
    target_compile_definitions(simllm_rnic_device_test PRIVATE
        SIMLLM_RNIC_SESSION_RECORD_EMBEDDED=1
    )
    target_include_directories(simllm_rnic_device_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_device_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_device_test PRIVATE -Werror)
        endif()
    endif()
    add_test(NAME simllm_rnic_device_test
             COMMAND simllm_rnic_device_test)

    add_executable(simllm_rnic_cmodel_test
        tests/rnic_cmodel_test.cpp
    )
    target_link_libraries(simllm_rnic_cmodel_test PRIVATE simllm::rnic)
    target_include_directories(simllm_rnic_cmodel_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_cmodel_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_cmodel_test PRIVATE -Werror)
        endif()
    endif()
    # The generated anomaly-table projection and the design document it
    # projects both live in the repository docs tree, so the byte comparison
    # runs against the sources rather than a copy.
    set(SIMLLM_RNIC_DOC_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../docs/design")
    add_test(NAME simllm_rnic_cmodel_test
             COMMAND simllm_rnic_cmodel_test
                     "${SIMLLM_RNIC_DOC_DIR}/rnic-anomaly-table.md"
                     "${SIMLLM_RNIC_DOC_DIR}/rnic-cmodel.md"
                     "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(SIMLLM_RNIC_BUILD_TOOLS)
    add_executable(simllm_rnic_pcie_probe
        tools/pcie_probe.cpp
    )
    target_link_libraries(simllm_rnic_pcie_probe PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_pcie_probe PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_pcie_probe PRIVATE -Werror)
        endif()
    endif()

    add_executable(simllm_rnic_cmodel_probe
        tools/cmodel_probe.cpp
    )
    target_link_libraries(simllm_rnic_cmodel_probe PRIVATE simllm::rnic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_cmodel_probe PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_cmodel_probe PRIVATE -Werror)
        endif()
    endif()

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

    add_executable(simllm_rnic_session_record_test
        tests/session_record_test.cpp
    )
    target_link_libraries(simllm_rnic_session_record_test PRIVATE simllm::rnic)
    target_include_directories(simllm_rnic_session_record_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simllm_rnic_session_record_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if(SIMLLM_RNIC_WARNINGS_AS_ERRORS)
            target_compile_options(simllm_rnic_session_record_test PRIVATE -Werror)
        endif()
    endif()
endif()

if(SIMLLM_RNIC_BUILD_TESTS AND SIMLLM_RNIC_BUILD_TOOLS)
    set(SIMLLM_RNIC_NEGATIVE_PROBE_CHECK
        "${CMAKE_CURRENT_BINARY_DIR}/check_negative_probe_$<CONFIG>.cmake")
    file(GENERATE
        OUTPUT "${SIMLLM_RNIC_NEGATIVE_PROBE_CHECK}"
        CONTENT [=[
execute_process(
    COMMAND "$<TARGET_FILE:simllm_rnic_wq_probe>"
            "--doorbell-service-ps" "-1"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(NOT result STREQUAL "2")
    message(FATAL_ERROR
        "negative probe returned ${result}, expected 2\n${output}\n${error}")
endif()
set(expected "--doorbell-service-ps must be an unsigned integer")
string(FIND "${error}" "${expected}" match)
if(match EQUAL -1)
    message(FATAL_ERROR
        "negative probe missed parser diagnostic\n${output}\n${error}")
endif()
]=]
    )
    add_test(NAME simllm_rnic_wq_probe_rejects_negative_service
             COMMAND ${CMAKE_COMMAND}
                     -P "${SIMLLM_RNIC_NEGATIVE_PROBE_CHECK}")
endif()
