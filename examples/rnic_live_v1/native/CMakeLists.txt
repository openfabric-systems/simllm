cmake_minimum_required(VERSION 3.20)

project(simllm_rnic_tier_a LANGUAGES CXX)

set(SIMLLM_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(SIMLLM_RNIC_SOURCE "${SIMLLM_REPOSITORY_ROOT}/simllm/backends/rnic")

set(SIMLLM_RNIC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SIMLLM_RNIC_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SIMLLM_RNIC_SOURCE}" "${CMAKE_BINARY_DIR}/rnic")

add_executable(simllm_rnic_tier_a
    fake_port_factory.cpp
    tier_a_producer.cpp
)
target_link_libraries(simllm_rnic_tier_a PRIVATE simllm::rnic)
target_include_directories(simllm_rnic_tier_a PRIVATE
    "${SIMLLM_RNIC_SOURCE}/tests"
)
target_compile_features(simllm_rnic_tier_a PRIVATE cxx_std_17)
set_target_properties(simllm_rnic_tier_a PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simllm_rnic_tier_a PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
endif()

include(CTest)
if(BUILD_TESTING)
    set(TIER_A_NATIVE_CHECK
        "${CMAKE_CURRENT_BINARY_DIR}/check_tier_a_native_$<CONFIG>.cmake")
    file(GENERATE
        OUTPUT "${TIER_A_NATIVE_CHECK}"
        CONTENT [=[
set(observations "${run_dir}/raw_observations.json")
file(MAKE_DIRECTORY "${run_dir}")
file(REMOVE "${observations}" "${observations}.tmp")
execute_process(
    COMMAND "$<TARGET_FILE:simllm_rnic_tier_a>"
            --factory fake
            --expectations "${expectations}"
            --observations "${observations}"
    RESULT_VARIABLE fake_result
    OUTPUT_VARIABLE fake_output
    ERROR_VARIABLE fake_error
)
if(NOT fake_result STREQUAL "0")
    message(FATAL_ERROR
        "fake Tier A producer returned ${fake_result}\n${fake_output}\n${fake_error}")
endif()
file(READ "${observations}" raw)
foreach(required
        "simllm-rnic-tier-a-observations-v1"
        "wrapper_bypass_control"
        "controlled_drop")
    string(FIND "${raw}" "${required}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "raw Tier A output is missing ${required}")
    endif()
endforeach()

execute_process(
    COMMAND "$<TARGET_FILE:simllm_rnic_tier_a>"
            --factory fake
            --expectations "${expectations}"
            --observations "${observations}"
    RESULT_VARIABLE repeat_result
    ERROR_VARIABLE repeat_error
)
if(NOT repeat_result STREQUAL "2")
    message(FATAL_ERROR
        "repeat publication returned ${repeat_result}, expected 2\n${repeat_error}")
endif()

set(htsim_observations "${run_dir}/htsim_observations.json")
file(REMOVE "${htsim_observations}" "${htsim_observations}.tmp")
execute_process(
    COMMAND "$<TARGET_FILE:simllm_rnic_tier_a>"
            --factory htsim
            --expectations "${expectations}"
            --observations "${htsim_observations}"
    RESULT_VARIABLE htsim_result
    ERROR_VARIABLE htsim_error
)
if(NOT htsim_result STREQUAL "2")
    message(FATAL_ERROR
        "unavailable htsim factory returned ${htsim_result}, expected 2")
endif()
string(FIND "${htsim_error}" "htsim Tier A port factory is unavailable" match)
if(match EQUAL -1)
    message(FATAL_ERROR "htsim factory rejection missed its diagnostic")
endif()
if(EXISTS "${htsim_observations}")
    message(FATAL_ERROR "unavailable htsim factory published observations")
endif()
]=]
    )
    add_test(
        NAME simllm_rnic_tier_a_native
        COMMAND "${CMAKE_COMMAND}"
                "-Drun_dir=${CMAKE_CURRENT_BINARY_DIR}/ctest_tier_a"
                "-Dexpectations=${CMAKE_CURRENT_SOURCE_DIR}/../tier_a_expectations.json"
                -P "${TIER_A_NATIVE_CHECK}"
    )
endif()
