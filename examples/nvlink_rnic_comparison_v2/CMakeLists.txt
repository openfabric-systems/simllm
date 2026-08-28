cmake_minimum_required(VERSION 3.20)
project(simllm_nvlink_rnic_comparison_v2 LANGUAGES CXX)

if(NOT DEFINED HTSIM_SOURCE_DIR)
    message(FATAL_ERROR "HTSIM_SOURCE_DIR must name an immutable htsim source export")
endif()
if(NOT DEFINED HTSIM_SOURCE_COMMIT)
    message(FATAL_ERROR "HTSIM_SOURCE_COMMIT must name the exported htsim commit")
endif()

set(HTSIM_SIM_DIR "${HTSIM_SOURCE_DIR}/htsim/sim")

add_executable(
    rnic_transport_schedule
    rnic_transport_schedule.cpp
    "${HTSIM_SIM_DIR}/rnic_fluid_manifold.cpp"
    "${HTSIM_SIM_DIR}/rnic_max_min_allocator.cpp"
    "${HTSIM_SIM_DIR}/rnic_packetized_manifold.cpp"
)
target_include_directories(rnic_transport_schedule PRIVATE "${HTSIM_SIM_DIR}")
target_compile_features(rnic_transport_schedule PRIVATE cxx_std_17)
target_compile_definitions(
    rnic_transport_schedule
    PRIVATE HTSIM_SOURCE_COMMIT="${HTSIM_SOURCE_COMMIT}"
)

if(MSVC)
    target_compile_options(rnic_transport_schedule PRIVATE /W4)
else()
    target_compile_options(rnic_transport_schedule PRIVATE -Wall -Wextra -Wpedantic)
endif()
