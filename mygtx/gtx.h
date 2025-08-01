#pragma once

// CUDA
#include <cuda.h> // CUDA Driver API
#include <cuda_runtime.h> // CUDA Runtime API

// Cpp
#include <array>
#include <cstdio> // printf
#include <string> // string support
#include <vector> // vector support

// Utility

#define CUDA_RT(call)                                                        \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if ( cudaSuccess != _err ) {                                         \
            fprintf(stderr, "CUDA Runtime Error in file '%s' line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err));           \
            return _err;                                                     \
        } } while (0)

#define CUDA_DRV(call)                                                                                                              \
    do {                                                                                                                            \
        CUresult _err = (call);                                                                                                     \
        const char *error_str;                                                                                                      \
        const char* error_name;                                                                                                     \
        if (CUDA_SUCCESS != _err) {                                                                                                 \
            CUresult ret_str = cuGetErrorString(_err, &error_str);                                                                  \
            CUresult ret_name = cuGetErrorName(_err, &error_name);                                                                  \
            if (ret_str == CUDA_ERROR_INVALID_VALUE) error_str = "Unknown error";                                                   \
            fprintf(stderr, "CUDA Driver Error in file '%s' line '%d' with %s : %s \n", __FILE__, __LINE__, error_name, error_str); \
            return _err;                                                                                                            \
        } } while (0)

CUresult retain_primary_context_1t1g(
    const int device_id,
    CUdevice& device,
    CUcontext& primary_context)
{
    // * Get Device
    CUDA_DRV(cuDeviceGet(&device, device_id));

    // * Device Name
    ::std::string device_name(100, '\0');
    CUDA_DRV(cuDeviceGetName(device_name.data(), 100, device));

    // * UUID
    CUuuid device_uuid;
    CUDA_DRV(cuDeviceGetUuid_v2(&device_uuid, device));
    const ::std::string uuid_str_decimal = [&]() {
        ::std::string uuid_str_decimal;
        for (int i = 0; i < 16; i++) {
            uuid_str_decimal += ::std::to_string(static_cast<int>(device_uuid.bytes[i]));
        }
        return uuid_str_decimal;
    }();
    printf("[Device %d] Name %s, UUID %s\n", device_id, device_name.c_str(), uuid_str_decimal.c_str()); fflush(stdout);

    // * Primary Context
    CUDA_DRV(cuDevicePrimaryCtxRetain(&primary_context, device));

    // * Context Id
    unsigned long long primary_context_id;
    CUDA_DRV(cuCtxGetId(primary_context, &primary_context_id));
    printf("[Device %d] Primary Context Id %llu\n", device_id, primary_context_id); fflush(stdout);

    return CUDA_SUCCESS;
}

CUresult verify_primary_context_1t1g(
    const int device_id,
    const CUcontext & primary_context)
{
    // * Description
    // For CUDA Runtime API, stream, memory, event, etc are banded with particular device
    //   under the hood, particular device is banded to each device's primary context.
    // For CUDA Driver API, stream, memory, event, etc are directly banded with particular context
    //   , including each device's primary context.

    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    // * Get Current Context
    CUcontext current_context;
    CUDA_DRV(cuCtxGetCurrent(&current_context));

    // * Get Current Context Id
    unsigned long long current_context_id;
    CUDA_DRV(cuCtxGetId(current_context, &current_context_id));
    printf("Current Context Id %llu\n", current_context_id);
    fflush(stdout);

    // * Get Current Device
    CUdevice current_device;
    CUDA_DRV(cuCtxGetDevice(&current_device));

    // * Get Current Device Name
    ::std::string current_device_name(100, '\0');
    CUDA_DRV(cuDeviceGetName(current_device_name.data(), 100, current_device));

    // * Get Current Device UUID
    CUuuid current_device_uuid;
    CUDA_DRV(cuDeviceGetUuid_v2(&current_device_uuid, current_device));
    ::std::string current_device_uuid_decimal;
    for (int i = 0; i < 16; i++) {
        current_device_uuid_decimal += ::std::to_string(static_cast<int>(current_device_uuid.bytes[i]));
    }

    // * Print Current Device Info
    printf("[Device %d] Name %s, UUID %s\n", device_id, current_device_name.c_str(), current_device_uuid_decimal.c_str()); fflush(stdout);

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_1t1g(
    const int device_id,
    const int intended_primary_partition_sm_count,
    const int primary_partition_gtx_stream_priority,
    const int remaining_partition_gtx_stream_priority,
    const CUcontext & primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    CUstream& remaining_partition_gtx_stream,
    int& primary_partition_sm_count,
    int& remaining_partition_sm_count,
    const int gtx_partition_flag = 0)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    // * Get Current Device
    CUdevice current_device;
    CUDA_DRV(cuCtxGetDevice(&current_device));

    // * Green Context Creation and Usage
    // 1. query set of SM resource (cuDeviceGetDevResource)
    // 2. partition resource (cuDevSmResourceSplitByCount)
    // 3. finalize resource by creating desc (cuDevResourceGenerateDesc)
    // 4. create green context (cuGreenCtxCreate)
    // 5. create stream for each green context (cuGreenCtxStreamCreate)

    // (1)
    CUdevResource device_resource;
    CUDA_DRV(cuDeviceGetDevResource(current_device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("[Device %d] [Device Resource] sm count: %d\n", device_id, device_resource.sm.smCount); fflush(stdout);
    if (intended_primary_partition_sm_count > device_resource.sm.smCount) {
        printf("[Device %d] intended_primary_partition_sm_count > device_resource.sm.smCount\n", device_id); fflush(stdout);
        CUDA_DRV(CUDA_ERROR_INVALID_VALUE);
    }
    const bool set_green_context = intended_primary_partition_sm_count < device_resource.sm.smCount;

    // (2)
    CUdevResource primary_partition_device_resource;
    CUdevResource remaining_partition_device_resource;
    if (set_green_context) {
        unsigned int primary_partition_device_resource_num_groups = 1;
        // 0 : default behavior
        // CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING: more fine grain partition with cost of cluster
        // CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE: sm90+, create group that max cluster
        const int partition_flag = gtx_partition_flag; // CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
        const int primary_partition_device_resource_min_sm_count = intended_primary_partition_sm_count;
        CUDA_DRV(cuDevSmResourceSplitByCount(
            &primary_partition_device_resource, 
            &primary_partition_device_resource_num_groups, 
            &device_resource, 
            &remaining_partition_device_resource, 
            partition_flag, 
            primary_partition_device_resource_min_sm_count));
    }
    else {
        primary_partition_device_resource = device_resource;
        remaining_partition_device_resource = device_resource;
    }
    printf("[Device %d] [Primary Partition] (for GEMM) sm count: %d\n", device_id, primary_partition_device_resource.sm.smCount); fflush(stdout);
    printf("[Device %d] [Remaining Partition] (for COMM) sm count: %d\n", device_id, remaining_partition_device_resource.sm.smCount); fflush(stdout);
    primary_partition_sm_count = primary_partition_device_resource.sm.smCount;
    remaining_partition_sm_count = remaining_partition_device_resource.sm.smCount;

    // (3)
    CUdevResourceDesc primary_partition_resource_desc;
    CUdevResourceDesc remaining_partition_resource_desc;
    const unsigned int num_resource = 1;
    CUDA_DRV(cuDevResourceGenerateDesc(&primary_partition_resource_desc, &primary_partition_device_resource, num_resource));
    CUDA_DRV(cuDevResourceGenerateDesc(&remaining_partition_resource_desc, &remaining_partition_device_resource, num_resource));

    // (4)
    constexpr unsigned int green_context_flags = CU_GREEN_CTX_DEFAULT_STREAM; // must be this val
    CUDA_DRV(cuGreenCtxCreate(&primary_partition_green_context, primary_partition_resource_desc, current_device, green_context_flags));
    CUDA_DRV(cuGreenCtxCreate(&remaining_partition_green_context, remaining_partition_resource_desc, current_device, green_context_flags));

    // (5)
    constexpr unsigned int green_context_stream_flags = CU_STREAM_NON_BLOCKING;
    CUDA_DRV(cuGreenCtxStreamCreate(
        &primary_partition_gtx_stream, 
        primary_partition_green_context, 
        green_context_stream_flags, 
        primary_partition_gtx_stream_priority));
    CUDA_DRV(cuGreenCtxStreamCreate(
        &remaining_partition_gtx_stream, 
        remaining_partition_green_context, 
        green_context_stream_flags, 
        remaining_partition_gtx_stream_priority));

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_allsm_1t1g(
    const int device_id,
    const int gtx_stream_priority,
    const CUcontext & primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream,
    const int gtx_partition_flag = 0)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));
    
    // * Get Device
    CUdevice current_device;
    CUDA_DRV(cuCtxGetDevice(&current_device));

    // * Green Context Creation and Usage
    // 1. query set of SM resource (cuDeviceGetDevResource)
    // 2. partition resource (cuDevSmResourceSplitByCount)
    // 3. finalize resource by creating desc (cuDevResourceGenerateDesc)

    // (1)
    CUdevResource device_resource;
    CUDA_DRV(cuDeviceGetDevResource(current_device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("[Device %d] [Device Resource] sm count: %d\n", device_id, device_resource.sm.smCount); fflush(stdout);

    // (2)
    // (3)
    CUdevResourceDesc green_context_resource_desc;
    CUDA_DRV(cuDevResourceGenerateDesc(&green_context_resource_desc, &device_resource, 1));

    // (4)
    constexpr unsigned int green_context_flags = CU_GREEN_CTX_DEFAULT_STREAM; // must be this val
    CUDA_DRV(cuGreenCtxCreate(&green_context, green_context_resource_desc, current_device, green_context_flags));

    // (5)
    constexpr unsigned int green_context_stream_flags = CU_STREAM_NON_BLOCKING;
    CUDA_DRV(cuGreenCtxStreamCreate(
        &gtx_stream, 
        green_context,
        green_context_stream_flags,
        gtx_stream_priority));

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_allsm_1t1g(
    const int device_id,
    const int gtx_stream_priority,
    const CUcontext & primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream,
    ::std::vector<CUstream>& per_split_gtx_stream,
    const int gtx_partition_flag = 0)
{
    CUDA_DRV(create_green_context_and_stream_allsm_1t1g(
        device_id,
        gtx_stream_priority,
        primary_context,
        green_context,
        gtx_stream,
        gtx_partition_flag));
    
    const int NUM_SPLITS = per_split_gtx_stream.size();
    const int green_context_stream_flags = CU_STREAM_NON_BLOCKING;

    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuGreenCtxStreamCreate(
            &per_split_gtx_stream.at(split_id), 
            green_context, 
            green_context_stream_flags, 
            gtx_stream_priority));
    }

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_allsm_1t1g(
    const int device_id,
    const int gtx_stream_priority_high,
    const int gtx_stream_priority_low,
    const CUcontext & primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream_high,
    CUstream& gtx_stream_low,
    ::std::vector<CUstream>& per_split_gtx_stream_high,
    ::std::vector<CUstream>& per_split_gtx_stream_low,
    const int gtx_partition_flag = 0)
{
    CUDA_DRV(create_green_context_and_stream_allsm_1t1g(
        device_id,
        gtx_stream_priority_high,
        primary_context,
        green_context,
        gtx_stream_high,
        gtx_partition_flag));
    
    const int NUM_SPLITS = per_split_gtx_stream_high.size();
    const int green_context_stream_flags = CU_STREAM_NON_BLOCKING;

    CUDA_DRV(cuGreenCtxStreamCreate(
        &gtx_stream_low, 
        green_context, 
        green_context_stream_flags, 
        gtx_stream_priority_low));

    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuGreenCtxStreamCreate(
            &per_split_gtx_stream_high.at(split_id), 
            green_context, 
            green_context_stream_flags, 
            gtx_stream_priority_high));
        CUDA_DRV(cuGreenCtxStreamCreate(
            &per_split_gtx_stream_low.at(split_id), 
            green_context, 
            green_context_stream_flags, 
            gtx_stream_priority_low));
    }

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_allsm_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(gtx_stream));
    CUDA_DRV(cuGreenCtxDestroy(green_context));

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_allsm_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream,
    ::std::vector<CUstream>& per_split_gtx_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(gtx_stream));

    const int NUM_SPLITS = per_split_gtx_stream.size();
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamDestroy(per_split_gtx_stream.at(split_id)));
    }

    CUDA_DRV(cuGreenCtxDestroy(green_context));

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_allsm_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& green_context,
    CUstream& gtx_stream_high,
    CUstream& gtx_stream_low,
    ::std::vector<CUstream>& per_split_gtx_stream_high,
    ::std::vector<CUstream>& per_split_gtx_stream_low)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(gtx_stream_high));
    CUDA_DRV(cuStreamDestroy(gtx_stream_low));

    const int NUM_SPLITS = per_split_gtx_stream_high.size();
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamDestroy(per_split_gtx_stream_high.at(split_id)));
        CUDA_DRV(cuStreamDestroy(per_split_gtx_stream_low.at(split_id)));
    }

    CUDA_DRV(cuGreenCtxDestroy(green_context));

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_1t1g(
    const int device_id,
    const int intended_primary_partition_sm_count,
    const int primary_partition_gtx_stream_priority,
    const int remaining_partition_gtx_stream_priority,
    const CUcontext & primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    ::std::vector<CUstream>& primary_partition_per_split_gtx_stream,
    CUstream& remaining_partition_gtx_stream,
    int& primary_partition_sm_count,
    int& remaining_partition_sm_count,
    const int gtx_partition_flag = 0)
{
    CUDA_DRV(create_green_context_and_stream_1t1g(
        device_id,
        intended_primary_partition_sm_count,
        primary_partition_gtx_stream_priority,
        remaining_partition_gtx_stream_priority,
        primary_context,
        primary_partition_green_context,
        remaining_partition_green_context,
        primary_partition_gtx_stream,
        remaining_partition_gtx_stream,
        primary_partition_sm_count,
        remaining_partition_sm_count,
        gtx_partition_flag));

    const int NUM_SPLITS = primary_partition_per_split_gtx_stream.size();
    constexpr unsigned int green_context_stream_flags = CU_STREAM_NON_BLOCKING;

    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuGreenCtxStreamCreate(
            &primary_partition_per_split_gtx_stream.at(split_id), 
            primary_partition_green_context, 
            green_context_stream_flags, 
            primary_partition_gtx_stream_priority));
    }

    return CUDA_SUCCESS;
}

CUresult create_green_context_and_stream_1t1g(
    const int device_id,
    const int intended_primary_partition_sm_count,
    const int primary_partition_gtx_stream_priority,
    const int remaining_partition_gtx_stream_priority,
    const CUcontext & primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    CUstream& remaining_partition_gtx_stream,
    ::std::vector<CUstream>& primary_partition_per_split_gtx_stream,
    ::std::vector<CUstream>& remaining_partition_per_split_gtx_stream,
    int& primary_partition_sm_count,
    int& remaining_partition_sm_count,
    const int gtx_partition_flag = 0)
{
    CUDA_DRV(create_green_context_and_stream_1t1g(
        device_id,
        intended_primary_partition_sm_count,
        primary_partition_gtx_stream_priority,
        remaining_partition_gtx_stream_priority,
        primary_context,
        primary_partition_green_context,
        remaining_partition_green_context,
        primary_partition_gtx_stream,
        remaining_partition_gtx_stream,
        primary_partition_sm_count,
        remaining_partition_sm_count,
        gtx_partition_flag));

    const int NUM_SPLITS = primary_partition_per_split_gtx_stream.size();
    constexpr unsigned int green_context_stream_flags = CU_STREAM_NON_BLOCKING;

    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuGreenCtxStreamCreate(
            &primary_partition_per_split_gtx_stream.at(split_id), 
            primary_partition_green_context, 
            green_context_stream_flags, 
            primary_partition_gtx_stream_priority));
        CUDA_DRV(cuGreenCtxStreamCreate(
            &remaining_partition_per_split_gtx_stream.at(split_id), 
            remaining_partition_green_context, 
            green_context_stream_flags, 
            remaining_partition_gtx_stream_priority));
    }

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    CUstream& remaining_partition_gtx_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(primary_partition_gtx_stream));
    CUDA_DRV(cuStreamDestroy(remaining_partition_gtx_stream));
    
    CUDA_DRV(cuGreenCtxDestroy(primary_partition_green_context));
    CUDA_DRV(cuGreenCtxDestroy(remaining_partition_green_context));

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    ::std::vector<CUstream>& primary_partition_per_split_gtx_stream,
    CUstream& remaining_partition_gtx_stream)
{
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(primary_partition_gtx_stream));
    CUDA_DRV(cuStreamDestroy(remaining_partition_gtx_stream));

    const int NUM_SPLITS = primary_partition_per_split_gtx_stream.size();
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamDestroy(primary_partition_per_split_gtx_stream.at(split_id)));
    }

    CUDA_DRV(cuGreenCtxDestroy(primary_partition_green_context));
    CUDA_DRV(cuGreenCtxDestroy(remaining_partition_green_context));

    return CUDA_SUCCESS;
}

CUresult destroy_green_context_and_stream_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context,
    CUgreenCtx& primary_partition_green_context,
    CUgreenCtx& remaining_partition_green_context,
    CUstream& primary_partition_gtx_stream,
    CUstream& remaining_partition_gtx_stream,
    ::std::vector<CUstream>& per_split_primary_partition_gtx_stream,
    ::std::vector<CUstream>& per_split_remaining_partition_gtx_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(primary_partition_gtx_stream));
    CUDA_DRV(cuStreamDestroy(remaining_partition_gtx_stream));

    const int NUM_SPLITS = per_split_primary_partition_gtx_stream.size();
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamDestroy(per_split_primary_partition_gtx_stream.at(split_id)));
        CUDA_DRV(cuStreamDestroy(per_split_remaining_partition_gtx_stream.at(split_id)));
    }

    CUDA_DRV(cuGreenCtxDestroy(primary_partition_green_context));
    CUDA_DRV(cuGreenCtxDestroy(remaining_partition_green_context));

    return CUDA_SUCCESS;
}

CUresult release_primary_context_1t1g(
    const CUdevice& device,
    const CUcontext& primary_context)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuDevicePrimaryCtxRelease(device));
    return CUDA_SUCCESS;
}

CUresult create_event_1t1g(
    const CUcontext & primary_context,
    CUevent& start_event,
    CUevent& stop_event)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));
    CUDA_DRV(cuEventCreate(&start_event, CU_EVENT_DEFAULT));
    CUDA_DRV(cuEventCreate(&stop_event, CU_EVENT_DEFAULT));
    return CUDA_SUCCESS;
}

CUresult destroy_event_1t1g(
    const CUcontext & primary_context,
    CUevent& start_event,
    CUevent& stop_event)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuEventDestroy(start_event));
    CUDA_DRV(cuEventDestroy(stop_event));
    return CUDA_SUCCESS;
}

CUresult create_primary_context_stream_1t1g(
    const int primary_context_stream_priority,
    const CUcontext& primary_context,
    CUstream& primary_context_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    // * Create Stream
    CUDA_DRV(cuStreamCreateWithPriority(&primary_context_stream, CU_STREAM_NON_BLOCKING, primary_context_stream_priority));
    return CUDA_SUCCESS;
}

CUresult create_primary_context_stream_1t1g(
    const int primary_context_stream_priority,
    const CUcontext& primary_context,
    CUstream& primary_context_stream,
    ::std::vector<CUstream>& primary_context_per_split_stream)
{
    const int NUM_SPLITS = primary_context_per_split_stream.size();
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    // * Create Stream
    CUDA_DRV(cuStreamCreateWithPriority(&primary_context_stream, CU_STREAM_NON_BLOCKING, primary_context_stream_priority));
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamCreateWithPriority(&primary_context_per_split_stream.at(split_id), CU_STREAM_NON_BLOCKING, primary_context_stream_priority));
    }
    return CUDA_SUCCESS;
}

CUresult destroy_primary_context_stream_1t1g(
    const CUcontext& primary_context,
    CUstream& primary_context_stream)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(primary_context_stream));
    return CUDA_SUCCESS;
}

CUresult destroy_primary_context_stream_1t1g(
    const CUcontext& primary_context,
    CUstream& primary_context_stream,
    ::std::vector<CUstream>& primary_context_per_split_stream)
{
    const int NUM_SPLITS = primary_context_per_split_stream.size();
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    CUDA_DRV(cuStreamDestroy(primary_context_stream));
    for (int split_id = 0; split_id < NUM_SPLITS; split_id++) {
        CUDA_DRV(cuStreamDestroy(primary_context_per_split_stream.at(split_id)));
    }
    return CUDA_SUCCESS;
}

CUresult get_primary_context_sm_count_1t1g(
    const CUcontext& primary_context,
    int& primary_partition_sm_count)
{
    // * Set Current Context
    CUDA_DRV(cuCtxSetCurrent(primary_context));

    // * Get Device
    CUdevice curr_device;
    CUDA_DRV(cuCtxGetDevice(&curr_device));

    // * Get Device Resource
    CUdevResource device_resource;
    CUDA_DRV(cuDeviceGetDevResource(curr_device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));

    // * Set Primary Partition SM Count
    primary_partition_sm_count = device_resource.sm.smCount;

    return CUDA_SUCCESS;
}
