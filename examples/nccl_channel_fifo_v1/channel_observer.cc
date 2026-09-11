// Record NCCL's published collective descriptors in diagnostic runs only.
// Timing is deliberately absent: this plugin identifies choices, not durations.
#include "nccl_profiler.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {
struct Context { int width; int rank; };
std::mutex output_mutex;
FILE* output = nullptr;

std::string quoted(const char* value) {
  std::string result = "\"";
  if (value != nullptr) {
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
      if (*p == '"' || *p == '\\') result += '\\';
      if (*p < 32) {
        char escape[7];
        std::snprintf(escape, sizeof(escape), "\\u%04x", *p);
        result += escape;
      } else result += static_cast<char>(*p);
    }
  }
  return result + "\"";
}

ncclResult_t initialize(void** context, uint64_t, int* mask, const char*,
                        int, int width, int rank, ncclDebugLogger_t) {
  std::lock_guard<std::mutex> lock(output_mutex);
  const char* path = std::getenv("NCCL_TRANSITION_SELECTION_LOG");
  if (path == nullptr) return ncclInvalidArgument;
  if (output == nullptr) output = std::fopen(path, "a");
  if (output == nullptr) return ncclSystemError;
  *context = new Context{width, rank};
  *mask = ncclProfileGroup | ncclProfileColl | ncclProfileCollApi | ncclProfileKernelLaunch;
  std::fprintf(output, "{\"kind\":\"init\",\"width\":%d,\"rank\":%d}\n", width, rank);
  std::fflush(output);
  return ncclSuccess;
}

ncclResult_t start(void* context, void** handle, ncclProfilerEventDescr_v7_t* event) {
  *handle = context;
  if(event->type == ncclProfileKernelLaunch) {
    const Context* ctx=static_cast<const Context*>(context);
    std::lock_guard<std::mutex> lock(output_mutex);
    std::fprintf(output,"{\"kind\":\"kernel_launch\",\"rank\":%d,\"stream\":\"%p\"}\n",ctx->rank,event->kernelLaunch.stream);
    std::fflush(output);return ncclSuccess;
  }
  if (event->type != ncclProfileColl) return ncclSuccess;
  const Context* ctx = static_cast<const Context*>(context);
  const auto& coll = event->coll;
  std::lock_guard<std::mutex> lock(output_mutex);
  const int written = std::fprintf(output,
      "{\"kind\":\"collective\",\"width\":%d,\"rank\":%d,\"sequence\":%llu,"
      "\"function\":%s,\"count\":%zu,\"datatype\":%s,\"in_place\":%s,"
      "\"algo\":%s,\"proto\":%s,\"#channels\":%u,\"#warps\":%u,"
      "\"kernelVariant\":%s,\"symmetric\":%s}\n",
      ctx->width, event->rank, static_cast<unsigned long long>(coll.seqNumber),
      quoted(coll.func).c_str(), coll.count, quoted(coll.datatype).c_str(),
      coll.sendBuff == coll.recvBuff ? "true" : "false", quoted(coll.algo).c_str(),
      quoted(coll.proto).c_str(), coll.nChannels, coll.nWarps,
      quoted(coll.kernelVariant).c_str(), coll.isSymColl ? "true" : "false");
  if (written < 0 || std::fflush(output) != 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t stop(void*) { return ncclSuccess; }
ncclResult_t state(void*, ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t*) {
  return ncclSuccess;
}
ncclResult_t finalize(void* context) {
  delete static_cast<Context*>(context);
  return ncclSuccess;
}
}  // namespace

extern "C" {
ncclProfiler_v7_t ncclProfiler_v7 = {
    "collective_selection_observer", initialize, start, stop, state, finalize};
}
