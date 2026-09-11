// Illustrative acknowledged GPU exchange, not an internal NCCL timing trace.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstdlib>
#define CUDA(x) do { auto cuda_status_=(x); if(cuda_status_!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(cuda_status_)); std::exit(2); } } while(0)
using Word=unsigned long long;
using Atomic=cuda::atomic_ref<Word,cuda::thread_scope_system>;
struct alignas(128) Inbox { Word ready; Word data; };
__device__ Word timer() { Word ns; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(ns)); return ns; }
__device__ void publish(Inbox* remote,Word sequence,bool separate) {
  if(separate) { remote->data=sequence; __threadfence_system(); Atomic(remote->ready).store(sequence,cuda::memory_order_release); }
  else Atomic(remote->ready).store((sequence<<32)|sequence,cuda::memory_order_release);
}
__device__ bool receive(Inbox* local,Word sequence,bool separate) {
  Word deadline=timer()+2000000000ULL,word;
  do { word=Atomic(local->ready).load(cuda::memory_order_acquire); if(timer()>deadline)return false; } while((separate ? word : word>>32)!=sequence);
  return (separate ? local->data : word & 0xffffffffULL)==sequence;
}
__global__ void exchange(Inbox* local,Inbox* remote,int rank,bool separate,Word* elapsed,int* failure) {
  constexpr int warmup=256,iterations=4096; Word start=0;
  for(Word i=1;i<=warmup+iterations;++i) {
    if(rank==0) {
      if(i==warmup+1) start=timer();
      publish(remote,i,separate);
      if(!receive(local,i,separate)) { *failure=1; return; }
    } else {
      if(!receive(local,i,separate)) { *failure=1; return; }
      publish(remote,i,separate);
    }
  }
  if(rank==0) *elapsed=timer()-start;
}
int main() {
  Inbox* boxes[2]; Word* elapsed[2]; int* failure[2]; cudaStream_t streams[2];
  for(int r=0;r<2;++r) { CUDA(cudaSetDevice(r)); int access; CUDA(cudaDeviceCanAccessPeer(&access,r,1-r));if(!access)return 3;
    CUDA(cudaDeviceEnablePeerAccess(1-r,0)); CUDA(cudaMalloc(&boxes[r],sizeof(Inbox)));CUDA(cudaMalloc(&elapsed[r],sizeof(Word)));CUDA(cudaMalloc(&failure[r],sizeof(int)));CUDA(cudaStreamCreate(&streams[r])); }
  std::puts("variant,exchanges,total_ns,rtt_ns,mismatches");
  for(int separate=0;separate<2;++separate) {
    for(int r=0;r<2;++r) { CUDA(cudaSetDevice(r));CUDA(cudaMemset(boxes[r],0,sizeof(Inbox)));CUDA(cudaMemset(failure[r],0,sizeof(int))); }
    for(int r=0;r<2;++r) { CUDA(cudaSetDevice(r));exchange<<<1,1,0,streams[r]>>>(boxes[r],boxes[1-r],r,separate,elapsed[r],failure[r]);CUDA(cudaGetLastError()); }
    int errors=0;
    for(int r=0;r<2;++r) { CUDA(cudaSetDevice(r));CUDA(cudaStreamSynchronize(streams[r]));int e;CUDA(cudaMemcpy(&e,failure[r],sizeof(int),cudaMemcpyDeviceToHost));errors+=e; }
    CUDA(cudaSetDevice(0));Word ns;CUDA(cudaMemcpy(&ns,elapsed[0],sizeof(Word),cudaMemcpyDeviceToHost));
    std::printf("%s,4096,%llu,%.9f,%d\n",separate ? "separate_fenced" : "embedded_ready",ns,ns/4096.0,errors);
    if(errors)return 4;
  }
  for(int r=0;r<2;++r) { CUDA(cudaSetDevice(r));CUDA(cudaFree(boxes[r]));CUDA(cudaFree(elapsed[r]));CUDA(cudaFree(failure[r]));CUDA(cudaStreamDestroy(streams[r])); }
}
