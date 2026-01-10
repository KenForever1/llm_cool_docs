
## SM

硬件实体：SM 是 GPU 的核心计算单元，负责执行线程块的指令。
每个 SM 一般包含：
CUDA Cores（如 A100 SM 有 64 FP32 cores）
Warp Schedulers（管理线程束调度）
Shared Memory/L1 Cache（片上存储）
Register File（线程私有寄存器）
示例：NVIDIA A100 GPU 有 108 个 SM。H100 GPU有132个SM。


## 逻辑层级

Grid -〉 Block -> Warp -> thread

### Block
同一个block中的thread可以同步，也可以通过shared memory进行通信。
一个block可以进行同步和共享内存，因此一个block中的thread只能在一个SM上调度。
一般一个block最大支持1024个threads。
但是SM一般可以调度多个线程块（如NVIDIA A100的每个SM最多支持32个Block，比如：若一个Grid包含100个Block，而GPU有10个SM，每个SM可能分配到10个Block（实际分配取决于资源限制））。
每个 SM 可以同时处理多个线程块。当一个线程块被分配到某个 SM 时，SM 会将其中的线程组织成更小的执行单元，称为线程束（warp），每个线程束通常包含 32 个线程。 这些线程束在 SM 内由硬件调度器管理，以实现高效的并行计算。 

### Warp

warp(线程束)是最基本的执行单元，一个warp包含32个并行thread，这些thread以不同数据资源执行相同的指令。
一个SM同时并发的warp是有限的，因为资源限制，SM要为每个线程块分配共享内存，而也要为每个线程束中的线程分配独立的寄存器，所以SM的配置会影响其所支持的线程块和warp并发数量。

查看gpu info：
```cpp
/* nvcc gpu_info.cpp -o gpu_info */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Number of CUDA devices detected: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);
        printf("  CUDA Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Thread Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], 
               deviceProp.maxThreadsDim[1], 
               deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], 
               deviceProp.maxGridSize[1], 
               deviceProp.maxGridSize[2]);
        printf("  Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total Constant Memory: %zu bytes\n", deviceProp.totalConstMem);
        printf("  Texture Alignment: %zu bytes\n", deviceProp.textureAlignment);
        printf("  Clock Rate: %.2f GHz\n", deviceProp.clockRate * 1e-6);
        printf("  Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6);
        printf("  Memory Bus Width: %d bit\n", deviceProp.memoryBusWidth);
        printf("  L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
        printf("  Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Registers per Multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
        printf("  Max Blocks per Multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("  Number of Async Engines: %d\n", deviceProp.asyncEngineCount);
        printf("  Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Max 1D Linear Texture Size: %d\n", deviceProp.maxTexture1DLinear);
        printf("  Max 2D Linear Texture Size: (%d, %d)\n", deviceProp.maxTexture2DLinear[0], deviceProp.maxTexture2DLinear[1]);
        printf("  Max 1D Texture Size: %d\n", deviceProp.maxTexture1D);
        printf("  Max 2D Texture Size: (%d, %d)\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
        printf("  Max 3D Texture Size: (%d, %d, %d)\n", 
               deviceProp.maxTexture3D[0], 
               deviceProp.maxTexture3D[1], 
               deviceProp.maxTexture3D[2]);
        printf("  Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
        printf("  TCC Driver: %s\n", deviceProp.tccDriver ? "Yes" : "No");
        printf("  Managed Memory: %s\n", deviceProp.managedMemory ? "Yes" : "No");
        printf("  Multi-GPU Board: %s\n", deviceProp.isMultiGpuBoard ? "Yes" : "No");
        if (deviceProp.isMultiGpuBoard) {
            printf("  Multi-GPU Board Group ID: %d\n", deviceProp.multiGpuBoardGroupID);
        }
        printf("  Stream Priorities Supported: %s\n", deviceProp.streamPrioritiesSupported ? "Yes" : "No");
        printf("  Global L1 Cache Supported: %s\n", deviceProp.globalL1CacheSupported ? "Yes" : "No");
        printf("  Local L1 Cache Supported: %s\n", deviceProp.localL1CacheSupported ? "Yes" : "No");
        printf("  Compute Preemption Supported: %s\n", deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Host Native Atomic Supported: %s\n", deviceProp.hostNativeAtomicSupported ? "Yes" : "No");
        printf("  Pageable Memory Access: %s\n", deviceProp.pageableMemoryAccess ? "Yes" : "No");
        printf("  Concurrent Managed Access: %s\n", deviceProp.concurrentManagedAccess ? "Yes" : "No");
        printf("  Compute Mode: %d\n", deviceProp.computeMode);
        printf("  PCI Bus ID: %d\n", deviceProp.pciBusID);
        printf("  PCI Device ID: %d\n", deviceProp.pciDeviceID);
        printf("  PCI Domain ID: %d\n", deviceProp.pciDomainID);
        printf("  Shared Memory per Block Optin: %s\n", deviceProp.sharedMemPerBlockOptin ? "Yes" : "No");
    }

    return 0;
}
```

https://harmanani.github.io/classes/csc447/Notes/Lecture15.pdf

