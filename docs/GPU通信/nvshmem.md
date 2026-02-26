
# nvshmem

## nvshmem

NVSHMEM 是基于 OpenSHMEM 的1.3并行编程接口规范并扩展完成的实现，为 NVIDIA GPU 集群提供了另外一种高效且可扩展的通信，属于NVIDIA magnum IO的重要一环，magnum IO包括：GDR、NCCL、NVSHMEM、UCX、GDS等IO套件。
NVSHMEM 为数据创建了一个全局的GPU地址空间（a.k.a.，PGAS 模型 Partitioned Global Address Space，分区全局地址空间），该空间跨越多个 GPU 内存。
使用NVSHMEM，可以编写包括通信和计算在内的长时间运行的内核，从而减少了与CPU同步的需要。由于GPU上的线程warp调度，这些联合计算和通信内核还允许计算与通信的细粒度重叠。NVSHMEM GPU启动的通信减少了内核启动、对CUDA API的调用和CPU-GPU同步带来的开销，减少这些开销可以显著提高应用程序工作负载的扩展性。
NVSHMEM在2.7.0版本上支持了IBGDA，支持**从GPU发起通信**，这在一定程度上支持了强扩展性(strong scaling)，

> 在GPU集群上运行的程序通常将计算阶段卸载到GPU上，并依靠CPU通过来管理集群节点之间的通信。由于重复内核启动、CPU-GPU同步、通信阶段GPU利用不足以及计算阶段网络利用不足的开销，依赖CPU进行通信限制了可扩展性。
其中一些问题可以通过重构应用程序代码来解决，以使用CUDA stream重叠独立的计算和通信阶段。这些优化可能会导致复杂的应用程序代码，并且随着每个GPU的问题大小变小，好处通常会减少。
> 而且根据Amdahl’s Law（阿姆达尔定律），就是程序中不可并行的部分会限制加速比。

主要的能力包括：
1. 将多个 GPU 的内存组合成一个分区的全局地址空间，可通过 NVSHMEM API 访问
2. 跨机通信单边操作/one-side，不需要对端参与，支持verbs API以及UCX
3. 包含一个低开销的内核通信 API，供 GPU 线程使用
4. 包括基于stream和 CPU 启动的通信 API
5. 支持 x86 和 Arm 处理器
6. 可与 MPI 和其他 OpenSHMEM 实现互操作


NVSHMEM 通过 PGAS 模型和 GPU 直接通信机制，为多 GPU 多节点应用提供了高效的编程接口，尤其适合需要频繁跨设备数据交换的高性能计算场景。尽管需开发者管理内存一致性，但其性能优势和编程便捷性使其成为替代传统 MPI 的重要选择。


NVSHMEM 3.0增加了对IBGDA中一种称为CPU-assisted(CPU辅助) IBGDA的新模式的支持(使用了GDRCopy)，该模式类似于Proxy代理网络和传统IBGDA之间的中间模式。它在GPU和CPU之间划分控制平面的职责。GPU生成工作请求（控制平面操作），CPU提交工作请求的NIC doorbell请求。它还允许在运行时动态选择是CPU还是GPU作为NIC的doorbell实施。
CPU-assisted（CPU辅助）的IBGDA放宽了IBGDA peer-mapping对现有驱动的影响（options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"），从而有助于提高IBGDA在非一致平台上的采用率，在这些平台上，管理级配置限制在大规模集群部署中更难实施。




http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.6.pdf
https://developer.nvidia.com/blog/tag/nvshmem/
https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/
https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/index.html
https://developer.nvidia.com/nvshmem-downloads
https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51705/


## GDRCopy
https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32039/ 一种CPU参与H2D和D2H的解决方案，比起cudaMemcpy来说，小数据会有极低的延迟，提供内存映射和拷贝的API。NVSHMEM和UCX中都用到了，其实NCCL里面也可以显式的编译使用，但是默认应该都没有使能。

## DeepEP

DeepEP内存编程模型：Partitioned Global Address Space（PGAS，分区全局地址空间）

用于并行计算和分布式内存系统的编程模型
逻辑上提供一个统一、连续的全局内存地址空间
物理上被分区（Partitioned）分布到不同的处理单元（Processing Elements，PEs）上
每个 PE 可以访问本地分区，也可以根据地址直接访问其他 PE 上的远程分区
单边通信：通信不需要对方感知和配合，put/get 非阻塞，需要显式同步原语


NVSHMEM操作GDR编程示例
```c++
nvshmem_put_nbi(remote_ptr, local_ptr, size, pe_id);
nvshmem_quiet(remote_pe_id);
nvshmem_amo_nonfetch_add(remote_ptr, value, pe_id);
nvshmem_sync_with_same_gpu_idx(rdma_team);
```
底层 通过 GPUDirect RDMA 把 GPU 上数据经过网络 NIC 直接发送到远端 GPU，全程无 CPU 干预
NVSHMEM → 调用通信库（如 UCX/NVLink/NCCL) → 触发 GDR → 网络接口 → 远端 GPU