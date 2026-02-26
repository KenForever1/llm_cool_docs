

# GPU 拓扑结构解析

```bash
nvidia-smi  topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PIX     PIX     PIX     SYS     SYS     SYS     SYS     PIX     PIX     0-23,48-71      0               N/A
GPU1    PIX      X      PIX     PIX     SYS     SYS     SYS     SYS     PIX     PIX     0-23,48-71      0               N/A
GPU2    PIX     PIX      X      PIX     SYS     SYS     SYS     SYS     PIX     PIX     0-23,48-71      0               N/A
GPU3    PIX     PIX     PIX      X      SYS     SYS     SYS     SYS     PIX     PIX     0-23,48-71      0               N/A
GPU4    SYS     SYS     SYS     SYS      X      PIX     PIX     PIX     SYS     SYS     24-47,72-95     1               N/A
GPU5    SYS     SYS     SYS     SYS     PIX      X      PIX     PIX     SYS     SYS     24-47,72-95     1               N/A
GPU6    SYS     SYS     SYS     SYS     PIX     PIX      X      PIX     SYS     SYS     24-47,72-95     1               N/A
GPU7    SYS     SYS     SYS     SYS     PIX     PIX     PIX      X      SYS     SYS     24-47,72-95     1               N/A
NIC0    PIX     PIX     PIX     PIX     SYS     SYS     SYS     SYS      X      PIX
NIC1    PIX     PIX     PIX     PIX     SYS     SYS     SYS     SYS     PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```

## 一、系统整体架构

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              双路服务器 (2 NUMA Nodes)                          │
├────────────────────────────────────┬───────────────────────────────────────────┤
│            NUMA 0                  │              NUMA 1                        │
│         CPU 0-23, 48-71            │          CPU 24-47, 72-95                  │
│                                    │                                            │
│  ┌──────────────────────────────┐  │  ┌──────────────────────────────┐         │
│  │     PCIe Switch / Bridge     │  │  │     PCIe Switch / Bridge     │         │
│  ├──────┬──────┬──────┬────────┤  │  ├──────┬──────┬──────┬────────┤         │
│  │ GPU0 │ GPU1 │ GPU2 │ GPU3   │  │  │ GPU4 │ GPU5 │ GPU6 │ GPU7   │         │
│  └──────┴──────┴──────┴────────┘  │  └──────┴──────┴──────┴────────┘         │
│                                    │                                            │
│  ┌──────────────────────────────┐  │                                            │
│  │  NIC0 (mlx5_0) │ NIC1 (mlx5_1)│  │            (无本地网卡)                    │
│  └──────────────────────────────┘  │                                            │
├────────────────────────────────────┴───────────────────────────────────────────┤
│                          QPI / UPI 互联 (跨 NUMA)                               │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 二、连接类型图例

| 标记 | 含义 | 延迟/带宽 |
|-----|------|----------|
| **X** | 自身 | - |
| **PIX** | 仅穿过单个 PCIe 桥（同一 PCIe Switch 下） | **最优** |
| **PXB** | 穿过多个 PCIe 桥（不经过 CPU） | 次优 |
| **PHB** | 经过 PCIe Host Bridge（经过 CPU） | 中等 |
| **NODE** | 同 NUMA 内，跨 PCIe Host Bridge | 较慢 |
| **SYS** | 跨 NUMA 节点（经过 QPI/UPI） | **最慢** |

## 三、关键发现

### 1. GPU 分组

| 组 | GPU | NUMA | CPU 亲和性 | 网卡亲和性 |
|----|-----|------|-----------|-----------|
| **组 A** | GPU0, GPU1, GPU2, GPU3 | NUMA 0 | CPU 0-23, 48-71 | NIC0, NIC1 (PIX) |
| **组 B** | GPU4, GPU5, GPU6, GPU7 | NUMA 1 | CPU 24-47, 72-95 | 无本地网卡 (SYS) |

### 2. 通信效率矩阵

```
           GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0        -    快    快    快    慢    慢    慢    慢
GPU1       快     -    快    快    慢    慢    慢    慢
GPU2       快    快     -    快    慢    慢    慢    慢
GPU3       快    快    快     -    慢    慢    慢    慢
GPU4       慢    慢    慢    慢     -    快    快    快
GPU5       慢    慢    慢    慢    快     -    快    快
GPU6       慢    慢    慢    慢    快    快     -    快
GPU7       慢    慢    慢    慢    快    快    快     -

快 = PIX (同 NUMA，同 PCIe Switch)
慢 = SYS (跨 NUMA，经过 QPI/UPI)
```

### 3. 网卡位置问题

**⚠️ 潜在性能瓶颈**：两个网卡（mlx5_0, mlx5_1）都在 **NUMA 0**

| 场景 | GPU → 网卡 | 路径类型 | 影响 |
|-----|-----------|---------|------|
| GPU0-3 → NIC | PIX | ✅ 最优路径 |
| GPU4-7 → NIC | SYS | ❌ 跨 NUMA，带宽受限，延迟增加 |

## 四、性能优化建议

### 1. NCCL 通信优化

```bash
# 让 NCCL 感知拓扑，自动选择最优路径
export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml  # 可选：导出拓扑调试

# 如果跨 NUMA 通信多，考虑启用 NVLink（如果有）或调整并行策略
```

### 2. 进程绑定建议

```bash
# GPU0-3 的进程绑定到 NUMA 0
numactl --cpunodebind=0 --membind=0 python train.py  # for GPU 0-3

# GPU4-7 的进程绑定到 NUMA 1
numactl --cpunodebind=1 --membind=1 python train.py  # for GPU 4-7
```

### 3. 分布式训练策略

| 并行方式 | 建议 |
|---------|------|
| **数据并行 (DP)** | 优先在同组 GPU 内进行 AllReduce |
| **张量并行 (TP)** | TP=4 在同一 NUMA 内（GPU0-3 或 GPU4-7） |
| **流水线并行 (PP)** | PP 跨 NUMA 可接受（通信量相对小） |
| **专家并行 (EP)** | 如果配置 EP=8，会跨 NUMA，主要瓶颈点 |


p2p建链
|1|2|3|
|---------|------|--|
|p2pTransport|有NVLink的同机GPU之间的通信|有NVLink时会走这个建立连接通信|
|shmTransport|没有NVLINK的GPU之间的通信|当不存在NVSwitch时候，通过CPU内存为中介，走PCIE通过CUDA IPC和共享内存进行数据交换|
|netTransport|跨机通信RDMA或者TCP通信|RoCE或者TCP的transport类型|
|collNetTransport|通过网络的IB sharp switch|需要NCCL_COLLNET_ENABLE=1|


[GPU分布式训练：NCCL性能解析（一）节点内性能分析](https://zhuanlan.zhihu.com/p/584500146)

https://developer.nvidia.com/zh-cn/blog/nvidia-hopper-architecture-in-depth/

https://www.nvidia.com/en-us/data-center/magnum-io/

https://developer.nvidia.cn/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/

# 并行
[深度学习的分布式训练与集合通信（1）](https://www.hiascend.com/zh/developer/techArticles/20241111-1)

[深度学习的分布式训练与集合通信（2）](https://www.hiascend.com/zh/developer/techArticles/20241122-1): DP，PP，TP，EP等多种并行策略及其通信模式

[深度学习的分布式训练与集合通信（3)](https://www.hiascend.com/developer/techArticles/20250207-1): 序列并行（SP），上下文并行（CP），混合序列并行Ulysess，ZeRO系列并行优化策略，完全分片数据并行（FSDP）

[LLM(31)：序列并行的典型方案与实现细节](https://zhuanlan.zhihu.com/p/14665512019)