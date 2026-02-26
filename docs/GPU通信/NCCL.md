# NCCL

### 基础概念

| 名词       | 概念                                                                                                                                                                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|通信组|通信组内的多个 GPU 相互通信，执行集合通信操作|
|channel|多个channel之间可以并行chunk数据传输。每个通信组会创建多个channel。每个channel负责一部分数据传输，不同的channel可以完全并行传输，更大化的利用硬件带宽，但是也会消耗更多的GPU计算和显存资源。|
| connection | GPU之间的通信连接抽象。类型可以是NVLINK，PCIE、RDMA、TCP等，这些可以理解为传输层协议                                                                                                                                                          |
| proto      | Proto是这里指的是应用层协议（NCCL），不是传输层协议（和rdma、socket、nvlink、pcie协议没有关系）。所谓上层协议，核心解决的问题是，数据发送应该切分成多大的chunk/slice进行发送，并且发送端如何通知接收端数据传输完毕了等等。LL、LL128、Simple。 一个集合通信操作，会选择何种协议，与通信msg的大小直接相关，是在任务enqueue的时候根据静态的规则匹配出来，Simple适合较大的msg，LL/LL128适合小msg。nccl 协议：https://zhuanlan.zhihu.com/p/699178659|


### 通信算法

通信算法。常见的包括ring、tree、NVLS、COLLNET
pattern 1、2、3代表tree，不同的树有什么区别？
pattern  4代表ring, 
pattern 5代表NVLS，对应HopperNVSwitchSHARP，NCCL2.17(2023.03)加入
pattern 6代表COLLNET，对应IBSHARP(交换机上直接reduce)，NCCL2.7(2020.06)加入

```bash
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
```

* Ring算法：GPU只和自己前后相邻的GPU建立连接
* Tree算法：根据实际的tree的连接关系进行建联

Ring-ALLReduce则分两大步骤实现：Scatter-Reduce和All-Gather。在这两个阶段的通信过程中，GPU都被安排在一个逻辑环中。每块GPU只从左邻GPU接受数据，并发送给右邻GPU。通过多轮迭代的Add就实现了Scatter-Reduce，再多轮迭代的直接替换就实现了All-Gather。

### 例子

假设如下12台单机8卡共96 GPU按照DP=3，PP=4，TP=8来组织训练：

|通信组类型|通信组个数|每个 comm 的 nRanks|详述|
|---|----|---|---|
|TP comm|12|8|张量并行：单机八卡为一个 MP comm|
|PP comm|24|4|流水线并行：DP 组内同号卡为一个 PP comm|
|DP comm|32|3|数据并行：DP 组间同位置卡为一个 DP comm|

由于训练单机/单卡放不下数据，采用跨机跨卡，计算-通信-计算，卡之间需要同步梯度等信息。
NCCL用统一的API，屏蔽了底层传输协议的差异（NVLINK、TCP、RDMA等），同机卡间采用NVLINK通信，跨卡采用RDMA或TCP通信，通过集合通信（Collective Communication）操作（Broadcast、Scatter、Reduce、AllReduce、AlltoAll等），实现并行+分布式+硬件，在GPU场景下，通信库就是NCCL。

