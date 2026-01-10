# 解析 EPLB (Expert Parallel Load Balancer) 算法原理

本文基于[deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB/tree/main)的源码实现，剖析 EPLB 算法的工作原理。

## 什么是 EPLB？

EPLB (Expert Parallel Load Balancer) 是一种用于大规模混合专家模型 (MoE) 的负载均衡算法。它的核心目标是将“逻辑专家”(Logical Experts) 及其负载 (Weight) 映射到“物理副本”(Physical Replicas)，并最终分配到具体的 GPU 上，以实现计算负载的均衡，从而提高并行训练或推理的效率。

对于分层负载均衡 (`rebalance_experts_hierarchical`)，它采用了 **Pack (分组) -> Replicate (复制) -> Pack (再分组)** 的三步策略。

## 核心流程概述

EPLB 的分层策略主要包含以下三个步骤：

```mermaid
graph TD
    Input[输入: Logical Experts & Loads] --> Step1
    
    subgraph Step 1: Pack Groups to Nodes
        Step1[将 Expert Groups 打包分配到 Nodes]
        Note1[目标: 均衡各 Node 的总负载]
    end
    
    Step1 --> Step2
    
    subgraph Step 2: Replicate Experts within Nodes
        Step2[在每个 Node 内部复制热门 Expert]
        Note2[目标: 将高负载 Expert 分裂为多个副本<br>填满该 Node 的物理槽位]
    end
    
    Step2 --> Step3
    
    subgraph Step 3: Pack Physical Experts to GPUs
        Step3[将物理副本打包分配到 GPU]
        Note3[目标: 均衡各 GPU 的总负载]
    end
    
    Step3 --> Output[输出: Physical -> Logical 映射表]
```

## 案例详解

我们结合 `main.py` 中的具体数据来演示这一过程：

*   **Logical Experts**: 12 个 (索引 0-11)
*   **Weight (负载)**: `[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86]`
*   **Topology (拓扑)**:
    *   16 Replicas (总物理槽位)
    *   4 Groups (专家分组)
    *   2 Nodes (服务器节点)
    *   8 GPUs (总 GPU 数)

这意味着：
*   每个 Group 包含 `12 / 4 = 3` 个 Logical Experts。
*   每个 Node 拥有 `16 / 2 = 8` 个物理槽位。
*   每个 GPU 拥有 `16 / 8 = 2` 个物理槽位。

### Step 1: Pack Groups to Nodes (组级均衡)

首先，将 12 个专家按顺序分为 4 个 Group，计算每组的总负载，然后利用贪心算法（Balanced Packing）将这 4 个 Group 分配给 2 个 Node，使得两个 Node 的负载尽可能接近。

*   **Group 0 (E0-E2)**: Load = 90+132+40 = 262
*   **Group 1 (E3-E5)**: Load = 61+104+165 = 330
*   **Group 2 (E6-E8)**: Load = 39+4+73 = 116
*   **Group 3 (E9-E11)**: Load = 56+183+86 = 325

分配逻辑（贪心策略）：优先处理大负载 Group。
1.  Node 0: [Group 1 (330)] -> Total: 330
2.  Node 1: [Group 3 (325)] -> Total: 325
3.  Node 1: [Group 3 (325), Group 0 (262)] -> Total: 587
4.  Node 0: [Group 1 (330), Group 2 (116)] -> Total: 446

*(注：实际代码中的 packing 顺序可能略有差异，这里展示的是逻辑过程)*

```mermaid
graph TB
    subgraph Groups
        G0[Group 0<br>Load: 262]
        G1[Group 1<br>Load: 330]
        G2[Group 2<br>Load: 116]
        G3[Group 3<br>Load: 325]
    end

    G1 --> N0
    G2 --> N0
    G3 --> N1
    G0 --> N1

    subgraph Nodes
        N0[Node 0<br>Target Load: ~446]
        N1[Node 1<br>Target Load: ~587]
    end
```

### Step 2: Construct Redundant Experts (节点内复制)

在确定了每个 Node 负责的 Logical Experts 后，我们需要在 Node 内部进行“复制”操作，将高负载的专家分裂成多个副本，直到填满该 Node 的 8 个物理槽位。

以 **Node 1** 为例，它负责 **Group 0 (E0, E1, E2)** 和 **Group 3 (E9, E10, E11)**。
共 6 个逻辑专家，需要填充 8 个物理槽位，因此需要增加 2 个副本。

**复制过程：**
1.  初始状态：每个专家 1 个副本。
2.  找到当前平均负载最高的专家：**E10 (183)**。
3.  **Split E10**: E10 副本数变为 2，每个副本负载降为 `183 / 2 = 91.5`。
4.  再次寻找最高负载：**E1 (132)**。
5.  **Split E1**: E1 副本数变为 2，每个副本负载降为 `132 / 2 = 66`。
6.  副本总数达到 8，停止。

```mermaid
graph TD
    subgraph "Node 1 Logical Experts (6个)"
        L9[E9: 56]
        L10[E10: 183]
        L11[E11: 86]
        L0[E0: 90]
        L1[E1: 132]
        L2[E2: 40]
    end

    L10 --"High Load: Split"--> P10_1[P_E10.1]
    L10 --> P10_2[P_E10.2]
    L1 --"High Load: Split"--> P1_1[P_E1.1]
    L1 --> P1_2[P_E1.2]
    L9 --> P9[P_E9]
    L11 --> P11[P_E11]
    L0 --> P0[P_E0]
    L2 --> P2[P_E2]
    
    subgraph "Node 1 Physical Experts (8个槽位)"
        P9
        P10_1
        P10_2
        P11
        P0
        P1_1
        P1_2
        P2
    end
    
    style L10 fill:#f96,stroke:#333
    style L1 fill:#f96,stroke:#333
```

### Step 3: Pack Physical Experts to GPUs (GPU 级均衡)

最后一步是将 Node 1 内生成的 8 个物理专家（带有拆分后的负载）分配到该 Node 下的 4 个 GPU 上（每个 GPU 2 个槽位）。

这再次使用 `balanced_packing` 算法，确保每个 GPU 处理的计算量尽可能一致。

**Node 1 物理专家负载：**
*   E10.1 (91.5), E10.2 (91.5)
*   E0 (90), E11 (86)
*   E1.1 (66), E1.2 (66)
*   E9 (56), E2 (40)

通过贪心打包，系统会将这些任务分配给 GPU 4, 5, 6, 7。

```mermaid
graph TB
    subgraph "Node 1 Physical Experts"
        PE1["E10.1 (91.5)"]
        PE2["E10.2 (91.5)"]
        PE3["E0 (90)"]
        PE4["E11 (86)"]
        PE5["E1.1 (66)"]
        PE6["E1.2 (66)"]
        PE7["E9 (56)"]
        PE8["E2 (40)"]
    end

    PE1 -.-> GPU4
    PE8 -.-> GPU4
    
    PE2 -.-> GPU5
    PE7 -.-> GPU5
    
    PE3 -.-> GPU6
    PE6 -.-> GPU6
    
    PE4 -.-> GPU7
    PE5 -.-> GPU7

    subgraph "Node 1 GPUs"
        GPU4["GPU 4"]
        GPU5["GPU 5"]
        GPU6["GPU 6"]
        GPU7["GPU 7"]
    end
```

## 总结

EPLB 算法通过 **Hierarchical Balancing (分层均衡)** 有效解决了 MoE 模型的负载均衡问题：

1.  **Node 间均衡**: 粗粒度分配，适应节点间带宽限制。
2.  **Node 内复制**: 利用高速互联，通过复制热门专家消除热点。
3.  **GPU 间均衡**: 细粒度调度，确保硬件算力不闲置。

最终输出的 `phy2log` 映射表（如 `[5, 6, 5, 7, ...]`）即指导系统：第 0 号物理位置加载 5 号逻辑专家，第 1 号位置加载 6 号专家，以此类推。