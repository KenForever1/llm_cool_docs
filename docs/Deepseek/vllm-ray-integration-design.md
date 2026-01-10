# vLLM 与 Ray 的深度集成：图编译与混合通信架构

在分布式大模型推理中，vLLM 通过深度集成 Ray 框架，利用 **Ray Compiled Graph** 消除 Python 调度开销，并结合自定义的 **RayPPCommunicator** 复用底层高效通信栈。本文深入剖析其代码级的实现逻辑。

## 1. Ray Compiled Graph：静态图构建与 SPMD 执行

在 `vllm/v1/executor/ray_executor.py` 的 `_compiled_ray_dag` 方法中，vLLM 构建了一个静态的计算图。这种设计避免了每次推理时动态创建任务的开销。

### 1.1 DAG 构建流程

代码逻辑采用了 **SPMD (Single Program, Multiple Data)** 模式。
*   **输入共享**：第一组 TP (Tensor Parallel) Workers 共享同一个 `InputNode`。
*   **流水线传递**：每一级 PP (Pipeline Parallel) 的输出通过 `bind` 操作成为下一级的输入。
*   **传输优化**：通过 `with_tensor_transport` 显式指定通信通道（如 NCCL 或 SHM）。

```mermaid
graph TD
    subgraph "Ray Compiled DAG Construction"
        Input[InputNode]
        
        subgraph "PP Stage 0 (TP Group)"
            W0_0[Worker 0<br/>TP Rank 0]
            W0_1[Worker 1<br/>TP Rank 1]
        end
        
        subgraph "PP Stage 1 (TP Group)"
            W1_0[Worker 2<br/>TP Rank 0]
            W1_1[Worker 3<br/>TP Rank 1]
        end
        
        Output[MultiOutputNode]

        %% Data Flow
        Input -->|ExecuteModelRequest| W0_0
        Input -->|ExecuteModelRequest| W0_1
        
        %% Pipeline Connections with Transport
        W0_0 -->|"bind() + with_tensor_transport(NCCL/SHM)"| W1_0
        W0_1 -->|"bind() + with_tensor_transport(NCCL/SHM)"| W1_1
        
        W1_0 --> Output
        W1_1 --> Output
    end
    
    style Input fill:#f9f,stroke:#333
    style Output fill:#f9f,stroke:#333
```

**关键代码映射**：
*   `outputs = [input_data for _ in self.pp_tp_workers[0]]`: 对应图中 Input 到 PP Stage 0 的连接。
*   `worker.execute_model_ray.bind(outputs[i])`: 对应图中 Stage 间的连接。
*   `output.with_tensor_transport(transport=...)`: 对应图中连接线上的传输协议配置。

## 2. RayPPCommunicator：动态 Rank 映射机制

Ray 的调度基于 128 位的 Actor ID（十六进制字符串），而底层 NCCL 通信依赖整数 Rank。`vllm/distributed/device_communicators/ray_communicator.py` 中的 `RayPPCommunicator` 类解决了这一映射问题。

### 2.1 适配器架构与 Rank 发现

`RayPPCommunicator` 并不创建新的通信连接，而是作为 vLLM 内部 `_PP` 组（Pipeline Parallel Group）的包装器。它必须在初始化阶段建立 Actor ID 到 NCCL Rank 的映射。

```mermaid
sequenceDiagram
    participant RayWorker as Ray Actor (Worker)
    participant Comm as RayPPCommunicator
    participant NCCL as vLLM DeviceCommunicator (NCCL)

    Note over RayWorker, NCCL: 初始化阶段 (_build_actor_rank_mapping)
    
    RayWorker->>Comm: __init__()
    Comm->>RayWorker: 获取 current_actor.actor_id (Hex String)
    RayWorker-->>Comm: 返回 "0x123abc..."
    
    Comm->>Comm: 将 Hex String 转为 Byte Tensor
    
    Comm->>NCCL: all_gather(actor_id_tensor)
    Note right of NCCL: 集合通信：交换所有 Worker 的 ID
    NCCL-->>Comm: 返回所有 Ranks 的 ID List
    
    Comm->>Comm: 构建映射表 {ActorID_String : NCCL_Rank}
    
    Note over RayWorker, NCCL: 运行时 (send/recv)
    
    RayWorker->>Comm: send(tensor, peer_rank)
    Comm->>NCCL: send(tensor, peer_rank)
    Note right of NCCL: 直接复用底层 NCCL 通道
```

**实现细节**：
*   **Rank 来源**：`self._rank = self._comm.rank_in_group`，强制使用 vLLM 内部的 Rank，忽略 Ray 传入的 rank 参数。
*   **映射构建**：`_build_actor_rank_mapping` 函数将 32 字节的 Actor ID 转换成 `torch.uint8` 张量，通过 `_comm.all_gather` 在所有 Worker 间交换，从而让每个 Worker 知道其他 Actor ID 对应的 Rank 是多少。

## 3. GroupCoordinator：双通道通信基石

vLLM 的分布式状态核心位于 `vllm/distributed/parallel_state.py` 中的 `GroupCoordinator`。它采用双层通信架构，确保了数据传输的高带宽和控制流的低延迟。

### 3.1 双层通信架构图

```mermaid
classDiagram
    class GroupCoordinator {
        +unique_name: str
        +rank: int
        +world_size: int
        +device_group: ProcessGroup (NCCL)
        +cpu_group: ProcessGroup (Gloo)
        +device_communicator: DeviceCommunicator
        +mq_broadcaster: MessageQueue
        +all_reduce()
        +broadcast_object()
    }

    class DeviceGroup_NCCL {
        <<Backend: NCCL>>
        +Role: Tensor Transmission
        +Hardware: GPU (NVLink/PCIe)
        +Ops: all_reduce, all_gather
    }

    class CPUGroup_Gloo {
        <<Backend: Gloo>>
        +Role: Control Plane & Metadata
        +Hardware: CPU / TCP
        +Ops: barrier, broadcast_object
    }
    
    class MessageQueue_SHM {
        <<Backend: SharedMemory>>
        +Role: Low Latency Broadcast
        +Scope: Intra-node
    }

    GroupCoordinator *-- DeviceGroup_NCCL : manages
    GroupCoordinator *-- CPUGroup_Gloo : manages
    GroupCoordinator *-- MessageQueue_SHM : uses (optional)
```

**代码级设计解析**：

1.  **初始化 (`__init__`)**：
    *   `device_group = torch.distributed.new_group(..., backend=backend)`: 创建 NCCL 组，用于 `all_reduce` 等重型 Tensor 操作。
    *   `cpu_group = torch.distributed.new_group(..., backend="gloo")`: 创建 Gloo 组，用于 `barrier`、`broadcast_object` 等轻量级控制操作。这一步使用了 `suppress_stdout` 来避免冗余日志。

2.  **分层调用**：
    *   调用 `all_reduce(tensor)` 时，请求被转发给 `device_communicator`，最终走 NCCL 通道。
    *   调用 `broadcast_object(obj)` 时，优先尝试 `mq_broadcaster` (共享内存)，如果不可用则回退到 `cpu_group` (Gloo)。

3.  **全局状态管理**：
    *   `_TP`, `_PP`, `_DP` 等全局变量本质上都是 `GroupCoordinator` 的实例。
    *   `init_model_parallel_group` 工厂函数负责根据 Rank 列表实例化这些协调器。

## 4. 总结

vLLM 的分布式架构并非简单的堆砌，而是针对不同通信需求做了精细的分层：
*   **计算层**：利用 Ray Compiled Graph 实现静态图优化和 SPMD 执行。
*   **调度适配层**：通过 `RayPPCommunicator` 的动态 Rank 映射，解决 Ray Actor 与 NCCL Rank 的寻址差异。
*   **通信基底层**：`GroupCoordinator` 提供的双通道（NCCL + Gloo/SHM）机制，保证了数据面和控制面的隔离与高效。