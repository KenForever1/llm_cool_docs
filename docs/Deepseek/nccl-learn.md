# 从 vLLM 源码看 NCCL 分布式通信

在深度学习大模型（LLM）的分布式推理与训练中，NCCL (NVIDIA Collective Communications Library) 是毫无疑问的通信基石。虽然 PyTorch 提供了 torch.distributed 这一高层封装，但在追求极致性能的推理框架 vLLM 中，开发者们选择了一条更底层的路——自己封装 NCCL。

## 为什么要重新封装 NCCL？
在 vLLM 的源码中，你不会看到它直接依赖 torch.distributed.all_reduce 来进行模型并行通信。相反，它实现了一套纯 Python 的封装库 pynccl。为什么？

答案藏在 pynccl_wrapper.py 的注释中：

CUDA Graph 兼容性：PyTorch 的原生 API 中包含许多辅助的 CUDA API 调用（如内存检查、CPU 同步等），这些操作往往无法被 CUDA Graph 捕获（Capture）。为了实现极致的推理速度，vLLM 需要一个“纯净”的 NCCL 调用接口。

版本解耦：通过 ctypes 动态加载 .so 库，vLLM 可以灵活地切换 NCCL 版本，甚至通过环境变量 VLLM_NCCL_SO_PATH 指定特定库，而无需重新编译 PyTorch。

```python
# 代码片段示意：加载动态库
self.lib = ctypes.CDLL(so_file)

# 映射 C 函数：ncclAllReduce
# 定义参数类型：发送缓冲、接收缓冲、数量、数据类型、操作类型、通信器、流
self._funcs["ncclAllReduce"].argtypes = [
    buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
    ncclRedOp_t, ncclComm_t, cudaStream_t
]
```

## PyNcclCommunicator

PyNcclCommunicator 类负责管理通信的生命周期。最精彩的部分在于初始化流程，这是所有 NCCL 应用的起点：

生成身份证 (Unique ID)：Rank 0 调用 ncclGetUniqueId 生成一个全局唯一的 ID。
广播身份证：通过 PyTorch 的 dist.broadcast (通常走 Gloo 后端) 将这个 ID 发送给所有其他 Rank。
建立连接：所有 Rank 拿到同一个 ID 后，各自调用 ncclCommInitRank，从而在底层建立起通信环路。


## 为什么需要 StatelessProcessGroup?

Unique ID 广播中有这样一个判断逻辑：
```
if not isinstance(group, StatelessProcessGroup):
    # 分支 A: 使用 PyTorch 标准的集合通信 API (dist.broadcast)
    # 这通常依赖于 Gloo 或 NCCL 后端
    tensor = torch.ByteTensor(list(self.unique_id.internal))
    dist.broadcast(tensor, src=ranks[0], group=group)
    # ...
else:
    # 分支 B: 使用 StatelessProcessGroup 自定义的 broadcast_obj 方法
    # 它是基于 TCPStore 实现的 pickle 对象传输
    self.unique_id = group.broadcast_obj(self.unique_id, src=0)
```

StatelessProcessGroup 是 vLLM 为了解决 PyTorch 原生 torch.distributed.init_process_group 产生的全局状态污染问题而设计的一个自定义类。

该判断 if not isinstance(group, StatelessProcessGroup): 的作用是区分“标准 PyTorch 分布式组”和“vLLM 自定义的无状态组”，从而决定如何获取分布式环境的基本信息（Rank, World Size 等）以及如何同步 NCCL Unique ID。

在 PyTorch 中，init_process_group 是一个全局性的操作，一旦调用，进程就被绑定到特定的 world_size 和 rank，很难再创建完全独立的、包含部分进程重叠的新组（尤其是在不知道全局 rank 的情况下）。

vLLM 的场景（例如 Ray 调度）非常灵活，可能需要动态地将某些进程组合在一起进行通信，而不希望受到全局 PyTorch 分布式状态的限制。StatelessProcessGroup 通过手动管理 TCPStore 和 socket，实现了一个不依赖全局状态的轻量级进程组，专门用于交换元数据（metadata）。

## NCCL 通信 Demo

```python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ==========================================
# 环境配置：将 vLLM 源码路径添加到 sys.path
# ==========================================
# 假设当前脚本位于 workspace/nccl_comm_demo/ 目录下
# 我们需要向上两级找到 vllm 包
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir)) # workspace/
# vllm_path = os.path.join(project_root, "vllm")

# if vllm_path not in sys.path:
#     sys.path.insert(0, vllm_path)

try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
except ImportError:
    print(f"错误: 无法导入 vLLM 模块。尝试的路径是: {vllm_path}")
    print("请确保您在正确的目录下运行脚本，或者 vllm 已经安装。")
    sys.exit(1)

def run_worker(rank, world_size):
    print(f"[Rank {rank}] 进程启动 (PID: {os.getpid()})")
    
    # 1. 初始化基础 PyTorch 分布式环境
    # PyNcclCommunicator 虽然使用独立的 NCCL 连接，但初始化时依赖
    # torch.distributed 来同步 NCCL Unique ID。
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # 使用 gloo 作为控制平面的后端，因为它对环境要求较低
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # 2. 设置当前进程使用的 CUDA 设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 3. 初始化 vLLM 的 PyNcclCommunicator
    # 这是一个对 NCCL C API 的封装，支持在 CUDA Graph 中使用
    print(f"[Rank {rank}] 正在初始化 PyNcclCommunicator...")
    try:
        # 传入 dist.group.WORLD 以获取全局 ranks 信息
        comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
        if comm.disabled:
             print(f"[Rank {rank}] PyNcclCommunicator 不可用 (可能是缺少 NCCL 库或不支持)")
             return
    except Exception as e:
        print(f"[Rank {rank}] 初始化 PyNccl 失败: {e}")
        return

    # ==========================================
    # Demo 1: All-Reduce (归约求和)
    # ==========================================
    # 场景：每个 rank 有一个数值，计算所有 rank 数值的总和，并同步给所有人。
    # Rank 0: [1.0], Rank 1: [2.0], ...
    tensor = torch.ones(1, device=device) * (rank + 1)
    print(f"[Rank {rank}] All-Reduce 前: {tensor.item()}")
    
    # 执行 inplace all_reduce
    tensor = comm.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # vllm 0.8.3 不支持设置out_tensor参数
    # comm.all_reduce(tensor, out_tensor=tensor, op=torch.distributed.ReduceOp.SUM)
    
    # 验证: 1 + 2 + ... + N = N*(N+1)/2
    expected = world_size * (world_size + 1) / 2
    print(f"[Rank {rank}] All-Reduce 后: {tensor.item()} (预期: {expected})")
    
    # ==========================================
    # Demo 2: All-Gather (全收集)
    # ==========================================
    # 场景：每个 rank 发送一个数据，最终所有 rank 都获得完整的数据列表。
    send_tensor = torch.tensor([float(rank)], device=device)
    
    # 接收 buffer 的大小必须是 world_size * send_tensor.size
    recv_tensor = torch.empty(world_size, device=device)
    
    comm.all_gather(recv_tensor, send_tensor)
    
    print(f"[Rank {rank}] All-Gather 结果: {recv_tensor.tolist()}")

    # ==========================================
    # Demo 3: Broadcast (广播)
    # ==========================================
    # 场景：Rank 0 将自己的数据发送给所有其他 Rank。
    if rank == 0:
        data = torch.tensor([888.0], device=device)
    else:
        data = torch.tensor([0.0], device=device)
        
    # src=0 表示数据来源是 rank 0
    comm.broadcast(data, src=0)
    
    if rank != 0:
        print(f"[Rank {rank}] Broadcast 收到: {data.item()}")
        assert data.item() == 888.0

    print(f"[Rank {rank}] 所有通信测试完成。")
    
    # 清理资源
    # 注意：PyNcclCommunicator 通常不需要显式销毁，Python GC 会处理
    dist.destroy_process_group()

def main():
    if not torch.cuda.is_available():
        print("错误: 本脚本需要 CUDA 环境才能运行 NCCL 通信。")
        return

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"警告: 检测到 {world_size} 个 GPU。NCCL 通信通常需要至少 2 个 GPU 才能看到交互效果。")
        # 即使 1 个 GPU 也可以运行代码逻辑（自发自收），不会报错。
    
    print(f"开始在 {world_size} 个 GPU 上运行 PyNccl Demo...")
    
    # 使用 spawn 启动多进程，这是 PyTorch 在 CUDA 环境下的推荐方式
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    
# [Rank 1] All-Reduce 前: 2.0
# [Rank 0] All-Reduce 前: 1.0
# [Rank 2] All-Reduce 前: 3.0
# [Rank 3] All-Reduce 前: 4.0
# [Rank 1] All-Reduce 后: 10.0 (预期: 10.0)
# [Rank 2] All-Reduce 后: 10.0 (预期: 10.0)
# [Rank 0] All-Reduce 后: 10.0 (预期: 10.0)
# [Rank 3] All-Reduce 后: 10.0 (预期: 10.0)
# [Rank 3] All-Gather 结果: [0.0, 1.0, 2.0, 3.0]
# [Rank 2] All-Gather 结果: [0.0, 1.0, 2.0, 3.0]
# [Rank 0] All-Gather 结果: [0.0, 1.0, 2.0, 3.0]
# [Rank 1] All-Gather 结果: [0.0, 1.0, 2.0, 3.0]
# [Rank 0] 所有通信测试完成。
# [Rank 1] Broadcast 收到: 888.0
# [Rank 2] Broadcast 收到: 888.0
# [Rank 3] Broadcast 收到: 888.0
# [Rank 1] 所有通信测试完成。
# [Rank 2] 所有通信测试完成。
# [Rank 3] 所有通信测试完成。
```