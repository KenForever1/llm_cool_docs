---
sidebar_position: 1
---


https://docs.vllm.ai/en/latest/serving/distributed_serving.html

# 分布式推理与服务

## 1 如何确定分布式推理策略？

在深入探讨分布式推理和服务的细节之前，我们首先需要明确何时使用分布式推理以及可用的策略是什么。常见的做法包括以下三类：
+ 单GPU（无分布式推理）：如果你的模型适合单个GPU，你可能不需要使用分布式推理。只需使用单个GPU来运行推理。
+ 单节点多GPU（张量并行推理）：如果您的模型太大，无法容纳在单个GPU中，但可以容纳在具有多个GPU的单个节点中，则可以使用张量并行。张量并行大小是您要使用的GPU数量。例如，如果在单个节点中有4个GPU，则可以将张量并行大小设置为4。
+ 多节点多GPU（张量并行加流水线并行推理）：如果你的模型太大，无法容纳在单个节点中，你可以将张量并行（**tensor parallel**）和流水线并行（**pipeline parallel**）结合使用。张量并行（对应常见的框架中tp参数，比如lmdeploy中也有tp参数）大小是您要在每个节点中使用的GPU数量，流水线并行大小是要使用的节点数量。例如，如果在2个节点中有16个GPU（每个节点8GPU），则可以将张量并行大小设置为8，将流水线并行大小设为2。
简而言之，你应该增加GPU的数量和节点的数量，直到你有足够的GPU内存来容纳模型。张量并行大小应该是每个节点中的GPU数量，流水线并行大小应该就是节点数量。

添加足够的GPU和节点来保存模型后，您可以先运行vLLM，它将打印一些日志，如#GPU块：790。将该数字乘以16（块大小），您可以大致得到当前配置上可以提供的最大令牌数量。如果这个数字不令人满意，例如你想要更高的吞吐量，你可以进一步增加GPU或节点的数量，直到块的数量足够。

注：
有一种边缘情况：如果模型适合具有多个GPU的单个节点，但GPU的数量不能均匀地划分模型大小，则可以使用流水线并行性，它可以沿层分割模型并支持不均匀分割。在这种情况下，张量并行大小应为1，流水线并行大小就等于GPU的数量。
## 2 vllm如何使用分布式推理和服务

vLLM支持分布式张量并行和流水线并行推理和服务。目前，我们支持[Megatron-LM](https://arxiv.org/pdf/1909.08053)的张量并行算法。我们使用Ray或python原生多进程的方式（multiprocessing库）来管理分布式运行时。在单个节点上部署时可以使用多进程的方式，多节点推理目前需要Ray。

在源码目录中可以看到：
```
$ ls vllm/executor/ -1
cpu_executor.py
distributed_gpu_executor.py
executor_base.py
gpu_executor.py
__init__.py
multiproc_gpu_executor.py
multiproc_worker_utils.py
neuron_executor.py
ray_gpu_executor.py
ray_utils.py
```

默认情况下，当不在Ray集群中运行时，如果同一节点上有足够的GPU可用于配置的tensor_paralle_size，则将使用多进程，否则将使用Ray。此默认值可以通过LLM类distributed-executor-backend参数或--distributed-executor-backend API服务器参数覆盖。将其设置为mp以采用多进程，或设置为ray以采用ray。对于多进程的方式，不需要安装Ray。

要使用LLM运行多GPU推理，需要将tensor_paralle_size参数设置为要使用的GPU数量。例如，要在4个GPU上运行推理：
```
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
```

要运行多GPU服务，需要在启动服务器时传入--tensor并行大小参数。例如，要在4个GPU上运行API服务器：

```
$ vllm serve facebook/opt-13b \
$     --tensor-parallel-size 4
```

您还可以额外指定--pipeline并行大小以启用流水线并行性。例如，要在具有流水线并行性和张量并行性的8个GPU上运行API服务器：

```
$ vllm serve gpt2 \
$     --tensor-parallel-size 4 \
$     --pipeline-parallel-size 2
```

## 3 采用Ray cluster进行多节点推理与服务

如果单个节点没有足够的GPU来容纳模型，则可以使用多个节点运行模型。重要的是要确保所有节点上的执行环境都是相同的，包括模型路径以及Python环境。推荐的方法是使用docker镜像来确保相同的环境，并通过将主机映射到相同的docker配置来隐藏主机的异构性。

第一步是启动容器并将其组织成集群。我们提供了一个帮助脚本来启动集群。请注意，此脚本在启动docker时没有管理权限，而在运行分析和跟踪工具时，访问GPU性能计数器需要管理权限。为此，脚本可以通过使用docker run命令中的--CAP-add选项将CAP_SYS_ADMIN发送到docker容器。

选择一个节点作为头节点，并运行以下命令：
```

$ bash run_cluster.sh \
$                   vllm/vllm-openai \
$                   ip_of_head_node \
$                   --head \
$                   /path/to/the/huggingface/home/in/this/node
```

在其余的工作节点上，运行以下命令：
```

$ bash run_cluster.sh \
$                   vllm/vllm-openai \
$                   ip_of_head_node \
$                   --worker \
$                   /path/to/the/huggingface/home/in/this/node
```

然后你会得到一个容器的Ray 集群（cluster）。你需要让运行这些命令的shell保持活动状态，以保持集群。任何shell断开连接都将终止集群。此外，请注意，参数ip_of_head_node应该是头节点的ip地址，所有工作节点都可以访问该地址。一个常见的误解是使用工作节点的IP地址，这是不正确的。

然后，在任何节点上，使用docker exec-it node/bin/bash进入容器，执行ray status以检查ray集群的状态。您应该看到正确数量的节点和GPU。

之后，在任何节点上，您都可以像往常一样使用vLLM，就像您在一个节点上拥有所有GPU一样。常见的做法是将张量并行大小设置为每个节点中的GPU数量，将流水线并行大小设置成节点数量。例如，如果您在2个节点中有16个GPU（每个节点8GPU），则可以将张量并行大小设置为8，管道并行大小设为2：

```
$ vllm serve /path/to/the/model/in/the/container \
$     --tensor-parallel-size 8 \
$     --pipeline-parallel-size 2
```

您还可以在不使用流水线并行的情况下使用张量并行，只需将张量并行大小设置为集群中的GPU数量即可。例如，如果在2个节点中有16个GPU（每个节点8GPU），则可以将张量并行大小设置为16：

```
$ vllm serve /path/to/the/model/in/the/container \
$     --tensor-parallel-size 16
```

为了使张量并行性能良好，您应该确保节点之间的通信是高效的，例如使用Infiniband等高速网卡。要正确设置集群以使用Infiniband，请在run_cluster.sh脚本中附加其他参数，如--privilege-e NCCL_IB_HCA=mlx5。确认Infiniband是否正常工作的一种方法是使用NCCL_DEBUG=TRACE环境变量集运行vLLM，例如NCCL_DEBUG=TRACE vLLM service。。。并检查NCCL版本和所用网络的日志。如果你在日志中发现**[send] via NET/Socket**，这意味着NCCL使用原始TCP Socket，这对于跨节点张量并行来说效率不高。如果你在日志中发现**send via NET/IB/GDRDA**，这意味着NCCL使用Infiniband和GPU Direct RDMA，这是高效的。

注意：
启动Ray集群后，最好还检查节点之间的GPU-GPU通信。设置起来可能并不容易。如果需要为通信配置设置一些环境变量，可以将它们附加到run_cluster.sh脚本中，例如-e NCCL_SOCKET_IFNAME=eth0。请注意，在shell中设置环境变量（例如NCCL_SOCKET_IFNAME=eth0-vllm-server…）仅适用于同一节点中的进程，不适用于其他节点中的过程。建议在创建集群时设置环境变量。

请确保您已将模型下载到所有节点（路径相同），或者将模型下载至所有节点都可以访问的某个分布式文件系统。
当您使用huggingface repo id引用模型时，您应该将huggingfacetoken附加到run_cluster.sh脚本中，例如-e HF_token=。推荐的方法是先下载模型，然后使用路径引用模型。