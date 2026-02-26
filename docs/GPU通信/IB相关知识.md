# HCA 与 RDMA 详解

## 一、HCA 是什么

查看你的设备上是否有IB网卡？
```bash
$ ls /dev/infiniband/
issm0  issm1  rdma_cm  umad0  umad1  uverbs0  uverbs1

$ ibv_devices
device                 node GUID
------              ----------------
mlx5_0              e8ebd303004325cc
mlx5_1              e8ebd303004325cd
```
查看IB设备详细信息：
```bash
$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.4200
        node_guid:                      e8eb:d303:0043:25cc
        sys_image_guid:                 e8eb:d303:0043:25cc
        vendor_id:                      0x02c9
        vendor_part_id:                 4119
        hw_ver:                         0x0
        board_id:                       MT_0000000080
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.4200
        node_guid:                      e8eb:d303:0043:25cd
        sys_image_guid:                 e8eb:d303:0043:25cc
        vendor_id:                      0x02c9
        vendor_part_id:                 4119
        hw_ver:                         0x0
        board_id:                       MT_0000000080
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
```



**HCA = Host Channel Adapter（主机通道适配器）**

HCA 是 InfiniBand 网络中**主机端的网络硬件设备**，相当于以太网中的网卡（NIC），但功能更强大。

| 对比项 | 以太网 | InfiniBand |
|-------|--------|------------|
| 主机端设备 | NIC（网卡） | HCA |
| 交换机端设备 | 交换机端口 | TCA（Target Channel Adapter） |
| 典型厂商 | Intel, Broadcom | Mellanox/NVIDIA, Intel |

## 二、HCA 的核心特性

```
┌─────────────────────────────────────────────────────┐
│                    应用程序                          │
├─────────────────────────────────────────────────────┤
│              用户态 RDMA 库 (libibverbs)             │
├─────────────────────────────────────────────────────┤
│                   内核驱动                           │
├─────────────────────────────────────────────────────┤
│                    HCA 硬件                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │ RDMA    │ │ DMA     │ │ 协议    │ │ 网络     │  │
│  │ 引擎    │ │ 引擎    │ │ 处理    │ │ 接口     │  │
│  └─────────┘ └─────────┘ └─────────┘ └──────────┘  │
└─────────────────────────────────────────────────────┘
```

HCA 硬件内置了：
- **RDMA 引擎**：硬件实现 RDMA 协议，无需 CPU 参与
- **DMA 引擎**：直接访问主机内存
- **协议卸载**：传输层协议由硬件处理
- **多队列支持**：支持大量并发 QP（Queue Pair）

## 三、RDMA 是什么

**RDMA = Remote Direct Memory Access（远程直接内存访问）**

RDMA 是一种**通信技术/协议**，允许一台机器**直接读写另一台机器的内存**，无需对方 CPU 参与。

### RDMA 的三种实现方式

| 技术 | 网络类型 | 硬件要求 | 性能 |
|-----|---------|---------|------|
| **InfiniBand** | 专用 IB 网络 | HCA + IB 交换机 | 最高（延迟 < 1μs） |
| **RoCE** (RDMA over Converged Ethernet) | 以太网 | 支持 RoCE 的网卡 | 高（延迟 ~ 2μs） |
| **iWARP** | 以太网 (TCP) | 支持 iWARP 的网卡 | 中等 |

## 四、HCA 与 RDMA 的关系

```
┌─────────────────────────────────────────────────────────────┐
│                         RDMA（概念/协议）                    │
│                                                             │
│   "远程直接内存访问" - 一种零拷贝、内核旁路的通信方式          │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 需要硬件支持
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      HCA / RNIC（硬件实现）                   │
│                                                             │
│   HCA = InfiniBand 网络的 RDMA 硬件                          │
│   RNIC = 以太网的 RDMA 网卡（支持 RoCE/iWARP）                │
└─────────────────────────────────────────────────────────────┘
```

**简单类比**：
- **RDMA** 类似于 "TCP/IP 协议" — 是一套通信规范
- **HCA** 类似于 "网卡硬件" — 是实现这套规范的硬件设备

## 五、RDMA 操作原理

### 传统网络 vs RDMA

```
传统网络通信：                      RDMA 通信：
                                   
App A        App B                 App A        App B
  │            │                     │            │
  ▼            ▼                     │            │
Socket       Socket                  │            │
  │            │                     ▼            ▼
  ▼            ▼                   ┌──────────────────┐
Kernel       Kernel                │   用户态直通     │
  │            │                   │  (Kernel Bypass) │
  ▼            ▼                   └──────────────────┘
 NIC ◄──────► NIC                  HCA ◄──────────► HCA
                                         │
                                   零拷贝 + CPU 不参与
```

### RDMA 核心操作

| 操作 | 说明 |
|-----|------|
| **RDMA Write** | 直接写入远端内存，远端 CPU 无感知 |
| **RDMA Read** | 直接读取远端内存，远端 CPU 无感知 |
| **Send/Recv** | 类似传统消息传递，但仍是零拷贝 |

## 六、为什么 GPU 训练用 RDMA/HCA

| 优势 | 说明 |
|-----|------|
| **超低延迟** | < 1μs，传统以太网 ~50-100μs |
| **高带宽** | 单端口 200/400 Gbps |
| **零拷贝** | 数据不经过 CPU，直达 GPU 显存 (GPUDirect RDMA) |
| **CPU 卸载** | CPU 不参与数据传输，可专注计算 |

这就是为什么你之前的配置中使用 `mlx5_*`（Mellanox HCA）来加速 NCCL 集合通信。

比如对deepseek R1进行Lora, 使用了NCCL_IB_HCA环境变量指定HCA：

```bash
IB_DEVICES=$(find /dev/infiniband/* -maxdepth 1 -not -type d | xargs -I{} echo '--device {}:{}')
DOCKER_NAME=lora
IMAGE=xxx/hpcaitech/colossalai:vxxx
docker run ${IB_DEVICES} --device /dev/gdrdrv:/dev/gdrdrv -v /etc/topo:/etc/topo -v /nfs:/nfs --name $DOCKER_NAME -d --rm -it  --gpus=all --cap-add SYS_NICE --cap-add IPC_LOCK --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $IMAGE bash

export NCCL_DEBUG_SUBSYS=init,graph,env
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export NCCL_SET_THREAD_NAME=1
colossalai run --hostfile hostfile --nproc_per_node 8 /nfs/LoRA/lora_finetune.py --pretrained /nfs/DeepSeek-R1-BF16/ --dataset /nfs/LoRA/dataset.jsonl --plugin moe --lr 2e-5 --max_length 256 -g --ep 8 --pp 3 --batch_size 24 --lora_rank 8 --lora_alpha 16 --num_epochs 2 --warmup_steps 8 --tensorboard_dir logs --save_dir /nfs/DeepSeek-R1-BF16-LoRA
```