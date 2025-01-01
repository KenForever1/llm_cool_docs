---
sidebar_position: 1
---
https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-class

## 1 系统交互方式

vLLM提供了许多与系统交互的入口点。主要包括LLM class和OpenAI API兼容Server接口，下图显示了它们之间的关系

![](https://docs.vllm.ai/en/stable/_images/entrypoints.excalidraw.png)

### 1.1 LLM class
LLM Class提供了进行离线推理的主要Python接口，离线推理是在不使用单独的模型推理服务器的情况下与模型交互。

以下是LLM类使用示例：

```
from vllm import LLM, SamplingParams

# Define a list of input prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The largest ocean is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the LLM engine with the OPT-125M model
llm = LLM(model="facebook/opt-125m")

# Generate outputs for the input prompts
outputs = llm.generate(prompts, sampling_params)

# Print the generated outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 1.2 兼容OpenAI的API服务器
vLLM的第二个主要接口是通过其兼容OpenAI的API服务器。此服务器可以使用vllm-server命令启动。
```
vllm serve <model>
```

或者，直接使用API server模块的入口点，而不是通过vllm CLI命令。例如：

```
python -m vllm.entrypoints.openai.api_server --model <model>
```

## 2 LLM Engine

LLMEngine和AsyncLLMEngine类是vLLM系统功能的核心，处理模型推理和异步请求处理。

![](https://docs.vllm.ai/en/stable/_images/llm_engine.excalidraw.png)

### 2.1 LLMEngine Class

LLMEngine类是vLLM发动机的核心部件。它负责接收来自客户端的请求并从模型生成输出。LLMEngine包括输入处理、模型执行（可能分布在多个主机和/或GPU上）、调度和输出处理。

+ 输入处理：使用指定的标记器处理输入文本的标记化。
+ 调度：选择在每个步骤中处理哪些请求。
+ 模型执行：管理语言模型的执行，包括跨多个GPU的分布式执行。
+ 输出处理：处理模型生成的输出，将语言模型中的令牌ID解码为人类可读的文本。

### 2.2 AsyncLLMEngine

AsyncLLMEngine类是LLMEngin类的异步包装器。它使用asyncio创建一个持续处理传入请求的后台循环。AsyncLLMEngine专为在线服务而设计，它可以处理多个并发请求并将输出流式传输到客户端。

### 2.3 Worker

worker是一个运行模型推理的进程。vLLM遵循使用一个进程来控制一个加速器设备（如GPU）的常见做法。例如，如果我们使用大小2的张量并行性和大小2的流水线并行性，我们总共将有4个worker。工人通过他们的级别和local_rank进行标识。rank用于全局编排，而localrank主要用于分配加速器设备和访问本地资源，如文件系统和共享内存。

### 2.4 Model Runner
每个worker都有一个model runner对象，负责加载和运行模型。大部分模型执行逻辑都存在于这里，例如准备输入张量和捕获柱状图。

### 2.5 Model

每个模型运行器对象都有一个模型对象，即实际的torch.nn。模块实例。请参阅[与HuggingFace的集成](https://docs.vllm.ai/en/stable/design/huggingface_integration.html#huggingface-integration)，了解各种配置如何影响我们最终得到的类。

你也可以改动ModelRunner模块，比如采用Tensorrtllm去加载模型和执行推理。

## 3 类层次结构

![](https://docs.vllm.ai/en/stable/_images/hierarchy.png)


这个类层次结构背后有几个重要的设计选择：



1.可扩展性：层次结构中的所有类都接受包含所有必要信息的配置对象。VllmConfig类是传递的主要配置对象。类层次结构相当深，每个类都需要读取它感兴趣的配置。通过将所有配置封装在一个对象中，我们可以很容易地传递配置对象并访问我们需要的配置。假设我们想添加一个只涉及模型运行器的新功能（考虑到LLM推理领域的发展速度，这种情况经常发生）。我们必须在VllmConfig类中添加一个新的配置选项。由于我们传递了整个配置对象，我们只需要将配置选项添加到VllmConfig类中，模型运行器就可以直接访问它。我们不需要更改引擎、worker或模型类的构造函数来传递新的配置选项。

2.统一性：模型运行器需要一个统一的接口来创建和初始化模型。vLLM支持50多种流行的开源模型。每个模型都有自己的初始化逻辑。如果构造函数签名随模型而变化，则模型运行者不知道如何相应地调用构造函数，而没有复杂且容易出错的检查逻辑。通过使模型类的构造函数统一，模型运行者可以在不知道具体模型类型的情况下轻松创建和初始化模型。这对于构建模型也很有用。视觉语言模型通常由视觉模型和语言模型组成。通过使构造函数统一，我们可以很容易地创建视觉模型和语言模型，并将它们组合成视觉语言模型。

注：
为了支持这一更改，所有vLLM模型的签名都已更新为：

```

def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

比如如下改动方式，这样，该模型可以与vLLM的旧版本和新版本一起使用。
```
class MyOldModel(nn.Module):
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        ...

from vllm.config import VllmConfig
class MyNewModel(MyOldModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        super().__init__(config, cache_config, quant_config, lora_config, prefix)

if __version__ >= "0.6.4":
    MyModel = MyNewModel
else:
    MyModel = MyOldModel
```

3.初始化时的分片和量化：某些特征需要更改模型权重。例如，张量并行需要分割模型权重，量化需要量化模型权重。实现此功能有两种可能的方法。一种方法是在模型初始化后更改模型权重。另一种方法是在模型初始化期间更改模型权重。vLLM选择后者。第一种方法不适用于大型模型。假设我们想用16个H100 80GB GPU运行一个405B型号（重量约为810GB）。理想情况下，每个GPU应该只加载50GB的权重。如果我们在模型初始化后更改模型权重，我们需要将整个810GB的权重加载到每个GPU，然后对权重进行分片，从而导致巨大的内存开销。相反，如果我们在模型初始化期间对权重进行分片，每一层都只会创建一个所需权重的分片，从而大大减少内存开销。同样的想法也适用于量化。请注意，我们还为模型的构造函数添加了一个额外的参数前缀，以便模型可以根据前缀进行不同的初始化。这对于非均匀量化非常有用，其中模型的不同部分被不同地量化。前缀通常是顶级模型的空字符串，子模型的前缀是类似“视觉”或“语言”的字符串。通常，它与检查点文件中模块的状态字典的名称相匹配。

这种设计的一个缺点是，很难在vLLM中为单个组件编写单元测试，因为每个组件都需要由一个完整的配置对象初始化。我们通过提供一个默认初始化函数来解决这个问题，该函数创建了一个默认配置对象，其中所有字段都设置为None。如果我们想要测试的组件只关心配置对象中的几个字段，我们可以创建一个默认的配置对象并设置我们关心的字段。这样，我们就可以单独测试组件。请注意，vLLM中的许多测试都是测试整个系统的端到端测试，因此这不是一个大问题。

总之，完整的配置对象VllmConfig可以被视为在所有vLLM类之间共享的引擎级全局状态。