---
sidebar_position: 1
---
https://docs.vllm.ai/en/stable/design/plugin_system.html
社区经常要求能够使用自定义功能扩展vLLM。为了促进这一点，vLLM包括一个插件系统，允许用户在不修改vLLM代码库的情况下添加自定义功能。本文档解释了插件如何在vLLM中工作，以及如何为vLLM创建插件。

## 1 插件在vLLM中的工作原理

插件是vLLM执行的用户注册代码。鉴于vLLM的架构（请参阅架构概述），可能涉及多个进程，特别是在使用具有各种并行技术的分布式推理时。为了成功启用插件，vLLM创建的每个进程都需要加载插件。这是由vllm.plugins模块中的load_general_plugins函数完成的。vLLM创建的每个进程在开始任何工作之前都会调用此函数。

## 2 vLLM如何发现插件

vLLM的插件系统使用标准的Python entry_points机制。这种机制允许开发人员在他们的Python包中注册函数，以供其他包使用。一个插件示例：

```
# inside `setup.py` file
from setuptools import setup

setup(name='vllm_add_dummy_model',
      version='0.1',
      packages=['vllm_add_dummy_model'],
      entry_points={
          'vllm.general_plugins':
          ["register_dummy_model = vllm_add_dummy_model:register"]
      })

# inside `vllm_add_dummy_model.py` file
def register():
    from vllm import ModelRegistry

    if "MyLlava" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyLlava",
                                        "vllm_add_dummy_model.my_llava:MyLlava")
```

每个插件都有三个部分：
插件组：入口点组的名称。vLLM使用入口点组vLLM.general_plugins来注册通用插件。这是setup.py文件中entry_points的键。对于vllm的通用插件，始终使用vllm.general_plugins。
插件名称：插件的名称。这是entry_points字典中的值。在上面的示例中，插件名称是register_dummy_model。插件可以使用VLLM_Plugins环境变量按名称进行过滤。要仅加载特定插件，请将VLLM_PLUGINS设置为插件名称。
插件值：要在插件系统中注册的函数的完全限定名。在上面的示例中，插件值是vllm_add_dummy_model:register，它引用vllm_add_dummy_mmodel模块中名为register的函数。
## 3 插件能做什么？
目前，插件的主要用例是将自定义的树外模型注册到vLLM中。这是通过调用ModelRegistry.register_model来注册模型来完成的。未来，插件系统可能会被扩展以支持更多功能，例如在vLLM中为某些类交换自定义实现。

## 4 插件编写指南

可重入：入口点中指定的函数应该是可重入的，这意味着它可以被多次调用而不会引起问题。这是必要的，因为在某些进程中可能会多次调用该函数。