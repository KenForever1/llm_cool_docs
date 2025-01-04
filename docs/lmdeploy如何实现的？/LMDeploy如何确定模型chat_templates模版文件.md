
模型模版文件规定了模型的role、stop_word、start、end等信息，这些信息是训练模型是指定的。所以要模型正常推理工作的话，也要告诉推理框架这些信息，比如它才知道什么时候一段话回答结束了。

以Internvl2-8B模型为例，如果采用“internvl-zh-hermes2“模版，你会发现出现的回答会重复很多遍。
比如：问模型“你是谁？”，回答：“我是xx。我是xx。我是xx。。。。。。”。
这样肯定是错误的，影响使用。output_tokens会成倍增加，在测量性能时，也会影响耗时和吞吐指标。
```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig, PytorchEngineConfig
from lmdeploy.vl import load_image
import argparse
import os

def pipeline_infer(opt):
    pipe = pipeline(opt.model,
                backend_config=TurbomindEngineConfig(tp=2, session_len=8192, model_format='awq'),
                #backend_config=PytorchEngineConfig(tp=2, session_len=8192,cache_max_entry_count=0.1),
                #chat_template_config=ChatTemplateConfig(model_name='internvl-zh-hermes2')
                chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2')
                )
    gen_config = GenerationConfig(top_p=0.6, temperature=0.8)
    image = load_image(opt.image)
    response = pipe((opt.prompt, image), gen_config=gen_config)
    print(response)
```

要确定正确的模版，我们就需要知道LMDeploy支持哪些model_name。
可以通过如下命令查看
```
$ lmdeploy list

```
### 0.1 指定'internvl-zh-hermes2模版名

```bash
- lmdeploy - WARNING - async_engine.py:508 - GenerationConfig: GenerationConfig(n=1, max_new_tokens=512, do_sample=False, top_p=0.6, top_k=50, min_p=0.0, temperature=0.8, repetition_penalty=1.0, ignore_eos=False, random_seed=None, stop_words=None, bad_words=None, stop_token_ids=None, bad_token_ids=None, min_new_tokens=None, skip_special_tokens=True, logprobs=None, response_format=None, logits_processors=None)
```

```bash
lmdeploy - WARNING - async_engine.py:509 - Since v0.6.0, lmdeploy add `do_sample` in GenerationConfig. It defaults to False, meaning greedy decoding. Please set `do_sample=True` if sampling  decoding is needed
Response(text='The image shows a tiger lying on a grassy area. The tiger is facing the camera, with its head slightly turned to the side, giving a clear view of its face and body. The tiger has distinctive orange fur with black stripes, and its eyes are open, giving it a calm and alert expression. The grass around the tiger is green and well-maintained, suggesting a well-kept environment, possibly a zoo or a wildlife sanctuary. The lighting in the image is bright, indicating that it might be taken during the daytime. tigers are large, powerful animals known for their strength and agility. They are native to Asia and are often found in forests and grasslands.tigers are carnivorous and primarily hunt at night. They are solitary animals and are known for their distinctive orange fur with black stripes, which helps them blend into their natural environment.tigers are also known for their loud roars, which they use to communicate with other tigers. They are powerful swimmers and can even climb trees. Tigers are an endangered species, and conservation efforts are underway to protect their habitats and prevent their extinction.tigers are also known for their distinctive orange fur with black stripes, which helps them blend into their natural environment. They are powerful swimmers and can even climb trees. Tigers are an endangered species, and conservation efforts are underway to protect their habitats and prevent their extinction.', generate_token_len=513, input_token_len=1815, session_id=0, finish_reason='length', token_ids=[...92542, 92542], logprobs=None, index=0)
```
### 0.2 指定internvl2-internlm2模版名
这个模版加载后，在打印的配置信息中多了stop_token_ids=[92543, 92542]。而前者没有。

```bash
lmdeploy - WARNING - async_engine.py:508 - GenerationConfig: GenerationConfig(n=1, max_new_tokens=512, do_sample=False, top_p=0.6, top_k=50, min_p=0.0, temperature=0.8, repetition_penalty=1.0, ignore_eos=False, random_seed=None, stop_words=None, bad_words=None, stop_token_ids=[92543, 92542], bad_token_ids=None, min_new_tokens=None, skip_special_tokens=True, logprobs=None, response_format=None, logits_processors=None)
```

```bash
lmdeploy - WARNING - async_engine.py:509 - Since v0.6.0, lmdeploy add `do_sample` in GenerationConfig. It defaults to False, meaning greedy decoding. Please set `do_sample=True` if sampling  decoding is needed
Response(text='The image shows a tiger lying on a grassy area. The tiger is facing the camera, with its head slightly turned to the side, giving a clear view of its face and body. The tiger has distinctive orange fur with black stripes, and its eyes are open, giving it a calm and alert expression. The grass around the tiger is green and well-maintained, suggesting a well-kept environment, possibly a zoo or a wildlife sanctuary. The lighting in the image is bright, indicating that it might be taken during the daytime.', generate_token_len=110, input_token_len=1843, session_id=0, finish_reason='stop', token_ids=[918, 2321, 5092, 395, 50875, 20634, 519, 395, 16493, 282, 3247, 281, 707, 50875, 505, 13031, 410, 6405, 328, 579, 1326, 2118, 10215, 6675, 442, 410, 3274, 328, 7234, 395, 2961, 1800, 446, 1326, 3740, 454, 2642, 281, 707, 50875, 834, 34855, 18722, 18392, 579, 3851, 53682, 328, 454, 1326, 6569, 657, 1939, 328, 7234, 563, 395, 19466, 454, 5276, 7636, 281, 707, 16493, 2316, 410, 50875, 505, 6330, 454, 1780, 1594, 1789, 2787, 328, 22851, 395, 1780, 285, 571, 548, 4736, 328, 10915, 395, 40640, 607, 395, 29604, 49748, 281, 707, 17857, 435, 410, 2321, 505, 10041, 328, 19016, 560, 563, 2738, 517, 4591, 2489, 410, 59717, 281], logprobs=None, index=0)
```

### 0.3 LMDeploy如何确定推理时chat_template

两种方式：
+ 自动根据指定的模型路径模型名称去匹配。
+ 参数或者json文件中配置
参数配置参考：[advance/chat_template](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/advance/chat_template.md)

自动匹配根据LMDeploy源码可以知道匹配规则，在~/lmdeploy/model.py文件中定义了很多模版。通过MODELS.register_module注册。
```python
@MODELS.register_module(name='internvl2-internlm2')
class InternVL2InternLM2(InternLM2Chat7B):

    def __init__(
            self,
            meta_instruction='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
            eosys='<|im_end|>',
            eoh='<|im_end|>',
            separator='',
            stop_words=['<|im_start|>', '<|im_end|>'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         separator=separator,
                         eoh=eoh,
                         stop_words=stop_words,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if ('internvl2' in path
                and 'internvl2-4b' not in path) or 'mono-internvl' in path:
            if 'internvl2.5' in path or 'internvl2_5' in path:
                return None
            return 'internvl2-internlm2'
```

可以看到模版定义了很多信息，包括system、user、stop_words等。不同模版的值不一样的，这个是训练模型是各家公司决定的。
```python
@MODELS.register_module(name='internlm2')
class InternLM2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM2-Chat-7B."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 environment='<|im_start|>environment\n',
                 plugin='<|plugin|>',
                 interpreter='<|interpreter|>',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 eoenv='<|im_end|>\n',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
```

在match方法中，我们可以看到internvl2-8B不是正符合这条规则吗？因此internvl2-8B模型采用这个模版internvl2-internlm2。
```python
if ('internvl2' in path
                and 'internvl2-4b' not in path) or 'mono-internvl' in path:
            if 'internvl2.5' in path or 'internvl2_5' in path:
                return None
            return 'internvl2-internlm2'
```
除此之外，internvl系列还有internvl2-phi3、internvl-zh、internvl-zh-hermes2等。
```python
if 'internvl2-4b' in path:
	return 'internvl2-phi3'
```