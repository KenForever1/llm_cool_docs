
[quantization/w4a16](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/w4a16.md)

```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com # 用于国内加速下载数据集
lmdeploy lite auto_awq /workspace/lm_deploy_repos/InternVL2-8B/ --work-dir /workspace/lm_deploy_repos/w4a16/ --calib-dataset 'wikitext2'
```

wiki2量化数据集：
```
https://huggingface.co/datasets/mindchain/wikitext2
```
WikiText语言建模数据集是从维基百科上经过验证的Good和Featured文章集中提取的超过1亿个令牌的集合。
与PTB的预处理版本相比，WikiText-2大了2倍多，WikiText-103大了110倍多。WikiText数据集还具有更大的词汇量，并保留了原始的大小写、标点符号和数字——所有这些都在PTB中被删除了。由于它由完整的文章组成，因此该数据集非常适合可以利用长期依赖关系的模型。

