"use strict";(self.webpackChunkllm_cool_docs=self.webpackChunkllm_cool_docs||[]).push([[996],{9121:(n,e,l)=>{l.r(e),l.d(e,{assets:()=>s,contentTitle:()=>d,default:()=>m,frontMatter:()=>r,metadata:()=>o,toc:()=>c});const o=JSON.parse('{"id":"lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f","title":"\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f","description":"\u5728lmdeploy\u4e2d\u652f\u6301\u4e86\u5f88\u591a\u6a21\u578b\u7684\u63a8\u7406\uff0c\u8fd9\u4e9b\u6a21\u578b\u662f\u5982\u4f55\u6ce8\u518c\u7ed9lmdeploy\u6846\u67b6\u7684\u5462\uff1f","source":"@site/docs/lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f.md","sourceDirName":"lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f","slug":"/lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f","permalink":"/llm_cool_docs/docs/lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u901a\u8fc7lmdepoly\u5b66\u4e60Python\u4e2d\u5982\u4f55\u5b9e\u73b0\u6ce8\u518c\u6a21\u5f0f.md","tags":[],"version":"current","frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"Intervl2_8B\u6a21\u578b\u7684\u9884\u5904\u7406preprocess\u6709\u4f55\u4e0d\u540c","permalink":"/llm_cool_docs/docs/lmdeploy\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/Intervl2_8B\u6a21\u578b\u7684\u9884\u5904\u7406preprocess\u6709\u4f55\u4e0d\u540c"},"next":{"title":"vllm distributed_serving","permalink":"/llm_cool_docs/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm distributed_serving"}}');var i=l(4848),t=l(8453);const r={},d=void 0,s={},c=[{value:"lmdepoy\u4e2d\u8c03\u7528\u6a21\u5757",id:"lmdepoy\u4e2d\u8c03\u7528\u6a21\u5757",level:3},{value:"\u5982\u4f55\u6ce8\u518cmodel\u7684\u5462\uff1f",id:"\u5982\u4f55\u6ce8\u518cmodel\u7684\u5462",level:3}];function a(n){const e={a:"a",blockquote:"blockquote",code:"code",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.R)(),...n.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(e.p,{children:"\u5728lmdeploy\u4e2d\u652f\u6301\u4e86\u5f88\u591a\u6a21\u578b\u7684\u63a8\u7406\uff0c\u8fd9\u4e9b\u6a21\u578b\u662f\u5982\u4f55\u6ce8\u518c\u7ed9lmdeploy\u6846\u67b6\u7684\u5462\uff1f\nlmdeploy\u6846\u67b6\u53c8\u662f\u5982\u4f55\u8c03\u7528\u5230\u6b63\u786e\u7684\u6a21\u578b\u63a8\u7406\u7684\u5462?"}),"\n",(0,i.jsxs)(e.blockquote,{children:["\n",(0,i.jsx)(e.p,{children:'MMEngine \u5b9e\u73b0\u7684\u6ce8\u518c\u5668\u53ef\u4ee5\u770b\u4f5c\u4e00\u4e2a\u6620\u5c04\u8868\u548c\u6a21\u5757\u6784\u5efa\u65b9\u6cd5\uff08build function\uff09\u7684\u7ec4\u5408\u3002\u6620\u5c04\u8868\u7ef4\u62a4\u4e86\u4e00\u4e2a\u5b57\u7b26\u4e32\u5230\u7c7b\u6216\u8005\u51fd\u6570\u7684\u6620\u5c04\uff0c\u4f7f\u5f97\u7528\u6237\u53ef\u4ee5\u501f\u52a9\u5b57\u7b26\u4e32\u67e5\u627e\u5230\u76f8\u5e94\u7684\u7c7b\u6216\u51fd\u6570\uff0c\u4f8b\u5982\u7ef4\u62a4\u5b57\u7b26\u4e32 "ResNet" \u5230 ResNet \u7c7b\u6216\u51fd\u6570\u7684\u6620\u5c04\uff0c\u4f7f\u5f97\u7528\u6237\u53ef\u4ee5\u901a\u8fc7 "ResNet" \u627e\u5230 ResNet \u7c7b\uff1b\u800c\u6a21\u5757\u6784\u5efa\u65b9\u6cd5\u5219\u5b9a\u4e49\u4e86\u5982\u4f55\u6839\u636e\u5b57\u7b26\u4e32\u67e5\u627e\u5230\u5bf9\u5e94\u7684\u7c7b\u6216\u51fd\u6570\u4ee5\u53ca\u5982\u4f55\u5b9e\u4f8b\u5316\u8fd9\u4e2a\u7c7b\u6216\u8005\u8c03\u7528\u8fd9\u4e2a\u51fd\u6570\uff0c\u4f8b\u5982\uff0c\u901a\u8fc7\u5b57\u7b26\u4e32 "bn" \u627e\u5230 nn.BatchNorm2d \u5e76\u5b9e\u4f8b\u5316 BatchNorm2d \u6a21\u5757\uff1b\u53c8\u6216\u8005\u901a\u8fc7\u5b57\u7b26\u4e32 "build_batchnorm2d" \u627e\u5230 build_batchnorm2d \u51fd\u6570\u5e76\u8fd4\u56de\u8be5\u51fd\u6570\u7684\u8c03\u7528\u7ed3\u679c\u3002'}),"\n"]}),"\n",(0,i.jsx)(e.h3,{id:"lmdepoy\u4e2d\u8c03\u7528\u6a21\u5757",children:"lmdepoy\u4e2d\u8c03\u7528\u6a21\u5757"}),"\n",(0,i.jsx)(e.p,{children:"\u4f7f\u7528\u6848\u4f8b\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-bash",children:"https://github1s.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/builder.py#L65\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u5982\u679cVISION_MODELS\u4e2d\u7684module\u548chf_config\u5339\u914d\uff0c\u5c31\u4f20\u9012\u53c2\u6570\u8c03\u7528\u6a21\u5757\u3002\u8c03\u7528\u6a21\u5757\u7684\u76f8\u5173\u51fd\u6570\u3002"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"from lmdeploy.vl.model.base import VISION_MODELS\ndef load_vl_model(model_path: str,\n                  backend: str,\n                  with_llm: bool = False,\n                  backend_config: Optional[Union[TurbomindEngineConfig,\n                                                 PytorchEngineConfig]] = None):\n    ...\n    for name, module in VISION_MODELS.module_dict.items():\n    try:\n        if module.match(hf_config):\n            logger.info(f'matching vision model: {name}')\n            model = module(**kwargs)\n            model.build_preprocessor()\n            # build the vision part of a VLM model when backend is\n            # turbomind, or load the whole VLM model when `with_llm==True`\n            if backend == 'turbomind' or with_llm:\n                model.build_model()\n            return model\n    except Exception as e:\n        logger.error(f'build vision model {name} failed, {e}')\n        raise\n\n    raise ValueError(f'unsupported vl model with config {hf_config}')\n                                        \n\n"})}),"\n",(0,i.jsx)(e.h3,{id:"\u5982\u4f55\u6ce8\u518cmodel\u7684\u5462",children:"\u5982\u4f55\u6ce8\u518cmodel\u7684\u5462\uff1f"}),"\n",(0,i.jsxs)(e.p,{children:["\u5728",(0,i.jsx)(e.strong,{children:"vl/models"}),"\u76ee\u5f55\u4e2d\u53ef\u4ee5\u770b\u5230\u6ce8\u518c\u7684\u5f88\u591a\u6a21\u578b\uff0c\u6bd4\u5982qwen\u3001internvl\u7b49\u3002"]}),"\n",(0,i.jsxs)(e.p,{children:["\u5728",(0,i.jsx)(e.strong,{children:"lmdeploy/vl/model/internvl.py"}),"\u6587\u4ef6\u4e2d\uff0c\u53ef\u4ee5\u770b\u5230\u5b9a\u4e49InternVLVisionModel\u7c7b\u65f6\uff0c\u901a\u8fc7@VISION_MODELS.register_module()\u8fdb\u884c\u4e86\u6ce8\u518c\u3002"]}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:'@VISION_MODELS.register_module()\nclass InternVLVisionModel(VisonModel):\n    """InternVL vision model."""\n\n    _arch = \'InternVLChatModel\'\n\n    def __init__(self,...):\n        ....\n'})}),"\n",(0,i.jsx)(e.p,{children:"VISION_MODELS\u5c31\u662fmmengine\u4e2dRegistry\u7684\u4e00\u4e2a\u5b9e\u4f8b\u7c7b\u578b\u3002"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"from mmengine import Registry\nVISION_MODELS = Registry('vision_model')\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u5728\u4f60\u7684\u9879\u76ee\u4e2d\u5982\u679c\u4f60\u4e5f\u9700\u8981\u6839\u636e\u914d\u7f6e\uff0c\u8c03\u7528\u4e0d\u540c\u7684\u6a21\u5757\uff0c\u4e5f\u53ef\u4ee5\u91c7\u7528\u8fd9\u79cd\u65b9\u6cd5\u3002\u6ce8\u518c\u7ed1\u5b9a\uff0c\u6700\u7b80\u5355\u7684\u5c31\u662f\u5efa\u7acb\u5b57\u7b26\u4e32\u548c\u7c7b\u7684\u6620\u5c04\u5173\u7cfb\u3002"}),"\n",(0,i.jsxs)(e.p,{children:["\u5728\u8fd9\u91cclmdeploy\u4f7f\u7528\u4e86",(0,i.jsx)(e.a,{href:"https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html",children:"mmengine\u4e2d\u7684Registry\u6a21\u5757"}),"\u3002\u652f\u6301\u7684\u529f\u80fd\u66f4\u52a0\u5168\u9762\uff0c\u5305\u62ec\uff1a"]}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"\u6a21\u5757\u7684\u6ce8\u518c\u548c\u8c03\u7528"}),"\n",(0,i.jsx)(e.li,{children:"\u51fd\u6570\u7684\u6ce8\u518c\u548c\u8c03\u7528"}),"\n",(0,i.jsx)(e.li,{children:"\u6a21\u5757\u95f4\u7236\u5b50\u5173\u7cfb\u5efa\u7acb\uff0c\u5982\u679c\u5b50\u8282\u70b9\u627e\u4e0d\u5230\uff0c\u5c31\u53bb\u7236\u8282\u70b9\u4e2d\u627e\u5bf9\u5e94\u6a21\u5757\u8c03\u7528"}),"\n",(0,i.jsx)(e.li,{children:"\u5144\u5f1f\u8282\u70b9\u5173\u7cfb\u5efa\u7acb"}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:"\u7b80\u5355\u4f7f\u7528\u4f8b\u5b50\uff1a\n\u4f7f\u7528\u6ce8\u518c\u5668\u7ba1\u7406\u4ee3\u7801\u5e93\u4e2d\u7684\u6a21\u5757\uff0c\u9700\u8981\u4ee5\u4e0b\u4e09\u4e2a\u6b65\u9aa4\u3002"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:"\u521b\u5efa\u6ce8\u518c\u5668"}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:"\u521b\u5efa\u4e00\u4e2a\u7528\u4e8e\u5b9e\u4f8b\u5316\u7c7b\u7684\u6784\u5efa\u65b9\u6cd5\uff08\u53ef\u9009\uff0c\u5728\u5927\u591a\u6570\u60c5\u51b5\u4e0b\u53ef\u4ee5\u53ea\u4f7f\u7528\u9ed8\u8ba4\u65b9\u6cd5\uff09"}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:"\u5c06\u6a21\u5757\u52a0\u5165\u6ce8\u518c\u5668\u4e2d"}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:"\u5047\u8bbe\u6211\u4eec\u8981\u5b9e\u73b0\u4e00\u7cfb\u5217\u6fc0\u6d3b\u6a21\u5757\u5e76\u4e14\u5e0c\u671b\u4ec5\u4fee\u6539\u914d\u7f6e\u5c31\u80fd\u591f\u4f7f\u7528\u4e0d\u540c\u7684\u6fc0\u6d3b\u6a21\u5757\u800c\u65e0\u9700\u4fee\u6539\u4ee3\u7801\u3002"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"from mmengine import Registry\n# scope \u8868\u793a\u6ce8\u518c\u5668\u7684\u4f5c\u7528\u57df\uff0c\u5982\u679c\u4e0d\u8bbe\u7f6e\uff0c\u9ed8\u8ba4\u4e3a\u5305\u540d\uff0c\u4f8b\u5982\u5728 mmdetection \u4e2d\uff0c\u5b83\u7684 scope \u4e3a mmdet\n# locations \u8868\u793a\u6ce8\u518c\u5728\u6b64\u6ce8\u518c\u5668\u7684\u6a21\u5757\u6240\u5b58\u653e\u7684\u4f4d\u7f6e\uff0c\u6ce8\u518c\u5668\u4f1a\u6839\u636e\u9884\u5148\u5b9a\u4e49\u7684\u4f4d\u7f6e\u5728\u6784\u5efa\u6a21\u5757\u65f6\u81ea\u52a8 import\nACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])\n\n\nimport torch.nn as nn\n\n# \u4f7f\u7528\u6ce8\u518c\u5668\u7ba1\u7406\u6a21\u5757\n@ACTIVATION.register_module()\nclass Sigmoid(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        print('call Sigmoid.forward')\n        return x\n\n@ACTIVATION.register_module()\nclass ReLU(nn.Module):\n    def __init__(self, inplace=False):\n        super().__init__()\n\n    def forward(self, x):\n        print('call ReLU.forward')\n        return x\n\n@ACTIVATION.register_module()\nclass Softmax(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        print('call Softmax.forward')\n        return x\n\nprint(ACTIVATION.module_dict)\n# {\n#     'Sigmoid': __main__.Sigmoid,\n#     'ReLU': __main__.ReLU,\n#     'Softmax': __main__.Softmax\n# }\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u6ce8\u518c\u540e\uff0c\u4f7f\u7528\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"import torch\n\ninput = torch.randn(2)\n\nact_cfg = dict(type='Sigmoid')\nactivation = ACTIVATION.build(act_cfg)\noutput = activation(input)\n# call Sigmoid.forward\nprint(output)\n\n\n\u5982\u679c\u6211\u4eec\u60f3\u4f7f\u7528 ReLU\uff0c\u4ec5\u9700\u4fee\u6539\u914d\u7f6e\u3002\n\nact_cfg = dict(type='ReLU', inplace=True)\nactivation = ACTIVATION.build(act_cfg)\noutput = activation(input)\n# call ReLU.forward\nprint(output)\n\n"})}),"\n",(0,i.jsxs)(e.p,{children:["\u8fdb\u9636\u529f\u80fd\u53c2\u8003\uff1a",(0,i.jsx)(e.a,{href:"https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html%E3%80%82",children:"https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html\u3002"})]})]})}function m(n={}){const{wrapper:e}={...(0,t.R)(),...n.components};return e?(0,i.jsx)(e,{...n,children:(0,i.jsx)(a,{...n})}):a(n)}},8453:(n,e,l)=>{l.d(e,{R:()=>r,x:()=>d});var o=l(6540);const i={},t=o.createContext(i);function r(n){const e=o.useContext(t);return o.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function d(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(i):n.components||i:r(n.components),o.createElement(t.Provider,{value:e},n.children)}}}]);