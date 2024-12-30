"use strict";(self.webpackChunkllm_cool_docs=self.webpackChunkllm_cool_docs||[]).push([[896],{3213:(n,e,l)=>{l.r(e),l.d(e,{assets:()=>c,contentTitle:()=>t,default:()=>m,frontMatter:()=>o,metadata:()=>i,toc:()=>a});const i=JSON.parse('{"id":"vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0","title":"vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0","description":"https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-class","source":"@site/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0.md","sourceDirName":"vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f","slug":"/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0","permalink":"/llm_cool_docs/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u8bbe\u8ba1\u67b6\u6784\u6982\u8ff0.md","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","previous":{"title":"vllm\u63d2\u4ef6\u7cfb\u7edf","permalink":"/llm_cool_docs/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/vllm\u63d2\u4ef6\u7cfb\u7edf"},"next":{"title":"\u591a\u5934\u6ce8\u610f\u529b\u673a\u5236","permalink":"/llm_cool_docs/docs/vllm\u5982\u4f55\u5b9e\u73b0\u7684\uff1f/\u591a\u5934\u6ce8\u610f\u529b\u673a\u5236"}}');var s=l(4848),r=l(8453);const o={sidebar_position:1},t=void 0,c={},a=[{value:"1 \u7cfb\u7edf\u4ea4\u4e92\u65b9\u5f0f",id:"1-\u7cfb\u7edf\u4ea4\u4e92\u65b9\u5f0f",level:2},{value:"1.1 LLM class",id:"11-llm-class",level:3},{value:"1.2 \u517c\u5bb9OpenAI\u7684API\u670d\u52a1\u5668",id:"12-\u517c\u5bb9openai\u7684api\u670d\u52a1\u5668",level:3},{value:"2 LLM Engine",id:"2-llm-engine",level:2},{value:"2.1 LLMEngine Class",id:"21-llmengine-class",level:3},{value:"2.2 AsyncLLMEngine",id:"22-asyncllmengine",level:3},{value:"2.3 Worker",id:"23-worker",level:3},{value:"2.4 Model Runner",id:"24-model-runner",level:3},{value:"2.5 Model",id:"25-model",level:3},{value:"3 \u7c7b\u5c42\u6b21\u7ed3\u6784",id:"3-\u7c7b\u5c42\u6b21\u7ed3\u6784",level:2}];function d(n){const e={a:"a",code:"code",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",pre:"pre",ul:"ul",...(0,r.R)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.p,{children:(0,s.jsx)(e.a,{href:"https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-class",children:"https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-class"})}),"\n",(0,s.jsx)(e.h2,{id:"1-\u7cfb\u7edf\u4ea4\u4e92\u65b9\u5f0f",children:"1 \u7cfb\u7edf\u4ea4\u4e92\u65b9\u5f0f"}),"\n",(0,s.jsx)(e.p,{children:"vLLM\u63d0\u4f9b\u4e86\u8bb8\u591a\u4e0e\u7cfb\u7edf\u4ea4\u4e92\u7684\u5165\u53e3\u70b9\u3002\u4e3b\u8981\u5305\u62ecLLM class\u548cOpenAI API\u517c\u5bb9Server\u63a5\u53e3\uff0c\u4e0b\u56fe\u663e\u793a\u4e86\u5b83\u4eec\u4e4b\u95f4\u7684\u5173\u7cfb"}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://docs.vllm.ai/en/stable/_images/entrypoints.excalidraw.png",alt:""})}),"\n",(0,s.jsx)(e.h3,{id:"11-llm-class",children:"1.1 LLM class"}),"\n",(0,s.jsx)(e.p,{children:"LLM Class\u63d0\u4f9b\u4e86\u8fdb\u884c\u79bb\u7ebf\u63a8\u7406\u7684\u4e3b\u8981Python\u63a5\u53e3\uff0c\u79bb\u7ebf\u63a8\u7406\u662f\u5728\u4e0d\u4f7f\u7528\u5355\u72ec\u7684\u6a21\u578b\u63a8\u7406\u670d\u52a1\u5668\u7684\u60c5\u51b5\u4e0b\u4e0e\u6a21\u578b\u4ea4\u4e92\u3002"}),"\n",(0,s.jsx)(e.p,{children:"\u4ee5\u4e0b\u662fLLM\u7c7b\u4f7f\u7528\u793a\u4f8b\uff1a"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{children:'from vllm import LLM, SamplingParams\n\n# Define a list of input prompts\nprompts = [\n    "Hello, my name is",\n    "The capital of France is",\n    "The largest ocean is",\n]\n\n# Define sampling parameters\nsampling_params = SamplingParams(temperature=0.8, top_p=0.95)\n\n# Initialize the LLM engine with the OPT-125M model\nllm = LLM(model="facebook/opt-125m")\n\n# Generate outputs for the input prompts\noutputs = llm.generate(prompts, sampling_params)\n\n# Print the generated outputs\nfor output in outputs:\n    prompt = output.prompt\n    generated_text = output.outputs[0].text\n    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")\n'})}),"\n",(0,s.jsx)(e.h3,{id:"12-\u517c\u5bb9openai\u7684api\u670d\u52a1\u5668",children:"1.2 \u517c\u5bb9OpenAI\u7684API\u670d\u52a1\u5668"}),"\n",(0,s.jsx)(e.p,{children:"vLLM\u7684\u7b2c\u4e8c\u4e2a\u4e3b\u8981\u63a5\u53e3\u662f\u901a\u8fc7\u5176\u517c\u5bb9OpenAI\u7684API\u670d\u52a1\u5668\u3002\u6b64\u670d\u52a1\u5668\u53ef\u4ee5\u4f7f\u7528vllm-server\u547d\u4ee4\u542f\u52a8\u3002"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{children:"vllm serve <model>\n"})}),"\n",(0,s.jsx)(e.p,{children:"\u6216\u8005\uff0c\u76f4\u63a5\u4f7f\u7528API server\u6a21\u5757\u7684\u5165\u53e3\u70b9\uff0c\u800c\u4e0d\u662f\u901a\u8fc7vllm CLI\u547d\u4ee4\u3002\u4f8b\u5982\uff1a"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{children:"python -m vllm.entrypoints.openai.api_server --model <model>\n"})}),"\n",(0,s.jsx)(e.h2,{id:"2-llm-engine",children:"2 LLM Engine"}),"\n",(0,s.jsx)(e.p,{children:"LLMEngine\u548cAsyncLLMEngine\u7c7b\u662fvLLM\u7cfb\u7edf\u529f\u80fd\u7684\u6838\u5fc3\uff0c\u5904\u7406\u6a21\u578b\u63a8\u7406\u548c\u5f02\u6b65\u8bf7\u6c42\u5904\u7406\u3002"}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://docs.vllm.ai/en/stable/_images/llm_engine.excalidraw.png",alt:""})}),"\n",(0,s.jsx)(e.h3,{id:"21-llmengine-class",children:"2.1 LLMEngine Class"}),"\n",(0,s.jsx)(e.p,{children:"LLMEngine\u7c7b\u662fvLLM\u53d1\u52a8\u673a\u7684\u6838\u5fc3\u90e8\u4ef6\u3002\u5b83\u8d1f\u8d23\u63a5\u6536\u6765\u81ea\u5ba2\u6237\u7aef\u7684\u8bf7\u6c42\u5e76\u4ece\u6a21\u578b\u751f\u6210\u8f93\u51fa\u3002LLMEngine\u5305\u62ec\u8f93\u5165\u5904\u7406\u3001\u6a21\u578b\u6267\u884c\uff08\u53ef\u80fd\u5206\u5e03\u5728\u591a\u4e2a\u4e3b\u673a\u548c/\u6216GPU\u4e0a\uff09\u3001\u8c03\u5ea6\u548c\u8f93\u51fa\u5904\u7406\u3002"}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsx)(e.li,{children:"\u8f93\u5165\u5904\u7406\uff1a\u4f7f\u7528\u6307\u5b9a\u7684\u6807\u8bb0\u5668\u5904\u7406\u8f93\u5165\u6587\u672c\u7684\u6807\u8bb0\u5316\u3002"}),"\n",(0,s.jsx)(e.li,{children:"\u8c03\u5ea6\uff1a\u9009\u62e9\u5728\u6bcf\u4e2a\u6b65\u9aa4\u4e2d\u5904\u7406\u54ea\u4e9b\u8bf7\u6c42\u3002"}),"\n",(0,s.jsx)(e.li,{children:"\u6a21\u578b\u6267\u884c\uff1a\u7ba1\u7406\u8bed\u8a00\u6a21\u578b\u7684\u6267\u884c\uff0c\u5305\u62ec\u8de8\u591a\u4e2aGPU\u7684\u5206\u5e03\u5f0f\u6267\u884c\u3002"}),"\n",(0,s.jsx)(e.li,{children:"\u8f93\u51fa\u5904\u7406\uff1a\u5904\u7406\u6a21\u578b\u751f\u6210\u7684\u8f93\u51fa\uff0c\u5c06\u8bed\u8a00\u6a21\u578b\u4e2d\u7684\u4ee4\u724cID\u89e3\u7801\u4e3a\u4eba\u7c7b\u53ef\u8bfb\u7684\u6587\u672c\u3002"}),"\n"]}),"\n",(0,s.jsx)(e.h3,{id:"22-asyncllmengine",children:"2.2 AsyncLLMEngine"}),"\n",(0,s.jsx)(e.p,{children:"AsyncLLMEngine\u7c7b\u662fLLMEngin\u7c7b\u7684\u5f02\u6b65\u5305\u88c5\u5668\u3002\u5b83\u4f7f\u7528asyncio\u521b\u5efa\u4e00\u4e2a\u6301\u7eed\u5904\u7406\u4f20\u5165\u8bf7\u6c42\u7684\u540e\u53f0\u5faa\u73af\u3002AsyncLLMEngine\u4e13\u4e3a\u5728\u7ebf\u670d\u52a1\u800c\u8bbe\u8ba1\uff0c\u5b83\u53ef\u4ee5\u5904\u7406\u591a\u4e2a\u5e76\u53d1\u8bf7\u6c42\u5e76\u5c06\u8f93\u51fa\u6d41\u5f0f\u4f20\u8f93\u5230\u5ba2\u6237\u7aef\u3002"}),"\n",(0,s.jsx)(e.h3,{id:"23-worker",children:"2.3 Worker"}),"\n",(0,s.jsx)(e.p,{children:"worker\u662f\u4e00\u4e2a\u8fd0\u884c\u6a21\u578b\u63a8\u7406\u7684\u8fdb\u7a0b\u3002vLLM\u9075\u5faa\u4f7f\u7528\u4e00\u4e2a\u8fdb\u7a0b\u6765\u63a7\u5236\u4e00\u4e2a\u52a0\u901f\u5668\u8bbe\u5907\uff08\u5982GPU\uff09\u7684\u5e38\u89c1\u505a\u6cd5\u3002\u4f8b\u5982\uff0c\u5982\u679c\u6211\u4eec\u4f7f\u7528\u5927\u5c0f2\u7684\u5f20\u91cf\u5e76\u884c\u6027\u548c\u5927\u5c0f2\u7684\u6d41\u6c34\u7ebf\u5e76\u884c\u6027\uff0c\u6211\u4eec\u603b\u5171\u5c06\u67094\u4e2aworker\u3002\u5de5\u4eba\u901a\u8fc7\u4ed6\u4eec\u7684\u7ea7\u522b\u548clocal_rank\u8fdb\u884c\u6807\u8bc6\u3002rank\u7528\u4e8e\u5168\u5c40\u7f16\u6392\uff0c\u800clocalrank\u4e3b\u8981\u7528\u4e8e\u5206\u914d\u52a0\u901f\u5668\u8bbe\u5907\u548c\u8bbf\u95ee\u672c\u5730\u8d44\u6e90\uff0c\u5982\u6587\u4ef6\u7cfb\u7edf\u548c\u5171\u4eab\u5185\u5b58\u3002"}),"\n",(0,s.jsx)(e.h3,{id:"24-model-runner",children:"2.4 Model Runner"}),"\n",(0,s.jsx)(e.p,{children:"\u6bcf\u4e2aworker\u90fd\u6709\u4e00\u4e2amodel runner\u5bf9\u8c61\uff0c\u8d1f\u8d23\u52a0\u8f7d\u548c\u8fd0\u884c\u6a21\u578b\u3002\u5927\u90e8\u5206\u6a21\u578b\u6267\u884c\u903b\u8f91\u90fd\u5b58\u5728\u4e8e\u8fd9\u91cc\uff0c\u4f8b\u5982\u51c6\u5907\u8f93\u5165\u5f20\u91cf\u548c\u6355\u83b7\u67f1\u72b6\u56fe\u3002"}),"\n",(0,s.jsx)(e.h3,{id:"25-model",children:"2.5 Model"}),"\n",(0,s.jsxs)(e.p,{children:["\u6bcf\u4e2a\u6a21\u578b\u8fd0\u884c\u5668\u5bf9\u8c61\u90fd\u6709\u4e00\u4e2a\u6a21\u578b\u5bf9\u8c61\uff0c\u5373\u5b9e\u9645\u7684torch.nn\u3002\u6a21\u5757\u5b9e\u4f8b\u3002\u8bf7\u53c2\u9605",(0,s.jsx)(e.a,{href:"https://docs.vllm.ai/en/stable/design/huggingface_integration.html#huggingface-integration",children:"\u4e0eHuggingFace\u7684\u96c6\u6210"}),"\uff0c\u4e86\u89e3\u5404\u79cd\u914d\u7f6e\u5982\u4f55\u5f71\u54cd\u6211\u4eec\u6700\u7ec8\u5f97\u5230\u7684\u7c7b\u3002"]}),"\n",(0,s.jsx)(e.p,{children:"\u4f60\u4e5f\u53ef\u4ee5\u6539\u52a8ModelRunner\u6a21\u5757\uff0c\u6bd4\u5982\u91c7\u7528Tensorrtllm\u53bb\u52a0\u8f7d\u6a21\u578b\u548c\u6267\u884c\u63a8\u7406\u3002"}),"\n",(0,s.jsx)(e.h2,{id:"3-\u7c7b\u5c42\u6b21\u7ed3\u6784",children:"3 \u7c7b\u5c42\u6b21\u7ed3\u6784"}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://docs.vllm.ai/en/stable/_images/hierarchy.png",alt:""})}),"\n",(0,s.jsx)(e.p,{children:"\u8fd9\u4e2a\u7c7b\u5c42\u6b21\u7ed3\u6784\u80cc\u540e\u6709\u51e0\u4e2a\u91cd\u8981\u7684\u8bbe\u8ba1\u9009\u62e9\uff1a"}),"\n",(0,s.jsx)(e.p,{children:"1.\u53ef\u6269\u5c55\u6027\uff1a\u5c42\u6b21\u7ed3\u6784\u4e2d\u7684\u6240\u6709\u7c7b\u90fd\u63a5\u53d7\u5305\u542b\u6240\u6709\u5fc5\u8981\u4fe1\u606f\u7684\u914d\u7f6e\u5bf9\u8c61\u3002VllmConfig\u7c7b\u662f\u4f20\u9012\u7684\u4e3b\u8981\u914d\u7f6e\u5bf9\u8c61\u3002\u7c7b\u5c42\u6b21\u7ed3\u6784\u76f8\u5f53\u6df1\uff0c\u6bcf\u4e2a\u7c7b\u90fd\u9700\u8981\u8bfb\u53d6\u5b83\u611f\u5174\u8da3\u7684\u914d\u7f6e\u3002\u901a\u8fc7\u5c06\u6240\u6709\u914d\u7f6e\u5c01\u88c5\u5728\u4e00\u4e2a\u5bf9\u8c61\u4e2d\uff0c\u6211\u4eec\u53ef\u4ee5\u5f88\u5bb9\u6613\u5730\u4f20\u9012\u914d\u7f6e\u5bf9\u8c61\u5e76\u8bbf\u95ee\u6211\u4eec\u9700\u8981\u7684\u914d\u7f6e\u3002\u5047\u8bbe\u6211\u4eec\u60f3\u6dfb\u52a0\u4e00\u4e2a\u53ea\u6d89\u53ca\u6a21\u578b\u8fd0\u884c\u5668\u7684\u65b0\u529f\u80fd\uff08\u8003\u8651\u5230LLM\u63a8\u7406\u9886\u57df\u7684\u53d1\u5c55\u901f\u5ea6\uff0c\u8fd9\u79cd\u60c5\u51b5\u7ecf\u5e38\u53d1\u751f\uff09\u3002\u6211\u4eec\u5fc5\u987b\u5728VllmConfig\u7c7b\u4e2d\u6dfb\u52a0\u4e00\u4e2a\u65b0\u7684\u914d\u7f6e\u9009\u9879\u3002\u7531\u4e8e\u6211\u4eec\u4f20\u9012\u4e86\u6574\u4e2a\u914d\u7f6e\u5bf9\u8c61\uff0c\u6211\u4eec\u53ea\u9700\u8981\u5c06\u914d\u7f6e\u9009\u9879\u6dfb\u52a0\u5230VllmConfig\u7c7b\u4e2d\uff0c\u6a21\u578b\u8fd0\u884c\u5668\u5c31\u53ef\u4ee5\u76f4\u63a5\u8bbf\u95ee\u5b83\u3002\u6211\u4eec\u4e0d\u9700\u8981\u66f4\u6539\u5f15\u64ce\u3001worker\u6216\u6a21\u578b\u7c7b\u7684\u6784\u9020\u51fd\u6570\u6765\u4f20\u9012\u65b0\u7684\u914d\u7f6e\u9009\u9879\u3002"}),"\n",(0,s.jsx)(e.p,{children:"2.\u7edf\u4e00\u6027\uff1a\u6a21\u578b\u8fd0\u884c\u5668\u9700\u8981\u4e00\u4e2a\u7edf\u4e00\u7684\u63a5\u53e3\u6765\u521b\u5efa\u548c\u521d\u59cb\u5316\u6a21\u578b\u3002vLLM\u652f\u630150\u591a\u79cd\u6d41\u884c\u7684\u5f00\u6e90\u6a21\u578b\u3002\u6bcf\u4e2a\u6a21\u578b\u90fd\u6709\u81ea\u5df1\u7684\u521d\u59cb\u5316\u903b\u8f91\u3002\u5982\u679c\u6784\u9020\u51fd\u6570\u7b7e\u540d\u968f\u6a21\u578b\u800c\u53d8\u5316\uff0c\u5219\u6a21\u578b\u8fd0\u884c\u8005\u4e0d\u77e5\u9053\u5982\u4f55\u76f8\u5e94\u5730\u8c03\u7528\u6784\u9020\u51fd\u6570\uff0c\u800c\u6ca1\u6709\u590d\u6742\u4e14\u5bb9\u6613\u51fa\u9519\u7684\u68c0\u67e5\u903b\u8f91\u3002\u901a\u8fc7\u4f7f\u6a21\u578b\u7c7b\u7684\u6784\u9020\u51fd\u6570\u7edf\u4e00\uff0c\u6a21\u578b\u8fd0\u884c\u8005\u53ef\u4ee5\u5728\u4e0d\u77e5\u9053\u5177\u4f53\u6a21\u578b\u7c7b\u578b\u7684\u60c5\u51b5\u4e0b\u8f7b\u677e\u521b\u5efa\u548c\u521d\u59cb\u5316\u6a21\u578b\u3002\u8fd9\u5bf9\u4e8e\u6784\u5efa\u6a21\u578b\u4e5f\u5f88\u6709\u7528\u3002\u89c6\u89c9\u8bed\u8a00\u6a21\u578b\u901a\u5e38\u7531\u89c6\u89c9\u6a21\u578b\u548c\u8bed\u8a00\u6a21\u578b\u7ec4\u6210\u3002\u901a\u8fc7\u4f7f\u6784\u9020\u51fd\u6570\u7edf\u4e00\uff0c\u6211\u4eec\u53ef\u4ee5\u5f88\u5bb9\u6613\u5730\u521b\u5efa\u89c6\u89c9\u6a21\u578b\u548c\u8bed\u8a00\u6a21\u578b\uff0c\u5e76\u5c06\u5b83\u4eec\u7ec4\u5408\u6210\u89c6\u89c9\u8bed\u8a00\u6a21\u578b\u3002"}),"\n",(0,s.jsx)(e.p,{children:"\u6ce8\uff1a\n\u4e3a\u4e86\u652f\u6301\u8fd9\u4e00\u66f4\u6539\uff0c\u6240\u6709vLLM\u6a21\u578b\u7684\u7b7e\u540d\u90fd\u5df2\u66f4\u65b0\u4e3a\uff1a"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{children:'\ndef __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n'})}),"\n",(0,s.jsx)(e.p,{children:"\u6bd4\u5982\u5982\u4e0b\u6539\u52a8\u65b9\u5f0f\uff0c\u8fd9\u6837\uff0c\u8be5\u6a21\u578b\u53ef\u4ee5\u4e0evLLM\u7684\u65e7\u7248\u672c\u548c\u65b0\u7248\u672c\u4e00\u8d77\u4f7f\u7528\u3002"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{children:'class MyOldModel(nn.Module):\n    def __init__(\n        self,\n        config,\n        cache_config: Optional[CacheConfig] = None,\n        quant_config: Optional[QuantizationConfig] = None,\n        lora_config: Optional[LoRAConfig] = None,\n        prefix: str = "",\n    ) -> None:\n        ...\n\nfrom vllm.config import VllmConfig\nclass MyNewModel(MyOldModel):\n    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n        config = vllm_config.model_config.hf_config\n        cache_config = vllm_config.cache_config\n        quant_config = vllm_config.quant_config\n        lora_config = vllm_config.lora_config\n        super().__init__(config, cache_config, quant_config, lora_config, prefix)\n\nif __version__ >= "0.6.4":\n    MyModel = MyNewModel\nelse:\n    MyModel = MyOldModel\n'})}),"\n",(0,s.jsx)(e.p,{children:"3.\u521d\u59cb\u5316\u65f6\u7684\u5206\u7247\u548c\u91cf\u5316\uff1a\u67d0\u4e9b\u7279\u5f81\u9700\u8981\u66f4\u6539\u6a21\u578b\u6743\u91cd\u3002\u4f8b\u5982\uff0c\u5f20\u91cf\u5e76\u884c\u9700\u8981\u5206\u5272\u6a21\u578b\u6743\u91cd\uff0c\u91cf\u5316\u9700\u8981\u91cf\u5316\u6a21\u578b\u6743\u91cd\u3002\u5b9e\u73b0\u6b64\u529f\u80fd\u6709\u4e24\u79cd\u53ef\u80fd\u7684\u65b9\u6cd5\u3002\u4e00\u79cd\u65b9\u6cd5\u662f\u5728\u6a21\u578b\u521d\u59cb\u5316\u540e\u66f4\u6539\u6a21\u578b\u6743\u91cd\u3002\u53e6\u4e00\u79cd\u65b9\u6cd5\u662f\u5728\u6a21\u578b\u521d\u59cb\u5316\u671f\u95f4\u66f4\u6539\u6a21\u578b\u6743\u91cd\u3002vLLM\u9009\u62e9\u540e\u8005\u3002\u7b2c\u4e00\u79cd\u65b9\u6cd5\u4e0d\u9002\u7528\u4e8e\u5927\u578b\u6a21\u578b\u3002\u5047\u8bbe\u6211\u4eec\u60f3\u752816\u4e2aH100 80GB GPU\u8fd0\u884c\u4e00\u4e2a405B\u578b\u53f7\uff08\u91cd\u91cf\u7ea6\u4e3a810GB\uff09\u3002\u7406\u60f3\u60c5\u51b5\u4e0b\uff0c\u6bcf\u4e2aGPU\u5e94\u8be5\u53ea\u52a0\u8f7d50GB\u7684\u6743\u91cd\u3002\u5982\u679c\u6211\u4eec\u5728\u6a21\u578b\u521d\u59cb\u5316\u540e\u66f4\u6539\u6a21\u578b\u6743\u91cd\uff0c\u6211\u4eec\u9700\u8981\u5c06\u6574\u4e2a810GB\u7684\u6743\u91cd\u52a0\u8f7d\u5230\u6bcf\u4e2aGPU\uff0c\u7136\u540e\u5bf9\u6743\u91cd\u8fdb\u884c\u5206\u7247\uff0c\u4ece\u800c\u5bfc\u81f4\u5de8\u5927\u7684\u5185\u5b58\u5f00\u9500\u3002\u76f8\u53cd\uff0c\u5982\u679c\u6211\u4eec\u5728\u6a21\u578b\u521d\u59cb\u5316\u671f\u95f4\u5bf9\u6743\u91cd\u8fdb\u884c\u5206\u7247\uff0c\u6bcf\u4e00\u5c42\u90fd\u53ea\u4f1a\u521b\u5efa\u4e00\u4e2a\u6240\u9700\u6743\u91cd\u7684\u5206\u7247\uff0c\u4ece\u800c\u5927\u5927\u51cf\u5c11\u5185\u5b58\u5f00\u9500\u3002\u540c\u6837\u7684\u60f3\u6cd5\u4e5f\u9002\u7528\u4e8e\u91cf\u5316\u3002\u8bf7\u6ce8\u610f\uff0c\u6211\u4eec\u8fd8\u4e3a\u6a21\u578b\u7684\u6784\u9020\u51fd\u6570\u6dfb\u52a0\u4e86\u4e00\u4e2a\u989d\u5916\u7684\u53c2\u6570\u524d\u7f00\uff0c\u4ee5\u4fbf\u6a21\u578b\u53ef\u4ee5\u6839\u636e\u524d\u7f00\u8fdb\u884c\u4e0d\u540c\u7684\u521d\u59cb\u5316\u3002\u8fd9\u5bf9\u4e8e\u975e\u5747\u5300\u91cf\u5316\u975e\u5e38\u6709\u7528\uff0c\u5176\u4e2d\u6a21\u578b\u7684\u4e0d\u540c\u90e8\u5206\u88ab\u4e0d\u540c\u5730\u91cf\u5316\u3002\u524d\u7f00\u901a\u5e38\u662f\u9876\u7ea7\u6a21\u578b\u7684\u7a7a\u5b57\u7b26\u4e32\uff0c\u5b50\u6a21\u578b\u7684\u524d\u7f00\u662f\u7c7b\u4f3c\u201c\u89c6\u89c9\u201d\u6216\u201c\u8bed\u8a00\u201d\u7684\u5b57\u7b26\u4e32\u3002\u901a\u5e38\uff0c\u5b83\u4e0e\u68c0\u67e5\u70b9\u6587\u4ef6\u4e2d\u6a21\u5757\u7684\u72b6\u6001\u5b57\u5178\u7684\u540d\u79f0\u76f8\u5339\u914d\u3002"}),"\n",(0,s.jsx)(e.p,{children:"\u8fd9\u79cd\u8bbe\u8ba1\u7684\u4e00\u4e2a\u7f3a\u70b9\u662f\uff0c\u5f88\u96be\u5728vLLM\u4e2d\u4e3a\u5355\u4e2a\u7ec4\u4ef6\u7f16\u5199\u5355\u5143\u6d4b\u8bd5\uff0c\u56e0\u4e3a\u6bcf\u4e2a\u7ec4\u4ef6\u90fd\u9700\u8981\u7531\u4e00\u4e2a\u5b8c\u6574\u7684\u914d\u7f6e\u5bf9\u8c61\u521d\u59cb\u5316\u3002\u6211\u4eec\u901a\u8fc7\u63d0\u4f9b\u4e00\u4e2a\u9ed8\u8ba4\u521d\u59cb\u5316\u51fd\u6570\u6765\u89e3\u51b3\u8fd9\u4e2a\u95ee\u9898\uff0c\u8be5\u51fd\u6570\u521b\u5efa\u4e86\u4e00\u4e2a\u9ed8\u8ba4\u914d\u7f6e\u5bf9\u8c61\uff0c\u5176\u4e2d\u6240\u6709\u5b57\u6bb5\u90fd\u8bbe\u7f6e\u4e3aNone\u3002\u5982\u679c\u6211\u4eec\u60f3\u8981\u6d4b\u8bd5\u7684\u7ec4\u4ef6\u53ea\u5173\u5fc3\u914d\u7f6e\u5bf9\u8c61\u4e2d\u7684\u51e0\u4e2a\u5b57\u6bb5\uff0c\u6211\u4eec\u53ef\u4ee5\u521b\u5efa\u4e00\u4e2a\u9ed8\u8ba4\u7684\u914d\u7f6e\u5bf9\u8c61\u5e76\u8bbe\u7f6e\u6211\u4eec\u5173\u5fc3\u7684\u5b57\u6bb5\u3002\u8fd9\u6837\uff0c\u6211\u4eec\u5c31\u53ef\u4ee5\u5355\u72ec\u6d4b\u8bd5\u7ec4\u4ef6\u3002\u8bf7\u6ce8\u610f\uff0cvLLM\u4e2d\u7684\u8bb8\u591a\u6d4b\u8bd5\u90fd\u662f\u6d4b\u8bd5\u6574\u4e2a\u7cfb\u7edf\u7684\u7aef\u5230\u7aef\u6d4b\u8bd5\uff0c\u56e0\u6b64\u8fd9\u4e0d\u662f\u4e00\u4e2a\u5927\u95ee\u9898\u3002"}),"\n",(0,s.jsx)(e.p,{children:"\u603b\u4e4b\uff0c\u5b8c\u6574\u7684\u914d\u7f6e\u5bf9\u8c61VllmConfig\u53ef\u4ee5\u88ab\u89c6\u4e3a\u5728\u6240\u6709vLLM\u7c7b\u4e4b\u95f4\u5171\u4eab\u7684\u5f15\u64ce\u7ea7\u5168\u5c40\u72b6\u6001\u3002"})]})}function m(n={}){const{wrapper:e}={...(0,r.R)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(d,{...n})}):d(n)}},8453:(n,e,l)=>{l.d(e,{R:()=>o,x:()=>t});var i=l(6540);const s={},r=i.createContext(s);function o(n){const e=i.useContext(r);return i.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function t(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:o(n.components),i.createElement(r.Provider,{value:e},n.children)}}}]);