
根据一次LMdeploy报错，一步步走入探究源码。

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
CURRENT_DIR=$(pwd)
lmdeploy serve api_server /workspace/lm_deploy_repos/InternVL2-8B  --server-name '0.0.0.0' --server-port 23333 --backend 'pytorch' --tp 1 --cache-max-entry-count 0.1
```

发现--cache-max-entry-count等于0.5、0.8都可以跑起来，0.1就报错了。 internal error happened为什么呢?

直接看打印看不出问题：
```bash
2024-12-30 12:43:33,111 - lmdeploy - DEBUG - request.py:382 - creating engine loop task.
INFO:     127.0.0.1:46540 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/opt/py3/lib/python3.10/site-packages/uvicorn/protocols/http/h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
......
  File "/opt/py3/lib/python3.10/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
  File "/opt/lmdeploy/lmdeploy/serve/openai/api_server.py", line 527, in chat_completions_v1
    choice_data = ChatCompletionResponseChoice(
  File "/opt/py3/lib/python3.10/site-packages/pydantic/main.py", line 214, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatCompletionResponseChoice
finish_reason
  Input should be 'stop', 'length' or 'tool_calls' [type=literal_error, input_value='error', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/literal_error
```

```bash
> /opt/lmdeploy/lmdeploy/serve/openai/api_server.py(526)chat_completions_v1()
-> choices = []
(Pdb) p text
'internal error happened'
```


通过pdb跟踪代码：在你想要加入断点的地方添加代码。 代码运行到这个地方就会停住。
```
import pdb
pdb.set_trace()
```

pdb常用方法和gdb类似：
```
n：next，下一步
w: where，代码当前位置
l: list,查看某个范围的代码情况
p：打印变量值
```


```python
~/lmdeploy/serve/async_engine.py

async for outputs in generator.async_stream_infer(

								session_id=session_id,
								
								**prompt_input,
								
								gen_config=gen_config,
								
								adapter_name=adapter_name,
								
								stream_output=stream_response,
								
								sequence_start=sequence_start,
								
								sequence_end=sequence_end,
								
								step=self.id2step[str(session_id)]):
	...
```

```bash
~/lmdeploy/pytorch/engine/engine_instance.py
async_stream_infer func
```

```bash
-> response='internal error happened',
(Pdb) list
622  	                        response = ''
623  	                    yield GenOut(response, self.id2step[str(session_id)],
624  	                                 len(input_ids), tokens, finish_reason)
625  	                else:
626  	                    yield GenOut(
627  ->	                        response='internal error happened',
628  	                        history_token_len=self.id2step[str(session_id)],
629  	                        input_token_len=len(input_ids),
630  	                        generate_token_len=0,
631  	                        finish_reason='error',
632  	                        token_ids=[])
(Pdb) p outputs
EngineOutput(status=<ResponseType.INPUT_LENGTH_ERROR: 7>, token_ids=[], num_token=0, logprobs=None)
```

通过上面的打印，我们知道了internal error实际上是INPUT_LENGTH_ERROR。

```bash
(Pdb) p len(input_ids)
3446
```

```python
~/lmdeploy/pytorch/engine/engine_instance.py
async def async_stream_infer(self,
                                session_id: int,
                                input_ids: List[int],
                                gen_config: GenerationConfig = None,
                                multimodal: InputMultiModalType = None,
                                adapter_name: str = None,
                                **kwargs):
    """Send stream inference request.

    Args:
        session_id (int): The session id.
        input_ids (List[int]): The input token ids.
        gen_config (GenerationConfig): The sampling parameters.
        adapter_name (str): The lora adapter name.

    Yields:
        int: Error flags. 0 if success.
        List[int]: The streaming output tokens.
        int: The number of the output tokens.
    """
    if len(input_ids) > self.max_input_len:
        yield EngineOutput(ResponseType.INPUT_LENGTH_ERROR, [], 0)
        return
```

```bash
(Pdb) p generator.max_input_len
1344
(Pdb) p generator.engine
<lmdeploy.pytorch.engine.engine.Engine object at 0x7f8f778d6800>
(Pdb) p generator.engine.max_session_len
1344
```

```bash
(Pdb) p generator.engine.engine_config
PytorchEngineConfig(dtype='auto', tp=1, session_len=None, max_batch_size=128, cache_max_entry_count=0.1, prefill_interval=16, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0, adapters=None, max_prefill_token_num=8192, thread_safe=False, enable_prefix_caching=False, device_type='cuda', eager_mode=False, custom_module_map=None, download_dir=None, revision=None, quant_policy=0)
```

```python
~/lmdeploy/pytorch/engine/engine.py
def _get_max_session_len(self):
    """get max session len."""
    session_len = self.scheduler_config.max_session_len
    max_tokens = (self.cache_config.num_gpu_blocks *
                    self.cache_config.block_size)
    window_size = self.cache_config.window_size
    if window_size > 0 and window_size <= max_tokens:
        max_tokens = (1 << 63) - 1
    if session_len is None:
        session_len = max_tokens
    else:
        session_len = min(max_tokens, session_len)
    return session_len
```

```bash
(Pdb) list
277  	                     err_msg=err_msg))
278
279  	    def _get_max_session_len(self):
280  	        """get max session len."""
281  	        pdb.set_trace()
282  ->	        session_len = self.scheduler_config.max_session_len
283  	        max_tokens = (self.cache_config.num_gpu_blocks *
284  	                      self.cache_config.block_size)
285  	        window_size = self.cache_config.window_size
286  	        if window_size > 0 and window_size <= max_tokens:
287  	            max_tokens = (1 << 63) - 1
(Pdb) p self.scheduler_config
SchedulerConfig(max_batches=128, max_session_len=None, max_request_output_len=512, eviction_type='recompute', prefill_interval=16, max_active_adapters=64)
```

```bash
(Pdb) n
> /opt/lmdeploy/lmdeploy/pytorch/engine/engine.py(283)_get_max_session_len()
-> max_tokens = (self.cache_config.num_gpu_blocks *
(Pdb) p self.cache_config
CacheConfig(max_batches=128, block_size=64, num_cpu_blocks=128, num_gpu_blocks=21, window_size=-1, cache_max_entry_count=0.1, max_prefill_token_num=8192, enable_prefix_caching=False, quant_policy=0, device_type='cuda')
(Pdb) p max_tokens
1344
```

修改参数--cache-max-entry-count 0.5
，从0.1改为0.5，发现num_gpu_blocks的值变了，从24变成了107。
```bash
(Pdb) p self.cache_config
CacheConfig(max_batches=128, block_size=64, num_cpu_blocks=128, num_gpu_blocks=107, window_size=-1, cache_max_entry_count=0.5, max_prefill_token_num=8192, enable_prefix_caching=False, quant_policy=0, device_type='cuda')

(Pdb) p max_tokens
6848
```

```python
~/lmdeploy/pytorch/engine/model_agent.py

def _update_cache_config(model_config: ModelConfig,

cache_config: CacheConfig,

gpu_id: int = 0,

host_mem_size: int = 1 * (1 << 30),

world_size: int = 1):
	if cache_config.num_gpu_blocks == 0:
	
		cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)
	
	if cache_config.num_gpu_blocks <= 0:
	
		raise RuntimeError('No enough gpu memory for kv cache.')
```

```bash
(Pdb) p cache_block_size
8388608
(Pdb) p gpu_mem
903548928.0
```
相除就是107。
cache_block_size的值是：
```
total = num_layers * (mem_key_block + mem_value_block)
```
mem_key_block size就是key的shape 乘以 element_size。如何是量化的模型，计算方式不同，首先element_size大小变小了，但还需要加上key_scale_zero_block、value_scale_zero_block的占用空间。shape表示为：
```python
def get_k_block_shape(
	block_size: int,
	num_heads: int,
	head_size: int,
	dtype: torch.dtype,
) -> Tuple[int, ...]:
	"""get block shape of k."""
	return (
		block_size,
		num_heads,
		head_size,
	)
```


总结，原因就是分配给kv_cache的GPU显存太少了。
```bash
num_gpu_blocks = gpu_mem / cache_block_size
block_size (int): The token numbers of the block.
```
num_gpu_blocks：表示分配给kv cache的gpu blocks。
因此最大支持：num_gpu_blocks * block_size个token。上面的报错正是超过了分配的kv_cache显存支持的最大token数量。
```bash
block_size (int): The token numbers of the block.
max_input = block_size * num_gpu_blocks
```
输入token需要小于max_input。

那参数cache_max_entry_count是如何影响的呢？
原因就是根据cache_max_entry_count该参数和可用内存（gpu的总内存 减去 prefill用到的内存）的乘积 决定的。

```python

def __get_free_gpu_mem_size(cache_block_size: int):
	"""get free gpu memory size."""
	torch.cuda.empty_cache()
	gpu_mem_physical_free, _ = get_gpu_memory(gpu_id)
	logger.debug(f'device<{gpu_id}> free gpu memory:'
				 f' {gpu_mem_physical_free>>20} mb')
	vocal_size = model_config.vocab_size

	runtime_cache_size, max_prefill_token_num = __get_runtime_size(
		gpu_mem_physical_free, cache_block_size, vocal_size)
	if cache_config.max_prefill_token_num != max_prefill_token_num:
		if max_prefill_token_num <= 0:
			raise RuntimeError('No enough gpu memory for runtime.')
		cache_config.max_prefill_token_num = max_prefill_token_num
		logger.warning(f'device<{gpu_id}> No enough memory. '
					   'update max_prefill_token_num='
					   f'{max_prefill_token_num}')
	gpu_mem_physical_free -= runtime_cache_size
	logger.debug('estimated max runtime memory:'
				 f' {runtime_cache_size>>20} mb')
	return gpu_mem_physical_free * cache_config.cache_max_entry_count
```


可以在运行的时候，设置--log-level=DEBUG打印更详细信息。
```bash
2024-12-30 12:41:32,785 - lmdeploy - DEBUG - model_agent.py:62 - device<0> free gpu memory: 6784 mb
2024-12-30 12:41:32,786 - lmdeploy - DEBUG - model_agent.py:76 - estimated max runtime memory: 5061 mb
2024-12-30 12:41:32,786 - lmdeploy - DEBUG - model_agent.py:108 - block num: 21
2024-12-30 12:41:32,786 - lmdeploy - INFO - cache_engine.py:36 - build CacheEngine with config:CacheConfig(max_batches=128, block_size=64, num_cpu_blocks=128, num_gpu_blocks=21, window_size=-1, cache_max_entry_count=0.1, max_prefill_token_num=8192, enable_prefix_caching=False, quant_policy=0, device_type='cuda')
2024-12-30 12:41:32,790 - lmdeploy - DEBUG - cache_engine.py:65 - Initialize cache engine with 21 gpu blocks and 128 cpu blocks.
2024-12-30 12:41:32,790 - lmdeploy - INFO - async_engine.py:169 - updated backend_config=PytorchEngineConfig(dtype='auto', tp=1, session_len=None, max_batch_size=128, cache_max_entry_count=0.1, prefill_interval=16, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0, adapters=None, max_prefill_token_num=8192, thread_safe=False, enable_prefix_caching=False, device_type='cuda', eager_mode=False, custom_module_map=None, download_dir=None, revision=None, quant_policy=0)
```