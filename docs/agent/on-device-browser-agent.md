# on-device-browser-agent
https://www.runanywhere.ai/blog/on-device-browser-agent

## 模型下载和保存

### 📋 整体流程图

```
[用户点击 Run Task]
        ↓
[TaskInput.tsx] handleSubmit() → onSubmit(task, modelId, ...)
        ↓
[App.tsx] handleSubmitTask() → port.postMessage({ type: 'START_TASK', payload })
        ↓  (Chrome Port 通信)
[background/index.ts] handleStartTask() → executor.executeTask()
        ↓
[executor.ts] Phase 1: llmEngine.initialize(modelId) ← 模型加载
              Phase 2: planner.createPlan()         ← AI 规划
              Phase 3: 执行循环                     ← 浏览器自动化
        ↓
[llm-engine.ts] → 创建 offscreen document → 发送 INIT_LLM 消息
        ↓
[offscreen.ts] 下载并加载模型 (HuggingFace / WebLLM)
```

### 🔧 模型下载机制

模型下载发生在 **offscreen document** ([`offscreen.ts`](workspace/on-device-browser-agent/src/offscreen/offscreen.ts)) 中，支持两种引擎：

#### 1. **Transformers.js** (用于 LFM2 等 ONNX 模型)
```typescript
// 第 209-213 行
const pipe = await pipeline('text-generation', modelId, {
  device: 'webgpu',  // 优先使用 WebGPU，否则回退到 WASM
  dtype: 'q4',
  progress_callback: progressCallback,  // 下载进度回调
});
```
- 从 **HuggingFace Hub** 下载模型
- 缓存到浏览器缓存 (`env.useBrowserCache = true`)

#### 2. **WebLLM** (用于 Qwen/Llama 等模型)
```typescript
// 第 302-316 行
const newEngine = await CreateMLCEngine(modelId, {
  initProgressCallback: (report) => {
    chrome.runtime.sendMessage({
      type: 'LLM_PROGRESS',
      progress: report.progress,
    });
  },
  appConfig: {
    useIndexedDBCache: true,  // 缓存到 IndexedDB
  },
});
```
- 从 **MLC 模型库** 下载预编译的 WebGPU 模型
- 缓存到 **IndexedDB**（下次启动无需重新下载）

代码更新测试，只刷新扩展，不要移除。如果你是"移除再重新加载"，模型缓存会丢失，需要重新下载。如果只是点击刷新按钮，模型还在。

## LLM Action决策

错误 `"No applicable action found (state machine, rules, and LLM exhausted)"` 会在**三层决策机制都无法返回有效动作**时触发：

### 决策流程

```
┌─────────────────────────────────────────┐
│ 1️⃣ 状态机 (Site Router)                  │
│    - 检查是否是 Amazon/YouTube 等已知网站  │
│    - 如果匹配，返回预定义动作              │
└────────────────┬────────────────────────┘
                 │ 没有匹配
                 ↓
┌─────────────────────────────────────────┐
│ 2️⃣ 规则引擎 (Navigator Rules)            │
│    - 基于页面元素的通用规则匹配            │
└────────────────┬────────────────────────┘
                 │ 没有匹配
                 ↓
┌─────────────────────────────────────────┐
│ 3️⃣ LLM 推理 (Navigator Agent)            │
│    - 但是有限制: llmCallsRemaining > 0   │
│    - MAX_LLM_CALLS_PER_TASK = 3          │
└────────────────┬────────────────────────┘
                 │ 次数用完 (llmCallsRemaining = 0)
                 ↓
         ❌ 触发这个错误
```

| 场景 | 原因 |
|------|------|
| **LLM 调用次数耗尽** | 任务已用完 3 次 LLM 调用（`MAX_LLM_CALLS_PER_TASK = 3`） |
| **未知网站** | 不是 Amazon/YouTube 等有状态机的网站 |
| **页面结构异常** | 规则引擎无法识别可交互元素 |
| **复杂任务** | 需要很多步骤，每步都依赖 LLM 判断 |

### 示例

用户输入：*"在某小众购物网站上搜索并购买商品"*

1. ❌ 状态机：只支持 Amazon/YouTube，不匹配
2. ❌ 规则引擎：页面结构特殊，无法匹配
3. ✅ LLM 第 1 次调用
4. ✅ LLM 第 2 次调用
5. ✅ LLM 第 3 次调用
6. ❌ **第 4 步需要 LLM，但次数已用完 → 报错**

这是一个**资源保护机制**，防止无限调用 LLM。

## bilibili

```bash
Go to "https://www.bilibili.com/"，search "llm", open first video.
```
实现一个状态机处理bilibili.com。
```bash
┌─────────────────────────────────────────────┐
│  NAVIGATING                                  │
│  → navigate to https://www.bilibili.com     │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  ON_HOMEPAGE                                 │
│  → type "llm" in search box                 │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  TYPED_QUERY                                 │
│  → press_enter / click search button        │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  ON_RESULTS                                  │
│  → click first video                        │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  ON_VIDEO                                    │
│  → done ✓                                   │
└─────────────────────────────────────────────┘
```
在site-router.ts中增加bilibili。
