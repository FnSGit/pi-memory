# pi-memory

pi 跨会话长期记忆扩展。分层记忆 + 语义检索，本地运行、中文友好、完全离线。

## 架构

```
pi (bun 单文件, jiti 加载)                node worker 子进程
┌──────────────────────────┐            ┌─────────────────────────┐
│ src/index.ts  工具+事件    │ JSON-RPC  │ src/worker.ts           │
│ src/client.ts 子进程管理   │ ─stdio──▶ │ src/store.ts  RuVector  │
│ src/paths.ts  sidecar 直读│            │ src/embed.ts  bge-zh    │
└──────────────────────────┘            └─────────────────────────┘
```

为什么要拆 worker：pi 以 bun 单文件可执行运行，extension 经 jiti 加载；
`@huggingface/transformers` 会被 jiti 编译成超长 data:URL，Bun resolver 抛
`NameTooLong`（ruvector 原生 `.node` 同理不可靠）。plain node 直跑全链路无此问题。

## 记忆分层

| 类型 | 存放 | 用途 |
|------|------|------|
| `preference` | 全局库（跨项目） | 用户偏好、习惯、固定要求 |
| `knowledge` | 全局库（跨项目） | 可复用经验、教训、事实 |
| `project` | 项目库（按 cwd 隔离） | 项目稳定事实、架构决策、约定 |

物理分库（global.db / projects/&lt;slug&gt;-&lt;hash&gt;.db），检索天然隔离，无需过滤。

## 三条通路

1. **手动工具**：`memory_save` / `memory_search` / `memory_forget`（走 worker）、`memory_list`（直读索引）
2. **自动注入**：会话首个 run 前，把偏好/常识/项目记忆以 `<user_memory>` section 注入系统提示（只读 sidecar，零 embedding 开销，每类条数可配）
3. **自动沉淀**：`agent_settled` 后把本段新对话交给提取模型（默认 `llmproxy-anthropic/Haiku`，`PI_MEMORY_EXTRACT_MODEL` 可覆盖）提取持久记忆，去重（相似度 ≥ 0.90 判重）后入库；节流：默认 10 分钟最小间隔 + 至少 2 条新增用户消息；print/json 模式不跑

## 存储布局

```
~/.pi/agent/pi-memory/
├── global.db + global.index.json     全局库
└── projects/<slug>-<hash>.db + .index.json
```

- 向量库 metadata 存完整条目（检索自包含）；sidecar JSON 镜像用于列表/注入/前缀删除
- 模型缓存：`~/.cache/huggingface`（首次自动下载 ~25MB，q8 量化）
- 调试日志：`~/.pi/agent/pi-memory-debug.log`（低频，仅异常路径）

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PI_MEMORY=0` | - | 禁用扩展 |
| `PI_MEMORY_ROOT` | `~/.pi/agent/pi-memory` | 存储根目录 |
| `PI_MEMORY_MODEL` | `Xenova/bge-small-zh-v1.5` | embedding 模型 |
| `PI_MEMORY_DIMS` | `512` | 向量维度，换模型必须同步改并清空旧库 |
| `PI_MEMORY_DTYPE` | `q8` | 量化档位（fp32 提质量） |
| `PI_MEMORY_HF_ENDPOINT` | 官方 | HF 端点覆盖（官方失败自动回退 hf-mirror 一次） |
| `PI_MEMORY_CACHE_DIR` | `~/.cache/huggingface` | 模型缓存 |
| `PI_MEMORY_INJECT=0` | 开 | 关闭会话注入 |
| `PI_MEMORY_INJECT_PREFS/KNOWLEDGE/PROJECT` | 10/6/8 | 注入条数 |
| `PI_MEMORY_EXTRACT=0` | 开 | 关闭自动沉淀 |
| `PI_MEMORY_EXTRACT_MODEL` | `llmproxy-anthropic/Haiku` | 沉淀提取用模型，`provider/modelId` 或 `modelId`；未配置/畸形/未找到时回退当前会话模型（原因进 debug log） |
| `PI_MEMORY_EXTRACT_MIN_GAP_MS` | 600000 | 沉淀最小间隔 |
| `PI_MEMORY_EXTRACT_MIN_NEW_USER_MSGS` | 2 | 触发沉淀的最少新用户消息 |
| `PI_MEMORY_EXTRACT_MAX_CHARS` | 12000 | 送入提取器的对话上限 |
| `PI_MEMORY_EXTRACT_TIMEOUT_MS` | 90000 | 提取 LLM 超时 |
| `PI_MEMORY_WORKER` | 按扩展路径推导 | worker 脚本路径覆盖 |
| `PI_MEMORY_NODE_BIN` | `node` | worker 用的 node 可执行（需 ≥ 23.6） |
| `PI_MEMORY_DEBUG=0` | 开 | 关闭调试日志 |

## 验证

```bash
npm install
npm run typecheck        # tsc --noEmit
npm run smoke            # store/embed 直连冒烟（相似度校准值在这里打印）
node test/worker-smoke.ts  # worker/client 协议冒烟
```

## 已知边界

- **去重阈值 0.90 偏保守**：完全相同 ≈1.0 拦截，同义改写 ≈0.85 不判重（误留重复可 `memory_forget`，误删真记忆不可逆）。校准值见 `src/store.ts` 的 `DEDUP_THRESHOLD`
- **score 语义**：ruvector Cosine 返回距离（1−相似度），store 已统一转为相似度（越高越相关）
- **多会话并行写**：sidecar 有 mtime 新鲜度检查 + 原子写，跨会话丢更新窗口极小但非零；向量库插入不受影响
- **自动沉淀成本**：用提取模型（默认 Haiku，可用 `PI_MEMORY_EXTRACT_MODEL` 换成 Sonnet/Opus 等），每次沉淀一次 LLM 调用（受节流约束）；其 token 不计入会话用量统计
- **换模型**：改 `PI_MEMORY_MODEL` + `PI_MEMORY_DIMS`，并删除 `~/.pi/agent/pi-memory/` 下旧 `.db`/`.index.json` 重建
- **首次模型下载**：需要网络。代理用户可 `NODE_USE_ENV_PROXY=1`（Node 24）前置启动 pi，或设 `PI_MEMORY_HF_ENDPOINT=https://hf-mirror.com`；下载后离线可用
