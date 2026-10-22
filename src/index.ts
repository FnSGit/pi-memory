/**
 * pi-memory — pi 跨会话长期记忆扩展
 *
 * 分层记忆（RuVector 向量库 + 本地 bge-small-zh-v1.5 embedding，中文友好、完全离线）：
 *   preference  用户偏好/习惯        → 全局库（跨项目）
 *   knowledge   可复用经验/常识      → 全局库（跨项目）
 *   project     项目稳定事实/决策    → 项目库（按 cwd 隔离）
 *
 * 三条通路：
 *   1. 手动工具：memory_save / memory_search / memory_forget（走 worker 子进程）
 *              / memory_list（sidecar 直读，零开销）
 *   2. 自动注入：首个 run 前把偏好/常识/项目记忆以 <user_memory> section 进系统提示
 *              （sidecar 直读，不触发 embedding 模型）
 *   3. 自动沉淀：agent_settled 后由提取模型（默认 Haiku，可配置）提取新对话中的持久记忆，去重后入库
 *
 * 进程架构：pi 以 bun 单文件运行、extension 经 jiti 加载，transformers/ruvector
 * 原生依赖无法在 jiti 内加载（NameTooLong），故向量与 embedding 全部在独立
 * node worker 子进程（src/worker.ts）中执行，本文件只做轻量编排。
 *
 * 存储布局（PI_MEMORY_ROOT，默认 ~/.pi/agent/pi-memory）：
 *   global.db + global.index.json          全局库
 *   projects/<slug>-<hash>.db + .index.json 项目库（slug 由 cwd 派生）
 *
 * 调试：异常路径追加写 ~/.pi/agent/pi-memory-debug.log（PI_MEMORY_DEBUG=0 关闭）。
 */
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createDebugLog, errMessage, loadConfig } from "./config.ts";
import { MemoryClient, type ForgetResult, type SaveResult } from "./client.ts";
import { Extractor } from "./extract.ts";
import { renderMemorySection } from "./inject.ts";
import { readIndex } from "./paths.ts";
import type { MemoryEntry, MemoryType, SaveInput, SearchHit } from "./types.ts";

const ID_PREFIX_LEN = 8;

function shortId(id: string): string {
	return id.slice(0, ID_PREFIX_LEN);
}

function clipText(text: string | undefined, max: number): string {
	const one = (text ?? "").replace(/\s+/g, " ").trim();
	return one.length <= max ? one : `${one.slice(0, max - 1)}…`;
}

function fmtDate(ts: number): string {
	return ts > 0 ? new Date(ts).toISOString().slice(0, 10) : "?";
}

function fmtEntryLine(e: MemoryEntry): string {
	return `(${e.type}, ${fmtDate(e.createdAt)}, ${e.source}) ${clipText(e.content, 160)} (id=${shortId(e.id)})`;
}

const TYPE_DESC = "preference=用户偏好(全局) knowledge=经验常识(全局) project=项目事实(当前项目)";

export default function (pi: ExtensionAPI) {
	if (process.env.PI_MEMORY === "0") return;
	const cfg = loadConfig();
	const log = createDebugLog(cfg);
	const client = new MemoryClient(cfg, log);
	const extract = new Extractor(cfg, log, async (input: SaveInput, cwd: string) =>
		client.request<SaveResult>("save", { ...input, cwd }),
	);

	let injected = false;

	// ---- 工具 ----

	pi.registerTool(
		defineTool({
			name: "memory_save",
			label: "Memory Save",
			description: `保存一条跨会话长期记忆。${TYPE_DESC}。仅用于值得长期记住的信息（用户明确要求记住的内容、稳定偏好、可复用经验、项目事实），不要存临时状态、任务细节或代码本身。`,
			parameters: Type.Object({
				content: Type.String({ description: "记忆内容，一句话，自包含、具体" }),
				type: Type.Union(
					[Type.Literal("preference"), Type.Literal("knowledge"), Type.Literal("project")],
					{ description: TYPE_DESC },
				),
				tags: Type.Optional(Type.Array(Type.String(), { description: '可选标签，如 ["npm","调试"]' })),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const r = await client.request<SaveResult>("save", {
					content: params.content,
					type: params.type,
					tags: params.tags ?? [],
					source: "manual",
					cwd: ctx.cwd ?? process.cwd(),
				});
				if (r.status === "duplicate") {
					return {
						content: [
							{ type: "text", text: `已存在近似记忆（相似度 ${(r.similarity ?? 1).toFixed(3)}），跳过保存（id=${shortId(r.id)}）` },
						],
						details: r,
					};
				}
				return {
					content: [{ type: "text", text: `已保存 ${params.type} 记忆（id=${shortId(r.id)}）` }],
					details: r,
				};
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "memory_search",
			label: "Memory Search",
			description:
				"语义检索跨会话长期记忆，返回相关记忆与相似度。回答用户之前聊过、但当前上下文没有的问题时使用；scope=all 同时查全局库与当前项目库。",
			parameters: Type.Object({
				query: Type.String({ description: "检索文本（自然语言）" }),
				scope: Type.Optional(
					Type.Union([Type.Literal("all"), Type.Literal("global"), Type.Literal("project")], {
						description: "检索范围，默认 all",
					}),
				),
				k: Type.Optional(Type.Number({ description: "返回条数，默认 5" })),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const { hits } = await client.request<{ hits: Array<SearchHit> }>("search", {
					query: params.query,
					scope: params.scope ?? "all",
					k: params.k ?? 5,
					cwd: ctx.cwd ?? process.cwd(),
				});
				if (hits.length === 0) {
					return { content: [{ type: "text", text: "没有相关记忆。" }], details: { hits: [] } };
				}
				const lines = hits.map(
					(h) =>
						`[${h.score.toFixed(3)}] (${h.entry.type}, ${fmtDate(h.entry.createdAt)}) ${clipText(h.entry.content, 200)} (id=${shortId(h.id)})`,
				);
				return {
					content: [{ type: "text", text: `找到 ${hits.length} 条相关记忆：\n${lines.join("\n")}` }],
					details: { hits: hits.map((h) => ({ id: h.id, score: h.score, type: h.entry.type })) },
				};
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "memory_forget",
			label: "Memory Forget",
			description: "删除一条长期记忆（按 id 或 id 前 8 位前缀）。用户要求忘掉某事时使用。",
			parameters: Type.Object({
				id: Type.String({ description: "记忆 id 或其前 8 位前缀" }),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const r = await client.request<ForgetResult>("forget", {
					idPrefix: params.id.trim(),
					cwd: ctx.cwd ?? process.cwd(),
				});
				if (r.status === "removed") {
					return {
						content: [{ type: "text", text: `已删除记忆（id=${shortId(r.id ?? "")}, scope=${r.scope}）` }],
						details: r,
					};
				}
				const text =
					r.status === "ambiguous" ? "该前缀命中多条记忆，请用更长的 id。" : "没有找到这条记忆。";
				return { content: [{ type: "text", text }], details: r };
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "memory_list",
			label: "Memory List",
			description: "按时间倒序列出长期记忆（不语义检索，直接读索引）。用户想看看记了什么时使用。",
			parameters: Type.Object({
				scope: Type.Optional(
					Type.Union([Type.Literal("all"), Type.Literal("global"), Type.Literal("project")], {
						description: "范围，默认 all",
					}),
				),
				type: Type.Optional(
					Type.Union(
						[Type.Literal("preference"), Type.Literal("knowledge"), Type.Literal("project")],
						{ description: "按类型过滤" },
					),
				),
				limit: Type.Optional(Type.Number({ description: "返回条数，默认 20" })),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const cwd = ctx.cwd ?? process.cwd();
				const scopes: Array<"global" | "project"> =
					params.scope === undefined || params.scope === "all" ? ["global", "project"] : [params.scope];
				let entries: MemoryEntry[] = [];
				for (const scope of scopes) {
					entries.push(...readIndex(cfg.root, scope, { cwd, dims: cfg.dims, warn: log }));
				}
				if (params.type) entries = entries.filter((e) => e.type === params.type);
				entries.sort((a, b) => b.createdAt - a.createdAt);
				entries = entries.slice(0, params.limit ?? 20);
				if (entries.length === 0) {
					return { content: [{ type: "text", text: "还没有任何记忆。" }], details: { entries: [] } };
				}
				return {
					content: [{ type: "text", text: `共 ${entries.length} 条记忆：\n${entries.map(fmtEntryLine).join("\n")}` }],
					details: { entries: entries.map((e) => ({ id: e.id, type: e.type, source: e.source })) },
				};
			},
		}),
	);

	// ---- 生命周期 ----

	pi.on("session_start", (_event, ctx) => {
		injected = false;
		extract.resetCursor(ctx);
		// 后台预热 worker（拉起子进程 + 加载/下载模型）；失败静默，首次真实使用时再暴露
		client.poke();
	});

	pi.on("before_agent_start", (event, ctx) => {
		if (!cfg.inject || injected) return;
		injected = true;
		try {
			const cwd = ctx.cwd ?? process.cwd();
			const read = (scope: "global" | "project", type?: MemoryType, limit?: number): MemoryEntry[] => {
				let entries = readIndex(cfg.root, scope, { cwd, dims: cfg.dims, warn: log });
				if (type) entries = entries.filter((e) => e.type === type);
				entries.sort((a, b) => b.createdAt - a.createdAt);
				return entries.slice(0, limit ?? 0);
			};
			const prefs = read("global", "preference", cfg.injectPrefs);
			const know = read("global", "knowledge", cfg.injectKnowledge);
			const proj = read("project", undefined, cfg.injectProject);
			if (prefs.length === 0 && know.length === 0 && proj.length === 0) return;
			event.systemPromptOptions.sections["user_memory"] = renderMemorySection(
				[...prefs, ...know],
				proj,
				cwd,
			);
		} catch (err) {
			log("inject_failed", { error: errMessage(err) });
		}
	});

	pi.on("agent_settled", async (_event, ctx) => {
		if (!cfg.extract) return;
		try {
			const saved = await extract.maybeRun(ctx);
			if (saved > 0 && ctx.hasUI) ctx.ui.notify(`pi-memory：沉淀 ${saved} 条记忆`, "info");
		} catch (err) {
			log("extract_failed", { error: errMessage(err) });
		}
	});

	pi.on("session_shutdown", () => {
		client.kill();
	});
}
