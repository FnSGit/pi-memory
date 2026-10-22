/**
 * pi-memory worker：独立 node 子进程，承载 embedding（transformers.js + ONNX）
 * 与 RuVector 原生库。
 *
 * 为什么拆进程：pi 以 bun 单文件可执行运行，extension 经 jiti 加载；jiti 会把
 * @huggingface/transformers 编译成超长 data:URL，Bun resolver 处理时抛
 * NameTooLong（原生 .node 模块同理不可靠）。plain node 直跑全链路无此问题
 * （test/smoke.ts 验证）。
 *
 * 协议：stdin/stdout 行分隔 JSON。请求 {id, method, params} → 响应
 * {id, ok, result|error}。stderr 仅诊断。方法：
 *   ping                      → {pong, dims, model}
 *   save {content,type,tags,source,cwd} → {status: saved|duplicate, id}
 *   search {query,scope,k,cwd,types?}   → {hits: SearchHit[]}
 *   forget {idPrefix,cwd}               → {status, id?, scope?}
 */
import * as crypto from "node:crypto";
import * as readline from "node:readline";
import { createDebugLog, errMessage, loadConfig } from "./config.ts";
import { embed } from "./embed.ts";
import { MemoryStore } from "./store.ts";
import type { MemoryType } from "./types.ts";

interface WorkerRequest {
	id?: unknown;
	method?: unknown;
	params?: unknown;
}

const cfg = loadConfig();
const log = createDebugLog(cfg);
const store = new MemoryStore(cfg, log);

log("worker_start", { pid: process.pid, model: cfg.model, dims: cfg.dims, root: cfg.root });

function send(msg: unknown): void {
	process.stdout.write(`${JSON.stringify(msg)}\n`);
}

async function handle(method: string, params: Record<string, unknown>): Promise<unknown> {
	switch (method) {
		case "ping":
			return { pong: true, dims: cfg.dims, model: cfg.model };
		case "save": {
			const content = typeof params.content === "string" ? params.content.trim() : "";
			const type = params.type as MemoryType;
			if (!content) throw new Error("save: content 为空");
			if (type !== "preference" && type !== "knowledge" && type !== "project") {
				throw new Error(`save: 非法类型 ${String(params.type)}`);
			}
			const tags = Array.isArray(params.tags) ? params.tags.filter((t): t is string => typeof t === "string").slice(0, 5) : [];
			const source = params.source === "manual" ? "manual" : "auto";
			const cwd = typeof params.cwd === "string" ? params.cwd : process.cwd();

			const vector = await embed(cfg, log, content);
			const dup = await store.findDuplicate(vector, type, cwd);
			if (dup) return { status: "duplicate", id: dup.id, similarity: dup.score };
			const entry = {
				id: crypto.randomUUID(),
				content,
				type,
				tags,
				createdAt: Date.now(),
				source: source as "manual" | "auto",
				...(type === "project" ? { projectPath: cwd } : {}),
			};
			await store.save(entry, vector, cwd);
			return { status: "saved", id: entry.id };
		}
		case "search": {
			const query = typeof params.query === "string" ? params.query : "";
			if (!query) throw new Error("search: query 为空");
			const scope = params.scope === "global" || params.scope === "project" ? params.scope : "all";
			const k = typeof params.k === "number" && params.k > 0 ? Math.min(Math.floor(params.k), 50) : 5;
			const cwd = typeof params.cwd === "string" ? params.cwd : process.cwd();
			const types = Array.isArray(params.types)
				? params.types.filter((t): t is MemoryType => t === "preference" || t === "knowledge" || t === "project")
				: undefined;
			const vector = await embed(cfg, log, query);
			return { hits: await store.search(vector, { scope, k, cwd, types }) };
		}
		case "forget": {
			const idPrefix = typeof params.idPrefix === "string" ? params.idPrefix.trim() : "";
			if (!idPrefix) throw new Error("forget: idPrefix 为空");
			const cwd = typeof params.cwd === "string" ? params.cwd : process.cwd();
			return await store.forget(idPrefix, cwd);
		}
		default:
			throw new Error(`未知方法: ${method}`);
	}
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line: string) => {
	const trimmed = line.trim();
	if (!trimmed) return;
	let req: WorkerRequest;
	try {
		req = JSON.parse(trimmed) as WorkerRequest;
	} catch {
		log("worker_bad_request", { line: trimmed.slice(0, 100) });
		return;
	}
	handle(String(req.method ?? ""), (req.params ?? {}) as Record<string, unknown>)
		.then((result) => send({ id: req.id, ok: true, result }))
		.catch((err) => {
			log("worker_method_failed", { method: String(req.method), error: errMessage(err) });
			send({ id: req.id, ok: false, error: errMessage(err) });
		});
});

process.on("disconnect", () => process.exit(0));
