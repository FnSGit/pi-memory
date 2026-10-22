/**
 * MemoryClient：extension 进程侧的 worker 子进程管理 + JSON-RPC 客户端。
 *
 * - 懒派生：首个向量操作（save/search/forget）才 spawn worker；注入与
 *   memory_list 走 sidecar 直读，不触发子进程
 * - 冷启动（首次 spawn，可能触发模型下载）给长超时；稳态短超时
 * - worker 意外退出：拒绝在途请求，下次调用自动重启
 * - session_shutdown 时 kill（幂等）
 */
import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as url from "node:url";
import { errMessage, type DebugLog, type MemoryConfig } from "./config.ts";
import type { SearchHit } from "./types.ts";

export interface SaveResult {
	status: "saved" | "duplicate";
	id: string;
	similarity?: number;
}

export interface ForgetResult {
	status: "removed" | "not_found" | "ambiguous";
	id?: string;
	scope?: string;
}

export interface SearchOptions {
	scope: "all" | "global" | "project";
	k: number;
	cwd: string;
}

const STEADY_TIMEOUT_MS = 60_000;
const COLD_TIMEOUT_MS = 600_000;
const MAX_STDERR_LOG = 2000;

interface Pending {
	resolve: (value: unknown) => void;
	reject: (err: Error) => void;
	timer: ReturnType<typeof setTimeout>;
	method: string;
}

export class MemoryClient {
	private readonly cfg: MemoryConfig;
	private readonly log: DebugLog;
	private proc: ChildProcess | null = null;
	private pending = new Map<string, Pending>();
	private buffer = "";
	private seq = 0;
	private killed = false;

	constructor(cfg: MemoryConfig, log: DebugLog) {
		this.cfg = cfg;
		this.log = log;
	}

	private workerPath(): string {
		if (process.env.PI_MEMORY_WORKER) return process.env.PI_MEMORY_WORKER;
		// jiti 提供 import.meta.url；失败则要求显式 PI_MEMORY_WORKER
		const here = url.fileURLToPath(new URL(".", import.meta.url));
		return path.join(here, "worker.ts");
	}

	private spawnWorker(): ChildProcess {
		const workerPath = this.workerPath();
		const nodeBin = process.env.PI_MEMORY_NODE_BIN ?? "node";
		const proc = spawn(nodeBin, [workerPath], {
			stdio: ["pipe", "pipe", "pipe"],
			env: process.env,
		});
		this.log("worker_spawn", { pid: proc.pid, nodeBin, workerPath });
		proc.stdout?.setEncoding("utf-8");
		proc.stdout?.on("data", (chunk: string) => this.onData(chunk));
		proc.stderr?.setEncoding("utf-8");
		proc.stderr?.on("data", (chunk: string) => {
			this.log("worker_stderr", { text: chunk.slice(0, MAX_STDERR_LOG) });
		});
		proc.on("exit", (code, signal) => {
			this.log("worker_exit", { code, signal });
			const pending = this.pending;
			this.pending = new Map();
			this.proc = null;
			this.buffer = "";
			for (const p of pending.values()) {
				clearTimeout(p.timer);
				p.reject(new Error(`pi-memory worker 意外退出 (code=${code}, signal=${signal})，于 ${p.method} 期间`));
			}
		});
		return proc;
	}

	private ensureProc(): ChildProcess {
		if (!this.proc) {
			if (this.killed) throw new Error("pi-memory client 已关闭");
			this.proc = this.spawnWorker();
		}
		return this.proc;
	}

	private onData(chunk: string): void {
		this.buffer += chunk;
		for (;;) {
			const nl = this.buffer.indexOf("\n");
			if (nl === -1) break;
			const line = this.buffer.slice(0, nl).trim();
			this.buffer = this.buffer.slice(nl + 1);
			if (!line) continue;
			let msg: { id?: unknown; ok?: unknown; result?: unknown; error?: unknown };
			try {
				msg = JSON.parse(line) as typeof msg;
			} catch {
				this.log("worker_bad_response", { line: line.slice(0, 100) });
				continue;
			}
			const p = this.pending.get(String(msg.id));
			if (!p) continue;
			this.pending.delete(String(msg.id));
			clearTimeout(p.timer);
			if (msg.ok === true) p.resolve(msg.result);
			else p.reject(new Error(String(msg.error ?? "worker 返回未知错误")));
		}
	}

	async request<T>(method: string, params: Record<string, unknown>): Promise<T> {
		const proc = this.ensureProc();
		const cold = this.pending.size === 0 && this.seq === 0;
		const id = `r${++this.seq}`;
		const timeout = cold ? COLD_TIMEOUT_MS : STEADY_TIMEOUT_MS;
		return await new Promise<T>((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`pi-memory ${method} 超时（${timeout / 1000}s）`));
			}, timeout);
			this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, timer, method });
			proc.stdin?.write(`${JSON.stringify({ id, method, params })}\n`, (err) => {
				if (err) {
					this.pending.delete(id);
					clearTimeout(timer);
					reject(new Error(`pi-memory worker 通信失败: ${errMessage(err)}`));
				}
			});
		});
	}

	/** 后台健康检查：预热 worker（如 session_start 时），失败静默 */
	poke(): void {
		this.request<{ pong: boolean }>("ping", {}).catch((err) => {
			this.log("poke_failed", { error: errMessage(err) });
		});
	}

	kill(): void {
		this.killed = true;
		if (!this.proc) return;
		const proc = this.proc;
		this.proc = null;
		const pending = this.pending;
		this.pending = new Map();
		for (const p of pending.values()) {
			clearTimeout(p.timer);
			p.reject(new Error("pi-memory client 已关闭"));
		}
		try {
			proc.stdin?.end();
			proc.kill();
		} catch (err) {
			this.log("worker_kill_failed", { error: errMessage(err) });
		}
	}
}

/** sidecar 索引文件是否存在（用于注入/列表前的快速判断，可选） */
export function indexExists(root: string, indexFile: string): boolean {
	return fs.existsSync(path.join(root, indexFile));
}
