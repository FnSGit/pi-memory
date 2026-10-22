/**
 * 路径与 sidecar 索引读取：双侧共享（extension 进程 / worker 进程）。
 *
 * extension 进程直接用 readIndex 做 memory_list 与会话注入（纯 fs，
 * 不派生 worker 子进程）；worker 进程的 MemoryStore 用同一套路径规则写库。
 */
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import type { MemoryEntry, MemoryType, Scope } from "./types.ts";
import { MEMORY_TYPES } from "./types.ts";

export interface ScopePaths {
	db: string;
	index: string;
	slug: string;
}

/** cwd → 稳定 slug：可读前缀 + 8 位哈希（防碰撞、限长） */
export function slugFor(cwd: string): string {
	const readable = cwd.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 120);
	const hash = crypto.createHash("md5").update(cwd).digest("hex").slice(0, 8);
	return `${readable || "project"}-${hash}`;
}

export function pathsFor(root: string, scope: Scope, cwd?: string): ScopePaths {
	if (scope === "global") {
		return {
			db: path.join(root, "global.db"),
			index: path.join(root, "global.index.json"),
			slug: "global",
		};
	}
	const slug = slugFor(cwd ?? process.cwd());
	return {
		db: path.join(root, "projects", `${slug}.db`),
		index: path.join(root, "projects", `${slug}.index.json`),
		slug,
	};
}

export function normalizeEntry(meta: Partial<MemoryEntry>, id: string): MemoryEntry {
	const type: MemoryType = MEMORY_TYPES.includes(meta.type as MemoryType) ? (meta.type as MemoryType) : "knowledge";
	return {
		id,
		content: typeof meta.content === "string" ? meta.content : "",
		type,
		tags: Array.isArray(meta.tags) ? meta.tags.filter((t): t is string => typeof t === "string") : [],
		createdAt: typeof meta.createdAt === "number" ? meta.createdAt : 0,
		source: meta.source === "manual" ? "manual" : "auto",
		projectPath: typeof meta.projectPath === "string" ? meta.projectPath : undefined,
	};
}

interface SidecarIndexFile {
	version?: unknown;
	dims?: unknown;
	entries?: unknown;
}

/** 读 sidecar 索引；文件缺失/损坏返回空数组。dims 不匹配视为换过模型，按空处理。 */
export function readIndex(
	root: string,
	scope: Scope,
	opts: { cwd?: string; dims: number; warn?: (event: string, detail: Record<string, unknown>) => void },
): MemoryEntry[] {
	const p = pathsFor(root, scope, opts.cwd);
	let parsed: SidecarIndexFile;
	try {
		parsed = JSON.parse(fs.readFileSync(p.index, "utf-8")) as SidecarIndexFile;
	} catch {
		return [];
	}
	if (parsed.version !== 1 || !Array.isArray(parsed.entries)) return [];
	if (typeof parsed.dims === "number" && parsed.dims !== opts.dims) {
		opts.warn?.("dims_mismatch", { scope, index: p.index, have: parsed.dims, want: opts.dims });
		return [];
	}
	return (parsed.entries as Partial<MemoryEntry>[]).map((e, i) =>
		normalizeEntry(e, typeof e.id === "string" ? e.id : `unknown-${i}`),
	);
}
