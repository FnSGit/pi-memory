/**
 * 记忆存储（仅 worker 进程加载）：RuVector VectorDB（向量检索）+
 * JSON sidecar 索引（列表/统计/注入）。
 *
 * 分层按 scope 分库（物理隔离，检索无需过滤——Node 侧 filtered search 不可用）：
 *   global    <root>/global.db                  preference + knowledge
 *   project   <root>/projects/<slug>.db         project（每项目一库）
 *
 * 向量库 metadata 存完整 entry（检索结果自包含）；sidecar 镜像用于
 * memory_list / session 注入（零 embedding 开销）/ 前缀 id 删除。
 * 索引读取带 mtime 新鲜度检查：多会话并行写同一 sidecar 时，本进程
 * 检测到外部修改即重读，缩小丢失更新窗口。
 *
 * score 语义：ruvector Cosine 后端返回距离（1 - 相似度），本类对外统一
 * 转换为相似度（越高越相关）后再排序/判定。
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { VectorDB } from "ruvector";
import { errMessage, type DebugLog, type MemoryConfig } from "./config.ts";
import { normalizeEntry, pathsFor, readIndex, type ScopePaths } from "./paths.ts";
import {
	scopeForType,
	type MemoryEntry,
	type MemoryType,
	type Scope,
	type SearchHit,
} from "./types.ts";

/** ruvector 将 VectorDB 导出为 const（typeof VectorDBWrapper），实例类型需单独取 */
type VectorDbInstance = InstanceType<typeof VectorDB>;

/**
 * 去重判定阈值：余弦相似度 ≥ 此值视为重复。
 * 校准参考（bge-small-zh-v1.5）：完全相同≈1.0，同义改写≈0.85，无关≈0.48。
 * 取 0.90 偏保守：拦住自动沉淀的近似重提取，又不误合并不同记忆
 * （误留重复可手动 forget，误删真记忆不可逆）。
 */
const DEDUP_THRESHOLD = 0.9;

interface SidecarIndex {
	version: 1;
	dims: number;
	entries: MemoryEntry[];
}

function freshIndex(dims: number): SidecarIndex {
	return { version: 1, dims, entries: [] };
}

export class MemoryStore {
	private readonly cfg: MemoryConfig;
	private readonly log: DebugLog;
	private readonly dbs = new Map<string, VectorDbInstance>();
	private readonly indices = new Map<string, SidecarIndex>();
	private readonly indexMtimes = new Map<string, number>();

	constructor(cfg: MemoryConfig, log: DebugLog) {
		this.cfg = cfg;
		this.log = log;
	}

	// ---- 路径与实例 ----

	private paths(scope: Scope, cwd?: string): ScopePaths {
		return pathsFor(this.cfg.root, scope, cwd);
	}

	private db(scope: Scope, cwd?: string): VectorDbInstance {
		const p = this.paths(scope, cwd);
		let db = this.dbs.get(p.slug);
		if (!db) {
			fs.mkdirSync(path.dirname(p.db), { recursive: true });
			db = new VectorDB({
				dimensions: this.cfg.dims,
				storagePath: p.db,
				distanceMetric: "Cosine",
			});
			this.dbs.set(p.slug, db);
		}
		return db;
	}

	private safeMtime(filePath: string): number | null {
		try {
			return fs.statSync(filePath).mtimeMs;
		} catch {
			return null;
		}
	}

	private loadIndex(p: ScopePaths): SidecarIndex {
		try {
			const parsed = JSON.parse(fs.readFileSync(p.index, "utf-8")) as Partial<SidecarIndex>;
			if (parsed.version === 1 && Array.isArray(parsed.entries)) {
				const idx = parsed as SidecarIndex;
				if (idx.dims === this.cfg.dims) return idx;
				// 换过模型：sidecar 与向量库维度不符，旧数据按不可用处理（README 说明重建方式）
				this.log("dims_mismatch", { index: p.index, have: idx.dims, want: this.cfg.dims });
				return freshIndex(this.cfg.dims);
			}
		} catch {
			// 缺失/损坏：按空库起步
		}
		return freshIndex(this.cfg.dims);
	}

	private index(scope: Scope, cwd?: string): SidecarIndex {
		const p = this.paths(scope, cwd);
		const mtime = this.safeMtime(p.index);
		const cachedMtime = this.indexMtimes.get(p.slug);
		const cached = this.indices.get(p.slug);
		if (!cached || cachedMtime === undefined || (mtime !== null && mtime !== cachedMtime)) {
			const idx = this.loadIndex(p);
			this.indices.set(p.slug, idx);
			this.indexMtimes.set(p.slug, mtime ?? -1);
			return idx;
		}
		return cached;
	}

	private writeIndexAtomic(scope: Scope, cwd?: string): void {
		const p = this.paths(scope, cwd);
		fs.mkdirSync(path.dirname(p.index), { recursive: true });
		const tmp = `${p.index}.tmp-${process.pid}`;
		fs.writeFileSync(tmp, JSON.stringify(this.index(scope, cwd)));
		fs.renameSync(tmp, p.index);
		this.indexMtimes.set(p.slug, this.safeMtime(p.index) ?? -1);
	}

	// ---- 写 ----

	async save(entry: MemoryEntry, vector: Float32Array, cwd?: string): Promise<void> {
		const scope = scopeForType(entry.type);
		await this.db(scope, cwd).insert({ id: entry.id, vector, metadata: { ...entry } });
		this.index(scope, cwd).entries.push(entry);
		this.writeIndexAtomic(scope, cwd);
	}

	// ---- 读 ----

	/**
	 * 语义检索。scope=all 时合并 global + 当前项目库，按相似度降序截断。
	 * 多取 3 倍候选做类型过滤，保证过滤后仍有 k 条。
	 */
	async search(
		vector: Float32Array,
		opts: { scope: "all" | Scope; k: number; cwd?: string; types?: MemoryType[] },
	): Promise<SearchHit[]> {
		const scopes: Scope[] = opts.scope === "all" ? ["global", "project"] : [opts.scope];
		const all: SearchHit[] = [];
		for (const scope of scopes) {
			if (this.index(scope, opts.cwd).entries.length === 0) continue;
			const hits = await this.db(scope, opts.cwd).search({ vector, k: Math.max(opts.k * 3, 12) });
			for (const h of hits) {
				const meta = h.metadata as Partial<MemoryEntry> | undefined;
				if (!meta?.content) continue;
				const entry = normalizeEntry(meta, h.id);
				if (opts.types && !opts.types.includes(entry.type)) continue;
				all.push({ id: h.id, score: 1 - h.score, entry });
			}
		}
		all.sort((a, b) => b.score - a.score);
		return all.slice(0, opts.k);
	}

	/** 保存前去重：同 scope 同类型 top1 相似度 ≥ 阈值即视为重复 */
	async findDuplicate(vector: Float32Array, type: MemoryType, cwd?: string): Promise<SearchHit | null> {
		const scope = scopeForType(type);
		if (this.index(scope, cwd).entries.length === 0) return null;
		const hits = await this.db(scope, cwd).search({ vector, k: 1 });
		const top = hits[0];
		if (!top) return null;
		const meta = top.metadata as Partial<MemoryEntry> | undefined;
		if (!meta?.content || meta.type !== type) return null;
		const similarity = 1 - top.score;
		if (similarity >= DEDUP_THRESHOLD) {
			return { id: top.id, score: similarity, entry: normalizeEntry(meta, top.id) };
		}
		return null;
	}

	/** 按 id 或 id 前 8 位前缀删除；前缀跨库命中多条则报 ambiguous */
	async forget(
		idOrPrefix: string,
		cwd?: string,
	): Promise<{ status: "removed" | "not_found" | "ambiguous"; id?: string; scope?: Scope }> {
		const found: Array<{ scope: Scope; entry: MemoryEntry }> = [];
		for (const scope of ["global", "project"] as Scope[]) {
			for (const e of this.index(scope, cwd).entries) {
				if (e.id === idOrPrefix || e.id.startsWith(idOrPrefix)) found.push({ scope, entry: e });
			}
		}
		if (found.length === 0) return { status: "not_found" };
		if (found.length > 1) return { status: "ambiguous" };
		const { scope, entry } = found[0];
		try {
			await this.db(scope, cwd).delete(entry.id);
		} catch (err) {
			// 向量库删除失败不阻断：索引已移除，向量成孤儿只在语义检索中多一条噪声
			this.log("forget_db_delete_failed", { id: entry.id, error: errMessage(err) });
		}
		const idx = this.index(scope, cwd);
		idx.entries = idx.entries.filter((e) => e.id !== entry.id);
		this.writeIndexAtomic(scope, cwd);
		return { status: "removed", id: entry.id, scope };
	}

	/** 最近优先列表（sidecar，零 embedding 开销） */
	list(opts: { scope: "all" | Scope; type?: MemoryType; limit: number; cwd?: string }): MemoryEntry[] {
		const scopes: Scope[] = opts.scope === "all" ? ["global", "project"] : [opts.scope];
		const entries: MemoryEntry[] = [];
		for (const scope of scopes) entries.push(...this.index(scope, opts.cwd).entries);
		const filtered = opts.type ? entries.filter((e) => e.type === opts.type) : entries;
		filtered.sort((a, b) => b.createdAt - a.createdAt);
		return filtered.slice(0, opts.limit);
	}

	count(scope: Scope, cwd?: string): number {
		return this.index(scope, cwd).entries.length;
	}
}
