/**
 * pi-memory 类型与常量：双侧共享（extension 进程 / worker 进程），保持零依赖。
 */

export type MemoryType = "preference" | "knowledge" | "project";
export const MEMORY_TYPES: readonly MemoryType[] = ["preference", "knowledge", "project"];

export type Scope = "global" | "project";

export interface MemoryEntry {
	id: string;
	content: string;
	type: MemoryType;
	tags: string[];
	createdAt: number;
	source: "manual" | "auto";
	/** project 类型记录来源项目路径 */
	projectPath?: string;
}

export interface SaveInput {
	content: string;
	type: MemoryType;
	tags: string[];
	source: "manual" | "auto";
}

export interface SearchHit {
	id: string;
	score: number;
	entry: MemoryEntry;
}

/** 类型 → 存储库：偏好/常识进全局库，项目事实进项目库 */
const TYPE_SCOPE: Record<MemoryType, Scope> = {
	preference: "global",
	knowledge: "global",
	project: "project",
};

export function scopeForType(type: MemoryType): Scope {
	return TYPE_SCOPE[type];
}
