/**
 * 会话开始注入：全局偏好/常识 + 当前项目记忆，以自定义 prompt section
 * （<user_memory>）注入系统提示。只读 sidecar 索引，不触发 embedding
 * 模型加载——会话启动零开销。
 */
import type { MemoryEntry } from "./types.ts";

const ITEM_MAX_CHARS = 160;

function clip(text: string, max: number): string {
	const one = text.replace(/\s+/g, " ").trim();
	return one.length <= max ? one : `${one.slice(0, max - 1)}…`;
}

/**
 * @param globalEntries 全局库条目（偏好 + 常识，已按类型分组截断）
 * @param projectEntries 当前项目库条目
 */
export function renderMemorySection(globalEntries: MemoryEntry[], projectEntries: MemoryEntry[], cwd: string): string {
	const prefs = globalEntries.filter((e) => e.type === "preference");
	const know = globalEntries.filter((e) => e.type === "knowledge");
	const lines: string[] = ["用户的长期记忆（pi-memory 自动注入，仅供参考；与当前文件、代码等实际情况冲突时以实际为准）："];
	if (prefs.length > 0) {
		lines.push("用户偏好：");
		for (const e of prefs) lines.push(`- ${clip(e.content, ITEM_MAX_CHARS)}`);
	}
	if (know.length > 0) {
		lines.push("经验/常识：");
		for (const e of know) lines.push(`- ${clip(e.content, ITEM_MAX_CHARS)}`);
	}
	if (projectEntries.length > 0) {
		lines.push(`项目记忆（${cwd}）：`);
		for (const e of projectEntries) lines.push(`- ${clip(e.content, ITEM_MAX_CHARS)}`);
	}
	return lines.join("\n");
}
