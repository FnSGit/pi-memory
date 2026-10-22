/**
 * 自动沉淀：agent_settled（自动处理真正结束）后，把本段新增对话交给
 * 当前会话模型提取持久记忆，经 embedding 去重后入库（source=auto）。
 *
 * 节流与安全阀：
 *   - 最小间隔 + 最少新增用户消息数，避免频繁 LLM 调用
 *   - 游标只前进不回退：提取失败不重试同一片段（防循环消耗）
 *   - running 互斥；print/json 模式不跑（一次性会话无沉淀价值）
 *   - LLM 调用带超时；异常只进 debug log
 */
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { MemoryConfig } from "./config.ts";
import type { DebugLog } from "./config.ts";
import type { MemoryType, SaveInput } from "./types.ts";

export const EXTRACT_SYSTEM_PROMPT = `你是跨会话记忆提取器。从对话片段中提取值得长期保存的信息。

分类标准：
- preference：用户的偏好、习惯、固定要求（如回复风格、语言、工具偏好、工作流约定）
- knowledge：可跨任务复用的经验、教训、事实（如踩坑记录、环境配置、某工具的关键行为）
- project：当前项目的稳定事实（架构决策、路径约定、构建/测试命令），仅在对话明确围绕某个项目时提取

规则：
- 只提取持久且可复用的信息；忽略一次性任务细节、临时调试状态、代码内容本身
- 每条 content 自包含、具体，一句话，不超过 100 字，不用代词指代（如"它"）
- 宁缺毋滥：没有确定值得记的就输出空数组
- tags 可选，1-3 个短标签

严格输出 JSON 数组，不要输出任何其他文字：
[{"type":"preference","content":"...","tags":["..."]}]
没有可提取内容时输出：[]`;

const USER_MSG_CAP = 800;
const ASSISTANT_MSG_CAP = 600;
const CONTENT_MAX_CHARS = 400;

interface ExtractItem {
	type?: unknown;
	content?: unknown;
	tags?: unknown;
}

function parseItems(text: string): ExtractItem[] {
	const cleaned = text.replace(/```[a-z]*\n?/gi, "");
	const start = cleaned.indexOf("[");
	const end = cleaned.lastIndexOf("]");
	if (start === -1 || end <= start) return [];
	try {
		const parsed: unknown = JSON.parse(cleaned.slice(start, end + 1));
		return Array.isArray(parsed) ? (parsed as ExtractItem[]) : [];
	} catch {
		return [];
	}
}

export interface ExtractSaveResult {
	status: "saved" | "duplicate";
	id: string;
}

export class Extractor {
	private cursor = 0;
	private lastRunAt = 0;
	private running = false;
	private readonly cfg: MemoryConfig;
	private readonly log: DebugLog;
	private readonly save: (input: SaveInput, cwd: string) => Promise<ExtractSaveResult>;

	constructor(
		cfg: MemoryConfig,
		log: DebugLog,
		save: (input: SaveInput, cwd: string) => Promise<ExtractSaveResult>,
	) {
		this.cfg = cfg;
		this.log = log;
		this.save = save;
	}

	/** 会话开始/替换时把游标对齐到当前分支末尾：只沉淀本会话新增内容 */
	resetCursor(ctx: ExtensionContext): void {
		try {
			this.cursor = ctx.sessionManager.getBranch().length;
		} catch {
			this.cursor = 0;
		}
	}

	/** 返回本次沉淀条数（0 = 未触发或无可沉淀） */
	async maybeRun(ctx: ExtensionContext): Promise<number> {
		if (this.running) return 0;
		const mode = ctx.mode;
		if (mode !== "tui" && mode !== "rpc") return 0;
		const now = Date.now();
		if (now - this.lastRunAt < this.cfg.extractMinGapMs) return 0;

		const branch = ctx.sessionManager.getBranch();
		const excerpt = this.buildExcerpt(branch);
		if (excerpt.userMsgCount < this.cfg.extractMinNewUserMsgs) return 0;

		// 游标先前进：LLM 提取失败不重试同一片段，避免反复消耗
		this.cursor = branch.length;
		this.lastRunAt = now;
		this.running = true;
		try {
			const model = ctx.model;
			if (!model) return 0;
			const response = await ctx.modelRegistry.complete(
				model,
				{
					systemPrompt: EXTRACT_SYSTEM_PROMPT,
					messages: [
						{ role: "user", content: [{ type: "text", text: excerpt.text }], timestamp: Date.now() },
					],
				},
				{ signal: AbortSignal.timeout(this.cfg.extractTimeoutMs) },
			);
			if (response.stopReason === "aborted") return 0;
			const text = response.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("");
			const items = parseItems(text);
			const cwd = ctx.cwd ?? process.cwd();
			let saved = 0;
			for (const item of items) {
				const content = typeof item.content === "string" ? item.content.trim() : "";
				if (!content || content.length > CONTENT_MAX_CHARS) continue;
				if (item.type !== "preference" && item.type !== "knowledge" && item.type !== "project") continue;
				const tags = Array.isArray(item.tags)
					? item.tags.filter((t): t is string => typeof t === "string").slice(0, 3)
					: [];
				try {
					const r = await this.save(
						{ content, type: item.type, tags, source: "auto" },
						cwd,
					);
					if (r.status === "saved") saved++;
				} catch (err) {
					this.log("extract_save_failed", { content: content.slice(0, 50), error: errText(err) });
				}
			}
			this.log("extract_done", { items: items.length, saved, excerptChars: excerpt.text.length });
			return saved;
		} finally {
			this.running = false;
		}
	}

	/** 从游标处向后收集对话摘要；超上限时保最近（倒序收集再反转） */
	private buildExcerpt(branch: unknown[]): { text: string; userMsgCount: number } {
		const parts: string[] = [];
		let userMsgCount = 0;
		let chars = 0;
		for (let i = branch.length - 1; i >= this.cursor && chars < this.cfg.extractMaxChars; i--) {
			const entry = branch[i] as { type?: unknown; message?: unknown } | undefined;
			if (!entry || entry.type !== "message" || !entry.message) continue;
			const msg = entry.message as { role?: unknown; content?: unknown };
			if (msg.role !== "user" && msg.role !== "assistant") continue;
			const content = Array.isArray(msg.content) ? msg.content : [];
			const text = content
				.map((c) =>
					c && typeof c === "object" && (c as { type?: unknown }).type === "text"
						? (c as { text?: unknown }).text
						: undefined,
				)
				.filter((t): t is string => typeof t === "string")
				.join(" ")
				.trim();
			if (!text) continue;
			const cap = msg.role === "user" ? USER_MSG_CAP : ASSISTANT_MSG_CAP;
			const clipped = text.length > cap ? `${text.slice(0, cap)}…` : text;
			parts.push(`${msg.role === "user" ? "用户" : "助手"}: ${clipped}`);
			chars += clipped.length;
			if (msg.role === "user") userMsgCount++;
		}
		return { text: parts.reverse().join("\n\n"), userMsgCount };
	}
}

function errText(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}
