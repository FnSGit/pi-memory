/**
 * pi-memory 配置：全部来自环境变量，缺省值开箱可用。
 *
 *   PI_MEMORY=0                            禁用本扩展
 *   PI_MEMORY_ROOT                         存储根目录，默认 ~/.pi/agent/pi-memory
 *   PI_MEMORY_MODEL                        embedding 模型，默认 Xenova/bge-small-zh-v1.5
 *   PI_MEMORY_DIMS                         向量维度，默认 512（bge-small-zh-v1.5）；换模型必须同步改
 *   PI_MEMORY_DTYPE                        量化档位，默认 q8（~25MB，快）；可换 fp32 提质量
 *   PI_MEMORY_HF_ENDPOINT                  HF 端点覆盖（如 https://hf-mirror.com）；默认官方，
 *                                          官方失败自动回退 hf-mirror 一次；代理用户可用 NODE_USE_ENV_PROXY=1
 *   PI_MEMORY_CACHE_DIR                    模型缓存目录，默认 ~/.cache/huggingface
 *   PI_MEMORY_INJECT=0                     关闭会话开始注入
 *   PI_MEMORY_INJECT_PREFS                 注入偏好条数，默认 10
 *   PI_MEMORY_INJECT_KNOWLEDGE             注入常识条数，默认 6
 *   PI_MEMORY_INJECT_PROJECT               注入项目记忆条数，默认 8
 *   PI_MEMORY_EXTRACT=0                    关闭会话 settle 后自动沉淀
 *   PI_MEMORY_EXTRACT_MIN_GAP_MS           两次沉淀最小间隔，默认 600000（10 分钟）
 *   PI_MEMORY_EXTRACT_MIN_NEW_USER_MSGS    触发沉淀的最少新增用户消息数，默认 2
 *   PI_MEMORY_EXTRACT_MAX_CHARS            送入提取器的对话字符上限，默认 12000
 *   PI_MEMORY_EXTRACT_TIMEOUT_MS           提取 LLM 调用超时，默认 90000
 *   PI_MEMORY_DEBUG=0                      关闭调试日志
 *   PI_MEMORY_DEBUG_LOG                    调试日志路径，默认 ~/.pi/agent/pi-memory-debug.log
 */
import * as os from "node:os";
import * as path from "node:path";
import * as fs from "node:fs";

/** 与 transformers.js 支持的量化档位对齐；非法值回退 q8 */
const DTYPES = new Set(["q8", "auto", "fp32", "fp16", "int8", "uint8", "q4", "bnb4", "q4f16"]);
export type EmbedDType = "q8" | "auto" | "fp32" | "fp16" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16";

export interface MemoryConfig {
	/** 记忆存储根目录 */
	root: string;
	/** 向量维度（bge-small-zh-v1.5 = 512），换模型必须同步改 */
	dims: number;
	/** embedding 模型（transformers.js 仓库名） */
	model: string;
	/** 量化档位：q8 / fp32 / fp16 ... */
	dtype: EmbedDType;
	/** HF 端点覆盖；默认官方，失败自动回退 hf-mirror */
	hfEndpoint?: string;
	/** 会话开始自动注入 */
	inject: boolean;
	injectPrefs: number;
	injectKnowledge: number;
	injectProject: number;
	/** 会话 settle 后 LLM 自动沉淀 */
	extract: boolean;
	extractMinGapMs: number;
	extractMinNewUserMsgs: number;
	extractMaxChars: number;
	extractTimeoutMs: number;
	/** 调试日志 */
	debugLog: string;
}

export type DebugLog = (event: string, detail?: Record<string, unknown>) => void;

function intEnv(name: string, fallback: number, min = 1): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const n = Number.parseInt(raw, 10);
	return Number.isFinite(n) && n >= min ? n : fallback;
}

function boolEnv(name: string, fallback: boolean): boolean {
	const raw = process.env[name];
	if (raw === undefined || raw === "") return fallback;
	return raw !== "0" && raw !== "false";
}

export function loadConfig(): MemoryConfig {
	return {
		root: process.env.PI_MEMORY_ROOT ?? path.join(os.homedir(), ".pi", "agent", "pi-memory"),
		dims: intEnv("PI_MEMORY_DIMS", 512),
		model: process.env.PI_MEMORY_MODEL ?? "Xenova/bge-small-zh-v1.5",
		dtype: DTYPES.has(process.env.PI_MEMORY_DTYPE ?? "")
			? (process.env.PI_MEMORY_DTYPE as EmbedDType)
			: "q8",
		hfEndpoint: process.env.PI_MEMORY_HF_ENDPOINT,
		inject: boolEnv("PI_MEMORY_INJECT", true),
		injectPrefs: intEnv("PI_MEMORY_INJECT_PREFS", 10, 0),
		injectKnowledge: intEnv("PI_MEMORY_INJECT_KNOWLEDGE", 6, 0),
		injectProject: intEnv("PI_MEMORY_INJECT_PROJECT", 8, 0),
		extract: boolEnv("PI_MEMORY_EXTRACT", true),
		extractMinGapMs: intEnv("PI_MEMORY_EXTRACT_MIN_GAP_MS", 600_000, 0),
		extractMinNewUserMsgs: intEnv("PI_MEMORY_EXTRACT_MIN_NEW_USER_MSGS", 2, 0),
		extractMaxChars: intEnv("PI_MEMORY_EXTRACT_MAX_CHARS", 12_000, 500),
		extractTimeoutMs: intEnv("PI_MEMORY_EXTRACT_TIMEOUT_MS", 90_000, 1000),
		debugLog: process.env.PI_MEMORY_DEBUG_LOG ?? path.join(os.homedir(), ".pi", "agent", "pi-memory-debug.log"),
	};
}

export function errMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}

const DEBUG_MAX_BYTES = 256 * 1024;

/** 异常路径低频诊断日志；超限直接截断重写（诊断日志允许丢历史）。 */
export function createDebugLog(cfg: MemoryConfig): DebugLog {
	const enabled = boolEnv("PI_MEMORY_DEBUG", true);
	return (event: string, detail: Record<string, unknown> = {}) => {
		if (!enabled) return;
		try {
			try {
				if (fs.statSync(cfg.debugLog).size > DEBUG_MAX_BYTES) fs.writeFileSync(cfg.debugLog, "");
			} catch {}
			const entry = { t: new Date().toISOString(), pid: process.pid, event, ...detail };
			fs.appendFileSync(cfg.debugLog, `${JSON.stringify(entry)}\n`);
		} catch {}
	};
}
