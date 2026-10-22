/**
 * Embedding：transformers.js 本地加载 bge-small-zh-v1.5（512 维，纯本地推理，中文友好）。
 *
 * 懒加载单例：首次 embed 才拉起 ONNX runtime 与模型（会话启动不阻塞；
 * session_start 只做后台预热，失败静默、留待首次真实使用时暴露错误）。
 * 官方 HF 端点失败（网络受限）自动回退 hf-mirror.com 一次。
 */
import * as os from "node:os";
import {
	env as hfEnv,
	pipeline,
	type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { errMessage, type DebugLog, type MemoryConfig } from "./config.ts";

const FALLBACK_ENDPOINT = "https://hf-mirror.com";

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;
let fallbackLogged = false;

function createExtractor(cfg: MemoryConfig, endpoint?: string): Promise<FeatureExtractionPipeline> {
	if (endpoint) hfEnv.remoteHost = endpoint;
	if (process.env.PI_MEMORY_CACHE_DIR) hfEnv.cacheDir = process.env.PI_MEMORY_CACHE_DIR;
	return pipeline("feature-extraction", cfg.model, { dtype: cfg.dtype });
}

export function getExtractor(cfg: MemoryConfig, log: DebugLog): Promise<FeatureExtractionPipeline> {
	if (!extractorPromise) {
		extractorPromise = createExtractor(cfg, cfg.hfEndpoint).catch((err) => {
			// 仅当用户未显式指定端点时才自动回退，避免掩盖配置错误
			if (!cfg.hfEndpoint) {
				if (!fallbackLogged) {
					log("embed_fallback", { reason: errMessage(err), to: FALLBACK_ENDPOINT });
					fallbackLogged = true;
				}
				return createExtractor(cfg, FALLBACK_ENDPOINT);
			}
			throw err;
		});
		// 失败后允许下次重试；成功后保持单例
		extractorPromise.catch(() => {
			extractorPromise = null;
		});
	}
	return extractorPromise;
}

/** 单条文本 → L2 归一化向量（均值池化）。 */
export async function embed(cfg: MemoryConfig, log: DebugLog, text: string): Promise<Float32Array> {
	const extractor = await getExtractor(cfg, log);
	const out = await extractor(text, { pooling: "mean", normalize: true });
	const dims = out.dims[out.dims.length - 1] ?? cfg.dims;
	if (dims !== cfg.dims) {
		throw new Error(
			`embedding 维度 ${dims} 与配置 PI_MEMORY_DIMS=${cfg.dims} 不一致：换模型需同步改 PI_MEMORY_DIMS 并清空旧库`,
		);
	}
	const data = out.data;
	return data instanceof Float32Array ? data : new Float32Array(data);
}
