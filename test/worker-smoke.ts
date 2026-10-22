/**
 * worker/client 协议冒烟测试：spawn 真实 worker 子进程（plain node + strip-types），
 * 走 save / search / forget 全链路。模型已缓存时秒级完成。
 * 运行：node test/worker-smoke.ts
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createDebugLog, loadConfig } from "../src/config.ts";
import { MemoryClient } from "../src/client.ts";
import type { SaveResult, ForgetResult } from "../src/client.ts";
import type { SearchHit } from "../src/types.ts";

let failed = 0;
function check(name: string, cond: boolean, detail?: string): void {
	console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
	if (!cond) failed++;
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-worker-"));
process.env.PI_MEMORY_ROOT = tmpRoot; // worker 继承本进程 env
const cfg = { ...loadConfig(), root: tmpRoot };
const log = createDebugLog(cfg);
const client = new MemoryClient(cfg, log);

console.log(`存储根目录: ${tmpRoot}`);

try {
	console.log("\n1. spawn worker + ping（首次含模型加载）...");
	const t0 = Date.now();
	const pong = await client.request<{ pong: boolean; dims: number; model: string }>("ping", {});
	console.log(`   ok  ${Date.now() - t0}ms  dims=${pong.dims} model=${pong.model}`);
	check("ping 返回 dims", pong.pong === true && pong.dims === cfg.dims);

	console.log("\n2. save（全局偏好 + 项目事实）...");
	const t1 = Date.now();
	const r1 = await client.request<SaveResult>("save", {
		content: "我喜欢简洁的回复，先给结论",
		type: "preference",
		tags: ["风格"],
		source: "manual",
		cwd: process.cwd(),
	});
	console.log(`   save1: ${r1.status} ${Date.now() - t1}ms`);
	check("save 偏好", r1.status === "saved");
	const r2 = await client.request<SaveResult>("save", {
		content: "pi-memory 测试项目使用 vitest",
		type: "project",
		tags: [],
		source: "auto",
		cwd: tmpRoot,
	});
	check("save 项目事实", r2.status === "saved");

	console.log("\n3. save 去重（同文本应判 duplicate）...");
	const r3 = await client.request<SaveResult>("save", {
		content: "我喜欢简洁的回复，先给结论",
		type: "preference",
		tags: [],
		source: "manual",
		cwd: process.cwd(),
	});
	check("重复保存判定 duplicate", r3.status === "duplicate", `sim=${r3.similarity?.toFixed(3)}`);

	console.log("\n4. search...");
	const { hits } = await client.request<{ hits: SearchHit[] }>("search", {
		query: "回复风格要求",
		scope: "all",
		k: 5,
		cwd: tmpRoot,
	});
	console.log(`   hits: ${hits.map((h) => `${h.id.slice(0, 8)}:${h.score.toFixed(3)}:${h.entry.type}`).join("  ")}`);
	check("语义检索命中偏好", hits.length > 0 && hits[0].entry.type === "preference");

	console.log("\n5. forget（前 8 位前缀）...");
	const f = await client.request<ForgetResult>("forget", { idPrefix: r1.id.slice(0, 8), cwd: process.cwd() });
	check("删除成功", f.status === "removed");

	const { hits: after } = await client.request<{ hits: SearchHit[] }>("search", {
		query: "我喜欢简洁的回复",
		scope: "all",
		k: 5,
		cwd: process.cwd(),
	});
	check("删除后检索不到", after.every((h) => h.id !== r1.id), `剩余 ${after.length} 条`);
} finally {
	client.kill();
	fs.rmSync(tmpRoot, { recursive: true, force: true });
}

console.log(`\n已清理临时目录 ${tmpRoot}`);
if (failed > 0) {
	console.log(`\nWORKER SMOKE FAILED: ${failed} 项未通过`);
	process.exit(1);
}
console.log("\nWORKER SMOKE OK");
