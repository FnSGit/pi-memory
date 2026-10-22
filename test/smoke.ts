/**
 * 冒烟测试：embedding 加载与中文相似度 → 分库存储 → 语义检索 → 去重 → 列表 → 删除。
 *
 * 运行：npm run smoke（Node >= 24 类型剥离直跑；首次会下载 ~25MB 模型，
 * 网络受限时前置 NODE_USE_ENV_PROXY=1 或设置 PI_MEMORY_HF_ENDPOINT）。
 * 全程使用临时目录，不触碰真实 ~/.pi/agent/pi-memory。
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createDebugLog, loadConfig } from "../src/config.ts";
import { embed } from "../src/embed.ts";
import { MemoryStore } from "../src/store.ts";

function dot(a: Float32Array, b: Float32Array): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

let failed = 0;
function check(name: string, cond: boolean, detail?: string): void {
	console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
	if (!cond) failed++;
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-smoke-"));
const cfg = { ...loadConfig(), root: tmpRoot, extract: false };
const log = createDebugLog(cfg);

console.log(`存储根目录: ${tmpRoot}`);
console.log(`模型: ${cfg.model} (dtype=${cfg.dtype}, dims=${cfg.dims})`);

console.log("\n1. embedding 加载（首次会下载模型，请稍候）...");
const t0 = Date.now();
const v1 = await embed(cfg, log, "我喜欢简洁的回复，先给结论再展开");
const v2 = await embed(cfg, log, "回复尽量简短，先说结论");
const v3 = await embed(cfg, log, "这个项目使用 pnpm 管理依赖，构建用 turbo");
const vSame = await embed(cfg, log, "我喜欢简洁的回复，先给结论再展开");
console.log(`   ok  ${v1.length} 维, 耗时 ${Date.now() - t0}ms`);

check("维度正确", v1.length === cfg.dims, `${v1.length}`);
const simNear = dot(v1, v2);
const simFar = dot(v1, v3);
const simSame = dot(v1, vSame);
console.log(`   同义改写相似度=${simNear.toFixed(3)}  无关文本相似度=${simFar.toFixed(3)}  完全相同=${simSame.toFixed(3)}`);
check("同义改写相似度显著高于无关文本", simNear > simFar + 0.1, `${simNear.toFixed(3)} vs ${simFar.toFixed(3)}`);
check("完全相同文本相似度≈1", simSame > 0.99, simSame.toFixed(3));

console.log("\n2. 分库存储...");
const store = new MemoryStore(cfg, log);
const now = Date.now();
await store.save(
	{ id: "t1-1111", content: "我喜欢简洁的回复，先给结论再展开", type: "preference", tags: ["风格"], createdAt: now, source: "manual" },
	v1,
);
await store.save(
	{ id: "t2-2222", content: "这个项目使用 pnpm 管理依赖，构建用 turbo", type: "project", tags: ["pnpm"], createdAt: now, source: "manual", projectPath: "/tmp/fake-proj" },
	v3,
	"/tmp/fake-proj",
);
check("全局库 1 条", store.count("global") === 1);
check("项目库 1 条", store.count("project", "/tmp/fake-proj") === 1);

console.log("\n3. 语义检索（用同义改写查偏好 + 无关 query 查项目）...");
const hitsPref = await store.search(v2, { scope: "all", k: 5, cwd: "/tmp/fake-proj" });
console.log(`   检索结果: ${hitsPref.map((h) => `${h.id}:${h.score.toFixed(3)}:${h.entry.type}`).join("  ")}`);
check("同义改写召回偏好", hitsPref.length > 0 && hitsPref[0].id === "t1-1111");

const hitsProj = await store.search(v3, { scope: "project", k: 5, cwd: "/tmp/fake-proj" });
check("项目库检索命中", hitsProj.length === 1 && hitsProj[0].id === "t2-2222");

const hitsGlobalOnly = await store.search(v3, { scope: "global", k: 5 });
console.log(`   global 范围查项目内容: ${hitsGlobalOnly.length} 条（物理隔离生效则不应高分命中）`);
check("分库隔离：global 检索不到 project 内容", hitsGlobalOnly.every((h) => h.id !== "t2-2222"));

console.log("\n4. 去重...");
const dup = await store.findDuplicate(v1, "preference");
const dupParaphrase = await store.findDuplicate(v2, "preference");
console.log(`   完全相同命中: ${dup ? `${dup.id} score=${dup.score.toFixed(3)}` : "无"}`);
console.log(`   同义改写命中: ${dupParaphrase ? `${dupParaphrase.id} score=${dupParaphrase.score.toFixed(3)}` : "无"}`);
check("完全相同判定重复", dup !== null);
// 保守方向：同义改写（~0.85）不判重，只拦近似完全重复；分数打印供后续校准 DEDUP_THRESHOLD
check("同义改写不判重（保守）", dupParaphrase === null, `score=${dupParaphrase ? dupParaphrase.score.toFixed(3) : "-"}`);

console.log("\n5. 列表（sidecar，零 embedding）...");
const listed = store.list({ scope: "all", limit: 10, cwd: "/tmp/fake-proj" });
check("列表 2 条", listed.length === 2);

console.log("\n6. 删除...");
const f1 = await store.forget("t1");
const f2 = await store.forget("nope");
console.log(`   t1 → ${f1.status};  nope → ${f2.status}`);
check("删除成功", f1.status === "removed");
check("删除后全局库为空", store.count("global") === 0);

fs.rmSync(tmpRoot, { recursive: true, force: true });
console.log(`\n已清理临时目录 ${tmpRoot}`);
if (failed > 0) {
	console.log(`\nSMOKE FAILED: ${failed} 项未通过`);
	process.exit(1);
}
console.log("\nSMOKE OK");
