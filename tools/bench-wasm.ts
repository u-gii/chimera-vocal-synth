/**
 * Real-time factor of the human voice in the wasm build (as the AudioWorklet runs it), on a
 * fixed utterance. Usage: bun tools/bench-wasm.ts (after `bun run wasm`).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { writeGestures } from "../src/wasm/wire.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const scenario = JSON.parse(readFileSync(`${ROOT}crates/chimera/tests/golden/voice.json`, "utf8")).find((s: { name: string }) => s.name === "speech-male");
const gestures = scenario.ops[0].gestures;
const FS = 48000, SECS = 4, BLOCKS = (FS * SECS) / 128;

const x = new WebAssembly.Instance(new WebAssembly.Module(readFileSync(`${ROOT}src/wasm/chimera.wasm`)), {}).exports as Record<string, (...a: number[]) => number> & { memory: WebAssembly.Memory };
const h = x.engine_new(FS);
writeGestures(x.memory.buffer, x.engine_records(h), gestures, 4000);
x.engine_schedule(h, gestures.length, 0.03, 0);
x.engine_set_gate(h, 1);
const t0 = performance.now();
for (let i = 0; i < BLOCKS; i++) x.engine_render(h, 128);
const wasm = (performance.now() - t0) / 1000;
console.log(`wasm   human male os=2: ${((100 * wasm) / SECS).toFixed(1)}% of one core`);

// The TypeScript engine this replaced measured, on the same speech (Ryzen 7 8745HS, 2026-11-05):
// 29% of a core in Bun (JavaScriptCore) and 46% in Chromium (V8), where the wasm build takes 32%.
