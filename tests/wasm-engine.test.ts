/**
 * The wasm build of the Rust engine: no imports, and the same samples as the native build (bit
 * for bit) on the voice scenarios, driven through the gesture-record codec the worklet uses.
 * Needs `bun run wasm` (the wasm build) and cargo (it renders the native reference).
 */
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { writeGestures, CONTROL_KEYS, GESTURE_RECORD } from "../src/wasm/wire.ts";
import type { Gesture } from "../src/wasm/types.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const WASM = `${ROOT}target/wasm32-unknown-unknown/release/chimera_wasm.wasm`;
const SOURCE_KEYS = ["f0", "glide", "pressure", "breathiness", "tension", "fry", "roughness", "vibratoDepth", "vibratoRate", "drift", "voiced", "brightness"];

interface Exports {
  memory: WebAssembly.Memory;
  [k: string]: unknown;
}

function instance(): Exports {
  const mod = new WebAssembly.Module(readFileSync(WASM));
  expect(WebAssembly.Module.imports(mod)).toEqual([]);
  return new WebAssembly.Instance(mod, {}).exports as unknown as Exports;
}

type Op = { at: number; op: string; [k: string]: unknown };
type F = (...a: number[]) => number;

function render(x: Exports, s: { body: string; fsOut: number; samples: number; ops: Op[] }): Float32Array {
  const fn = (name: string) => x[name] as F;
  const h = fn("engine_new")(s.fsOut);
  const [female, child] = { male: [0, 0], female: [1, 0], child: [0, 1] }[s.body] ?? [0, 0];
  if (female || child) fn("engine_set_body")(h, female, child);
  const out = new Float32Array(s.samples);
  const ops = [...s.ops].sort((a, b) => a.at - b.at);
  let i = 0, oi = 0;
  while (i < s.samples) {
    while (oi < ops.length && ops[oi].at <= i) {
      const o = ops[oi++];
      const vals = () => new Float64Array(x.memory.buffer, fn("engine_values")(h), 32).fill(NaN);
      if (o.op === "gate") fn("engine_set_gate")(h, o.level as number);
      else if (o.op === "artic") {
        const v = vals();
        for (const [k, val] of Object.entries(o.values as object)) v[CONTROL_KEYS.indexOf(k as never)] = val as number;
        fn("engine_set_artic")(h, o.tipFree === undefined ? 2 : o.tipFree ? 1 : 0);
      } else if (o.op === "source") {
        const v = vals();
        for (const [k, val] of Object.entries(o.values as object)) v[SOURCE_KEYS.indexOf(k)] = typeof val === "boolean" ? (val ? 1 : 0) : val;
        fn("engine_set_source")(h);
      } else if (o.op === "schedule") {
        const gs = o.gestures as Gesture[];
        const n = writeGestures(x.memory.buffer, fn("engine_records")(h), gs, 4000);
        fn("engine_schedule")(h, n, o.offset as number, 0);
      } else if (o.op === "release") fn("engine_release")(h, o.tag as number);
      else fn("engine_stop_speech")(h);
    }
    const next = oi < ops.length ? Math.max(i + 1, ops[oi].at) : Infinity;
    const m = Math.min(128, s.samples - i, next - i);
    const p = fn("engine_render")(h, m);
    out.set(new Float32Array(x.memory.buffer, p, m), i);
    i += m;
  }
  fn("engine_free")(h);
  return out;
}

describe("wasm engine", () => {
  test.skipIf(!existsSync(WASM))("renders the voice scenarios bit-identically to the native build", () => {
    const proc = Bun.spawnSync(["cargo", "run", "-q", "--release", "-p", "chimera-wasm", "--example", "render_scenarios"], { cwd: ROOT });
    expect(proc.exitCode).toBe(0);
    const scenarios = JSON.parse(readFileSync(`${ROOT}crates/chimera/tests/golden/voice.json`, "utf8"));
    const x = instance();
    for (const s of scenarios) {
      const wasm = render(x, s);
      const native = new Float32Array(readFileSync(`${ROOT}target/native-render/${s.name}.f32`).buffer.slice(0));
      expect(wasm.length).toBe(native.length);
      let diff = 0, first = -1, maxd = 0;
      for (let i = 0; i < wasm.length; i++)
        if (Object.is(wasm[i], native[i]) === false) {
          diff++;
          if (first < 0) first = i;
          maxd = Math.max(maxd, Math.abs(wasm[i] - native[i]));
        }
      if (diff) console.log(`${s.name}: first difference at ${first} (${(first / s.fsOut).toFixed(4)} s): wasm ${wasm[first]} native ${native[first]}; max |Δ| ${maxd}`);
      expect(`${s.name}: ${diff} samples differ`).toBe(`${s.name}: 0 samples differ`);
    }
  }, 120000);
  test("gesture records are 272 bytes", () => {
    expect(GESTURE_RECORD).toBe(272);
    expect(CONTROL_KEYS.length).toBe(22);
  });
});
