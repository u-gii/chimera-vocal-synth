/**
 * Test harness for the wasm engine in bun: loads the build (`bun run wasm`) for the main-thread
 * facade, and runs separate engine instances the way the AudioWorklet does.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { initChimera } from "../src/wasm/chimera.ts";
import { writeGestures, fromUtf8 } from "../src/wasm/wire.ts";
import type { Gesture } from "../src/wasm/types.ts";

export const WASM_PATH = fileURLToPath(new URL("../src/wasm/chimera.wasm", import.meta.url));
const bytes = readFileSync(WASM_PATH);
export const wasmModule = new WebAssembly.Module(bytes);

/** Load the engine for the main-thread facade (src/wasm/engine.ts can be imported after this). */
export function initEngine(): Promise<void> {
  return initChimera(bytes);
}

type Fn = (...a: number[]) => number;
type Exports = { memory: WebAssembly.Memory } & Record<string, Fn>;

/** A human voice through the engine's audio-thread ABI (its own instance, as in the worklet). */
export class WasmVoice {
  private readonly x: Exports;
  private readonly h: number;
  constructor(readonly fs = 48000) {
    this.x = new WebAssembly.Instance(wasmModule, {}).exports as unknown as Exports;
    this.h = this.x.engine_new(fs);
  }
  get clock(): number {
    return this.x.engine_clock(this.h);
  }
  schedule(gestures: readonly Gesture[], lead: number, replace = false): number {
    const n = writeGestures(this.x.memory.buffer, this.x.engine_records(this.h), gestures, 4000);
    return this.x.engine_schedule(this.h, n, lead, replace ? 1 : 0);
  }
  release(tag: number): void {
    this.x.engine_release(this.h, tag);
  }
  stop(): void {
    this.x.engine_stop_speech(this.h);
  }
  setGate(level: number): void {
    this.x.engine_set_gate(this.h, level);
  }
  /** Source values by name (crates/chimera-wasm engine_set_source order). */
  setSource(values: Record<string, number | boolean>): void {
    const keys = ["f0", "glide", "pressure", "breathiness", "tension", "fry", "roughness", "vibratoDepth", "vibratoRate", "drift", "voiced", "brightness"];
    const v = new Float64Array(this.x.memory.buffer, this.x.engine_values(this.h), 32).fill(NaN);
    for (const [k, val] of Object.entries(values)) v[keys.indexOf(k)] = typeof val === "boolean" ? (val ? 1 : 0) : val;
    this.x.engine_set_source(this.h);
  }
  render(n: number): Float32Array {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 128) {
      const m = Math.min(128, n - i);
      out.set(new Float32Array(this.x.memory.buffer, this.x.engine_render(this.h, m), m), i);
    }
    return out;
  }
  /** f0, voicing and the oral tube's areas now (resets the telemetry's RMS measurements). */
  state(): { f0: number; voicing: number; orAreas: Float64Array } {
    const p = this.x.engine_telemetry(this.h);
    const head = new Float64Array(this.x.memory.buffer, p, 16);
    const [nArtic, nPh, nOr] = [head[11], head[12], head[13]];
    const orAreas = new Float64Array(this.x.memory.buffer, p + 8 * (16 + nArtic + nPh), nOr).slice();
    return { f0: head[3], voicing: head[4], orAreas };
  }
  /** The gesture score (times on the voice clock). */
  score(): Gesture[] {
    this.x.engine_score_json(this.h);
    return JSON.parse(fromUtf8(new Uint8Array(this.x.memory.buffer, this.x.chimera_result_ptr(), this.x.chimera_result_len())));
  }
}
