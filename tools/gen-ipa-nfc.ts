/**
 * Unicode NFC data for the Rust port of `tokenizeIpa` (crates/chimera/src/phonetics/ipa.rs), which
 * normalizes its input to NFC as `String.prototype.normalize("NFC")` does, without a Unicode crate.
 * Only what can change the tokenizer's output is kept:
 *   - `ccc`: canonical-ordering rank (order-isomorphic to the combining class) of every code point
 *     with a non-zero combining class, as [first, last, rank] ranges;
 *   - `decomp`: the canonical decomposition of every code point that NFC rewrites on its own
 *     (singletons, non-starter decompositions, composition exclusions), CJK compatibility
 *     ideographs left out;
 *   - `comp`: the primary compositions whose first character is one the tokenizer can emit (an
 *     IPA symbol, an alias source, or a delimiter); every other starter is dropped by the
 *     tokenizer whether or not it composes, so its compositions do not matter.
 * The data comes from the JS engine's own ICU, so it matches what the TypeScript engine did.
 * Usage: bun tools/gen-ipa-nfc.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { initChimera, call } from "../src/wasm/chimera.ts";

// the consonant inventory, from the engine (run `bun run wasm` first)
await initChimera(readFileSync(new URL("../src/wasm/chimera.wasm", import.meta.url)));
const CONSONANT_SYMBOLS = call<{ consonantSymbols: string[] }>("catalog").consonantSymbols;

const VOWEL_CHARS = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɝɚ";
const MULTI = ["tʃ", "dʒ", "ts", "dz", "pf", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"];
const ALIAS_SOURCES = ["t͡ʃ", "d͡ʒ", "t͡s", "d͡z", "ɡ", "ʧ", "ʤ", "r̥", "ɐ̯"];
const DELIMS = "ˈ'ˌ .|,ː:";

const ch = (cp: number) => String.fromCodePoint(cp);
const isSurrogate = (cp: number) => cp >= 0xd800 && cp < 0xe000;
const reorders = (a: string, b: string) => ("x" + a + b).normalize("NFD") !== "x" + a + b;

// non-zero combining classes (decomposable code points are replaced before ordering)
const marks: number[] = [];
for (let cp = 0x80; cp < 0x110000; cp++) {
  if (isSurrogate(cp)) continue;
  const m = ch(cp);
  if (m.normalize("NFD") !== m) continue;
  if (cp === 0x334 || cp === 0x345 || reorders(m, "̴") || reorders("̴", m) || reorders(m, "ͅ") || reorders("ͅ", m)) marks.push(cp);
}
// classes: equal combining class ⇔ neither order is reordered
const classes: number[][] = [];
for (const cp of marks) {
  const c = classes.find((k) => !reorders(ch(k[0]), ch(cp)) && !reorders(ch(cp), ch(k[0])));
  if (c) c.push(cp);
  else classes.push([cp]);
}
classes.sort((a, b) => (reorders(ch(a[0]), ch(b[0])) ? 1 : -1)); // a after b ⇔ ccc(a) > ccc(b)
const rankOf = new Map<number, number>();
classes.forEach((k, r) => k.forEach((cp) => rankOf.set(cp, r + 1)));
const ccc: [number, number, number][] = [];
for (const cp of marks) {
  const r = rankOf.get(cp)!;
  const last = ccc[ccc.length - 1];
  if (last && last[1] === cp - 1 && last[2] === r) last[1] = cp;
  else ccc.push([cp, cp, r]);
}

const decomp: [number, number[]][] = [];
for (let cp = 0; cp < 0x110000; cp++) {
  if (isSurrogate(cp) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f)) continue;
  const s = ch(cp);
  if (s.normalize("NFC") !== s) decomp.push([cp, [...s.normalize("NFD")].map((c) => c.codePointAt(0)!)]);
}

const starters = new Set<string>([...CONSONANT_SYMBOLS, ...VOWEL_CHARS, ...MULTI, ...ALIAS_SOURCES, ...DELIMS].flatMap((s) => [...s]).filter((c) => !rankOf.has(c.codePointAt(0)!)));
const comp: [number, number, number][] = [];
for (const s of starters) {
  for (let cp = 0; cp < 0x110000; cp++) {
    if (isSurrogate(cp) || ch(cp).normalize("NFD") !== ch(cp)) continue; // replaced before composing
    const out = (s + ch(cp)).normalize("NFC");
    if ([...out].length !== 1) continue;
    if (!rankOf.has(cp)) throw new Error(`${s} composes with a starter U+${cp.toString(16)}`);
    comp.push([s.codePointAt(0)!, cp, out.codePointAt(0)!]);
  }
}
// what the Rust side relies on: ç is the only tokenizer symbol with a decomposition, and c composes
// with nothing ordered before the cedilla
const significant = new Set([...CONSONANT_SYMBOLS, ...VOWEL_CHARS, ...MULTI, ...DELIMS].flatMap((s) => [...s]));
for (const s of significant) if (s.normalize("NFD") !== s && s !== "ç") throw new Error(`decomposable symbol ${s}`);
for (const [f, m] of comp) if (f === 0x63 && rankOf.get(m)! < rankOf.get(0x327)!) throw new Error("c composes before the cedilla");
for (const [, d] of decomp) for (const c of d) if (significant.has(ch(c)) && !rankOf.has(c)) console.warn(`decomposition yields symbol ${ch(c)}`);

const path = fileURLToPath(new URL("../crates/chimera/src/phonetics/ipaNfc.generated.json", import.meta.url));
const text = JSON.stringify({ ccc, decomp, comp });
writeFileSync(path, text + "\n");
console.log(`${marks.length} marks in ${classes.length} classes (${ccc.length} ranges), ${decomp.length} decompositions, ${comp.length} compositions, ${(text.length / 1024).toFixed(1)} KB`);
