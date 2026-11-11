/**
 * tools/musicxml-to-notes.ts, run as a user runs it, on small hand-written scores: the form is
 * sung in play order, words come back together for the dictionary, and words the dictionary
 * can't fit to their notes stop the tool rather than shift every later syllable.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const TOOL = fileURLToPath(new URL("../tools/musicxml-to-notes.ts", import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "musicxml-"));

/** a one-part score in 4/4, one beat per division, at 100 bpm */
function score(measures: string[]): string {
  const body = measures.map((m, i) => `<measure number="${i + 1}">${i ? "" : '<attributes><divisions>1</divisions></attributes><direction><sound tempo="100"/></direction>'}${m}</measure>`).join("");
  return `<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1">${body}</part></score-partwise>`;
}

/** a note: step, beats, then lyric lines as [syllabic, text] (line 1, 2, ...), and extra XML */
function note(step: string, beats: number, lyrics: [string, string][] = [], extra = ""): string {
  const ly = lyrics.map(([s, t], i) => `<lyric number="${i + 1}"><syllabic>${s}</syllabic><text>${t}</text></lyric>`).join("");
  return `<note><pitch><step>${step}</step><octave>4</octave></pitch><duration>${beats}</duration>${extra}${ly}</note>`;
}

function run(xml: string, ...args: string[]): { code: number; out: string; err: string } {
  const path = join(dir, `${Math.random().toString(36).slice(2)}.musicxml`);
  writeFileSync(path, xml);
  const p = Bun.spawnSync(["bun", TOOL, path, "P1", ...args]);
  return { code: p.exitCode, out: p.stdout.toString(), err: p.stderr.toString() };
}

describe("musicxml-to-notes", () => {
  test("repeats and endings are sung in play order, each pass with its own verse", () => {
    const r = run(
      score([
        `<barline location="left"><repeat direction="forward"/></barline>${note("C", 4, [["single", "one"], ["single", "two"]])}`,
        `<barline location="left"><ending number="1" type="start"/></barline>${note("D", 4, [["single", "go"]])}<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline>`,
        `<barline location="left"><ending number="2" type="start"/></barline>${note("E", 4, [["single", "end"]])}<barline location="right"><ending number="2" type="discontinue"/></barline>`,
        `${note("F", 2, [["single", "last"]])}<note><rest/><duration>2</duration></note>`,
      ]),
    );
    expect(r.code).toBe(0);
    expect(JSON.parse(r.out)).toEqual({ bpm: 100, lyrics: "one go two end last", notes: [[60, 4], [62, 4], [60, 4], [64, 4], [65, 2], [0, 2]] });
  });

  test("the same form with its attributes in another order and quoting is sung the same", () => {
    // MuseScore 0.9.5 writes `<ending type="start" number="1">`; XML allows either order and quote
    const r = run(
      score([
        `<barline location="left"><repeat direction='forward'/></barline>${note("C", 4, [["single", "one"], ["single", "two"]])}`,
        `<barline location="left"><ending type="start" number="1"/></barline>${note("D", 4, [["single", "go"]])}<barline location="right"><ending type="stop" number="1"/><repeat times="2" direction="backward"/></barline>`,
        `<barline location="left"><ending type='start' number='2'/></barline>${note("E", 4, [["single", "end"]])}<barline location="right"><ending type="discontinue" number="2"/></barline>`,
        `${note("F", 2, [["single", "last"]])}<note><rest/><duration>2</duration></note>`,
      ]),
    );
    expect(r.code).toBe(0);
    expect(JSON.parse(r.out)).toEqual({ bpm: 100, lyrics: "one go two end last", notes: [[60, 4], [62, 4], [60, 4], [64, 4], [65, 2], [0, 2]] });
  });

  test("a word the score splits is joined for the dictionary, and tied notes are one note", () => {
    const r = run(
      score([
        note("C", 1, [["begin", "to"]]) + note("D", 1, [["middle", "ge"]]) + note("E", 1, [["end", "ther"]]) + note("G", 1, [["single", "we"]]),
        note("G", 2, [["single", "sing"]], '<tie type="start"/>') + note("G", 2, [], '<tie type="stop"/>'),
      ]),
    );
    expect(r.code).toBe(0);
    expect(JSON.parse(r.out)).toMatchObject({ lyrics: "together we sing", notes: [[60, 1], [62, 1], [64, 1], [67, 1], [67, 4]] });
  });

  test("a word with more syllables than notes stops the tool, once per word whatever its case, until respelled", () => {
    const xml = score([note("C", 2, [["single", "Ba"]]) + note("D", 2, [["single", "ba"]])]);
    const bad = run(xml);
    expect(bad.code).toBe(1);
    expect(bad.err.match(/"ba": 2 syllable/gi)).toHaveLength(1);
    const good = run(xml, "--spell", "ba=bah");
    expect(good.code).toBe(0);
    expect(JSON.parse(good.out).lyrics).toBe("bah bah");
  });

  test("a D.S. jump is refused, not sung straight through", () => {
    const r = run(score([note("C", 4, [["single", "one"]]), `<direction><sound dalsegno="1"/></direction>${note("D", 4, [["single", "two"]])}`]));
    expect(r.code).toBe(1);
    expect(r.err).toContain("not supported");
  });
});
