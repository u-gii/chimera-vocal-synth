//! `chimera compare csd`: sing the English songs of the Children's Song Dataset (Choi et al.,
//! ISMIR 2020; CC BY-NC-SA 4.0, fetch with `tools/fetch-csd.sh`) from their own note and
//! syllable annotations, then measure the synth against the recording note by note.
//!
//! One professional female singer; each song in two keys (`a`, `b`). Per song CSD gives a CSV of
//! notes (start, end, MIDI pitch, syllable as `_`-joined phonemes in CSD's romanisation). We map
//! those phonemes to the engine's IPA, so the synth sings exactly the phonemes the singer sang,
//! with her note timing and pitch (G2P is not under test). The render is laid on the recording's
//! timeline, so the two WAVs line up sample for sample in time.
//!
//! Measures, per note, over the note's steady part (30 % in to 20 % before its end; melisma
//! notes 10 %–10 %):
//!  - pitch: median f0 vs the score, in cents, for singer and synth (YIN)
//!  - vibrato: depth (peak cents, from the detrended contour's RMS) and rate, notes ≥ 0.35 s steady
//!  - aperiodicity: median YIN aperiodicity (breathiness/roughness proxy; 0 = perfectly periodic)
//!  - H1−H2: first minus second harmonic level (dB), the classic voice-quality measure
//!  - voicing: share of steady frames that are voiced
//!  - spectrum: band levels (0–0.5, 0.5–1, 1–2, 2–4, 4–8 kHz) relative to the 0–8 kHz total, and a
//!    note-level mel-cepstral distance (13 coefficients of the note's averaged log-mel spectrum)
//!  - voice onset: first voiced frame relative to the note start, for phrase-initial notes
//!  - timing: the shift that best lines up the two level envelopes, over the whole song
//!
//! "Voiced" means YIN finds a period and the frame is within 35 dB of the signal's loud level.
//!
//! `score.json` (see `score.rs`) folds the measures into one number, how far the synth is from the
//! singer in units of her own song-to-song spread, over training songs and held-out songs; with
//! `--against <earlier score.json>` it also gives the verdict on a change.
//!
//! The band and cepstral measures compare averaged spectra, so they see vowel colour, spectral
//! tilt and noise, not frame-level detail. At this singer's pitch (C4–C6) harmonics are 260 Hz+
//! apart, which is why there are no LPC formants here.

use crate::js::fixed;
use crate::{Ctx, FS, body};
use chimera::analysis::fft::{Fft, hann, next_pow2};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::math::{js_round, midi_to_hz};
use chimera::phonetics::Segment;
use chimera::phonetics::Syllable;
use chimera::phonetics::planner::diphthong_parts;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

mod score;
pub mod tsp;

/// A gap between notes longer than this (s) ends a phrase: the synth stops and breathes.
const PHRASE_GAP: f64 = 0.08;
/// Two notes this close (s) with a vowel-only second syllable on the same vowel are a melisma.
const MELISMA_GAP: f64 = 0.02;
const VIBRATO_CENTS: f64 = 35.0;
const BANDS: [(f64, f64); 5] = [
    (0.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
];
const BAND_NAMES: [&str; 5] = ["0-.5k", ".5-1k", "1-2k", "2-4k", "4-8k"];
const N_MEL: usize = 40;
/// pitch frame hop (s)
const HOP: f64 = 0.005;
/// shortest unbroken voiced run (s) searched for vibrato
const VIB_MIN_RUN: f64 = 0.4;
/// rates (Hz) accepted as vibrato
const VIB_RATE: (f64, f64) = (4.0, 9.0);
/// a note whose planned vowel start is this far (s) from its annotated start is left out
const MAX_MISALIGN: f64 = 0.03;
/// frames this far (dB) below a signal's loud level count as silent
const GATE_DB: f64 = 35.0;
const N_CEP: usize = 13;

/// CSD's phoneme romanisation → engine IPA, and whether it is a vowel.
fn csd_ipa(sym: &str) -> Option<(&'static str, bool)> {
    Some(match sym {
        "a" => ("ɑ", true),
        "ae" => ("æ", true),
        "ai" => ("aɪ", true),
        "ao" => ("ʌ", true),
        "au" => ("aʊ", true),
        "e" => ("ɛ", true),
        "ei" => ("eɪ", true),
        "eo" => ("ə", true),
        "er" => ("ɝ", true),
        "i" => ("ɪ", true),
        "ii" => ("i", true),
        "oi" => ("ɔɪ", true),
        "oo" => ("ɔ", true),
        "ou" => ("oʊ", true),
        "u" => ("ʊ", true),
        "uu" => ("u", true),
        "b" => ("b", false),
        "ch" => ("tʃ", false),
        "d" => ("d", false),
        "dh" => ("ð", false),
        "f" => ("f", false),
        "g" => ("g", false),
        "h" => ("h", false),
        "j" => ("dʒ", false),
        "k" => ("k", false),
        "l" => ("l", false),
        "m" => ("m", false),
        "n" => ("n", false),
        "ng" => ("ŋ", false),
        "p" => ("p", false),
        "r" => ("ɹ", false),
        "s" => ("s", false),
        "sh" => ("ʃ", false),
        "t" => ("t", false),
        "th" => ("θ", false),
        "v" => ("v", false),
        "w" => ("w", false),
        "y" => ("j", false),
        "z" => ("z", false),
        "zh" => ("ʒ", false),
        _ => return None,
    })
}

/// IPA onsets with no voicing of their own: after them, the first voiced frame is the vowel's.
const VOICELESS: [&str; 9] = ["p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ"];

/// What a phrase starts with, which decides what its first voiced frame belongs to.
#[derive(Clone, Copy, PartialEq)]
enum Start {
    Vowel,
    /// a voiceless consonant: the voicing starts with the vowel, whichever convention places it
    Voiceless,
    /// a voiced consonant: the voicing starts with the consonant, which `plan_song` sings before
    /// the note while CSD's note starts mark the start of the syllable
    Voiced,
}

struct Note {
    start: f64,
    end: f64,
    midi: f64,
    /// the CSD syllable as written (`k_y_uu`)
    csd: String,
    syl: Syllable,
    /// first note after a phrase gap (or of the song)
    phrase_start: bool,
    /// end of the previous note (−∞ for the first)
    prev_end: f64,
}

impl Note {
    fn starts_with(&self) -> Start {
        match self.syl.onset.first() {
            None => Start::Vowel,
            Some(c) if VOICELESS.contains(&c.ipa.as_str()) => Start::Voiceless,
            Some(_) => Start::Voiced,
        }
    }
}

/// Read `csv/<id>.csv` into notes with engine syllables, melismas and phrase ends.
fn load_notes(path: &str) -> Result<Vec<Note>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut notes: Vec<Note> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    for (ln, line) in text.lines().enumerate().skip(1) {
        let f: Vec<&str> = line.trim().split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let parse = |s: &str| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| format!("{path}:{}: {e}", ln + 1))
        };
        let (start, end, midi) = (parse(f[0])?, parse(f[1])?, parse(f[2])?);
        let csd = f[3].trim().to_string();
        let mut onset = Vec::new();
        let mut nucleus: Option<Segment> = None;
        let mut coda = Vec::new();
        for p in csd.split('_').filter(|p| !p.is_empty()) {
            let Some((ipa, vowel)) = csd_ipa(p) else {
                unknown.push(format!("{p} (in {csd}, line {})", ln + 1));
                continue;
            };
            match (&nucleus, vowel) {
                (None, true) => nucleus = Some(Segment::new(ipa)),
                (None, false) => onset.push(Segment::new(ipa)),
                // a second vowel in one syllable sings as part of the coda
                (Some(_), _) => coda.push(Segment::new(ipa)),
            }
        }
        let nucleus = nucleus.unwrap_or_else(|| Segment::new("ə"));
        notes.push(Note {
            start,
            end,
            midi,
            csd: csd.clone(),
            syl: Syllable {
                text: csd,
                onset,
                nucleus,
                coda,
                melisma: None,
                word_end: true,
                phrase_end: false,
            },
            phrase_start: false,
            prev_end: f64::NEG_INFINITY,
        });
    }
    if !unknown.is_empty() {
        return Err(format!(
            "{path}: unmapped CSD phonemes: {}",
            unknown.join(", ")
        ));
    }
    let n = notes.len();
    for i in 0..n {
        let gap_after = if i + 1 < n {
            notes[i + 1].start - notes[i].end
        } else {
            f64::INFINITY
        };
        notes[i].syl.phrase_end = gap_after > PHRASE_GAP;
        notes[i].phrase_start = i == 0 || notes[i - 1].syl.phrase_end;
        if i > 0 {
            notes[i].prev_end = notes[i - 1].end;
        }
        if i > 0 && !notes[i].phrase_start {
            let (prev, cur) = (&notes[i - 1], &notes[i]);
            let melisma = cur.syl.onset.is_empty()
                && cur.syl.coda.is_empty()
                && cur.syl.nucleus.ipa == prev.syl.nucleus.ipa
                && cur.start - prev.end < MELISMA_GAP;
            if melisma {
                // plan_song holds the chain's first coda until its last note
                notes[i].syl.melisma = Some(true);
            }
        }
    }
    Ok(notes)
}

/// A render on the recording's timeline.
struct Render {
    out: Vec<f32>,
    /// gestures the score had no room for
    dropped: u64,
    /// planned start of each note's vowel. `plan_song` lays a phrase's segments end to end and
    /// gives each vowel at least 50 ms, so a note too short for its consonants pushes the
    /// following vowels late.
    vowel_t0: Vec<f64>,
}

/// Sing the notes on the recording's timeline: one `plan_song` per phrase, 60 bpm so beats are
/// seconds; a note lasts until the next one starts (its last note, to its own end).
fn render_notes(notes: &[Note], body_name: &str, seconds: f64) -> Result<Render, String> {
    let b = body(body_name)?;
    let mut v = HumanVoice::new(FS, b, 2);
    v.set_source(&SourcePatch {
        f0: Some(midi_to_hz(notes[0].midi)),
        voiced: Some(true),
        pressure: Some(9.0),
        tension: Some(0.55),
        ..Default::default()
    });
    let mut vowel_t0 = Vec::with_capacity(notes.len());
    let mut i = 0;
    while i < notes.len() {
        let mut j = i;
        while !notes[j].syl.phrase_end {
            j += 1;
        }
        let sung: Vec<SungNote> = (i..=j)
            .map(|k| SungNote {
                midi: notes[k].midi,
                beats: if k < j {
                    notes[k + 1].start - notes[k].start
                } else {
                    notes[k].end - notes[k].start
                },
                syl: notes[k].syl.clone(),
            })
            .collect();
        let plan = plan_song(
            &sung,
            &SongOptions {
                body: body_name.into(),
                bpm: 60.0,
                t0: notes[i].start,
                vibrato: VIBRATO_CENTS,
            },
        );
        v.schedule(&plan.gestures);
        // each note sings one vowel segment, two for a diphthong that is not carried on into a
        // melisma (as plan_song builds them); CSD syllables have exactly one vowel each
        let mut vowels = plan.segments.iter().filter(|s| s.kind == "vowel");
        for k in i..=j {
            let next_melisma = k < j && notes[k + 1].syl.melisma == Some(true);
            let parts = if diphthong_parts(&notes[k].syl.nucleus.ipa).is_some() && !next_melisma {
                2
            } else {
                1
            };
            let first = vowels.next().ok_or("plan has fewer vowels than notes")?;
            vowel_t0.push(first.t0);
            for _ in 1..parts {
                vowels.next().ok_or("plan has fewer vowels than notes")?;
            }
        }
        if vowels.next().is_some() {
            return Err("plan has more vowels than notes".into());
        }
        i = j + 1;
    }
    let mut out = vec![0f32; js_round(FS * seconds) as usize];
    for block in out.chunks_mut(128) {
        v.render(block);
    }
    Ok(Render {
        out,
        dropped: v.score.dropped,
        vowel_t0,
    })
}

/// Mono samples and rate of a PCM (16/24/32-bit) or float WAV; channels are averaged.
pub(crate) fn read_wav(path: &str) -> Result<(Vec<f64>, f64), String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let bad = || format!("{path}: not a WAV file this reader understands");
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    // (format tag, channels, rate, bits per sample)
    let mut fmt: Option<(u16, usize, u32, usize)> = None;
    let mut data: Option<&[u8]> = None;
    let mut p = 12;
    while p + 8 <= b.len() {
        let size = u32_at(p + 4) as usize;
        let body = &b[p + 8..(p + 8 + size).min(b.len())];
        match &b[p..p + 4] {
            b"fmt " if body.len() >= 16 => {
                let mut tag = u16_at(p + 8);
                if tag == 0xFFFE && body.len() >= 26 {
                    tag = u16_at(p + 8 + 24);
                }
                fmt = Some((
                    tag,
                    u16_at(p + 10) as usize,
                    u32_at(p + 12),
                    u16_at(p + 22) as usize,
                ));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        p += 8 + size + (size & 1);
    }
    let ((tag, ch, rate, bits), data) = (fmt.ok_or_else(bad)?, data.ok_or_else(bad)?);
    let bytes = bits / 8;
    if ch == 0 || bytes == 0 {
        return Err(bad());
    }
    let sample = |s: &[u8]| -> Option<f64> {
        Some(match (tag, bits) {
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f64 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f64 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
            _ => return None,
        })
    };
    let mut out = Vec::with_capacity(data.len() / (bytes * ch));
    for frame in data.chunks_exact(bytes * ch) {
        let mut s = 0.0;
        for c in frame.chunks_exact(bytes) {
            s += sample(c).ok_or_else(|| format!("{path}: unsupported format {tag}/{bits}-bit"))?;
        }
        out.push(s / ch as f64);
    }
    Ok((out, rate as f64))
}

/// What one signal does on one note.
#[derive(Clone, Default)]
struct NoteMeas {
    /// median f0 vs the score (cents); NaN if no voiced frame
    cents: f64,
    /// (peak cents, Hz); NaN unless periodic vibrato was found (see `vibrato`)
    vib_depth: f64,
    vib_rate: f64,
    /// the note had a voiced run long enough to look for vibrato
    vib_eligible: bool,
    aperiodicity: f64,
    voiced: f64,
    /// dB re the 0–8 kHz total
    bands: [f64; 5],
    /// first harmonic minus second (dB): median over voiced frames, each at its own f0
    h1h2: f64,
    cep: [f64; N_CEP],
    /// first voiced frame relative to the note start (s); NaN if none or not phrase-initial
    onset: f64,
}

fn median(xs: &[f64]) -> f64 {
    let mut s: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(f64::total_cmp);
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        0.5 * (s[m - 1] + s[m])
    }
}

fn mean(xs: &[f64]) -> f64 {
    let s: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        f64::NAN
    } else {
        s.iter().sum::<f64>() / s.len() as f64
    }
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

struct Analyzer {
    fs: f64,
    fft: Fft,
    win: Vec<f64>,
    /// triangular mel filters over the FFT bins, 60 Hz–8 kHz
    mel: Vec<Vec<(usize, f64)>>,
    yin_len: usize,
    yin: YinOpts,
    /// a frame quieter than this (mean square) is unvoiced whatever YIN says: YIN ignores level
    gate: f64,
}

impl Analyzer {
    fn new(x: &[f64], fs: f64, min_hz: f64, max_hz: f64) -> Self {
        let n = next_pow2(0.04 * fs);
        let bin = fs / n as f64;
        let (m0, m1) = (hz_to_mel(60.0), hz_to_mel(8000.0));
        let edge = |k: usize| {
            let m = m0 + (m1 - m0) * k as f64 / (N_MEL + 1) as f64;
            700.0 * (10f64.powf(m / 2595.0) - 1.0)
        };
        let mel = (0..N_MEL)
            .map(|k| {
                let (lo, mid, hi) = (edge(k), edge(k + 1), edge(k + 2));
                (0..=n / 2)
                    .filter_map(|i| {
                        let f = i as f64 * bin;
                        let w = if f > lo && f <= mid {
                            (f - lo) / (mid - lo)
                        } else if f > mid && f < hi {
                            (hi - f) / (hi - mid)
                        } else {
                            0.0
                        };
                        (w > 0.0).then_some((i, w))
                    })
                    .collect()
            })
            .collect();
        Analyzer {
            fs,
            fft: Fft::new(n),
            win: hann(n),
            mel,
            yin_len: js_round(2.5 * fs / min_hz) as usize,
            yin: YinOpts {
                min_f0: Some(min_hz),
                max_f0: Some(max_hz),
                threshold: Some(0.2),
            },
            gate: {
                // GATE_DB below the loud level (95th percentile of 10 ms frames)
                let mut e = envelope_db(x, fs);
                e.sort_by(f64::total_cmp);
                let loud = e.get(e.len() * 95 / 100).copied().unwrap_or(0.0);
                10f64.powf((loud - GATE_DB) / 10.0)
            },
        }
    }

    /// (f0 Hz or 0, aperiodicity) for the frame centred at t, searched within ±7 semitones of
    /// `target` (the score knows the note; unbounded YIN jumps octaves on a weak fundamental)
    fn pitch(&self, x: &[f64], t: f64, target: f64) -> (f64, f64) {
        let c = (t * self.fs) as isize - (self.yin_len / 2) as isize;
        if c < 0 || c as usize + self.yin_len > x.len() {
            return (0.0, 1.0);
        }
        let frame = &x[c as usize..c as usize + self.yin_len];
        let ms = frame.iter().map(|v| v * v).sum::<f64>() / frame.len() as f64;
        let range = 2f64.powf(7.0 / 12.0);
        let opts = YinOpts {
            min_f0: Some(target / range),
            max_f0: Some(target * range),
            ..self.yin
        };
        let p = yin_pitch(frame, self.fs, opts);
        (if ms < self.gate { 0.0 } else { p.f0 }, p.aperiodicity)
    }

    /// Power spectrum (linear, n/2+1 bins) of the frame centred at t, if it fits in x.
    fn power(&self, x: &[f64], t: f64) -> Option<Vec<f64>> {
        let n = self.fft.n;
        let c = (t * self.fs) as isize - (n / 2) as isize;
        if c < 0 || c as usize + n > x.len() {
            return None;
        }
        let mut re: Vec<f64> = (0..n).map(|i| x[c as usize + i] * self.win[i]).collect();
        let mut im = vec![0.0; n];
        self.fft.transform(&mut re, &mut im);
        Some((0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect())
    }

    /// Level (dB) of the harmonic near `hz` in one frame's power spectrum: the strongest bin
    /// within ±3 % (at least ±2 bins), refined by a parabola through its neighbours (dB).
    fn harmonic_db(&self, p: &[f64], hz: f64) -> f64 {
        let bin = self.fs / self.fft.n as f64;
        let half = (0.03 * hz / bin).max(2.0);
        let (lo, hi) = (
            ((hz / bin - half).floor().max(1.0)) as usize,
            ((hz / bin + half).ceil()) as usize,
        );
        if hi + 1 >= p.len() {
            return f64::NAN;
        }
        let k = (lo..=hi).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        let db = |i: usize| 10.0 * (p[i] + 1e-30).log10();
        let (a, b, c) = (db(k - 1), db(k), db(k + 1));
        let den = a - 2.0 * b + c;
        if den < 0.0 {
            b - 0.125 * (a - c) * (a - c) / den
        } else {
            b
        }
    }

    fn note(&self, x: &[f64], n: &Note) -> NoteMeas {
        let d = n.end - n.start;
        let (a, b) = if n.syl.melisma == Some(true) {
            (n.start + 0.1 * d, n.end - 0.1 * d)
        } else {
            (n.start + 0.3 * d, n.end - 0.2 * d)
        };
        let target = midi_to_hz(n.midi);
        // pitch every HOP over the steady part; (frame index, cents) of the voiced frames
        let mut voiced: Vec<(usize, f64)> = Vec::new();
        let mut aper = Vec::new();
        let mut h1h2 = Vec::new();
        let frames = if b >= a {
            ((b - a) / HOP).floor() as usize + 1
        } else {
            0
        };
        for i in 0..frames {
            let t = a + i as f64 * HOP;
            let (f0, ap) = self.pitch(x, t, target);
            aper.push(ap);
            if f0 > 0.0 {
                voiced.push((i, 1200.0 * (f0 / target).log2()));
                // H1−H2 per frame at that frame's own f0 (every other frame)
                if i % 2 == 0
                    && let Some(p) = self.power(x, t)
                {
                    h1h2.push(self.harmonic_db(&p, f0) - self.harmonic_db(&p, 2.0 * f0));
                }
            }
        }
        let cents: Vec<f64> = voiced.iter().map(|v| v.1).collect();
        let (vib_depth, vib_rate, vib_eligible) = vibrato(&voiced);
        let mut m = NoteMeas {
            cents: median(&cents),
            vib_depth,
            vib_rate,
            vib_eligible,
            aperiodicity: median(&aper),
            voiced: if frames > 0 {
                voiced.len() as f64 / frames as f64
            } else {
                f64::NAN
            },
            h1h2: median(&h1h2),
            onset: f64::NAN,
            bands: [f64::NAN; 5],
            cep: [f64::NAN; N_CEP],
        };
        // spectrum, averaged over the steady part
        let mut acc = vec![0.0; self.fft.n / 2 + 1];
        let mut count = 0;
        let mut t = a;
        while t <= b {
            if let Some(p) = self.power(x, t) {
                acc.iter_mut().zip(&p).for_each(|(s, v)| *s += v);
                count += 1;
            }
            t += 0.01;
        }
        if count > 0 {
            let bin = self.fs / self.fft.n as f64;
            let band = |lo: f64, hi: f64| {
                acc.iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        let f = *i as f64 * bin;
                        f >= lo && f < hi
                    })
                    .map(|(_, p)| p)
                    .sum::<f64>()
            };
            let total = band(0.0, 8000.0) + 1e-30;
            for (k, (lo, hi)) in BANDS.iter().enumerate() {
                m.bands[k] = 10.0 * ((band(*lo, *hi) + 1e-30) / total).log10();
            }
            let logmel: Vec<f64> = self
                .mel
                .iter()
                .map(|f| (f.iter().map(|(i, w)| acc[*i] * w).sum::<f64>() + 1e-20).ln())
                .collect();
            // cosine transform scaled as a real cepstrum of ln|X| (½ ln power, ×2 for the
            // two-sided sum), so the distance is on the usual MCD scale; c0 (level) excluded
            for q in 1..=N_CEP {
                let s: f64 = logmel
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        v * (std::f64::consts::PI * q as f64 * (k as f64 + 0.5) / N_MEL as f64)
                            .cos()
                    })
                    .sum();
                m.cep[q - 1] = s / N_MEL as f64;
            }
        }
        // voice onset of a phrase-initial note, searched from after the previous note
        if n.phrase_start {
            let mut t = (n.start - 0.25).max(n.prev_end + 0.03);
            while t < n.start + (0.3f64).min(0.5 * d) {
                if self.pitch(x, t, target).0 > 0.0 {
                    m.onset = t - n.start;
                    break;
                }
                t += HOP;
            }
        }
        m
    }
}

/// Vibrato of a note from its voiced pitch frames (index on the HOP grid, cents): over the
/// longest unbroken voiced run, if it lasts VIB_MIN_RUN, the contour is detrended and its
/// normalised autocorrelation searched for its fundamental period. A peak of at least 0.5 at a
/// rate within VIB_RATE is vibrato: rate from the peak's lag, depth (peak cents) from the residual's RMS (a sine of
/// peak A has RMS A/√2). Returns (depth, rate, eligible); depth and rate are NaN without vibrato.
fn vibrato(voiced: &[(usize, f64)]) -> (f64, f64, bool) {
    let none = (f64::NAN, f64::NAN);
    // longest run of consecutive frame indices
    let (mut best, mut start) = ((0, 0), 0);
    for i in 1..=voiced.len() {
        if i == voiced.len() || voiced[i].0 != voiced[i - 1].0 + 1 {
            if i - start > best.1 - best.0 {
                best = (start, i);
            }
            start = i;
        }
    }
    let run: Vec<f64> = voiced[best.0..best.1].iter().map(|v| v.1).collect();
    let n = run.len();
    if (n as f64) * HOP < VIB_MIN_RUN {
        return (none.0, none.1, false);
    }
    let k = n as f64;
    let (mt, mc) = ((k - 1.0) / 2.0, run.iter().sum::<f64>() / k);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, c) in run.iter().enumerate() {
        sxy += (i as f64 - mt) * (c - mc);
        sxx += (i as f64 - mt) * (i as f64 - mt);
    }
    let slope = sxy / sxx;
    let r: Vec<f64> = run
        .iter()
        .enumerate()
        .map(|(i, c)| c - mc - slope * (i as f64 - mt))
        .collect();
    let energy = r.iter().map(|v| v * v).sum::<f64>() / k;
    if energy <= 0.0 {
        return (none.0, none.1, true);
    }
    // unbiased normalised autocorrelation
    let ac = |lag: usize| {
        (0..n - lag).map(|i| r[i] * r[i + lag]).sum::<f64>() / (n - lag) as f64 / energy
    };
    // the fundamental period is the first local maximum of at least 0.5, searched from the
    // shortest lag the grid resolves up to the slowest rate accepted (a later peak can be a
    // multiple of a faster period); its rate must then fall within VIB_RATE
    let hi = (1.0 / (VIB_RATE.0 * HOP)).ceil() as usize;
    let vals: Vec<f64> = (0..=(hi + 1).min(n - 1))
        .map(|l| if l == 0 { 1.0 } else { ac(l) })
        .collect();
    let peak = (2..vals.len() - 1)
        .find(|&j| vals[j] >= 0.5 && vals[j] >= vals[j - 1] && vals[j] >= vals[j + 1]);
    let Some(j) = peak else {
        return (none.0, none.1, true);
    };
    let (a, b, c) = (vals[j - 1], vals[j], vals[j + 1]);
    let den = a - 2.0 * b + c;
    let shift = if den < 0.0 { 0.5 * (a - c) / den } else { 0.0 };
    let rate = 1.0 / ((j as f64 + shift) * HOP);
    if rate < VIB_RATE.0 || rate > VIB_RATE.1 {
        return (none.0, none.1, true);
    }
    (energy.sqrt() * std::f64::consts::SQRT_2, rate, true)
}

/// Level envelope: RMS in dB (floor −80) of 10 ms frames.
fn envelope_db(x: &[f64], fs: f64) -> Vec<f64> {
    let n = js_round(0.01 * fs) as usize;
    x.chunks(n)
        .map(|c| {
            let ms = c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64;
            (10.0 * (ms + 1e-30).log10()).max(-80.0)
        })
        .collect()
}

/// The shift (s, positive = synth late) within ±0.3 s that best correlates the two level
/// envelopes, and that correlation.
fn envelope_lag(h: &[f64], s: &[f64]) -> (f64, f64) {
    let n = h.len().min(s.len());
    let mut best = (f64::NAN, f64::NEG_INFINITY);
    for lag in -30i64..=30 {
        let pairs: Vec<(f64, f64)> = (0..n as i64)
            .filter_map(|i| {
                let j = i + lag;
                (j >= 0 && (j as usize) < n).then(|| (h[i as usize], s[j as usize]))
            })
            .collect();
        let k = pairs.len() as f64;
        let (mh, ms) = (
            pairs.iter().map(|p| p.0).sum::<f64>() / k,
            pairs.iter().map(|p| p.1).sum::<f64>() / k,
        );
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in &pairs {
            sxy += (a - mh) * (b - ms);
            sxx += (a - mh) * (a - mh);
            syy += (b - ms) * (b - ms);
        }
        let r = sxy / (sxx * syy).sqrt();
        if r > best.1 {
            best = (lag as f64 * 0.01, r);
        }
    }
    best
}

/// `fixed`, with a dash for a missing value
fn fx(x: f64, d: usize) -> String {
    if x.is_finite() {
        fixed(x, d)
    } else {
        "—".into()
    }
}

/// Mel-cepstral distance (dB) between two note cepstra, the usual 10/ln10·√(2Σ(Δc)²).
fn mcd(a: &[f64; N_CEP], b: &[f64; N_CEP]) -> f64 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    10.0 / std::f64::consts::LN_10 * (2.0 * s).sqrt()
}

struct SongResult {
    id: String,
    summary: Value,
    notes: Vec<Value>,
    /// (vowel IPA, mcd, band diffs synth − human)
    vowels: Vec<(String, f64, [f64; 5])>,
    line: String,
    score: score::SongScore,
}

fn r2(x: f64) -> Value {
    if x.is_finite() {
        json!((x * 100.0).round() / 100.0)
    } else {
        Value::Null
    }
}

/// One voice's measures on a note, as the score keeps them (`score::NoteRow`).
fn note_row(m: &NoteMeas) -> score::NoteRow {
    [
        m.bands[0],
        m.bands[1],
        m.bands[2],
        m.bands[3],
        m.bands[4],
        m.h1h2,
        if m.vib_eligible { 1.0 } else { 0.0 },
        m.vib_depth,
        m.vib_rate,
        m.voiced,
        m.onset,
        m.aperiodicity,
        m.cents,
    ]
}

fn r3(x: f64) -> Value {
    if x.is_finite() {
        json!((x * 1000.0).round() / 1000.0)
    } else {
        Value::Null
    }
}

fn compare_song(
    ctx: &Ctx,
    csd: &str,
    out_dir: &str,
    id: &str,
    body_name: &str,
) -> Result<SongResult, String> {
    let notes = load_notes(&format!("{csd}/english/csv/{id}.csv"))?;
    if notes.is_empty() {
        return Err(format!("{id}: no notes"));
    }
    let (human, hfs) = read_wav(&format!("{csd}/english/wav/{id}.wav"))?;
    let seconds = human.len() as f64 / hfs;
    let Render {
        out,
        dropped,
        vowel_t0,
    } = render_notes(&notes, body_name, seconds).map_err(|e| format!("{id}: {e}"))?;
    ctx.write_wav(&format!("{out_dir}/{id}-synth.wav"), &out)?;
    let synth: Vec<f64> = out.iter().map(|&v| v as f64).collect();

    let lo = notes.iter().map(|n| n.midi).fold(f64::INFINITY, f64::min);
    let hi = notes
        .iter()
        .map(|n| n.midi)
        .fold(f64::NEG_INFINITY, f64::max);
    let (min_hz, max_hz) = (midi_to_hz(lo - 7.0), midi_to_hz(hi + 7.0));
    let ha = Analyzer::new(&human, hfs, min_hz, max_hz);
    let sa = Analyzer::new(&synth, FS, min_hz, max_hz);

    let mut rows = Vec::new();
    let mut vowels = Vec::new();
    let mut per: Vec<(NoteMeas, NoteMeas, f64)> = Vec::new();
    // notes measured on both sides; those the planner could not place on time are reported
    // but left out of every statistic (their windows would hold the wrong sound)
    let mut kept: Vec<&Note> = Vec::new();
    // every note's measures on both voices, compared or not, for the score
    let (mut singer_rows, mut synth_rows) = (Vec::new(), Vec::new());
    let (mut starts, mut aligned_all) = (Vec::new(), Vec::new());
    for (n, &vt0) in notes.iter().zip(&vowel_t0) {
        let h = ha.note(&human, n);
        let s = sa.note(&synth, n);
        let d = mcd(&h.cep, &s.cep);
        let bd: [f64; 5] = std::array::from_fn(|k| s.bands[k] - h.bands[k]);
        let aligned = (vt0 - n.start).abs() <= MAX_MISALIGN;
        singer_rows.push(note_row(&h));
        synth_rows.push(note_row(&s));
        starts.push(if !n.phrase_start {
            0
        } else {
            match n.starts_with() {
                Start::Vowel => score::PHRASE_VOWEL,
                Start::Voiceless => score::PHRASE_VOICELESS,
                Start::Voiced => score::PHRASE_VOICED,
            }
        });
        aligned_all.push(aligned);
        rows.push(json!({
            "start": n.start, "end": n.end, "midi": n.midi, "csd": n.csd,
            "ipa": format!("{}{}{}",
                n.syl.onset.iter().map(|s| s.ipa.as_str()).collect::<String>(),
                n.syl.nucleus.ipa,
                n.syl.coda.iter().map(|s| s.ipa.as_str()).collect::<String>()),
            "melisma": n.syl.melisma == Some(true), "phraseStart": n.phrase_start,
            "human": meas_json(&h), "synth": meas_json(&s), "mcd": r2(d),
            "bandDiffDb": bd.iter().map(|v| r2(*v)).collect::<Vec<_>>(),
            "plannedVowelStart": r2(vt0), "aligned": aligned,
        }));
        if aligned {
            vowels.push((n.syl.nucleus.ipa.clone(), d, bd));
            per.push((h, s, d));
            kept.push(n);
        }
    }
    let misaligned = notes.len() - kept.len();
    let compared = per.iter().filter(|p| p.2.is_finite()).count();
    // share of notes long enough to look for vibrato that have it
    let vib_share = |side: usize| {
        let e: Vec<bool> = per
            .iter()
            .map(|p| if side == 0 { &p.0 } else { &p.1 })
            .filter(|m| m.vib_eligible)
            .map(|m| m.vib_depth.is_finite())
            .collect();
        if e.is_empty() {
            f64::NAN
        } else {
            e.iter().filter(|&&v| v).count() as f64 / e.len() as f64
        }
    };

    let col = |f: &dyn Fn(&NoteMeas) -> f64, side: usize| -> Vec<f64> {
        per.iter()
            .map(|p| f(if side == 0 { &p.0 } else { &p.1 }))
            .collect()
    };
    let abs_cents = |side| col(&|m: &NoteMeas| m.cents.abs(), side);
    let vs_human: Vec<f64> = per.iter().map(|p| (p.1.cents - p.0.cents).abs()).collect();
    let mcds: Vec<f64> = per.iter().map(|p| p.2).collect();
    // phrase starts, by what the first voiced frame belongs to
    let onset_of = |side: usize, start: Start| -> Vec<f64> {
        kept.iter()
            .zip(&per)
            .filter(|(n, _)| n.phrase_start && n.starts_with() == start)
            .map(|(_, p)| if side == 0 { p.0.onset } else { p.1.onset })
            .collect()
    };
    let (lag, lag_r) = envelope_lag(&envelope_db(&human, hfs), &envelope_db(&synth, FS));
    let band_mean: Vec<f64> = (0..5)
        .map(|k| {
            mean(
                &per.iter()
                    .map(|p| p.1.bands[k] - p.0.bands[k])
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let summary = json!({
        "id": id, "notes": notes.len(), "notesMisaligned": misaligned, "notesCompared": compared,
        "seconds": r2(seconds), "droppedGestures": dropped,
        "pitchAbsCentsVsScore": {"human": r2(median(&abs_cents(0))), "synth": r2(median(&abs_cents(1)))},
        "pitchAbsCentsSynthVsHuman": r2(median(&vs_human)),
        "vibratoShare": {"human": r2(vib_share(0)), "synth": r2(vib_share(1))},
        "vibratoDepthCents": {"human": r2(median(&col(&|m| m.vib_depth, 0))), "synth": r2(median(&col(&|m| m.vib_depth, 1)))},
        "vibratoRateHz": {"human": r2(median(&col(&|m| m.vib_rate, 0))), "synth": r2(median(&col(&|m| m.vib_rate, 1)))},
        "aperiodicity": {"human": r2(median(&col(&|m| m.aperiodicity, 0))), "synth": r2(median(&col(&|m| m.aperiodicity, 1)))},
        "h1h2Db": {"human": r2(median(&col(&|m| m.h1h2, 0))), "synth": r2(median(&col(&|m| m.h1h2, 1)))},
        "voicedShare": {"human": r2(mean(&col(&|m| m.voiced, 0))), "synth": r2(mean(&col(&|m| m.voiced, 1)))},
        "vowelOnsetSec": {"human": r3(median(&onset_of(0, Start::Vowel))), "synth": r3(median(&onset_of(1, Start::Vowel)))},
        "voicelessOnsetSec": {"human": r3(median(&onset_of(0, Start::Voiceless))), "synth": r3(median(&onset_of(1, Start::Voiceless)))},
        "voicedOnsetSec": {"human": r3(median(&onset_of(0, Start::Voiced))), "synth": r3(median(&onset_of(1, Start::Voiced)))},
        "noteMcdDb": r2(mean(&mcds)),
        "envelopeLagSec": r2(lag), "envelopeCorr": r2(lag_r),
        "bandDiffDb": BAND_NAMES.iter().zip(&band_mean).map(|(n, v)| (n.to_string(), r2(*v))).collect::<serde_json::Map<_, _>>(),
    });
    let song_score = score::SongScore {
        id: id.to_string(),
        starts,
        aligned: aligned_all,
        singer: singer_rows,
        synth: synth_rows,
        dropped,
        mcd: mean(&mcds),
        env_corr: lag_r,
    };
    let line = format!(
        "{id}  {:>3} notes ({misaligned:>2} off-time)  pitch |c| H {:>5} S {:>5}  vib H {:>4}c/{:>3}Hz S {:>4}c/{:>3}Hz  aper H {} S {}  H1-H2 H {:>5} S {:>5}  MCD {:>4}  lag {:+.2}s r {}  bands {}",
        notes.len(),
        fx(median(&abs_cents(0)), 1),
        fx(median(&abs_cents(1)), 1),
        fx(median(&col(&|m| m.vib_depth, 0)), 0),
        fx(median(&col(&|m| m.vib_rate, 0)), 1),
        fx(median(&col(&|m| m.vib_depth, 1)), 0),
        fx(median(&col(&|m| m.vib_rate, 1)), 1),
        fx(median(&col(&|m| m.aperiodicity, 0)), 2),
        fx(median(&col(&|m| m.aperiodicity, 1)), 2),
        fx(median(&col(&|m| m.h1h2, 0)), 1),
        fx(median(&col(&|m| m.h1h2, 1)), 1),
        fx(mean(&mcds), 1),
        lag,
        fx(lag_r, 2),
        band_mean
            .iter()
            .map(|v| format!("{:+.1}", v))
            .collect::<Vec<_>>()
            .join(" "),
    );
    Ok(SongResult {
        id: id.to_string(),
        summary,
        notes: rows,
        vowels,
        line,
        score: song_score,
    })
}

fn meas_json(m: &NoteMeas) -> Value {
    json!({
        "cents": r2(m.cents), "vibDepth": r2(m.vib_depth), "vibRate": r2(m.vib_rate),
        "aperiodicity": r2(m.aperiodicity), "voiced": r2(m.voiced), "onset": r2(m.onset),
        "h1h2": r2(m.h1h2),
        "bandsDb": m.bands.iter().map(|v| r2(*v)).collect::<Vec<_>>(),
    })
}

/// `compare csd [csd-dir] [outdir] [--songs en001a,en002b|all|a|b] [--body female] [--jobs n]
/// [--against earlier/score.json]`
pub fn csd(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut pos: Vec<&str> = Vec::new();
    let mut songs = "a".to_string();
    let mut body_name = "female".to_string();
    let mut jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut against: Option<String> = None;
    let mut it = a.iter();
    while let Some(x) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match x.as_str() {
            "--songs" => songs = val("--songs")?,
            "--body" => body_name = val("--body")?,
            "--jobs" => jobs = val("--jobs")?.parse().map_err(|e| format!("--jobs: {e}"))?,
            "--against" => against = Some(val("--against")?),
            _ => pos.push(x),
        }
    }
    body(&body_name)?;
    let default_dir = format!(
        "{}/datasets/csd/CSD",
        std::env::var("HOME").unwrap_or_default()
    );
    let csd = pos.first().copied().unwrap_or(&default_dir).to_string();
    let out_dir = pos.get(1).copied().unwrap_or("out/csd").to_string();
    let csv_dir = format!("{csd}/english/csv");
    let mut all: Vec<String> = std::fs::read_dir(&csv_dir)
        .map_err(|e| format!("{csv_dir}: {e} (fetch the dataset with tools/fetch-csd.sh)"))?
        .filter_map(|e| {
            e.ok()?
                .file_name()
                .to_str()?
                .strip_suffix(".csv")
                .map(String::from)
        })
        .collect();
    all.sort();
    let ids: Vec<String> = match songs.as_str() {
        "all" => all,
        "a" | "b" => all
            .into_iter()
            .filter(|s| s.ends_with(songs.as_str()))
            .collect(),
        list => list.split(',').map(|s| s.trim().to_string()).collect(),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    println!(
        "CSD {csd}: {} songs, body {body_name}, {jobs} jobs → {out_dir}",
        ids.len()
    );

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<SongResult>> = Mutex::new(Vec::new());
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(id) = ids.get(i) else { break };
                    match compare_song(ctx, &csd, &out_dir, id, &body_name) {
                        Ok(r) => {
                            println!("{}", r.line);
                            results.lock().unwrap().push(r);
                        }
                        Err(e) => {
                            eprintln!("{id}: {e}");
                            errors.lock().unwrap().push(e);
                        }
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    let errors = errors.into_inner().unwrap();
    results.sort_by(|x, y| x.id.cmp(&y.id));
    if results.is_empty() {
        return Err(format!("no song compared:\n{}", errors.join("\n")));
    }

    // per vowel, over every song
    let mut by_vowel: BTreeMap<String, Vec<(f64, [f64; 5])>> = BTreeMap::new();
    for r in &results {
        for (v, d, b) in &r.vowels {
            by_vowel.entry(v.clone()).or_default().push((*d, *b));
        }
    }
    let vowel_rows: Vec<(String, usize, f64, Vec<f64>)> = by_vowel
        .iter()
        .map(|(v, xs)| {
            let d = mean(&xs.iter().map(|x| x.0).collect::<Vec<_>>());
            let b = (0..5)
                .map(|k| mean(&xs.iter().map(|x| x.1[k]).collect::<Vec<_>>()))
                .collect();
            (v.clone(), xs.len(), d, b)
        })
        .collect();
    let overall = |key: &str, side: Option<&str>| -> f64 {
        median(
            &results
                .iter()
                .map(|r| {
                    let v = &r.summary[key];
                    side.map_or(v, |s| &v[s]).as_f64().unwrap_or(f64::NAN)
                })
                .collect::<Vec<_>>(),
        )
    };

    let mut md = String::new();
    md += "# CHIMERA vs the Children's Song Dataset\n\n";
    md += &format!(
        "{} songs, body `{body_name}`. Medians over songs of each song's median (MCD and bands: means). \
         H = the singer, S = the synth. Per-note detail is in `report.json`; renders are `<id>-synth.wav`, \
         on the same timeline as `english/wav/<id>.wav`.\n\n\
         {} of {} notes are compared. {} were left out because the planner could not start their vowel \
         within {} ms of the annotated start (a note too short for its consonants pushes the \
         following vowels late); the rest lack a spectrum on one side.\n\n",
        results.len(),
        results
            .iter()
            .map(|r| r.summary["notesCompared"].as_u64().unwrap_or(0))
            .sum::<u64>(),
        results
            .iter()
            .map(|r| r.summary["notes"].as_u64().unwrap_or(0))
            .sum::<u64>(),
        results
            .iter()
            .map(|r| r.summary["notesMisaligned"].as_u64().unwrap_or(0))
            .sum::<u64>(),
        MAX_MISALIGN * 1000.0,
    );
    let (score_json, score_md, verdict) = score::report(
        &body_name,
        &results.iter().map(|r| &r.score).collect::<Vec<_>>(),
        against.as_deref(),
    )?;
    md += &score_md;
    md += "## Measures\n\n";
    md += "| measure | human | synth |\n|---|---|---|\n";
    for (label, key) in [
        ("pitch error vs score (|cents|)", "pitchAbsCentsVsScore"),
        ("long notes with vibrato (share)", "vibratoShare"),
        ("vibrato depth (peak cents)", "vibratoDepthCents"),
        ("vibrato rate (Hz)", "vibratoRateHz"),
        ("aperiodicity (YIN)", "aperiodicity"),
        ("H1−H2 (dB; higher = breathier, lower = pressed)", "h1h2Db"),
        ("voiced share of steady frames", "voicedShare"),
        (
            "voice onset re note start, vowel-initial phrases (s)",
            "vowelOnsetSec",
        ),
        (
            "voice onset re note start, phrases opening on a voiceless consonant (s)",
            "voicelessOnsetSec",
        ),
        (
            "voice onset re note start, phrases opening on a voiced consonant (s; not scored)",
            "voicedOnsetSec",
        ),
    ] {
        let d = if key.ends_with("OnsetSec") { 3 } else { 2 };
        md += &format!(
            "| {label} | {} | {} |\n",
            fx(overall(key, Some("human")), d),
            fx(overall(key, Some("synth")), d)
        );
    }
    md += &format!(
        "\nPitch, synth vs singer (|cents|): {}. Note-level mel-cepstral distance: {} dB. \
         Level-envelope offset (synth late if positive): {} s, correlation {}.\n\n",
        fx(overall("pitchAbsCentsSynthVsHuman", None), 1),
        fx(overall("noteMcdDb", None), 2),
        fx(overall("envelopeLagSec", None), 2),
        fx(overall("envelopeCorr", None), 2)
    );
    md += "## Spectral balance, synth − singer (dB re the 0–8 kHz level)\n\n";
    md += &format!(
        "| vowel | notes | MCD dB | {} |\n|---|---|---|{}\n",
        BAND_NAMES.join(" | "),
        "---|".repeat(5)
    );
    let all_b: Vec<f64> = (0..5)
        .map(|k| {
            mean(
                &results
                    .iter()
                    .flat_map(|r| r.vowels.iter().map(move |v| v.2[k]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    md += &format!(
        "| **all** | {} | {} | {} |\n",
        results.iter().map(|r| r.vowels.len()).sum::<usize>(),
        fixed(
            mean(
                &results
                    .iter()
                    .flat_map(|r| r.vowels.iter().map(|v| v.1))
                    .collect::<Vec<_>>()
            ),
            2
        ),
        all_b
            .iter()
            .map(|v| format!("{v:+.1}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    for (v, n, d, b) in &vowel_rows {
        md += &format!(
            "| {v} | {n} | {} | {} |\n",
            fixed(*d, 2),
            b.iter()
                .map(|x| format!("{x:+.1}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    md += "\n## Songs\n\n```\n";
    for r in &results {
        md += &r.line;
        md += "\n";
    }
    md += "```\n";
    if !errors.is_empty() {
        md += &format!("\n## Errors\n\n```\n{}\n```\n", errors.join("\n"));
    }
    let report = json!({
        "body": body_name,
        "songs": results.iter().map(|r| json!({"summary": r.summary, "notes": r.notes})).collect::<Vec<_>>(),
        "errors": errors,
    });
    let write = |name: &str, text: String| {
        let p = format!("{out_dir}/{name}");
        std::fs::write(&p, text).map_err(|e| format!("{p}: {e}"))
    };
    write("report.md", md.clone())?;
    write(
        "report.json",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    write(
        "score.json",
        serde_json::to_string_pretty(&score_json).map_err(|e| e.to_string())?,
    )?;
    println!("\n{md}");
    println!("wrote {out_dir}/report.md, {out_dir}/report.json, {out_dir}/score.json");
    if let Some(v) = verdict {
        println!("{v}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 44100.0;

    fn held_note(midi: f64, start: f64, end: f64) -> Note {
        Note {
            start,
            end,
            midi,
            csd: "a".into(),
            syl: Syllable {
                text: "a".into(),
                onset: vec![],
                nucleus: Segment::new("ɑ"),
                coda: vec![],
                melisma: None,
                word_end: true,
                phrase_end: true,
            },
            phrase_start: true,
            prev_end: f64::NEG_INFINITY,
        }
    }

    /// 2 s of harmonics `amps` on `midi`, pitch modulated by `vib(t)` cents
    fn tone(midi: f64, amps: &[f64], vib: impl Fn(f64) -> f64) -> Vec<f64> {
        let f = midi_to_hz(midi);
        let mut phase = 0.0;
        (0..(2.0 * RATE) as usize)
            .map(|i| {
                let t = i as f64 / RATE;
                phase += 2.0 * std::f64::consts::PI * f * 2f64.powf(vib(t) / 1200.0) / RATE;
                0.2 * amps
                    .iter()
                    .enumerate()
                    .map(|(h, a)| a * ((h + 1) as f64 * phase).sin())
                    .sum::<f64>()
            })
            .collect()
    }

    fn measure(x: &[f64], midi: f64) -> NoteMeas {
        let a = Analyzer::new(x, RATE, midi_to_hz(midi - 7.0), midi_to_hz(midi + 7.0));
        a.note(x, &held_note(midi, 0.1, 1.9))
    }

    fn sine_vibrato(rate: f64) -> NoteMeas {
        let x = tone(67.0, &[1.0], |t| {
            50.0 * (2.0 * std::f64::consts::PI * rate * t).sin()
        });
        measure(&x, 67.0)
    }

    #[test]
    fn vibrato_reads_rate_and_depth_of_a_periodic_modulation() {
        for rate in [4.5, 5.5, 8.5] {
            let m = sine_vibrato(rate);
            assert!(m.vib_eligible);
            assert!(
                (m.vib_rate - rate).abs() < 0.3,
                "{rate} Hz: rate {}",
                m.vib_rate
            );
            assert!(
                (m.vib_depth - 50.0).abs() < 8.0,
                "{rate} Hz: depth {}",
                m.vib_depth
            );
        }
    }

    #[test]
    fn periodic_modulation_outside_the_vibrato_range_is_not_vibrato() {
        // a 10 Hz or 40 Hz wobble also correlates at multiples of its period, inside 4–9 Hz
        for rate in [10.0, 12.0, 40.0] {
            let m = sine_vibrato(rate);
            assert!(
                m.vib_depth.is_nan(),
                "{rate} Hz read as {} c at {} Hz",
                m.vib_depth,
                m.vib_rate
            );
        }
    }

    #[test]
    fn vibrato_is_absent_from_a_straight_tone_with_pitch_jitter_or_a_scoop() {
        // ±8 cents of random jitter (a new value every 2.5 ms, interpolated): not vibrato
        let mut seed = 12345u32;
        let jitter: Vec<f64> = (0..1000)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                8.0 * ((seed >> 8) as f64 / (1u32 << 24) as f64 * 2.0 - 1.0)
            })
            .collect();
        let x = tone(67.0, &[1.0], |t| {
            let u = t / 0.0025;
            let (i, f) = (u.floor() as usize, u - u.floor());
            jitter[i] * (1.0 - f) + jitter[i + 1] * f
        });
        let m = measure(&x, 67.0);
        assert!(
            m.vib_eligible && m.vib_depth.is_nan(),
            "{} c at {} Hz",
            m.vib_depth,
            m.vib_rate
        );
        // a slow 60-cent scoop up and back over the note
        let x = tone(67.0, &[1.0], |t| {
            -60.0 * (std::f64::consts::PI * t / 2.0).sin()
        });
        let m = measure(&x, 67.0);
        assert!(
            m.vib_depth.is_nan(),
            "{} c at {} Hz",
            m.vib_depth,
            m.vib_rate
        );
    }

    #[test]
    fn h1h2_reads_the_harmonic_ratio_whatever_the_vibrato() {
        // H2 at half the amplitude of H1: 6.02 dB, with and without a wide vibrato
        for depth in [0.0, 100.0] {
            let x = tone(62.0, &[1.0, 0.5], |t| {
                depth * (2.0 * std::f64::consts::PI * 6.0 * t).sin()
            });
            let m = measure(&x, 62.0);
            assert!(
                (m.h1h2 - 6.02).abs() < 0.7,
                "vibrato {depth} c: H1−H2 {}",
                m.h1h2
            );
        }
    }

    #[test]
    fn a_note_with_no_spectrum_has_no_cepstrum() {
        let x = tone(67.0, &[1.0], |_| 0.0);
        let a = Analyzer::new(&x, RATE, 200.0, 600.0);
        let m = a.note(&x, &held_note(67.0, 5.0, 6.0)); // past the end of the signal
        assert!(m.cep.iter().all(|c| c.is_nan()) && mcd(&m.cep, &m.cep).is_nan());
    }
}
