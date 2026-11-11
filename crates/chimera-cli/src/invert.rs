//! Articulatory inversion of a recording, and copy synthesis from it.
//!
//! Every 10 ms the recording's harmonics (their levels at k·f0) are matched against a codebook of
//! vocal-tract postures: the calibrated vowels of a body and blends between them, each with its
//! radiated-pressure transfer function from the frequency-domain airway model (as `calibrate`
//! uses). Sampling both at the harmonics makes the match work at high f0, where LPC formants lock
//! onto harmonics. Per frame the residual is taken after a level and a spectral tilt are removed
//! (the source's), and a Viterbi pass over each voiced run keeps the tongue from jumping.
//!
//! The copy synthesis drives the real-time voice with the recording's f0, its voicing and the
//! inverted postures, then matches the recording's loudness frame by frame. It has no consonants:
//! unvoiced frames are silent.

use crate::FS;
use chimera::airway::frequency::transfer_function;
use chimera::analysis::fft::{Fft, hann, next_pow2, power_spectrum_db};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::{ArticKey, ArticPatch, BodyMix, HumanArtic, NEUTRAL, rest_tip};
use chimera::math::{clamp, js_max, js_round, log10, log2, pow};
use chimera::phonetics::planner::vowel_artic;
use chimera::voice::human_airway::HumanAirway;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

/// Analysis frames: 40 ms every 10 ms.
pub const HOP_S: f64 = 0.01;
pub const WIN_S: f64 = 0.04;
/// Harmonics are used up to here (Hz).
const F_MAX: f64 = 5000.0;
/// Postures lead the recording's timing by this much (s): the articulators take time to arrive.
const LEAD_S: f64 = 0.025;
/// Voicing and pitch lead it too: with a 20 ms glide and no lead, 15 % of a speaker's voiced
/// frames came out more than a semitone flat (median 6 %) on fast pitch moves, and onsets came
/// out unvoiced for their first frames.
const GATE_LEAD_S: f64 = 0.04;
const F0_LEAD_S: f64 = 0.01;
const GLIDE_S: f64 = 0.005;

/// One analysis frame.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Hz; 0 if unvoiced or too quiet
    pub f0: f64,
    /// mean power (dB)
    pub level_db: f64,
    /// level of harmonic k·f0 (dB), k = 1…
    pub peaks_db: Vec<f64>,
    /// lowest level between harmonics k and k+1 (dB): the noise between them
    pub valleys_db: Vec<f64>,
}

/// The `p` quantile (0..1) of the finite values, NaN if none.
pub fn quantile(v: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let i = clamp(js_round(p * (v.len() - 1) as f64), 0.0, (v.len() - 1) as f64);
    v[i as usize]
}

/// Frame `k` starts at `k·HOP_S` s.
pub fn frames(x: &[f64], fs: f64) -> Vec<Frame> {
    let win = js_round(WIN_S * fs) as usize;
    let hop = js_round(HOP_S * fs) as usize;
    let n = if x.len() >= win { (x.len() - win) / hop + 1 } else { 0 };
    let level: Vec<f64> = (0..n)
        .map(|k| {
            let f = &x[k * hop..k * hop + win];
            10.0 * log10(f.iter().map(|v| v * v).sum::<f64>() / win as f64 + 1e-20)
        })
        .collect();
    let gate = quantile(&level, 0.95) - 35.0;
    let n_fft = next_pow2(2.0 * win as f64);
    let fft = Fft::new(n_fft);
    let window = hann(win);
    let bin_hz = fs / n_fft as f64;
    (0..n)
        .map(|k| {
            let mut fr = Frame { level_db: level[k], ..Default::default() };
            if level[k] < gate {
                return fr;
            }
            let f = &x[k * hop..k * hop + win];
            let p = yin_pitch(f, fs, YinOpts { min_f0: Some(70.0), max_f0: Some(900.0), threshold: Some(0.3) });
            if p.f0 <= 0.0 {
                return fr;
            }
            let spec = power_spectrum_db(f, &fft, &window);
            let band = |lo: f64, hi: f64, max: bool| {
                let a = (lo / bin_hz).ceil() as usize;
                let b = ((hi / bin_hz) as usize).min(spec.len() - 1);
                let it = spec[a.min(b)..=b].iter().copied();
                if max { it.fold(f64::NEG_INFINITY, f64::max) } else { it.fold(f64::INFINITY, f64::min) }
            };
            let k_max = (F_MAX / p.f0) as usize;
            fr.f0 = p.f0;
            fr.peaks_db = (1..=k_max).map(|h| band((h as f64 - 0.3) * p.f0, (h as f64 + 0.3) * p.f0, true)).collect();
            fr.valleys_db = (1..k_max).map(|h| band((h as f64 + 0.3) * p.f0, (h as f64 + 0.7) * p.f0, false)).collect();
            fr
        })
        .collect()
}

/// The articulators the codebook varies, and their calibration ranges (for distances).
const KEYS: [ArticKey; 8] = [
    ArticKey::TbX,
    ArticKey::TbY,
    ArticKey::Jaw,
    ArticKey::LipAp,
    ArticKey::LipProt,
    ArticKey::LipWidth,
    ArticKey::Root,
    ArticKey::Larynx,
];
const KEY_RANGE: [f64; 8] = [3.4, 3.8, 2.25, 1.88, 1.4, 1.0, 1.8, 2.6];
/// Japanese vowels and schwa: their pairs get extra blends (¼, ¾).
const JP: [&str; 6] = ["a", "i", "ɯ", "e", "o", "ə"];

pub struct Entry {
    pub label: String,
    pub patch: ArticPatch,
    key: [f64; 8],
    /// radiated pressure (dB) on the codebook's grid
    db: Vec<f64>,
}

pub struct Codebook {
    pub entries: Vec<Entry>,
    log_lo: f64,
    log_step: f64,
}

const GRID_LO: f64 = 60.0;
const GRID_HI: f64 = 6000.0;
const GRID_N: usize = 500;

impl Codebook {
    /// Entry `e`'s level at `f` (dB), linear in log frequency.
    fn db_at(&self, e: usize, f: f64) -> f64 {
        let db = &self.entries[e].db;
        let x = clamp((log2(f) - self.log_lo) / self.log_step, 0.0, (GRID_N - 1) as f64);
        let i = (x as usize).min(GRID_N - 2);
        let t = x - i as f64;
        db[i] * (1.0 - t) + db[i + 1] * t
    }
}

/// A posture from a vowel patch, with the tip resting behind the teeth as the voice puts it.
fn posture(p: &ArticPatch) -> HumanArtic {
    let mut a = p.apply_to(&NEUTRAL);
    if a.tt_curl <= 0.05 {
        let t = rest_tip(a.jaw, a.tb_x, a.tb_y);
        a.tt_x = t.x;
        a.tt_y = t.y;
    }
    a
}

/// The body's calibrated Japanese vowels and schwa, and every pair blended at ¼, ½ and ¾, with
/// their transfer functions. Only these: with all 23 calibrated vowels a Japanese speaker's
/// frames went to [ø y æ] a third of the time, near-equivalent spectra in postures foreign to
/// the language.
pub fn codebook(mix: BodyMix, parallel: bool) -> Codebook {
    let base: Vec<(String, ArticPatch)> =
        JP.iter().filter_map(|n| vowel_artic(n, mix).map(|p| (n.to_string(), p))).collect();
    let mut list = base.clone();
    for i in 0..base.len() {
        for j in i + 1..base.len() {
            let (a, b) = (&base[i], &base[j]);
            // labelled dominant vowel first
            list.push((format!("{}~{}", a.0, b.0), ArticPatch::blend(&[(0.5, a.1), (0.5, b.1)])));
            list.push((format!("{}~~{}", a.0, b.0), ArticPatch::blend(&[(0.75, a.1), (0.25, b.1)])));
            list.push((format!("{}~~{}", b.0, a.0), ArticPatch::blend(&[(0.25, a.1), (0.75, b.1)])));
        }
    }
    let log_lo = log2(GRID_LO);
    let log_step = (log2(GRID_HI) - log_lo) / (GRID_N - 1) as f64;
    let freqs: Vec<f64> = (0..GRID_N).map(|i| pow(2.0, log_lo + i as f64 * log_step)).collect();
    let body = mix.body();
    let make = |part: &[(String, ArticPatch)]| -> Vec<Entry> {
        let mut air = HumanAirway::new(96000.0, body);
        part.iter()
            .map(|(label, patch)| {
                let a = posture(patch);
                air.apply(&a, 0.0);
                let el = air.net.linear_elements();
                let h = transfer_function(&el, air.pharynx.first_node(), &freqs, &[air.glottis_port]);
                Entry {
                    label: label.clone(),
                    patch: *patch,
                    key: std::array::from_fn(|i| a.get(KEYS[i]) / KEY_RANGE[i]),
                    db: h.pressure_db,
                }
            })
            .collect()
    };
    let entries = if parallel { par_chunks(&list, make) } else { make(&list) };
    Codebook { entries, log_lo, log_step }
}

/// `f` over chunks of `items` on all cores, concatenated in order.
pub fn par_chunks<T: Sync, R: Send>(items: &[T], f: impl Fn(&[T]) -> Vec<R> + Sync) -> Vec<R> {
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
    let chunk = items.len().div_ceil(jobs).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(chunk).map(|part| s.spawn(|| f(part))).collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker")).collect()
    })
}

/// `f` over `items` on all cores, in order.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    par_chunks(items, |part| part.iter().map(&f).collect())
}

pub struct Inversion {
    /// the codebook entry per frame (None where unvoiced)
    pub path: Vec<Option<usize>>,
    /// RMS of the per-frame residual along the path (dB): how well the body's tract explains
    /// the recording's spectra
    pub residual_db: f64,
}

/// Fewest harmonics a frame needs to be inverted.
const MIN_HARMONICS: usize = 3;
/// Viterbi cost of moving the articulators by one calibration range, in dB² of residual.
const SMOOTH: f64 = 40.0;

/// Per-frame residual (dB², mean over harmonics) of each entry after removing level and tilt.
fn local_costs(cb: &Codebook, fr: &Frame) -> Vec<f64> {
    let k = fr.peaks_db.len();
    let xs: Vec<f64> = (1..=k).map(|h| log2(h as f64 * fr.f0)).collect();
    let xm = xs.iter().sum::<f64>() / k as f64;
    let sxx: f64 = xs.iter().map(|x| (x - xm) * (x - xm)).sum();
    (0..cb.entries.len())
        .map(|e| {
            let d: Vec<f64> = (0..k).map(|h| fr.peaks_db[h] - cb.db_at(e, (h + 1) as f64 * fr.f0)).collect();
            let dm = d.iter().sum::<f64>() / k as f64;
            let slope = if sxx > 0.0 { xs.iter().zip(&d).map(|(x, y)| (x - xm) * (y - dm)).sum::<f64>() / sxx } else { 0.0 };
            let r: f64 = xs.iter().zip(&d).map(|(x, y)| pow(y - dm - slope * (x - xm), 2.0)).sum::<f64>() / k as f64;
            // a source tilts −6…−18 dB/octave once radiation is in; more is a vowel being faked
            let out = slope - clamp(slope, -24.0, 0.0);
            r + 2.0 * out * out
        })
        .collect()
}

pub fn invert(cb: &Codebook, fr: &[Frame]) -> Inversion {
    let n_e = cb.entries.len();
    let trans: Vec<f64> = (0..n_e * n_e)
        .map(|ij| {
            let (a, b) = (&cb.entries[ij / n_e].key, &cb.entries[ij % n_e].key);
            SMOOTH * a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>()
        })
        .collect();
    let usable = |f: &Frame| f.f0 > 0.0 && f.peaks_db.len() >= MIN_HARMONICS;
    let mut path = vec![None; fr.len()];
    let (mut sum, mut count) = (0.0, 0usize);
    let mut i = 0;
    while i < fr.len() {
        if !usable(&fr[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < fr.len() && usable(&fr[i]) {
            i += 1;
        }
        let local: Vec<Vec<f64>> = fr[start..i].iter().map(|f| local_costs(cb, f)).collect();
        let mut acc = local[0].clone();
        let mut back: Vec<Vec<usize>> = Vec::with_capacity(local.len());
        back.push(vec![0; n_e]);
        for l in &local[1..] {
            let mut next = vec![0.0; n_e];
            let mut from = vec![0usize; n_e];
            for e in 0..n_e {
                let (mut best, mut arg) = (f64::INFINITY, 0);
                for p in 0..n_e {
                    let c = acc[p] + trans[p * n_e + e];
                    if c < best {
                        best = c;
                        arg = p;
                    }
                }
                next[e] = best + l[e];
                from[e] = arg;
            }
            acc = next;
            back.push(from);
        }
        let mut e = (0..n_e).min_by(|&a, &b| acc[a].total_cmp(&acc[b])).unwrap();
        for t in (0..local.len()).rev() {
            path[start + t] = Some(e);
            sum += local[t][e];
            count += 1;
            e = back[t][e];
        }
    }
    Inversion { path, residual_db: if count > 0 { (sum / count as f64).sqrt() } else { f64::NAN } }
}

/// The recording's frames `range` resynthesised: its f0 and voicing, the inverted postures, the
/// body `mix` and the source `source`. Returns (raw, loudness-matched) at `FS`.
pub fn copy_synth(
    mix: BodyMix,
    source: &SourcePatch,
    cb: &Codebook,
    fr: &[Frame],
    path: &[Option<usize>],
    range: (usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let n_frames = range.1 - range.0;
    let total = js_round((n_frames as f64 * HOP_S + WIN_S) * FS) as usize;
    let mut out = vec![0f32; total];
    let mut voice = HumanVoice::new(FS, mix.body(), 2);
    let mut src = *source;
    src.glide = Some(GLIDE_S);
    voice.set_source(&src);
    let frame_at = |t: f64| -> usize {
        let k = js_round((t - WIN_S / 2.0) / HOP_S);
        range.0 + clamp(k, 0.0, (n_frames - 1) as f64) as usize
    };
    // start in the first voiced posture
    if let Some(e) = path[range.0..range.1].iter().flatten().next() {
        voice.set_articulation(&cb.entries[*e].patch, Some(true));
    }
    let mut f0 = fr[range.0..range.1].iter().map(|f| f.f0).find(|&f| f > 0.0).unwrap_or(200.0);
    let mut last_e = usize::MAX;
    let block = 120;
    let mut pos = 0;
    while pos < total {
        let t = pos as f64 / FS;
        let ka = frame_at(t + LEAD_S);
        let kg = frame_at(t + GATE_LEAD_S);
        let kn = frame_at(t + F0_LEAD_S);
        if let Some(e) = path[ka]
            && e != last_e
        {
            voice.set_articulation(&cb.entries[e].patch, Some(true));
            last_e = e;
        }
        let voiced = fr[kg].f0 > 0.0 && path[kg].is_some();
        voice.set_gate(if voiced { 1.0 } else { 0.0 });
        if fr[kn].f0 > 0.0 {
            f0 = fr[kn].f0;
        }
        voice.set_source(&SourcePatch::f0(f0));
        let len = block.min(total - pos);
        voice.render(&mut out[pos..pos + len]);
        pos += len;
    }
    let matched = match_loudness(&out, &fr[range.0..range.1]);
    (out, matched)
}

/// Scale `x` so its 40 ms level follows the recording's frames (voiced frames exactly; elsewhere
/// never louder than the recording).
fn match_loudness(x: &[f32], rec: &[Frame]) -> Vec<f32> {
    let win = js_round(WIN_S * FS) as usize;
    let hop = js_round(HOP_S * FS) as usize;
    let syn: Vec<f64> = (0..rec.len())
        .map(|k| {
            let a = (k * hop).min(x.len());
            let b = (k * hop + win).min(x.len());
            let p = x[a..b].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / win as f64;
            10.0 * log10(p + 1e-20)
        })
        .collect();
    let mut gain = vec![0.0; rec.len()];
    let mut last = 0.0;
    for k in 0..rec.len() {
        let g = clamp(rec[k].level_db - syn[k], -60.0, 30.0);
        if rec[k].f0 > 0.0 {
            last = g;
            gain[k] = g;
        } else {
            gain[k] = if g < last { g } else { last };
        }
    }
    // smooth over 3 frames, then interpolate between frame centres
    let sm: Vec<f64> = (0..gain.len())
        .map(|k| {
            let a = k.saturating_sub(1);
            let b = (k + 1).min(gain.len() - 1);
            gain[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = (i as f64 - win as f64 / 2.0) / hop as f64;
            let k = clamp(t.floor(), 0.0, (sm.len() - 1) as f64) as usize;
            let k2 = (k + 1).min(sm.len() - 1);
            let w = clamp(t - k as f64, 0.0, 1.0);
            let db = sm[k] * (1.0 - w) + sm[k2] * w;
            (v as f64 * pow(10.0, db / 20.0)) as f32
        })
        .collect()
}

/// Frame-aligned distance between the recording's frames and the copy's (raw) frames: RMS over
/// matched voiced frames of the harmonic levels after removing each frame's mean, and of the
/// between-harmonic noise relative to that mean (dB). Frames the copy leaves unvoiced or off
/// pitch count in `missed` (fraction of the recording's voiced frames).
pub struct Distance {
    pub harmonics_db: f64,
    pub noise_db: f64,
    pub missed: f64,
}

impl Distance {
    pub fn cost(&self) -> f64 {
        self.harmonics_db * self.harmonics_db + 0.5 * self.noise_db * self.noise_db + 200.0 * self.missed
    }
}

pub fn distance(rec: &[Frame], syn: &[Frame]) -> Distance {
    let (mut h, mut nh, mut v, mut nv, mut voiced, mut missed) = (0.0, 0usize, 0.0, 0usize, 0usize, 0usize);
    for (r, s) in rec.iter().zip(syn) {
        if r.f0 <= 0.0 || r.peaks_db.len() < MIN_HARMONICS {
            continue;
        }
        voiced += 1;
        if s.f0 <= 0.0 || (log2(s.f0 / r.f0)).abs() > 1.0 / 12.0 {
            missed += 1;
            continue;
        }
        let k = r.peaks_db.len().min(s.peaks_db.len());
        let rm = r.peaks_db[..k].iter().sum::<f64>() / k as f64;
        let sm = s.peaks_db[..k].iter().sum::<f64>() / k as f64;
        for i in 0..k {
            h += pow((s.peaks_db[i] - sm) - (r.peaks_db[i] - rm), 2.0);
            nh += 1;
        }
        let kv = r.valleys_db.len().min(s.valleys_db.len());
        for i in 0..kv {
            v += pow((s.valleys_db[i] - sm) - (r.valleys_db[i] - rm), 2.0);
            nv += 1;
        }
    }
    Distance {
        harmonics_db: if nh > 0 { (h / nh as f64).sqrt() } else { f64::NAN },
        noise_db: if nv > 0 { (v / nv as f64).sqrt() } else { f64::NAN },
        missed: if voiced > 0 { missed as f64 / voiced as f64 } else { 1.0 },
    }
}

/// The `n`-frame window with the most invertible frames.
pub fn busiest_window(fr: &[Frame], path: &[Option<usize>], n: usize) -> (usize, usize) {
    if fr.len() <= n {
        return (0, fr.len());
    }
    let on: Vec<usize> = path.iter().map(|p| p.is_some() as usize).collect();
    let mut s: usize = on[..n].iter().sum();
    let (mut best, mut at) = (s, 0);
    for i in n..fr.len() {
        s = s + on[i] - on[i - n];
        if s > best {
            best = s;
            at = i + 1 - n;
        }
    }
    (at, at + n)
}

/// The share of voiced frames per base vowel along a path (a blend counts for its first vowel).
pub fn vowel_shares(cb: &Codebook, path: &[Option<usize>]) -> Vec<(String, f64)> {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut n = 0;
    for e in path.iter().flatten() {
        let v = cb.entries[*e].label.split('~').next().unwrap_or("").to_string();
        *counts.entry(v).or_default() += 1;
        n += 1;
    }
    let mut v: Vec<(String, f64)> = counts.into_iter().map(|(k, c)| (k, c as f64 / js_max(1.0, n as f64))).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}
