//! `chimera fit voice`: estimate a body and a voice source from a recording, by articulatory
//! inversion and copy synthesis (see `invert.rs`).
//!
//! 1. Body: on a grid of the female and child axes, the body whose あいうえお has the
//!    recording's median formants (F3 and F4 weighing most: the tract's length).
//! 2. Source: breathiness, tension, brightness, roughness and lung pressure by Nelder–Mead, so
//!    the copy synthesis (the
//!    recording's f0, voicing and inverted postures, through the real-time voice) matches the
//!    recording frame by frame: harmonic levels and the noise between them.
//!
//! Both run on the 10 s of the recording with the most voice. The outputs are the copy synthesis
//! of the whole recording with the fitted voice and with the nearest preset, an A/B file
//! (recording, fitted, preset) and the fitted voice singing あいうえお.

use crate::invert::{
    Codebook, Frame, HOP_S, codebook, copy_synth, distance, frames, invert, par_map, quantile, vowel_shares,
};
use crate::compare::read_wav;
use crate::js::fixed;
use crate::{Ctx, FS, arg, wav_name};
use chimera::analysis::fft::{Fft, hann, next_pow2, power_spectrum_db};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::BodyMix;
use chimera::math::{clamp, js_max, js_round, log10, pow};
use chimera::phonetics::planner::vowel_artic;
use chimera::util::optimize::{NelderMeadOpts, nelder_mead};
use chimera::voice::human_voice::{DEFAULT_SOURCE, HumanVoice, SourcePatch};
use serde::Serialize;

/// Long-term spectrum bands, log-spaced between these (Hz). They start above most voices' f0:
/// a band below the lowest harmonic holds only room noise in a recording and nothing in a
/// synthesis (150–208 Hz read +4 dB against −13 dB on a 212–393 Hz voice), so it measured the room.
const BAND_LO: f64 = 250.0;
const BAND_HI: f64 = 7500.0;
const N_BANDS: usize = 10;
/// Speed of sound (cm/s), as the airway's.
const C: f64 = 35000.0;
/// Frames the fit works on (10 s).
const FIT_FRAMES: usize = 1000;

/// Summary measurements of a voice, for the report.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    /// f0 of the voiced frames (Hz): 10th, 50th and 90th percentile
    pub f0_low: f64,
    pub f0_median: f64,
    pub f0_high: f64,
    /// median F1…F4 over the voiced frames (LPC, Hz); NaN when too few frames had one
    pub formants: [f64; 4],
    /// the uniform tube with the same formant spacing (cm), from F2…F4
    pub tract_length_cm: f64,
    /// long-term spectrum's least-squares slope (dB per octave), 250–7500 Hz
    pub tilt_db_per_octave: f64,
    /// median YIN aperiodicity of the voiced frames (0 periodic … 1 noise)
    pub aperiodicity: f64,
    /// median level of the first harmonic over the second (dB): high = breathy/lax, low = pressed
    pub h1_h2_db: f64,
    pub voiced_frames: usize,
}

/// Measure a mono signal at rate `fs` (40 ms frames every 10 ms within 30 dB of the loud ones).
pub fn analyze(x: &[f64], fs: f64) -> Features {
    let win = js_round(0.04 * fs) as usize;
    let hop = js_round(0.01 * fs) as usize;
    let n_frames = if x.len() >= win { (x.len() - win) / hop + 1 } else { 0 };
    let level: Vec<f64> = (0..n_frames)
        .map(|k| {
            let f = &x[k * hop..k * hop + win];
            10.0 * log10(f.iter().map(|v| v * v).sum::<f64>() / win as f64 + 1e-20)
        })
        .collect();
    let gate = quantile(&level, 0.95) - 30.0;
    let n_fft = next_pow2(win as f64);
    let fft = Fft::new(n_fft);
    let window = hann(win);
    let bin_hz = fs / n_fft as f64;
    let dec = js_max(1.0, js_round(fs / 11025.0)) as usize;
    let lpc_len = win * 5 / 8;
    let (mut f0s, mut aps, mut h12) = (Vec::new(), Vec::new(), Vec::new());
    let mut fm: [Vec<f64>; 4] = Default::default();
    let mut power = vec![0.0; n_fft / 2 + 1];
    for k in 0..n_frames {
        if level[k] < gate {
            continue;
        }
        let f = &x[k * hop..k * hop + win];
        let p = yin_pitch(f, fs, YinOpts { min_f0: Some(70.0), max_f0: Some(900.0), threshold: Some(0.3) });
        if p.f0 <= 0.0 {
            continue;
        }
        f0s.push(p.f0);
        aps.push(p.aperiodicity);
        let spec = power_spectrum_db(f, &fft, &window);
        for (acc, db) in power.iter_mut().zip(&spec) {
            *acc += pow(10.0, db / 10.0);
        }
        let peak = |hz: f64| {
            let lo = (hz * 0.8 / bin_hz) as usize;
            let hi = ((hz * 1.2 / bin_hz) as usize).min(spec.len() - 1);
            spec[lo..=hi].iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        h12.push(peak(p.f0) - peak(2.0 * p.f0));
        let mid = &f[(win - lpc_len) / 2..][..lpc_len];
        let ds: Vec<f64> = mid.chunks_exact(dec).map(|c| c.iter().sum::<f64>() / dec as f64).collect();
        let found = estimate_formants(
            &ds,
            fs / dec as f64,
            FormantOpts { order: Some(12), min_freq: Some(150.0), max_bw: Some(500.0), ..Default::default() },
        );
        for (i, fr) in found.iter().take(4).enumerate() {
            fm[i].push(fr.freq);
        }
    }
    let voiced = f0s.len();
    let formants: [f64; 4] =
        std::array::from_fn(|i| if fm[i].len() * 4 >= voiced.max(1) { quantile(&fm[i], 0.5) } else { f64::NAN });
    // a tube closed at the glottis: F_i = (2i−1)·ΔF/2, least squares over F2…F4; L = c/(2ΔF)
    let (mut num, mut den) = (0.0, 0.0);
    for (i, f) in formants.iter().enumerate().skip(1) {
        if f.is_finite() {
            let k = (2 * i + 1) as f64 / 2.0;
            num += f * k;
            den += k * k;
        }
    }
    let tract_length_cm = if den > 0.0 { C / (2.0 * num / den) } else { f64::NAN };
    let octaves_per_band = log10(BAND_HI / BAND_LO) / log10(2.0) / N_BANDS as f64;
    let bands: Vec<(f64, f64)> = (0..N_BANDS)
        .filter_map(|b| {
            let lo = BAND_LO * pow(BAND_HI / BAND_LO, b as f64 / N_BANDS as f64);
            let hi = BAND_LO * pow(BAND_HI / BAND_LO, (b + 1) as f64 / N_BANDS as f64);
            let bins: Vec<f64> = (0..power.len())
                .filter(|&k| (k as f64 * bin_hz) >= lo && (k as f64 * bin_hz) < hi)
                .map(|k| power[k])
                .collect();
            (voiced > 0 && !bins.is_empty())
                .then(|| ((b as f64 + 0.5) * octaves_per_band, 10.0 * log10(bins.iter().sum::<f64>() / bins.len() as f64 + 1e-30)))
        })
        .collect();
    let n = bands.len() as f64;
    let xm = bands.iter().map(|b| b.0).sum::<f64>() / n;
    let ym = bands.iter().map(|b| b.1).sum::<f64>() / n;
    let sxy: f64 = bands.iter().map(|(x, y)| (x - xm) * (y - ym)).sum();
    let sxx: f64 = bands.iter().map(|(x, _)| (x - xm) * (x - xm)).sum();
    Features {
        f0_low: quantile(&f0s, 0.1),
        f0_median: quantile(&f0s, 0.5),
        f0_high: quantile(&f0s, 0.9),
        formants,
        tract_length_cm,
        tilt_db_per_octave: if sxx > 0.0 { sxy / sxx } else { f64::NAN },
        aperiodicity: quantile(&aps, 0.5),
        h1_h2_db: quantile(&h12, 0.5),
        voiced_frames: voiced,
    }
}

/// The estimated voice: a body and the source's controls (all 0..1 but the lung pressure).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Voice {
    pub female: f64,
    pub child: f64,
    pub breathiness: f64,
    pub tension: f64,
    pub brightness: f64,
    /// jitter, shimmer and (above 0.5) period doubling
    pub roughness: f64,
    /// lung pressure (cmH₂O)
    pub pressure: f64,
}

/// The lung pressures the fit may use (cmH₂O): soft speech to loud.
const PRESSURE: (f64, f64) = (4.0, 14.0);

impl Voice {
    /// A body with the default source.
    pub fn body(female: f64, child: f64) -> Voice {
        Voice {
            female,
            child,
            breathiness: DEFAULT_SOURCE.breathiness,
            tension: DEFAULT_SOURCE.tension,
            brightness: DEFAULT_SOURCE.brightness,
            roughness: DEFAULT_SOURCE.roughness,
            pressure: DEFAULT_SOURCE.pressure,
        }
    }
    pub fn mix(&self) -> BodyMix {
        BodyMix::new(self.female, self.child)
    }
    pub fn source(&self, f0: f64) -> SourcePatch {
        SourcePatch {
            f0: Some(f0),
            breathiness: Some(self.breathiness),
            tension: Some(self.tension),
            brightness: Some(self.brightness),
            roughness: Some(self.roughness),
            pressure: Some(self.pressure),
            ..Default::default()
        }
    }
    /// The source as the optimiser's coordinates, each 0..1.
    fn source_x(&self) -> Vec<f64> {
        let p = (self.pressure - PRESSURE.0) / (PRESSURE.1 - PRESSURE.0);
        vec![self.breathiness, self.tension, self.brightness, self.roughness, p]
    }
    fn with_source(self, x: &[f64]) -> Voice {
        let c = |v: f64| clamp(v, 0.0, 1.0);
        Voice {
            breathiness: c(x[0]),
            tension: c(x[1]),
            brightness: c(x[2]),
            roughness: c(x[3]),
            pressure: PRESSURE.0 + c(x[4]) * (PRESSURE.1 - PRESSURE.0),
            ..self
        }
    }
}

/// The engine singing あいうえお as `v`, 0.5 s each after a 0.2 s onset, its pitch sweeping
/// geometrically low → high → low over `f0` (Hz). Samples at `FS`.
pub fn probe(v: &Voice, f0: (f64, f64)) -> Vec<f32> {
    const VOWELS: [&str; 5] = ["a", "i", "ɯ", "e", "o"];
    let mix = v.mix();
    let mut voice = HumanVoice::new(FS, mix.body(), 2);
    voice.set_source(&v.source(f0.0));
    let n_seg = js_round(0.5 * FS) as usize;
    let lead = js_round(0.2 * FS) as usize;
    let total = lead + n_seg * VOWELS.len();
    let mut out = vec![0f32; total];
    let mut current = usize::MAX;
    voice.set_gate(1.0);
    let mut pos = 0;
    while pos < total {
        let k = if pos < lead { 0 } else { ((pos - lead) / n_seg).min(VOWELS.len() - 1) };
        if k != current {
            voice.set_articulation(&vowel_artic(VOWELS[k], mix).expect("calibrated Japanese vowel"), Some(true));
            current = k;
        }
        let tri = 1.0 - (2.0 * pos as f64 / total as f64 - 1.0).abs();
        voice.set_source(&SourcePatch::f0(f0.0 * pow(f0.1 / f0.0, tri)));
        let len = 128.min(total - pos);
        voice.render(&mut out[pos..pos + len]);
        pos += len;
    }
    out
}

fn to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&s| s as f64).collect()
}

/// How far a probe's median formants are from the recording's, in units of 3 %. F3 and F4 weigh
/// most: they follow the tract's length and depend least on the vowel.
fn formant_cost(t: &Features, s: &Features) -> f64 {
    const W: [f64; 4] = [0.5, 1.0, 2.0, 2.0];
    let mut c = 0.0;
    for ((w, ft), fs) in W.iter().zip(&t.formants).zip(&s.formants) {
        if ft.is_finite() {
            let r = if fs.is_finite() { (fs / ft).ln() / 0.03 } else { 5.0 };
            c += w * r * r;
        }
    }
    c
}

/// Stage 1: the body whose あいうえお, sung over the recording's pitch range, has the
/// recording's median formants: a 5×5 grid of the female and child axes, then a 3×3 grid at
/// half the step around the best. Returns the body and every point tried with its cost.
///
/// Not by inversion residual: the codebook can explain a spectrum with another body's tract by
/// picking another vowel, so the residual hardly varied over the bodies (6.97–7.90 dB on a
/// 46 s recording, not monotonic) and the pick moved between runs.
pub fn fit_body(target: &Features) -> (Voice, Vec<(Voice, f64)>) {
    let f0 = (target.f0_low, js_max(target.f0_high, target.f0_low * 1.05));
    let measure = |v: &Voice| formant_cost(target, &analyze(&to_f64(&probe(v, f0)), FS));
    let coarse: Vec<Voice> = (0..25).map(|i| Voice::body((i / 5) as f64 / 4.0, (i % 5) as f64 / 4.0)).collect();
    let mut tried: Vec<(Voice, f64)> = coarse.iter().copied().zip(par_map(&coarse, measure)).collect();
    let best = tried.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let fine: Vec<Voice> = (0..9)
        .map(|i| {
            let fe = clamp(best.female + ((i / 3) as f64 - 1.0) * 0.125, 0.0, 1.0);
            let ch = clamp(best.child + ((i % 3) as f64 - 1.0) * 0.125, 0.0, 1.0);
            Voice::body(fe, ch)
        })
        .filter(|v| !tried.iter().any(|(t, _)| t.female == v.female && t.child == v.child))
        .collect();
    tried.extend(fine.iter().copied().zip(par_map(&fine, measure)));
    let best = *tried.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    (best.0, tried)
}

/// One copy synthesis of `rec` (all of it) as `v` with `cb`'s postures along `path`, scored.
fn score(v: &Voice, cb: &Codebook, rec: &[Frame], path: &[Option<usize>], f0: f64) -> (f64, crate::invert::Distance) {
    let (raw, _) = copy_synth(v.mix(), &v.source(f0), cb, rec, path, (0, rec.len()));
    let d = distance(rec, &frames(&to_f64(&raw), FS));
    (d.cost(), d)
}

/// Stage 2: breathiness, tension, brightness, roughness and lung pressure for a fixed body, by
/// Nelder–Mead on the copy
/// synthesis' distance to `rec`.
pub fn fit_source(body: Voice, rec: &[Frame], f0: f64, max_iter: usize, mut progress: impl FnMut(usize, f64)) -> (Voice, f64) {
    let cb = codebook(body.mix(), true);
    let inv = invert(&cb, rec);
    let mut evals = 0;
    let mut best = f64::INFINITY;
    let r = nelder_mead(
        |x| {
            let c = score(&body.with_source(x), &cb, rec, &inv.path, f0).0;
            evals += 1;
            best = best.min(c);
            progress(evals, best);
            c
        },
        &body.source_x(),
        NelderMeadOpts {
            step: Some(vec![0.2; 5]),
            lo: Some(vec![0.0; 5]),
            hi: Some(vec![1.0; 5]),
            max_iter: Some(max_iter),
            tol: Some(1e-4),
        },
    );
    (body.with_source(&r.x), r.fx)
}

fn f(x: f64, d: usize) -> String {
    if x.is_finite() { fixed(x, d) } else { "-".into() }
}

/// Linear resampling to `FS` (for the A/B file only).
fn resample(x: &[f64], fs: f64) -> Vec<f32> {
    let n = js_round(x.len() as f64 * FS / fs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 * fs / FS;
            let k = (t as usize).min(x.len() - 1);
            let k2 = (k + 1).min(x.len() - 1);
            let w = t - k as f64;
            (x[k] * (1.0 - w) + x[k2] * w) as f32
        })
        .collect()
}

/// `fit voice <in.wav> [outdir] [--iters n]`
pub fn voice(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut iters = 30usize;
    let mut pos: Vec<String> = Vec::new();
    let mut it = a.iter();
    while let Some(x) = it.next() {
        if x == "--iters" {
            iters = it.next().and_then(|v| v.parse().ok()).ok_or("--iters needs a number")?;
        } else {
            pos.push(x.clone());
        }
    }
    let input = pos.first().ok_or("usage: chimera fit voice <in.wav> [outdir] [--iters n]")?;
    let stem = std::path::Path::new(input)
        .file_stem()
        .map(|s| s.to_string_lossy().replace(' ', "-"))
        .unwrap_or_else(|| "voice".into());
    let default_out = format!("out/fit/{stem}");
    let out_dir = arg(&pos, 1, &default_out).to_string();
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let (x, fs) = read_wav(input)?;
    let target = analyze(&x, fs);
    let rec = frames(&x, fs);
    let n_voiced = rec.iter().filter(|r| r.f0 > 0.0).count();
    println!(
        "{input}: {} s at {} Hz, {n_voiced} voiced frames, f0 {}–{}–{} Hz",
        fixed(x.len() as f64 / fs, 1),
        fs,
        f(target.f0_low, 0),
        f(target.f0_median, 0),
        f(target.f0_high, 0)
    );
    if n_voiced < 50 {
        return Err(format!("{input}: only {n_voiced} voiced frames; need half a second of voice or more"));
    }
    // the 10 s with the most voice
    let on: Vec<Option<usize>> = rec.iter().map(|r| (r.f0 > 0.0).then_some(0)).collect();
    let win = crate::invert::busiest_window(&rec, &on, FIT_FRAMES);
    let part = &rec[win.0..win.1];
    let f0 = target.f0_median;
    println!("fitting on {}–{} s", fixed(win.0 as f64 * HOP_S, 1), fixed(win.1 as f64 * HOP_S, 1));

    println!("stage 1: body (formants of あいうえお over the recording's pitch range)…");
    let (body, grid) = fit_body(&target);
    println!("  best: female {} child {}", fixed(body.female, 2), fixed(body.child, 2));
    println!("stage 2: source (Nelder–Mead on the copy synthesis, up to {iters} iterations)…");
    let (fitted, _) = fit_source(body, part, f0, iters, |n, best| {
        if n % 10 == 0 {
            println!("  {n} renders, best cost {}", fixed(best, 1));
        }
    });

    // the preset a user would pick by ear
    let preset = if target.f0_median > 250.0 {
        Voice::body(0.0, 1.0)
    } else if target.f0_median > 165.0 {
        Voice::body(1.0, 0.0)
    } else {
        Voice::body(0.0, 0.0)
    };
    let preset_name = match (preset.female, preset.child) {
        (_, c) if c > 0.5 => "child",
        (fe, _) if fe > 0.5 => "female",
        _ => "male",
    };

    println!("rendering the whole recording with both voices…");
    let runs = par_map(&[preset, fitted], |v| {
        let cb = codebook(v.mix(), false);
        let inv = invert(&cb, &rec);
        let (raw, matched) = copy_synth(v.mix(), &v.source(f0), &cb, &rec, &inv.path, (0, rec.len()));
        let d = distance(&rec, &frames(&to_f64(&raw), FS));
        let feats = analyze(&to_f64(&matched), FS);
        let shares = vowel_shares(&cb, &inv.path);
        (matched, d, feats, inv.residual_db, shares)
    });
    let (p_wav, p_d, p_feat, p_res, _) = &runs[0];
    let (f_wav, f_d, f_feat, f_res, f_shares) = &runs[1];

    println!();
    println!("  {:<22}{:>11}{:>11}{:>11}", "", "recording", preset_name, "fitted");
    type Row = (&'static str, fn(&Features) -> String);
    let rows: [Row; 8] = [
        ("F1 (Hz)", |x| f(x.formants[0], 0)),
        ("F2 (Hz)", |x| f(x.formants[1], 0)),
        ("F3 (Hz)", |x| f(x.formants[2], 0)),
        ("F4 (Hz)", |x| f(x.formants[3], 0)),
        ("tract (cm)", |x| f(x.tract_length_cm, 1)),
        ("tilt (dB/oct)", |x| f(x.tilt_db_per_octave, 1)),
        ("aperiodicity", |x| f(x.aperiodicity, 3)),
        ("H1-H2 (dB)", |x| f(x.h1_h2_db, 1)),
    ];
    for (name, g) in &rows {
        println!("  {name:<22}{:>11}{:>11}{:>11}", g(&target), g(p_feat), g(f_feat));
    }
    println!("  {:<22}{:>11}{:>11}{:>11}", "inversion residual dB", "", f(*p_res, 2), f(*f_res, 2));
    println!("  {:<22}{:>11}{:>11}{:>11}", "harmonics off (dB)", "", f(p_d.harmonics_db, 2), f(f_d.harmonics_db, 2));
    println!("  {:<22}{:>11}{:>11}{:>11}", "noise off (dB)", "", f(p_d.noise_db, 2), f(f_d.noise_db, 2));
    println!("  {:<22}{:>11}{:>11}{:>11}", "voiced frames missed", "", f(p_d.missed * 100.0, 1) + "%", f(f_d.missed * 100.0, 1) + "%");
    println!();
    println!(
        "fitted: female {}  child {}  breathiness {}  tension {}  brightness {}  roughness {}  pressure {}  f0 {} Hz",
        fixed(fitted.female, 2),
        fixed(fitted.child, 2),
        fixed(fitted.breathiness, 2),
        fixed(fitted.tension, 2),
        fixed(fitted.brightness, 2),
        fixed(fitted.roughness, 2),
        fixed(fitted.pressure, 1),
        fixed(f0, 0)
    );
    let top: Vec<String> = f_shares.iter().take(6).map(|(v, s)| format!("{v} {}%", fixed(s * 100.0, 0))).collect();
    println!("inverted postures, by nearest vowel: {}", top.join(", "));

    // A/B: the fit window of the recording, the fitted copy, the preset copy
    let hop_out = js_round(HOP_S * FS) as usize;
    let cut = |w: &[f32]| w[(win.0 * hop_out).min(w.len())..(win.1 * hop_out).min(w.len())].to_vec();
    let hop_in = js_round(HOP_S * fs) as usize;
    let rec_part = resample(&x[(win.0 * hop_in).min(x.len())..(win.1 * hop_in).min(x.len())], fs);
    let gap = vec![0f32; js_round(0.8 * FS) as usize];
    let mut ab = rec_part;
    for w in [cut(f_wav), cut(p_wav)] {
        ab.extend_from_slice(&gap);
        ab.extend_from_slice(&w);
    }
    let peak = ab.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.99 {
        for v in &mut ab {
            *v *= 0.99 / peak;
        }
    }

    let json = serde_json::json!({
        "recording": input,
        "body": { "female": fitted.female, "child": fitted.child },
        "source": {
            "f0": f0,
            "breathiness": fitted.breathiness,
            "tension": fitted.tension,
            "brightness": fitted.brightness,
            "roughness": fitted.roughness,
            "pressure": fitted.pressure,
        },
        "fitWindowSeconds": [win.0 as f64 * HOP_S, win.1 as f64 * HOP_S],
        "bodyGrid": grid.iter().map(|(v, c)| serde_json::json!({ "female": v.female, "child": v.child, "formantCost": c })).collect::<Vec<_>>(),
        "preset": { "name": preset_name, "voice": preset, "residualDb": p_res, "harmonicsDb": p_d.harmonics_db, "noiseDb": p_d.noise_db, "missed": p_d.missed, "features": p_feat },
        "fitted": { "voice": fitted, "residualDb": f_res, "harmonicsDb": f_d.harmonics_db, "noiseDb": f_d.noise_db, "missed": f_d.missed, "features": f_feat },
        "recordingFeatures": target,
        "vowelShares": f_shares,
    });
    let json_path = format!("{out_dir}/voice.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| format!("{json_path}: {e}"))?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/copy-fitted")), f_wav)?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/copy-{preset_name}")), p_wav)?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/ab")), &ab)?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/aiueo")), &probe(&fitted, (target.f0_low, js_max(target.f0_high, target.f0_low * 1.05))))?;
    println!("wrote {json_path}, copy-fitted.wav, copy-{preset_name}.wav, ab.wav (recording | fitted | {preset_name}), aiueo.wav");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's own あいうえお, inverted with its own body's codebook, is explained closely,
    /// and あいえお come back as those vowels (or blends led by them) in most frames. う is left
    /// out of the labels: half [i] half [o] has its formants (seen picked for 2/3 of it), which
    /// sounds the same, so either is right.
    #[test]
    fn inversion_recovers_the_engines_own_vowels() {
        let v = Voice::body(1.0, 0.0);
        let x = to_f64(&probe(&v, (190.0, 240.0)));
        let fr = frames(&x, FS);
        let cb = codebook(v.mix(), true);
        let inv = invert(&cb, &fr);
        assert!(inv.residual_db < 4.0, "residual {} dB", inv.residual_db);
        for (k, vowel) in ["a", "i", "ɯ", "e", "o"].iter().enumerate() {
            if *vowel == "ɯ" {
                continue;
            }
            // the middle 0.3 s of each 0.5 s vowel (after the 0.2 s onset)
            let (a, b) = (((0.2 + 0.5 * k as f64 + 0.1) / HOP_S) as usize, ((0.2 + 0.5 * k as f64 + 0.4) / HOP_S) as usize);
            let hits = inv.path[a..b]
                .iter()
                .flatten()
                .filter(|&&e| cb.entries[e].label.split('~').next() == Some(vowel))
                .count();
            let got: Vec<&str> = inv.path[a..b].iter().flatten().map(|&e| cb.entries[e].label.as_str()).collect();
            assert!(hits * 2 > b - a, "{vowel}: {hits} of {} frames: {got:?}", b - a);
        }
    }

    /// The body grid finds the body of an engine voice.
    #[test]
    fn fit_body_recovers_an_engine_body() {
        for (fe, ch, f0) in [(1.0, 0.0, (200.0, 250.0)), (0.0, 0.0, (105.0, 135.0)), (0.0, 1.0, (260.0, 320.0))] {
            let mut v = Voice::body(fe, ch);
            v.breathiness = 0.4;
            let (got, _) = fit_body(&analyze(&to_f64(&probe(&v, f0)), FS));
            assert!(
                (got.female - fe).abs() <= 0.25 && (got.child - ch).abs() <= 0.25,
                "body ({fe}, {ch}) fitted as ({}, {})",
                got.female,
                got.child
            );
        }
    }
}
