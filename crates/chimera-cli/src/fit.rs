//! `chimera fit voice`: estimate a body and a voice source from a recording.
//!
//! The recording and the engine's own voice go through one analysis: the pitch range, the median
//! formants, the long-term average spectrum, the aperiodicity and H1−H2. The body (the female and
//! child axes of `BodyMix`) and the source (breathiness, tension, brightness) are chosen so that
//! the engine singing あいうえお across the recording's pitch range measures like the recording.
//!
//! No transcript is needed on purpose: medians over the recording's many vowels are compared with
//! medians over the five Japanese vowels. That ties the estimate to what does not depend on the
//! words: the length of the tract (the upper formants), the spectral tilt and the noise of the
//! source. The vowels' own shapes stay the calibrated ones.

use crate::compare::read_wav;
use crate::js::fixed;
use crate::{Ctx, FS, arg, wav_name};
use chimera::analysis::fft::{Fft, hann, next_pow2, power_spectrum_db};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::BodyMix;
use chimera::math::{clamp, js_max, js_round, ln, log10, pow};
use chimera::phonetics::planner::vowel_artic;
use chimera::util::optimize::{NelderMeadOpts, nelder_mead};
use chimera::voice::human_voice::{DEFAULT_SOURCE, HumanVoice, SourcePatch};
use serde::Serialize;

/// The probe's vowels: Japanese あ い う え お.
const VOWELS: [&str; 5] = ["a", "i", "ɯ", "e", "o"];
/// Seconds per probe vowel, and the onset left out of the analysis.
const SEG: f64 = 0.5;
const LEAD: f64 = 0.2;
/// Long-term spectrum bands, log-spaced between these (Hz). They start above most voices' f0:
/// a band below the lowest harmonic holds only room noise in a recording and nothing in the
/// probe (150–208 Hz read +4 dB against −13 dB on a 212–393 Hz voice), so it measured the room.
const BAND_LO: f64 = 250.0;
const BAND_HI: f64 = 7500.0;
const N_BANDS: usize = 10;
/// Speed of sound (cm/s), as the airway's.
const C: f64 = 35000.0;

/// What the analysis measures of a voice.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    /// f0 of the voiced frames (Hz): 10th, 50th and 90th percentile
    pub f0_low: f64,
    pub f0_median: f64,
    pub f0_high: f64,
    /// median F1…F4 over the voiced frames (Hz); NaN when too few frames had one
    pub formants: [f64; 4],
    /// the uniform tube with the same formant spacing (cm), from F2…F4
    pub tract_length_cm: f64,
    /// long-term average spectrum of the voiced frames, per band, dB re the bands' mean
    pub ltas_db: Vec<f64>,
    /// the long-term spectrum's least-squares slope (dB per octave)
    pub tilt_db_per_octave: f64,
    /// median YIN aperiodicity of the voiced frames (0 periodic … 1 noise)
    pub aperiodicity: f64,
    /// median level of the first harmonic over the second (dB): high = breathy/lax, low = pressed
    pub h1_h2_db: f64,
    pub voiced_frames: usize,
}

/// The `p` quantile (0..1) of the finite values, NaN if none.
fn quantile(v: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let i = clamp(js_round(p * (v.len() - 1) as f64), 0.0, (v.len() - 1) as f64);
    v[i as usize]
}

/// Measure a mono signal at rate `fs`. Frames are 40 ms every 10 ms; only frames within 30 dB of
/// the loud ones and voiced by YIN count.
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
    // formants on ~11 kHz audio (as `render vowels`): order 12 then finds F1…F4 below 5.5 kHz
    let dec = js_max(1.0, js_round(fs / 11025.0)) as usize;
    let lpc_len = win * 5 / 8;
    let mut f0s = Vec::new();
    let mut aps = Vec::new();
    let mut h12 = Vec::new();
    let mut fm: [Vec<f64>; 4] = Default::default();
    let mut power = vec![0.0; n_fft / 2 + 1];
    for k in 0..n_frames {
        if level[k] < gate {
            continue;
        }
        let f = &x[k * hop..k * hop + win];
        let p = yin_pitch(
            f,
            fs,
            YinOpts {
                min_f0: Some(70.0),
                max_f0: Some(900.0),
                threshold: Some(0.3),
            },
        );
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
        let ds: Vec<f64> = mid
            .chunks_exact(dec)
            .map(|c| c.iter().sum::<f64>() / dec as f64)
            .collect();
        let found = estimate_formants(
            &ds,
            fs / dec as f64,
            FormantOpts {
                order: Some(12),
                min_freq: Some(150.0),
                max_bw: Some(500.0),
                ..Default::default()
            },
        );
        for (i, fr) in found.iter().take(4).enumerate() {
            fm[i].push(fr.freq);
        }
    }
    let voiced = f0s.len();
    let formants: [f64; 4] = std::array::from_fn(|i| {
        if fm[i].len() * 4 >= voiced.max(1) { quantile(&fm[i], 0.5) } else { f64::NAN }
    });
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
    let mut ltas_db: Vec<f64> = (0..N_BANDS)
        .map(|b| {
            let lo = BAND_LO * pow(BAND_HI / BAND_LO, b as f64 / N_BANDS as f64);
            let hi = BAND_LO * pow(BAND_HI / BAND_LO, (b + 1) as f64 / N_BANDS as f64);
            let bins: Vec<f64> = (0..power.len())
                .filter(|&k| (k as f64 * bin_hz) >= lo && (k as f64 * bin_hz) < hi)
                .map(|k| power[k])
                .collect();
            if voiced == 0 || bins.is_empty() {
                f64::NAN
            } else {
                10.0 * log10(bins.iter().sum::<f64>() / bins.len() as f64 + 1e-30)
            }
        })
        .collect();
    let mean = ltas_db.iter().sum::<f64>() / N_BANDS as f64;
    for v in &mut ltas_db {
        *v -= mean;
    }
    // slope against octaves above BAND_LO, at the bands' centres
    let octaves_per_band = log10(BAND_HI / BAND_LO) / log10(2.0) / N_BANDS as f64;
    let xs: Vec<f64> = (0..N_BANDS).map(|b| (b as f64 + 0.5) * octaves_per_band).collect();
    let xm = xs.iter().sum::<f64>() / N_BANDS as f64;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (x, y) in xs.iter().zip(&ltas_db) {
        sxy += (x - xm) * y;
        sxx += (x - xm) * (x - xm);
    }
    let tilt_db_per_octave = sxy / sxx;
    Features {
        f0_low: quantile(&f0s, 0.1),
        f0_median: quantile(&f0s, 0.5),
        f0_high: quantile(&f0s, 0.9),
        formants,
        tract_length_cm,
        ltas_db,
        tilt_db_per_octave,
        aperiodicity: quantile(&aps, 0.5),
        h1_h2_db: quantile(&h12, 0.5),
        voiced_frames: voiced,
    }
}

/// The estimated voice: a body and the source's timbre controls, all 0..1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Voice {
    pub female: f64,
    pub child: f64,
    pub breathiness: f64,
    pub tension: f64,
    pub brightness: f64,
}

impl Voice {
    fn from_x(x: &[f64]) -> Voice {
        let c = |v: f64| clamp(v, 0.0, 1.0);
        Voice { female: c(x[0]), child: c(x[1]), breathiness: c(x[2]), tension: c(x[3]), brightness: c(x[4]) }
    }
    fn x(&self) -> Vec<f64> {
        vec![self.female, self.child, self.breathiness, self.tension, self.brightness]
    }
    /// A body with the default source.
    fn body(female: f64, child: f64) -> Voice {
        Voice {
            female,
            child,
            breathiness: DEFAULT_SOURCE.breathiness,
            tension: DEFAULT_SOURCE.tension,
            brightness: DEFAULT_SOURCE.brightness,
        }
    }
    pub fn mix(&self) -> BodyMix {
        BodyMix::new(self.female, self.child)
    }
    fn source(&self, f0: f64) -> SourcePatch {
        SourcePatch {
            f0: Some(f0),
            breathiness: Some(self.breathiness),
            tension: Some(self.tension),
            brightness: Some(self.brightness),
            ..Default::default()
        }
    }
}

/// The engine singing あいうえお as `v`, its pitch sweeping geometrically low → high → low over
/// `f0` (Hz). Samples at `FS`; the first `LEAD` s are the onset.
pub fn probe(v: &Voice, f0: (f64, f64)) -> Vec<f32> {
    let mix = v.mix();
    let mut voice = HumanVoice::new(FS, mix.body(), 2);
    voice.set_source(&v.source(f0.0));
    let n_seg = js_round(SEG * FS) as usize;
    let lead = js_round(LEAD * FS) as usize;
    let total = lead + n_seg * VOWELS.len();
    let mut out = vec![0f32; total];
    let mut current = usize::MAX;
    voice.set_gate(1.0);
    let mut pos = 0;
    while pos < total {
        let k = if pos < lead { 0 } else { ((pos - lead) / n_seg).min(VOWELS.len() - 1) };
        if k != current {
            let artic = vowel_artic(VOWELS[k], mix).expect("calibrated Japanese vowel");
            voice.set_articulation(&artic, Some(true));
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

/// Analyse the probe of `v` (without its onset).
pub fn measure(v: &Voice, f0: (f64, f64)) -> Features {
    let out = probe(v, f0);
    let lead = js_round(LEAD * FS) as usize;
    let x: Vec<f64> = out[lead..].iter().map(|&s| s as f64).collect();
    analyze(&x, FS)
}

/// How far the probe's features are from the target's: squared errors in units of a just-audible
/// difference (3 % per formant, 3 dB per band and for H1−H2, 0.05 aperiodicity). F3 and F4 weigh
/// most: they follow the tract's length and depend least on the vowel. With `body_only`, the
/// formants alone.
pub fn cost(t: &Features, s: &Features, body_only: bool) -> f64 {
    const W: [f64; 4] = [0.5, 1.0, 2.0, 2.0];
    let mut c = 0.0;
    for ((w, ft), fs) in W.iter().zip(&t.formants).zip(&s.formants) {
        if !ft.is_finite() {
            continue;
        }
        let r = if fs.is_finite() { ln(fs / ft) / 0.03 } else { 5.0 };
        c += w * r * r;
    }
    if body_only {
        return c;
    }
    let d: Vec<f64> = t
        .ltas_db
        .iter()
        .zip(&s.ltas_db)
        .map(|(a, b)| b - a)
        .filter(|x| x.is_finite())
        .collect();
    if !d.is_empty() {
        let m = d.iter().sum::<f64>() / d.len() as f64;
        let ms = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / d.len() as f64;
        c += 2.0 * ms / 9.0;
    }
    let sq = |a: f64, b: f64, unit: f64| if a.is_finite() && b.is_finite() { ((a - b) / unit) * ((a - b) / unit) } else { 0.0 };
    c += sq(s.aperiodicity, t.aperiodicity, 0.05);
    c += sq(s.h1_h2_db, t.h1_h2_db, 3.0);
    c
}

/// `f` over `items` on all cores, in order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
    let chunk = items.len().div_ceil(jobs).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| s.spawn(|| part.iter().map(&f).collect::<Vec<R>>()))
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("fit worker")).collect()
    })
}

/// The pitch range the probe sweeps for a target.
fn sweep(t: &Features) -> (f64, f64) {
    (t.f0_low, js_max(t.f0_high, t.f0_low * 1.05))
}

/// Stage 1: the body on a 5×5 grid of the female and child axes, by the formants alone.
pub fn fit_body(target: &Features) -> Voice {
    let f0 = sweep(target);
    let grid: Vec<Voice> = (0..25).map(|i| Voice::body((i / 5) as f64 / 4.0, (i % 5) as f64 / 4.0)).collect();
    let costs = par_map(&grid, |v| cost(target, &measure(v, f0), true));
    let best = (0..grid.len()).min_by(|&a, &b| costs[a].total_cmp(&costs[b])).unwrap();
    grid[best]
}

/// Stage 2: from the grid's best body, everything at once by Nelder–Mead on the full cost.
pub fn fit_voice(target: &Features, start: Voice, max_iter: usize, mut progress: impl FnMut(usize, f64)) -> (Voice, f64) {
    let f0 = sweep(target);
    let mut evals = 0;
    let mut best = f64::INFINITY;
    let r = nelder_mead(
        |x| {
            let c = cost(target, &measure(&Voice::from_x(x), f0), false);
            evals += 1;
            best = best.min(c);
            progress(evals, best);
            c
        },
        &start.x(),
        NelderMeadOpts {
            step: Some(vec![0.15; 5]),
            lo: Some(vec![0.0; 5]),
            hi: Some(vec![1.0; 5]),
            max_iter: Some(max_iter),
            tol: Some(1e-4),
        },
    );
    (Voice::from_x(&r.x), r.fx)
}

fn f(x: f64, d: usize) -> String {
    if x.is_finite() { fixed(x, d) } else { "-".into() }
}

fn row(name: &str, t: &Features, cols: &[&Features], fmt: impl Fn(&Features) -> String) {
    let mut s = format!("  {name:<16}{:>12}", fmt(t));
    for c in cols {
        s += &format!("{:>12}", fmt(c));
    }
    println!("{s}");
}

/// `fit voice <in.wav> [outdir] [--iters n]`
pub fn voice(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut iters = 60usize;
    let mut pos: Vec<String> = Vec::new();
    let mut it = a.iter();
    while let Some(x) = it.next() {
        if x == "--iters" {
            iters = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or("--iters needs a number")?;
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
    println!(
        "{input}: {} s at {} Hz, {} voiced frames, f0 {}–{}–{} Hz",
        fixed(x.len() as f64 / fs, 1),
        fs,
        target.voiced_frames,
        f(target.f0_low, 0),
        f(target.f0_median, 0),
        f(target.f0_high, 0)
    );
    if target.voiced_frames < 50 {
        return Err(format!("{input}: only {} voiced frames; need half a second of voice or more", target.voiced_frames));
    }
    let f0 = sweep(&target);

    // the preset a user would pick by ear: female or child above ~165 Hz, else male
    let preset = if target.f0_median > 250.0 {
        Voice::body(0.0, 1.0)
    } else if target.f0_median > 165.0 {
        Voice::body(1.0, 0.0)
    } else {
        Voice::body(0.0, 0.0)
    };
    let preset_feat = measure(&preset, f0);
    let preset_cost = cost(&target, &preset_feat, false);

    println!("stage 1: body grid (25 renders)…");
    let body = fit_body(&target);
    println!("  body female {} child {}", fixed(body.female, 2), fixed(body.child, 2));
    println!("stage 2: body and source (Nelder–Mead, up to {iters} iterations)…");
    let (fitted, fitted_cost) = fit_voice(&target, body, iters, |n, best| {
        if n % 20 == 0 {
            println!("  {n} renders, best cost {}", fixed(best, 2));
        }
    });
    let fitted_feat = measure(&fitted, f0);

    println!();
    println!("  {:<16}{:>12}{:>12}{:>12}", "", "recording", "preset", "fitted");
    let cols = [&preset_feat, &fitted_feat];
    for i in 0..4 {
        row(&format!("F{} (Hz)", i + 1), &target, &cols, |x| f(x.formants[i], 0));
    }
    row("tract (cm)", &target, &cols, |x| f(x.tract_length_cm, 1));
    row("aperiodicity", &target, &cols, |x| f(x.aperiodicity, 3));
    row("H1-H2 (dB)", &target, &cols, |x| f(x.h1_h2_db, 1));
    row("tilt (dB/oct)", &target, &cols, |x| f(x.tilt_db_per_octave, 1));
    println!("  {:<16}{:>12}{:>12}{:>12}", "cost", "", fixed(preset_cost, 2), fixed(fitted_cost, 2));
    println!();
    println!(
        "fitted: female {}  child {}  breathiness {}  tension {}  brightness {}  f0 {} Hz",
        fixed(fitted.female, 2),
        fixed(fitted.child, 2),
        fixed(fitted.breathiness, 2),
        fixed(fitted.tension, 2),
        fixed(fitted.brightness, 2),
        fixed(target.f0_median, 0)
    );

    let json = serde_json::json!({
        "recording": input,
        "body": { "female": fitted.female, "child": fitted.child },
        "source": {
            "f0": target.f0_median,
            "breathiness": fitted.breathiness,
            "tension": fitted.tension,
            "brightness": fitted.brightness,
        },
        "cost": { "preset": preset_cost, "fitted": fitted_cost },
        "presetVoice": preset,
        "recordingFeatures": target,
        "presetFeatures": preset_feat,
        "fittedFeatures": fitted_feat,
    });
    let json_path = format!("{out_dir}/voice.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| format!("{json_path}: {e}"))?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/fitted")), &probe(&fitted, f0))?;
    ctx.write_wav(&wav_name(&format!("{out_dir}/preset")), &probe(&preset, f0))?;
    println!("wrote {json_path}, {out_dir}/fitted.wav, {out_dir}/preset.wav");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe's own analysis finds the pitch it swept and a tract of the body's size.
    #[test]
    fn the_probe_measures_its_own_pitch_and_size() {
        let male = measure(&Voice::body(0.0, 0.0), (110.0, 140.0));
        let child = measure(&Voice::body(0.0, 1.0), (260.0, 320.0));
        assert!(male.f0_median > 105.0 && male.f0_median < 145.0, "male f0 {}", male.f0_median);
        assert!(child.f0_median > 250.0 && child.f0_median < 330.0, "child f0 {}", child.f0_median);
        assert!(
            male.tract_length_cm > child.tract_length_cm + 1.5,
            "male tract {} cm vs child {} cm",
            male.tract_length_cm,
            child.tract_length_cm
        );
    }

    /// The body grid finds the body of an engine voice it is shown.
    #[test]
    fn fit_body_recovers_an_engine_body() {
        for (fe, ch) in [(1.0, 0.0), (0.0, 0.0)] {
            let mut v = Voice::body(fe, ch);
            v.breathiness = 0.3;
            let f0 = if fe > 0.5 { (200.0, 260.0) } else { (105.0, 135.0) };
            let target = measure(&v, f0);
            let got = fit_body(&target);
            assert!(
                (got.female - fe).abs() <= 0.25 && (got.child - ch).abs() <= 0.25,
                "body ({fe}, {ch}) fitted as ({}, {})",
                got.female,
                got.child
            );
        }
    }
}
