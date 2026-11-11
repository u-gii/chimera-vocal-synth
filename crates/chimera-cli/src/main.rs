//! `chimera`: offline renders to WAV, speech diagnostics and table calibration over the CHIMERA
//! engine. Ports of the TypeScript tools (`tools/render-*.ts`, `speech-diagnostics.ts`,
//! `calibrate-*.ts`) with their arguments, defaults and printed summaries. Spectrograms of the
//! WAVs: `bun tools/spectrogram.ts in.wav out.png [title] [maxFreq] [windowMs]`.

mod calibrate;
mod compare;
mod diagnose;
mod fit;
mod invert;
mod js;
mod render;
mod wav;

use chimera::anatomy::human::{BODY_CHILD, BODY_FEMALE, BODY_MALE, BodyMix, HumanBody};
use std::process::ExitCode;

const USAGE: &str = "\
usage: chimera <command> [args] [--float]

  render speech  <out> [\"ipa / ipa\"] [male|female|child]
  render text    <out> [\"English text.\"] [male|female|child]
  render song    [outdir] [male|female|child]
  render notes   <in.json> <out> [male|female|child] [--artic tracks.json] [--tele voice.tele]
                 (melody file; midi <= 0 is a rest; --artic also writes the articulators at 100 Hz)
  render beatbox [outdir] [preset name | pattern] [male|female|child] [bpm] [bars]
  render creature [creatureId] [outdir]
  render vowels  [male|female|child] [out]
  diagnose       [outdir]
  compare csd    [csd-dir] [outdir] [--songs a|b|all|en001a,…] [--body female] [--jobs n]
                 [--against earlier/score.json]  (score.json: distance to the singer, and the verdict)
  compare tsp    [tsp-dir] [outdir] [--speakers FA,MA,…] [--jobs n] [--dump]
                 [--against earlier/score.json]  (distance to 23 real speakers, phone by phone)
  calibrate vowels     [male|female|child|all] [--out file]
  calibrate consonants [--out file]
  fit voice      <in.wav> [outdir] [--iters n]
                 (inverts the recording's vocal tract postures, then fits a body and a voice source
                 so its copy synthesis matches; writes voice.json, copy-fitted.wav, copy-<preset>.wav,
                 ab.wav (recording | fitted | preset) and aiueo.wav)

Output names are stems (<out>.wav). --float writes 32-bit float WAVs instead of 16-bit PCM.
Calibration writes the tables the engine embeds, in crates/chimera/src/phonetics/.";

/// The output rate of every render.
pub const FS: f64 = 48000.0;

/// The repository root (data files and the generated tables are found from here).
pub fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

/// Options shared by every command.
pub struct Ctx {
    /// write 32-bit float WAVs
    pub float: bool,
}

impl Ctx {
    pub fn write_wav(&self, path: &str, x: &[f32]) -> Result<(), String> {
        wav::write(path, x, FS as u32, self.float).map_err(|e| format!("{path}: {e}"))
    }
}

/// The body for a name (`male`, `female`, `child`).
pub fn body(name: &str) -> Result<HumanBody, String> {
    match name {
        "male" => Ok(BODY_MALE),
        "female" => Ok(BODY_FEMALE),
        "child" => Ok(BODY_CHILD),
        _ => Err(format!("unknown body {name:?}; have male, female, child")),
    }
}

/// The app's body mix for a body preset (female 0..1, child 0..1), as `src/ui/body.ts` PRESETS.
pub fn body_mix(name: &str) -> Result<BodyMix, String> {
    match name {
        "male" => Ok(BodyMix::new(0.0, 0.0)),
        "female" => Ok(BodyMix::new(1.0, 0.0)),
        "child" => Ok(BodyMix::new(0.0, 1.0)),
        _ => Err(format!("unknown body {name:?}; have male, female, child")),
    }
}

/// `<out>.wav` for an output stem (a trailing `.wav` is accepted too).
pub fn wav_name(stem: &str) -> String {
    format!("{}.wav", stem.strip_suffix(".wav").unwrap_or(stem))
}

pub fn peak(x: &[f32]) -> f64 {
    x.iter()
        .fold(0.0f64, |p, &v| chimera::math::js_max(p, (v as f64).abs()))
}

/// Positional argument `i`, or a default.
pub fn arg<'a>(a: &'a [String], i: usize, default: &'a str) -> &'a str {
    a.get(i).map(String::as_str).unwrap_or(default)
}

fn run(args: Vec<String>) -> Result<(), String> {
    let float = args.iter().any(|a| a == "--float");
    let args: Vec<String> = args.into_iter().filter(|a| a != "--float").collect();
    let ctx = Ctx { float };
    let cmd: Vec<&str> = args.iter().take(2).map(String::as_str).collect();
    match cmd.as_slice() {
        ["render", kind, ..] => {
            let rest = &args[2..];
            match *kind {
                "speech" => render::speech(&ctx, rest),
                "text" => render::text(&ctx, rest),
                "song" => render::song(&ctx, rest),
                "notes" => render::notes(&ctx, rest),
                "beatbox" => render::beatbox(&ctx, rest),
                "creature" => render::creature(&ctx, rest),
                "vowels" => render::vowels(&ctx, rest),
                _ => Err(format!("unknown render kind {kind:?}\n\n{USAGE}")),
            }
        }
        ["compare", "csd", ..] => compare::csd(&ctx, &args[2..]),
        ["compare", "tsp", ..] => compare::tsp::run(&ctx, &args[2..]),
        ["diagnose", ..] => diagnose::run(&ctx, &args[1..]),
        ["calibrate", "vowels", ..] => calibrate::vowels(&args[2..]),
        ["calibrate", "consonants", ..] => calibrate::consonants(&args[2..]),
        ["fit", "voice", ..] => fit::voice(&ctx, &args[2..]),
        ["help" | "--help" | "-h", ..] => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
