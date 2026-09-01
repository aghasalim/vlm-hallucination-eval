//! The whole trade-off curve, an exhaustive recalibration and a large bootstrap.
//!
//! reports/results.md publishes six rows of the threshold sweep and
//! reports/figures/tradeoff.png draws all of it. Both read
//! reports/verification.json. This rebuilds every row of that sweep, for all
//! three phrasings, from the z-scores and the per-probe answers, and requires
//! an exact match.
//!
//! It then does two things the Python does not, because in Python they would
//! be slow enough that I would have talked myself out of them:
//!
//!   * a 0.0005-step recalibration over the whole threshold range instead of
//!     the 0.05 grid that was actually fitted, so I can say how wide the
//!     F1-optimal plateau is and whether the coarse grid missed anything;
//!   * a leave-one-image-out refit over the 11 calibration images, and a
//!     1,000,000-resample image-level bootstrap of the relative hallucination
//!     reduction, which is the number the README quotes as "25 to 40%".
//!
//! Run: cargo run --release --quiet -- <repo-root>

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::exit;

// ------------------------------------------------------------------ JSON ---

#[derive(Debug, Clone)]
enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn need(&self, key: &str) -> &J {
        self.get(key)
            .unwrap_or_else(|| die(&format!("missing key {key:?}")))
    }
    fn num(&self) -> f64 {
        match self {
            J::Num(n) => *n,
            _ => die("expected a number"),
        }
    }
    fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => die("expected a string"),
        }
    }
    fn arr(&self) -> &[J] {
        match self {
            J::Arr(a) => a,
            _ => die("expected an array"),
        }
    }
    fn truthy(&self) -> bool {
        matches!(self, J::Bool(true))
    }
}

fn die(msg: &str) -> ! {
    eprintln!("sweep: {msg}");
    exit(1);
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn string(&mut self) -> String {
        if self.b.get(self.i) != Some(&b'"') {
            die("expected a string");
        }
        self.i += 1;
        let mut out = String::new();
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            let c = self.b[self.i];
            self.i += 1;
            if c == b'\\' {
                let e = *self.b.get(self.i).unwrap_or_else(|| die("truncated escape"));
                self.i += 1;
                out.push(match e {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    _ => die("unsupported string escape"),
                });
            } else {
                out.push(c as char);
            }
        }
        if self.i >= self.b.len() {
            die("unterminated string");
        }
        self.i += 1;
        out
    }
    fn value(&mut self) -> J {
        self.ws();
        match *self.b.get(self.i).unwrap_or_else(|| die("unexpected end of input")) {
            b'{' => {
                self.i += 1;
                let mut m = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return J::Obj(m);
                }
                loop {
                    self.ws();
                    let k = self.string();
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        die("expected ':' after a key");
                    }
                    self.i += 1;
                    let v = self.value();
                    m.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(&b',') => self.i += 1,
                        Some(&b'}') => {
                            self.i += 1;
                            break;
                        }
                        _ => die("expected ',' or '}'"),
                    }
                }
                J::Obj(m)
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return J::Arr(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    match self.b.get(self.i) {
                        Some(&b',') => self.i += 1,
                        Some(&b']') => {
                            self.i += 1;
                            break;
                        }
                        _ => die("expected ',' or ']'"),
                    }
                }
                J::Arr(a)
            }
            b'"' => J::Str(self.string()),
            b't' => {
                self.expect("true");
                J::Bool(true)
            }
            b'f' => {
                self.expect("false");
                J::Bool(false)
            }
            b'n' => {
                self.expect("null");
                J::Null
            }
            _ => {
                let start = self.i;
                if self.b[self.i] == b'-' {
                    self.i += 1;
                }
                while self.i < self.b.len()
                    && matches!(self.b[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    self.i += 1;
                }
                let s = std::str::from_utf8(&self.b[start..self.i]).unwrap_or_else(|_| die("bad utf8"));
                J::Num(s.parse::<f64>().unwrap_or_else(|_| die("bad number")))
            }
        }
    }
    fn expect(&mut self, lit: &str) {
        if self.b.len() < self.i + lit.len() || &self.b[self.i..self.i + lit.len()] != lit.as_bytes() {
            die("bad literal");
        }
        self.i += lit.len();
    }
}

fn load(path: &str) -> J {
    let raw = fs::read(path).unwrap_or_else(|e| die(&format!("cannot read {path}: {e}")));
    let mut p = P { b: &raw, i: 0 };
    let v = p.value();
    p.ws();
    if p.i != raw.len() {
        die(&format!("trailing content in {path}"));
    }
    v
}

// -------------------------------------------------------------- the work ---

#[derive(Clone)]
struct Probe {
    file: String,
    truth: bool,
    pred: bool,
    z: f64,
}

/// xorshift64*, so the bootstrap does not depend on a crate either.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn f1_of(tp: f64, fp: f64, fn_: f64) -> f64 {
    let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let rec = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    }
}

fn clip_f1(probes: &[Probe], t: f64) -> f64 {
    let (mut tp, mut fp, mut fnn) = (0.0, 0.0, 0.0);
    for p in probes {
        let acc = p.z >= t;
        if p.truth && acc {
            tp += 1.0;
        } else if !p.truth && acc {
            fp += 1.0;
        } else if p.truth {
            fnn += 1.0;
        }
    }
    f1_of(tp, fp, fnn)
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let ver = load(&format!("{root}/reports/verification.json"));
    let sty = load(&format!("{root}/reports/prompt_styles.json"));

    let zs = ver.need("z_scores");
    let cal: BTreeSet<&str> = ver.need("calibration_images").arr().iter().map(|v| v.str()).collect();
    let test: BTreeSet<&str> = ver.need("test_images").arr().iter().map(|v| v.str()).collect();
    let styles = ["neutral", "leading", "presupposing"];

    let probes_for = |style: &str| -> Vec<Probe> {
        sty.need(style)
            .need("probes")
            .arr()
            .iter()
            .map(|p| {
                let file = p.need("file_name").str().to_string();
                let obj = p.need("object").str();
                let z = zs
                    .get(&file)
                    .unwrap_or_else(|| die(&format!("no z-scores for {file}")))
                    .get(obj)
                    .unwrap_or_else(|| die(&format!("no z-score for {obj}")))
                    .num();
                if !z.is_finite() {
                    die(&format!("z-score for {obj} is not finite"));
                }
                Probe { file, truth: p.need("truth").truthy(), pred: p.need("pred").truthy(), z }
            })
            .collect()
    };

    let mut compared = 0usize;
    let mut mismatches = 0usize;

    // ---- 1. every row of the published trade-off curve ---------------------
    // src/vlmhall/verify.py sweeps numpy.arange(-1.5, 3.01, 0.1), which is
    // start + i*step in f64, and rounds only the threshold it prints.
    println!("rebuilding the trade-off curve from the z-scores and the probe answers");
    for style in styles {
        let all = probes_for(style);
        let held: Vec<&Probe> = all.iter().filter(|p| test.contains(p.file.as_str())).collect();
        if held.is_empty() {
            die("no held-out probes");
        }
        let n_present = held.iter().filter(|p| p.truth).count() as f64;
        let n_absent = held.iter().filter(|p| !p.truth).count() as f64;
        let published = ver.need("results").need(style).need("tradeoff").arr();
        if published.len() != 46 {
            die("the published trade-off is not 46 rows");
        }
        for (i, row) in published.iter().enumerate() {
            let t = -1.5 + (i as f64) * 0.1;
            let (mut tp, mut fp) = (0.0, 0.0);
            for p in &held {
                if p.pred && p.z >= t {
                    if p.truth {
                        tp += 1.0;
                    } else {
                        fp += 1.0;
                    }
                }
            }
            let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
            let rec = if n_present > 0.0 { tp / n_present } else { 0.0 };
            let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
            let hall = if n_absent > 0.0 { fp / n_absent } else { 0.0 };
            let got = [(t * 100.0).round() / 100.0, hall, rec, prec, f1];
            let want = [
                row.need("threshold").num(),
                row.need("hallucination_rate").num(),
                row.need("recall").num(),
                row.need("precision").num(),
                row.need("f1").num(),
            ];
            let names = ["threshold", "hallucination_rate", "recall", "precision", "f1"];
            for k in 0..5 {
                compared += 1;
                if (got[k] - want[k]).abs() > 1e-12 {
                    println!(
                        "MISMATCH {style} row {i} {}: published {:.17} recomputed {:.17}",
                        names[k], want[k], got[k]
                    );
                    mismatches += 1;
                }
            }
        }
    }
    println!("  compared {compared} published figures across 3 phrasings x 46 rows, {mismatches} mismatches");

    // ---- 2. exhaustive recalibration --------------------------------------
    // The fit that produced the published threshold only looked at 81 points.
    // 9001 of them cost nothing here, and answer whether the coarse grid was
    // lucky: how wide is the plateau it landed on, and is there a better one?
    let probes = probes_for("neutral"); // CLIP alone does not depend on phrasing
    let cal_probes: Vec<Probe> = probes.iter().filter(|p| cal.contains(p.file.as_str())).cloned().collect();
    let pub_t = ver.need("results").need("neutral").need("threshold").num();
    let pub_f1 = ver.need("results").need("neutral").need("calibration_f1").num();

    let steps = 9001;
    let mut best = f64::NEG_INFINITY;
    let mut best_t = 0.0;
    for i in 0..steps {
        let t = -1.5 + (i as f64) * 0.0005;
        let f1 = clip_f1(&cal_probes, t);
        if f1 > best {
            best = f1;
            best_t = t;
        }
    }
    // The contiguous run of thresholds that reach the published F1 and contain
    // the published threshold: the width of that run is how much slack the
    // fitted number has.
    let mut lo = pub_t;
    let mut hi = pub_t;
    {
        let step = 0.0005;
        let at = |t: f64| (clip_f1(&cal_probes, t) - pub_f1).abs() <= 1e-12;
        if !at(pub_t) {
            println!("MISMATCH published threshold {pub_t} does not reach the published calibration F1 {pub_f1}");
            mismatches += 1;
        }
        while lo - step >= -1.5 && at(lo - step) {
            lo -= step;
        }
        while hi + step <= 3.0 && at(hi + step) {
            hi += step;
        }
    }
    compared += 1;
    println!(
        "exhaustive recalibration: {steps} thresholds, best F1 {best:.10} at z >= {best_t:+.4}; \
published z >= {pub_t:+.4} scores {pub_f1:.10}"
    );
    println!("  the published threshold sits in a plateau [{lo:+.4}, {hi:+.4}], width {:.4}", hi - lo);
    if best > pub_f1 + 1e-12 {
        println!("  a finer grid finds a better calibration F1 than the 0.05 grid that was fitted");
    } else {
        println!("  no threshold anywhere beats the one the 0.05 grid found");
    }

    // Leave one calibration image out and refit on the grid that was used.
    let cal_files: Vec<&str> = cal.iter().copied().collect();
    let mut same = 0;
    let mut loo_lo = f64::INFINITY;
    let mut loo_hi = f64::NEG_INFINITY;
    for drop in &cal_files {
        let sub: Vec<Probe> = cal_probes.iter().filter(|p| p.file != *drop).cloned().collect();
        let mut bf = f64::NEG_INFINITY;
        let mut bt = 0.0;
        for i in 0..81 {
            let t = -1.0 + (i as f64) * 0.05;
            let f1 = clip_f1(&sub, t);
            if f1 > bf {
                bf = f1;
                bt = t;
            }
        }
        if (bt - pub_t).abs() < 1e-9 {
            same += 1;
        }
        loo_lo = loo_lo.min(bt);
        loo_hi = loo_hi.max(bt);
    }
    println!(
        "leave-one-image-out refits: {same} of {} keep z >= {pub_t:+.2}, range [{loo_lo:+.2}, {loo_hi:+.2}]",
        cal_files.len()
    );

    // ---- 3. a bootstrap big enough to see the shape ------------------------
    // R does 10,000 of these. A million costs about a second here, and the
    // quantity being resampled is a difference of one or two events, so the
    // sampling distribution is lumpy and worth resolving properly.
    const B: usize = 1_000_000;
    println!("image-level bootstrap, {B} resamples per phrasing");
    for style in styles {
        let all = probes_for(style);
        let t = ver.need("results").need(style).need("threshold").num();
        let held: Vec<&Probe> = all
            .iter()
            .filter(|p| test.contains(p.file.as_str()) && !p.truth)
            .collect();
        let files: Vec<String> = {
            let mut f: Vec<String> = held.iter().map(|p| p.file.clone()).collect();
            f.sort();
            f.dedup();
            f
        };
        // Per image: how many absent probes the VLM answered yes to, and how
        // many survive the CLIP veto.
        let per: Vec<(u32, u32)> = files
            .iter()
            .map(|f| {
                let b = held.iter().filter(|p| &p.file == f && p.pred).count() as u32;
                let v = held.iter().filter(|p| &p.file == f && p.pred && p.z >= t).count() as u32;
                (b, v)
            })
            .collect();
        let fp_b: u32 = per.iter().map(|x| x.0).sum();
        let fp_v: u32 = per.iter().map(|x| x.1).sum();
        let point = 1.0 - f64::from(fp_v) / f64::from(fp_b);

        let mut rng = Rng(20_260_814);
        let mut vals: Vec<f64> = Vec::with_capacity(B);
        let mut n_zero_or_worse = 0usize;
        let mut degenerate = 0usize;
        let k = per.len() as u64;
        for _ in 0..B {
            let (mut b, mut v) = (0u32, 0u32);
            for _ in 0..k {
                let (x, y) = per[rng.below(k) as usize];
                b += x;
                v += y;
            }
            if b == 0 {
                degenerate += 1;
                continue;
            }
            let r = 1.0 - f64::from(v) / f64::from(b);
            if r <= 0.0 {
                n_zero_or_worse += 1;
            }
            vals.push(r);
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| vals[((vals.len() as f64 - 1.0) * p).round() as usize];
        println!(
            "  {style:<13} fp {fp_b} -> {fp_v}, reduction {point:.3}, 95% [{:.3}, {:.3}], \
P(no reduction) {:.3}, {degenerate} resamples had no baseline error",
            q(0.025),
            q(0.975),
            n_zero_or_worse as f64 / vals.len() as f64
        );
    }

    println!("\n{compared} checked figures, {mismatches} mismatches");
    if mismatches > 0 {
        exit(1);
    }
    if compared != 691 {
        println!("FAIL expected 691 comparisons (3 x 46 x 5 plus the calibration point)");
        exit(1);
    }
}
