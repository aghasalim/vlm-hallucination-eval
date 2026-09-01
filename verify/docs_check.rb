# Every number printed in the write-up, checked against the file it came from.
#
# The other checks in verify/ recompute the JSON from rawer JSON. None of them
# looks at the prose, and the prose is where I am most likely to be wrong: a
# table copied by hand, a percentage quoted in an abstract written before the
# last run, a claim like "more than doubles" that was true of an earlier
# measurement. Markdown does not get regenerated when the numbers move.
#
# So this walks README.md, reports/results.md and notes/METHODS.md, pulls out
# every published figure, and requires each to be the correct rounding of the
# value in reports/*.json or data/eval_set.json. Percentages are checked to the
# printed decimal place, not to full precision, because that is the claim being
# made.
#
# Run: ruby verify/docs_check.rb <repo-root>

require "json"

# Ruby 2.6 opens files as US-ASCII. These documents carry arrows and minus
# signs, so every read below is explicit about UTF-8: without it a regex
# raises on the first non-ASCII byte instead of reporting a mismatch.

ROOT = ARGV[0] || "."
def load_json(rel)
  JSON.parse(File.read(File.join(ROOT, rel), encoding: "UTF-8"))
end

V = load_json("reports/verification.json")
S = load_json("reports/prompt_styles.json")
E = load_json("data/eval_set.json")
B = load_json("reports/baseline.json")

STYLES = %w[neutral leading presupposing].freeze

$checked = 0
$problems = []

def fail!(msg)
  $problems << msg
end

# The claim a printed percentage makes is "this rounds to here", so that is what
# is checked. 4.8% must be the right one-decimal rounding of 0.047619...
def pct(where, printed, value, dp = 1)
  $checked += 1
  got = Float(printed.to_s.sub("%", "").strip)
  want = value * 100.0
  tol = 0.5 * 10.0**(-dp) + 1e-9
  fail!("#{where}: prints #{got}% but the data gives #{format('%.4f', want)}%") if (got - want).abs > tol
end

def frac(where, printed, value, dp = 3)
  $checked += 1
  got = Float(printed.to_s.strip)
  tol = 0.5 * 10.0**(-dp) + 1e-9
  fail!("#{where}: prints #{got} but the data gives #{format('%.6f', value)}") if (got - value).abs > tol
end

def int(where, printed, value)
  $checked += 1
  got = Integer(printed.to_s.strip)
  fail!("#{where}: prints #{got} but the data gives #{value}") if got != value
end

# --- markdown tables -------------------------------------------------------
# Cells with the emphasis, the unicode minus and the arrow normalised away, so
# "**4.8%**" and "4.8%" are the same published figure.
def table_rows(path)
  File.read(File.join(ROOT, path), encoding: "UTF-8").lines.select { |l| l.strip.start_with?("|") }.map do |l|
    l.strip.sub(/\A\|/, "").sub(/\|\z/, "")
      .split("|").map { |c| c.gsub("*", "").gsub("−", "-").gsub("→", "->").strip }
  end.reject { |cells| cells.all? { |c| c.match?(/\A:?-+:?\z/) } }
end

def row!(rows, path, size, &pred)
  hit = rows.select { |c| c.size == size && pred.call(c) }
  fail!("#{path}: no #{size}-column row matched") if hit.empty?
  fail!("#{path}: #{hit.size} rows matched where one was expected") if hit.size > 1
  hit.first
end

results = table_rows("reports/results.md")
methods = table_rows("notes/METHODS.md")

# reports/results.md section 1: the baseline table.
STYLES.each do |st|
  m = S[st]["metrics"]
  r = row!(results, "results.md s1", 6) { |c| c[0] == st }
  next if r.nil?
  pct("results.md s1 #{st} accuracy",   r[1], m["overall"]["accuracy"])
  pct("results.md s1 #{st} yes-rate",   r[2], m["overall"]["yes_rate"])
  pct("results.md s1 #{st} recall",     r[3], m["present"]["recall"])
  pct("results.md s1 #{st} adversarial", r[4], m["adversarial"]["hallucination_rate"])
  pct("results.md s1 #{st} absent",     r[5], m["absent_all"]["hallucination_rate"])

  # section 2: the phrasing table, same figure quoted a second time.
  r2 = row!(results, "results.md s2", 3) { |c| c[0] == st }
  pct("results.md s2 #{st}", r2[2], m["absent_all"]["hallucination_rate"]) if r2

  # notes/METHODS.md quotes it a third time.
  r3 = row!(methods, "METHODS.md phrasing", 3) { |c| c[0] == st }
  pct("METHODS.md #{st}", r3[2], m["absent_all"]["hallucination_rate"]) if r3
end

# reports/results.md section 3: the three decision rules.
RULES = { "baseline (VLM alone)" => "baseline", "CLIP alone" => "clip_only", "VLM AND CLIP" => "verified" }.freeze
STYLES.each do |st|
  RULES.each do |label, key|
    res = V["results"][st][key]
    r = row!(results, "results.md s3", 7) { |c| c[0] == st && c[1] == label }
    next if r.nil?
    pct("results.md s3 #{st}/#{label} accuracy", r[2], res["accuracy"])
    frac("results.md s3 #{st}/#{label} precision", r[3], res["precision"])
    frac("results.md s3 #{st}/#{label} recall",    r[4], res["recall"])
    frac("results.md s3 #{st}/#{label} F1",        r[5], res["f1"])
    pct("results.md s3 #{st}/#{label} hallucination", r[6], res["hallucination_rate"])
  end
end

# notes/METHODS.md quotes recall, F1 and hallucination for two of the rules.
{ "BLIP alone" => "baseline", "BLIP AND CLIP" => "verified" }.each do |label, key|
  STYLES.each do |st|
    res = V["results"][st][key]
    r = row!(methods, "METHODS.md rules", 5) { |c| c[0] == st && c[1] == label }
    next if r.nil?
    frac("METHODS.md #{st}/#{label} recall", r[2], res["recall"])
    frac("METHODS.md #{st}/#{label} F1",     r[3], res["f1"])
    pct("METHODS.md #{st}/#{label} hallucination", r[4], res["hallucination_rate"])
  end
end
r = row!(methods, "METHODS.md CLIP alone", 5) { |c| c[1] == "CLIP alone" }
if r
  res = V["results"]["neutral"]["clip_only"]
  frac("METHODS.md any/CLIP alone recall", r[2], res["recall"])
  frac("METHODS.md any/CLIP alone F1",     r[3], res["f1"])
  pct("METHODS.md any/CLIP alone hallucination", r[4], res["hallucination_rate"])
end

# The threshold sweep, quoted in both documents.
sweep = V["results"]["neutral"]["tradeoff"]
[["results.md", results], ["METHODS.md", methods]].each do |name, rows|
  hits = rows.select { |c| c.size == 4 && c[0].match?(/\A-?[\d.]+/) && c[3].match?(/\A[\d.]+\z/) }
  fail!("#{name}: found no trade-off rows") if hits.empty?
  hits.each do |c|
    t = Float(c[0][/\A-?[\d.]+/])
    row = sweep.find { |x| (x["threshold"] - t).abs < 1e-9 }
    if row.nil?
      fail!("#{name}: threshold #{t} is not a point on the published sweep")
      next
    end
    pct("#{name} sweep #{t} hallucination", c[1], row["hallucination_rate"])
    pct("#{name} sweep #{t} recall",        c[2], row["recall"])
    frac("#{name} sweep #{t} F1",           c[3], row["f1"])
  end
end

# --- prose -----------------------------------------------------------------
def prose(path)
  # Emphasis markers are stripped along with the arrows and the unicode minus.
  # Otherwise a pattern fails the moment someone bolds a number inside the
  # sentence it is checking, which reads as "the claim is gone" when the claim
  # is present and correct.
  File.read(File.join(ROOT, path), encoding: "UTF-8")
      .gsub(/\s+/, " ").gsub("→", "->").gsub("−", "-").gsub(/\*+/, "")
end

def claim!(text, where, re)
  m = re.match(text)
  fail!("#{where}: the sentence this checks is no longer in the document") if m.nil?
  m
end

n_present = E.sum { |r| r["present"].size }
n_absent  = E.sum { |r| r["absent"].size }
n_adv     = E.sum { |r| r["absent"].values.count("adversarial") }
n_rand    = E.sum { |r| r["absent"].values.count("random") }
n_pop     = E.sum { |r| r["absent"].values.count("popular") }
held      = V["results"]["neutral"]["baseline"]
n_held_absent = held["n"] - (held["tp"] + held["fn"])
rel = STYLES.to_h do |st|
  b = V["results"][st]["baseline"]["fp"].to_f
  v = V["results"][st]["verified"]["fp"].to_f
  [st, 1.0 - v / b]
end

readme = prose("README.md")

if (m = claim!(readme, "README abstract accuracy", /accuracy is nearly invariant to phrasing, ([\d.]+)%, ([\d.]+)%, ([\d.]+)% for neutral, presupposing and leading/))
  pct("README abstract neutral accuracy",      m[1], S["neutral"]["metrics"]["overall"]["accuracy"])
  pct("README abstract presupposing accuracy", m[2], S["presupposing"]["metrics"]["overall"]["accuracy"])
  pct("README abstract leading accuracy",      m[3], S["leading"]["metrics"]["overall"]["accuracy"])
end

if (m = claim!(readme, "README abstract hallucination", /more than doubles across the same three, from ([\d.]+)% to ([\d.]+)%/))
  lo = S["neutral"]["metrics"]["absent_all"]["hallucination_rate"]
  hi = S["leading"]["metrics"]["absent_all"]["hallucination_rate"]
  pct("README abstract lowest hallucination",  m[1], lo)
  pct("README abstract highest hallucination", m[2], hi)
  $checked += 1
  fail!("README says the rate more than doubles, but #{hi} is not over twice #{lo}") if hi <= 2 * lo
end

if (m = claim!(readme, "README relative reduction", /cuts the hallucination rate by (\d+) to (\d+)% relative, (\d+)% on presupposing phrasing, (\d+)% on neutral and (\d+)% on leading/))
  pct("README reduction, presupposing", m[3], rel["presupposing"], 0)
  pct("README reduction, neutral",      m[4], rel["neutral"], 0)
  pct("README reduction, leading",      m[5], rel["leading"], 0)
  int("README reduction range, low",  m[1], (rel.values.min * 100).round)
  int("README reduction range, high", m[2], (rel.values.max * 100).round)
end

deltas = STYLES.map { |st| V["results"][st]["baseline"]["fp"] - V["results"][st]["verified"]["fp"] }
[[readme, "README"], [prose("reports/results.md"), "results.md"]].each do |text, where|
  m = claim!(text, "#{where} false-positive counts", /(\d+) to (\d+) fewer false positives out of (\d+) verified-absent probes/)
  next if m.nil?
  int("#{where} fewest removed", m[1], deltas.min)
  int("#{where} most removed",   m[2], deltas.max)
  int("#{where} absent probes",  m[3], n_held_absent)
end

if (m = claim!(readme, "README eval set", /(\d+) images, (\d+) objects verified present and (\d+) verified absent \((\d+) of them adversarial\), so (\d+) yes\/no probes/))
  int("README images",      m[1], E.size)
  int("README present",     m[2], n_present)
  int("README absent",      m[3], n_absent)
  int("README adversarial", m[4], n_adv)
  int("README probes",      m[5], n_present + n_absent)
end

if (m = claim!(readme, "README small numbers", /(\d+) probes over (\d+) held-out images, (\d+) of them verified-absent/))
  int("README held-out probes", m[1], held["n"])
  int("README held-out images", m[2], V["test_images"].size)
  int("README held-out absent", m[3], n_held_absent)
end

if (m = claim!(readme, "README two errors", /([\d.]+)% is two errors/))
  pct("README two-error rate", m[1], V["results"]["neutral"]["verified"]["hallucination_rate"])
  int("README two errors", "2", V["results"]["neutral"]["verified"]["fp"])
end

int("README popular split", claim!(readme, "README popular split", /popular` probe split has only (\d+) verified items/)&.[](1) || "0", n_pop)

res_prose = prose("reports/results.md")
if (m = claim!(res_prose, "results.md header", /(\d+) hand-verified images, (\d+) objects verified present and (\d+) verified absent \((\d+) adversarial, (\d+) random, (\d+) popular\), (\d+) yes\/no probes in total\. Mean (\d+) annotated object instances per image/))
  int("results.md images",      m[1], E.size)
  int("results.md present",     m[2], n_present)
  int("results.md absent",      m[3], n_absent)
  int("results.md adversarial", m[4], n_adv)
  int("results.md random",      m[5], n_rand)
  int("results.md popular",     m[6], n_pop)
  int("results.md probes",      m[7], n_present + n_absent)
  int("results.md mean instances", m[8], (E.sum { |r| r["n_instances"] }.to_f / E.size).round)
end

if (m = claim!(res_prose, "results.md misses", /misses (\d+)% of objects that are genuinely present/))
  pct("results.md miss rate", m[1], 1.0 - S["neutral"]["metrics"]["present"]["recall"], 0)
end

if (m = claim!(res_prose, "results.md captions", /Zero of (\d+) captions mentioned a verified-absent object/))
  int("results.md caption count", m[1], B["captions"].size)
  $checked += 1
  n = B["captions"].count { |c| !c["hallucinated"].empty? }
  fail!("results.md says zero captions, but #{n} name a verified-absent object") if n != 0
end

if (m = claim!(res_prose, "results.md before and after", /Verification cuts hallucination from ([\d.]+)% to ([\d.]+)% on neutral phrasing and from ([\d.]+)% to ([\d.]+)% on leading phrasing/))
  pct("results.md neutral before", m[1], V["results"]["neutral"]["baseline"]["hallucination_rate"])
  pct("results.md neutral after",  m[2], V["results"]["neutral"]["verified"]["hallucination_rate"])
  pct("results.md leading before", m[3], V["results"]["leading"]["baseline"]["hallucination_rate"])
  pct("results.md leading after",  m[4], V["results"]["leading"]["verified"]["hallucination_rate"])
end

if (m = claim!(res_prose, "results.md recall cost", /Recall falls from ([\d.]+) to ([\d.]+), and F1 actually drops \(([\d.]+) -> ([\d.]+)\)/))
  frac("results.md recall before", m[1], V["results"]["neutral"]["baseline"]["recall"])
  frac("results.md recall after",  m[2], V["results"]["neutral"]["verified"]["recall"])
  frac("results.md F1 before",     m[3], V["results"]["neutral"]["baseline"]["f1"])
  frac("results.md F1 after",      m[4], V["results"]["neutral"]["verified"]["f1"])
end

if (m = claim!(res_prose, "results.md CLIP alone", /Thresholding CLIP on its own hallucinates on ([\d.]+)% of verified-absent objects/))
  pct("results.md CLIP alone", m[1], V["results"]["neutral"]["clip_only"]["hallucination_rate"])
end

if (m = claim!(res_prose, "results.md zero cost", /exactly zero.{0,40}?cost of roughly (\d+) points of recall/))
  off = sweep.find { |x| (x["threshold"] + 1.5).abs < 1e-9 }["recall"]
  zero = sweep.select { |x| x["hallucination_rate"] == 0.0 }.max_by { |x| x["recall"] }
  fail!("results.md claims hallucination reaches exactly zero, but no swept threshold does") if zero.nil?
  pct("results.md recall cost", m[1], off - (zero ? zero["recall"] : 0.0), 0) if zero
end

puts "#{$checked} published figures checked across README.md, reports/results.md and notes/METHODS.md"
if $problems.empty?
  puts "every one is the correct rounding of the value in the data"
else
  puts "#{$problems.size} problems:"
  $problems.each { |p| puts "  #{p}" }
  exit 1
end
exit 1 if $checked < 150
