// Structural validation of every tracked results file, plus a recompute of the
// evaluation-set counts the README publishes.
//
// The other checks in verify/ each recompute one published table. This one
// checks the things that would make those recomputes agree on a wrong answer:
// a probe that is in the results but not in the hand-built ground truth, a
// duplicate key that json silently collapses, a NaN in the z-scores, a
// calibration image that also appears in the held-out split. It reads every
// tracked JSON file under data/ and reports/.
//
// Run: cd verify/gocheck && go run . -root ..
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Counts published in README.md section 2 and in the header of
// reports/results.md. Changing the eval set without changing the prose must
// fail here.
const (
	wantImages      = 33
	wantPresent     = 172
	wantAbsent      = 67
	wantAdversarial = 44
	wantRandom      = 21
	wantPopular     = 2
	wantProbes      = 239 // present + absent
	wantMeanInst    = 23  // "Mean 23 annotated object instances per image"
	wantVocab       = 80  // COCO's 80 categories
	wantCalibration = 11
	wantHeldOut     = 22
	wantTradeoff    = 46
	wantCaptionHall = 0 // "Zero of 33 captions mentioned a verified-absent object"
)

var problems []string

func fail(format string, a ...any) {
	problems = append(problems, fmt.Sprintf(format, a...))
}

func check(cond bool, format string, a ...any) {
	if !cond {
		fail(format, a...)
	}
}

type evalRow struct {
	FileName   string            `json:"file_name"`
	Scene      string            `json:"scene"`
	NInstances int               `json:"n_instances"`
	Present    []string          `json:"present"`
	Absent     map[string]string `json:"absent"`
}

type probe struct {
	FileName string `json:"file_name"`
	Object   string `json:"object"`
	Truth    bool   `json:"truth"`
	Kind     string `json:"kind"`
	Answer   string `json:"answer"`
	Pred     *bool  `json:"pred"`
}

type caption struct {
	FileName     string   `json:"file_name"`
	Caption      string   `json:"caption"`
	Mentioned    []string `json:"mentioned"`
	Grounded     []string `json:"grounded"`
	Hallucinated []string `json:"hallucinated"`
}

type stylePayload struct {
	Probes  []probe                       `json:"probes"`
	Metrics map[string]map[string]float64 `json:"metrics"`
}

type baselineFile struct {
	Probes   []probe   `json:"probes"`
	Captions []caption `json:"captions"`
}

type tradeoffRow struct {
	Threshold         float64 `json:"threshold"`
	HallucinationRate float64 `json:"hallucination_rate"`
	Recall            float64 `json:"recall"`
	Precision         float64 `json:"precision"`
	F1                float64 `json:"f1"`
}

type styleResult struct {
	Threshold float64       `json:"threshold"`
	Tradeoff  []tradeoffRow `json:"tradeoff"`
}

type verificationFile struct {
	Results     map[string]styleResult        `json:"results"`
	Calibration []string                      `json:"calibration_images"`
	TestImages  []string                      `json:"test_images"`
	ZScores     map[string]map[string]float64 `json:"z_scores"`
}

func readJSON(path string, v any) []byte {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "cannot read", path, err)
		os.Exit(1)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		fmt.Fprintln(os.Stderr, "cannot parse", path, err)
		os.Exit(1)
	}
	return raw
}

// encoding/json keeps the last of a repeated key, exactly as Python's json
// does, so a file with "pred" twice decodes without complaint in both and the
// duplicate is invisible. Walk the token stream and reject it.
func rejectDuplicateKeys(path string, raw []byte) {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	var walk func(trail string)
	walk = func(trail string) {
		tok, err := dec.Token()
		if err != nil {
			if err != io.EOF {
				fail("%s: %v", path, err)
			}
			return
		}
		delim, isDelim := tok.(json.Delim)
		if !isDelim {
			return
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for dec.More() {
				k, err := dec.Token()
				if err != nil {
					fail("%s: %v", path, err)
					return
				}
				key := k.(string)
				if seen[key] {
					fail("%s: duplicate key %q under %s", path, key, trail)
				}
				seen[key] = true
				walk(trail + "/" + key)
			}
			dec.Token() // '}'
		case '[':
			i := 0
			for dec.More() {
				walk(fmt.Sprintf("%s[%d]", trail, i))
				i++
			}
			dec.Token() // ']'
		}
	}
	walk("$")
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func finite(vals ...float64) bool {
	for _, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()
	p := func(rel string) string { return filepath.Join(*root, rel) }

	var rows []evalRow
	var styles map[string]stylePayload
	var base baselineFile
	var ver verificationFile

	files := map[string]any{
		"data/eval_set.json":         &rows,
		"reports/prompt_styles.json": &styles,
		"reports/baseline.json":      &base,
		"reports/verification.json":  &ver,
	}
	for _, rel := range sortedKeys(files) {
		raw := readJSON(p(rel), files[rel])
		rejectDuplicateKeys(rel, raw)
	}

	// --- the vocabulary, taken from the data rather than transcribed ---------
	// Every image's z-score block is CLIP scored against all 80 COCO prompts,
	// so the key set of any one block is the vocabulary. Requiring all 33 to
	// agree means a typo'd category name cannot hide in one image.
	var vocab map[string]bool
	var vocabFrom string
	for _, f := range sortedKeys(ver.ZScores) {
		keys := ver.ZScores[f]
		if vocab == nil {
			vocab = map[string]bool{}
			for k := range keys {
				vocab[k] = true
			}
			vocabFrom = f
			continue
		}
		if len(keys) != len(vocab) {
			fail("z_scores: %s has %d categories, %s has %d", f, len(keys), vocabFrom, len(vocab))
			continue
		}
		for k := range keys {
			if !vocab[k] {
				fail("z_scores: %s has category %q that %s does not", f, k, vocabFrom)
			}
		}
	}
	check(len(vocab) == wantVocab, "vocabulary is %d categories, README says %d", len(vocab), wantVocab)
	check(len(ver.ZScores) == wantImages, "z_scores covers %d images, README says %d", len(ver.ZScores), wantImages)

	// --- z-scores are what they claim to be ---------------------------------
	// Each block is (sim - mean) / (std + 1e-8) over the same 80 prompts, so
	// every block must have mean 0 and standard deviation 1 to within the
	// epsilon. A block copied from another image, rescaled or truncated fails.
	worstMean, worstSD := 0.0, 0.0
	for _, f := range sortedKeys(ver.ZScores) {
		var sum, sumsq float64
		n := 0.0
		for _, v := range ver.ZScores[f] {
			if !finite(v) {
				fail("z_scores: %s has a non-finite score", f)
				continue
			}
			sum += v
			sumsq += v * v
			n++
		}
		mean := sum / n
		sd := math.Sqrt(sumsq/n - mean*mean)
		worstMean = math.Max(worstMean, math.Abs(mean))
		worstSD = math.Max(worstSD, math.Abs(sd-1))
	}
	check(worstMean < 1e-9, "a z-score block has mean %.3g, expected 0", worstMean)
	check(worstSD < 1e-6, "a z-score block has sd off by %.3g, expected 1", worstSD)

	// --- the evaluation set, and the counts the README quotes ---------------
	nPresent, nAbsent, totalInst := 0, 0, 0
	byKind := map[string]int{}
	truth := map[string]bool{} // file|object -> present
	kindOf := map[string]string{}
	seenFile := map[string]bool{}
	for _, r := range rows {
		check(!seenFile[r.FileName], "eval_set: %s appears twice", r.FileName)
		seenFile[r.FileName] = true
		totalInst += r.NInstances
		check(r.NInstances > 0, "eval_set: %s has no annotated instances", r.FileName)
		check(strings.TrimSpace(r.Scene) != "", "eval_set: %s has no scene description", r.FileName)
		seenObj := map[string]bool{}
		for _, o := range r.Present {
			check(vocab[o], "eval_set: %s present %q is not a COCO category", r.FileName, o)
			check(!seenObj[o], "eval_set: %s probes %q twice", r.FileName, o)
			seenObj[o] = true
			truth[r.FileName+"|"+o] = true
			kindOf[r.FileName+"|"+o] = "present"
			nPresent++
		}
		for _, o := range sortedKeys(r.Absent) {
			k := r.Absent[o]
			check(vocab[o], "eval_set: %s absent %q is not a COCO category", r.FileName, o)
			check(!seenObj[o], "eval_set: %s marks %q both present and absent", r.FileName, o)
			seenObj[o] = true
			check(k == "adversarial" || k == "random" || k == "popular",
				"eval_set: %s %q has unknown split %q", r.FileName, o, k)
			byKind[k]++
			truth[r.FileName+"|"+o] = false
			kindOf[r.FileName+"|"+o] = k
			nAbsent++
		}
	}
	check(len(rows) == wantImages, "eval_set has %d images, README says %d", len(rows), wantImages)
	check(nPresent == wantPresent, "eval_set has %d present probes, README says %d", nPresent, wantPresent)
	check(nAbsent == wantAbsent, "eval_set has %d absent probes, README says %d", nAbsent, wantAbsent)
	check(byKind["adversarial"] == wantAdversarial, "adversarial is %d, README says %d", byKind["adversarial"], wantAdversarial)
	check(byKind["random"] == wantRandom, "random is %d, README says %d", byKind["random"], wantRandom)
	check(byKind["popular"] == wantPopular, "popular is %d, README says %d", byKind["popular"], wantPopular)
	check(nPresent+nAbsent == wantProbes, "eval_set yields %d probes, README says %d", nPresent+nAbsent, wantProbes)
	meanInst := float64(totalInst) / float64(len(rows))
	check(int(math.Round(meanInst)) == wantMeanInst,
		"mean annotated instances is %.2f, README rounds to %d", meanInst, wantMeanInst)

	// --- probes agree with the ground truth, in every phrasing --------------
	styleNames := sortedKeys(styles)
	check(len(styleNames) == 3, "expected 3 question phrasings, found %d", len(styleNames))
	for _, s := range styleNames {
		ps := styles[s].Probes
		check(len(ps) == wantProbes, "%s has %d probes, expected %d", s, len(ps), wantProbes)
		seen := map[string]bool{}
		for _, pr := range ps {
			key := pr.FileName + "|" + pr.Object
			check(!seen[key], "%s: probe %s repeated", s, key)
			seen[key] = true
			want, ok := truth[key]
			if !ok {
				fail("%s: probe %s is not in the hand-built eval set", s, key)
				continue
			}
			check(pr.Truth == want, "%s: probe %s truth is %v, eval set says %v", s, key, pr.Truth, want)
			check(pr.Kind == kindOf[key], "%s: probe %s split is %q, eval set says %q", s, key, pr.Kind, kindOf[key])
			check(pr.Pred != nil, "%s: probe %s has an unparsed answer, but unparsed is published as 0", s, key)
			check(strings.TrimSpace(pr.Answer) != "", "%s: probe %s has an empty answer", s, key)
		}
		check(len(seen) == wantProbes, "%s covers %d distinct probes, expected %d", s, len(seen), wantProbes)
	}

	// --- baseline.json is the neutral phrasing ------------------------------
	// The README's section 1 table and the neutral row of the phrasing table
	// are the same run. If they ever diverge, one of the two is stale.
	neutral := styles["neutral"].Probes
	check(len(base.Probes) == len(neutral), "baseline has %d probes, neutral has %d", len(base.Probes), len(neutral))
	if len(base.Probes) == len(neutral) {
		for i := range neutral {
			a, b := base.Probes[i], neutral[i]
			if a.FileName != b.FileName || a.Object != b.Object || a.Truth != b.Truth ||
				a.Kind != b.Kind || a.Answer != b.Answer ||
				(a.Pred == nil) != (b.Pred == nil) || (a.Pred != nil && *a.Pred != *b.Pred) {
				fail("baseline.json probe %d differs from prompt_styles neutral", i)
				break
			}
		}
	}

	// --- captions --------------------------------------------------------
	check(len(base.Captions) == wantImages, "%d captions, expected %d", len(base.Captions), wantImages)
	nHall := 0
	for _, c := range base.Captions {
		check(strings.TrimSpace(c.Caption) != "", "caption for %s is empty", c.FileName)
		ment := map[string]bool{}
		for _, m := range c.Mentioned {
			ment[m] = true
			check(vocab[m], "caption %s mentions %q which is not a COCO category", c.FileName, m)
		}
		for _, g := range c.Grounded {
			check(ment[g], "caption %s: grounded %q is not in mentioned", c.FileName, g)
			check(truth[c.FileName+"|"+g], "caption %s: grounded %q is not verified present", c.FileName, g)
		}
		for _, h := range c.Hallucinated {
			check(ment[h], "caption %s: hallucinated %q is not in mentioned", c.FileName, h)
			v, ok := truth[c.FileName+"|"+h]
			check(ok && !v, "caption %s: hallucinated %q is not verified absent", c.FileName, h)
		}
		if len(c.Hallucinated) > 0 {
			nHall++
		}
	}
	check(nHall == wantCaptionHall, "%d captions name a verified-absent object, README says %d", nHall, wantCaptionHall)

	// --- the calibration / held-out split -----------------------------------
	check(len(ver.Calibration) == wantCalibration, "%d calibration images, expected %d", len(ver.Calibration), wantCalibration)
	check(len(ver.TestImages) == wantHeldOut, "%d held-out images, expected %d", len(ver.TestImages), wantHeldOut)
	inSplit := map[string]int{}
	for _, f := range ver.Calibration {
		inSplit[f]++
	}
	for _, f := range ver.TestImages {
		inSplit[f] += 10
	}
	for _, f := range sortedKeys(inSplit) {
		check(inSplit[f] == 1 || inSplit[f] == 10, "%s is in both splits or listed twice", f)
		check(seenFile[f], "%s is in a split but not in the eval set", f)
	}
	for _, r := range rows {
		check(inSplit[r.FileName] != 0, "%s is in the eval set but in neither split", r.FileName)
	}

	// --- the trade-off curve is a curve -------------------------------------
	// Raising the threshold can only delete accepted answers, so hallucination
	// rate and recall must both be non-increasing. A curve that goes back up
	// would mean the sweep is not doing what results.md says it does.
	for _, s := range styleNames {
		res, ok := ver.Results[s]
		if !ok {
			fail("verification.json has no results for %s", s)
			continue
		}
		rowsT := res.Tradeoff
		check(len(rowsT) == wantTradeoff, "%s trade-off has %d rows, expected %d", s, len(rowsT), wantTradeoff)
		for i, r := range rowsT {
			check(finite(r.Threshold, r.HallucinationRate, r.Recall, r.Precision, r.F1),
				"%s trade-off row %d has a non-finite value", s, i)
			check(r.HallucinationRate >= 0 && r.HallucinationRate <= 1 &&
				r.Recall >= 0 && r.Recall <= 1 && r.Precision >= 0 && r.Precision <= 1 &&
				r.F1 >= 0 && r.F1 <= 1, "%s trade-off row %d has a rate outside [0,1]", s, i)
			if i > 0 {
				prev := rowsT[i-1]
				check(r.Threshold > prev.Threshold, "%s trade-off thresholds are not increasing at row %d", s, i)
				check(r.HallucinationRate <= prev.HallucinationRate+1e-12,
					"%s hallucination rises from %.4f to %.4f at threshold %.2f",
					s, prev.HallucinationRate, r.HallucinationRate, r.Threshold)
				check(r.Recall <= prev.Recall+1e-12,
					"%s recall rises from %.4f to %.4f at threshold %.2f",
					s, prev.Recall, r.Recall, r.Threshold)
			}
		}
		check(res.Threshold >= -1.0 && res.Threshold <= 3.0,
			"%s fitted threshold %.4f is outside the grid that was swept", s, res.Threshold)
	}

	fmt.Printf("eval set: %d images, %d present and %d absent probes (%d adversarial, %d random, %d popular), %d in total\n",
		len(rows), nPresent, nAbsent, byKind["adversarial"], byKind["random"], byKind["popular"], nPresent+nAbsent)
	fmt.Printf("mean annotated instances per image: %.2f\n", meanInst)
	fmt.Printf("vocabulary: %d categories, identical across all %d z-score blocks\n", len(vocab), len(ver.ZScores))
	fmt.Printf("z-score blocks: worst |mean| %.3g, worst |sd - 1| %.3g\n", worstMean, worstSD)
	fmt.Printf("split: %d calibration + %d held-out images, disjoint and covering the set\n",
		len(ver.Calibration), len(ver.TestImages))
	fmt.Printf("probes: %d per phrasing x %d phrasings, all matched to the eval set\n", wantProbes, len(styleNames))

	if len(problems) > 0 {
		fmt.Printf("\n%d structural problems:\n", len(problems))
		for _, p := range problems {
			fmt.Println("  " + p)
		}
		os.Exit(1)
	}
	fmt.Println("no structural problems")
}
