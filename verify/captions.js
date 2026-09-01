// Re-extract the COCO objects each caption mentions, from the caption text.
//
// The caption side of the measurement is the one claim in the README with no
// confusion matrix behind it: "Zero of 33 captions mentioned a verified-absent
// object", and the argument that follows it, that a 0% rate is not evidence of
// grounding. That zero is produced by one regex in
// src/vlmhall/evaluate.py:mentioned_objects. If that regex quietly matched
// nothing, the zero would look exactly the same.
//
// So this re-derives `mentioned`, `grounded` and `hallucinated` for all 33
// captions from the caption strings and data/eval_set.json, and requires them
// to equal what reports/baseline.json publishes. The matching rule has two
// details worth a second implementation: a naive plural, so "dogs" counts and
// "hotdog" does not, and word boundaries built from spaces rather than \b.
//
// Honest limitation: the synonym table below is transcribed from the Python,
// so a wrong entry there would be wrong in both. What is independently checked
// is the matching rule and every count that comes out of it. The 80 category
// names are not transcribed, they are read out of reports/verification.json.
//
// Run: node verify/captions.js <repo-root>

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const read = (rel) => JSON.parse(fs.readFileSync(path.join(root, rel), "utf8"));

const baseline = read("reports/baseline.json");
const evalSet = read("data/eval_set.json");
const vocabBlocks = read("reports/verification.json").z_scores;

// The 80 COCO categories, taken from the data rather than copied out of the
// source: every image was CLIP-scored against all of them.
const anyBlock = vocabBlocks[Object.keys(vocabBlocks)[0]];
const COCO = Object.keys(anyBlock);
if (COCO.length !== 80) {
  console.log(`FAIL vocabulary has ${COCO.length} categories, expected 80`);
  process.exit(1);
}

const SYNONYMS = {
  sofa: "couch", settee: "couch", tv: "tv", television: "tv",
  monitor: "tv", screen: "tv", cellphone: "cell phone",
  "mobile phone": "cell phone", phone: "cell phone",
  motorbike: "motorcycle", bike: "bicycle", cycle: "bicycle",
  plane: "airplane", aeroplane: "airplane", aircraft: "airplane",
  jet: "airplane", lorry: "truck", van: "truck", auto: "car",
  automobile: "car", table: "dining table", desk: "dining table",
  doughnut: "donut", hotdog: "hot dog", puppy: "dog", kitten: "cat",
  cattle: "cow", calf: "cow", lamb: "sheep", pony: "horse",
  man: "person", woman: "person", boy: "person", girl: "person",
  child: "person", people: "person", men: "person", women: "person",
  player: "person", skier: "person", surfer: "person", rider: "person",
  glass: "wine glass", mug: "cup", plant: "potted plant",
  fridge: "refrigerator", stove: "oven", computer: "laptop",
  "remote control": "remote", purse: "handbag", bag: "handbag",
  luggage: "suitcase", racket: "tennis racket", bat: "baseball bat",
  glove: "baseball glove", ball: "sports ball", board: "surfboard",
  signal: "traffic light", stoplight: "traffic light",
};

for (const canon of Object.values(SYNONYMS)) {
  if (!COCO.includes(canon)) {
    console.log(`FAIL synonym maps to ${canon}, which is not a COCO category`);
    process.exit(1);
  }
}

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function mentionedObjects(text) {
  let t = " " + text.toLowerCase().replace(/[^a-z ]/g, " ") + " ";
  t = t.replace(/\s+/g, " ");
  const found = new Set();
  const pairs = Object.entries(SYNONYMS).concat(COCO.map((c) => [c, c]));
  for (const [surface, canon] of pairs) {
    if (new RegExp(` ${escape(surface)}(s|es)? `).test(t)) found.add(canon);
  }
  return found;
}

const groundTruth = new Map(); // file -> {present:Set, absent:Set}
for (const r of evalSet) {
  groundTruth.set(r.file_name, {
    present: new Set(r.present),
    absent: new Set(Object.keys(r.absent)),
  });
}

const sortedArray = (s) => [...s].sort();
const sameList = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

let problems = 0;
let checked = 0;
let withHallucination = 0;
let totalMentions = 0;

for (const c of baseline.captions) {
  const gt = groundTruth.get(c.file_name);
  if (!gt) {
    console.log(`FAIL caption for ${c.file_name} has no eval-set entry`);
    problems++;
    continue;
  }
  const mentioned = mentionedObjects(c.caption);
  const grounded = sortedArray(new Set([...mentioned].filter((m) => gt.present.has(m))));
  const hallucinated = sortedArray(new Set([...mentioned].filter((m) => gt.absent.has(m))));
  const mentionedList = sortedArray(mentioned);
  totalMentions += mentionedList.length;
  if (hallucinated.length > 0) withHallucination++;

  const compare = [
    ["mentioned", mentionedList, c.mentioned],
    ["grounded", grounded, c.grounded],
    ["hallucinated", hallucinated, c.hallucinated],
  ];
  for (const [field, got, want] of compare) {
    checked++;
    if (!sameList(got, want)) {
      console.log(
        `MISMATCH ${c.file_name} ${field}: published [${want}] recomputed [${got}]`
      );
      problems++;
    }
  }
}

console.log(`${baseline.captions.length} captions, ${totalMentions} object mentions re-extracted`);
console.log(
  `captions naming a verified-absent object: ${withHallucination} of ${baseline.captions.length}`
);
console.log(`${checked} published lists compared, ${problems} mismatches`);

// A few cases where the plural rule and the word boundaries decide the answer.
// If these ever stop holding the re-extraction above is not the same matcher.
const rules = [
  ["two dogs on a couch", ["couch", "dog"]],
  ["a hotdog on a plate", ["hot dog"]],
  ["a man riding a motorbike", ["motorcycle", "person"]],
  ["scissors and a toothbrush", ["scissors", "toothbrush"]],
  ["nothing recognisable here", []],
];
for (const [text, want] of rules) {
  const got = sortedArray(mentionedObjects(text));
  checked++;
  if (!sameList(got, want.slice().sort())) {
    console.log(`MISMATCH matcher rule "${text}": expected [${want}] got [${got}]`);
    problems++;
  }
}

if (checked !== baseline.captions.length * 3 + rules.length) {
  console.log(`FAIL only ${checked} comparisons ran`);
  process.exit(1);
}
if (problems > 0) process.exit(1);
