#!/usr/bin/env bash
# Recompute the published evaluation numbers in every language here.
#
# Everything in reports/ comes out of one Python path, and the tests check that
# the path runs rather than that the metrics are right. On an eval set this
# small a counting error moves a headline by several points and still looks
# entirely reasonable, so each implementation below rederives something
# published from the raw per-probe records.
#
# Each is skipped with a message if its toolchain is missing. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else printf 'FAILED: %s\n' "$name"; fail=$((fail + 1)); fi
}

# stdin must be closed: sqlite3 reads it, and inside this script that is the
# script itself. Its CSV writer emits CRLF, so strip the carriage returns
# before anchoring a pattern to end of line.
check_sql () {
    local out
    out=$(sqlite3 -init verify/metrics.sql :memory: "" < /dev/null 2>&1 | tr -d '\r') || return 1
    printf '%s\n' "$out"
    ! printf '%s' "$out" | grep -qiE '(^|,)(FAIL|mismatch)'
}

check_c ()    { cc -std=c99 -O2 -Wall -Wextra -Wpedantic -o "$tmp/cascade" verify/cascade.c -lm && "$tmp/cascade" "$root"; }
check_go ()   { ( cd verify/gocheck && go run . -root "$root" ); }
check_rust () { ( cd verify/sweep && cargo run --release --quiet -- "$root" ); }

run "SQL, the per-probe aggregation"        sqlite3 check_sql
run "C, the verification cascade"           cc      check_c
run "Go, files, splits and probe coverage"  go      check_go
run "R, calibration and bootstrap"          Rscript Rscript verify/calibrate.R "$root"
run "Ruby, the prose against the data"      ruby    ruby verify/docs_check.rb "$root"
run "Node, captions against verified labels" node   node verify/captions.js "$root"
run "Rust, the whole threshold sweep"       cargo   check_rust

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
