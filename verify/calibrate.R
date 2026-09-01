# Refit the CLIP threshold and put error bars on what verification buys.
#
# reports/results.md says the threshold was "fitted on 11 calibration images"
# and reports/verification.json records threshold -0.4 with a calibration F1 of
# 0.8676. That fit is the one step in the pipeline that chooses a number rather
# than measuring one, and it is the step most able to flatter the result, so it
# is the one worth refitting independently. This sweeps the same grid over the
# calibration images and requires the same argmax.
#
# It then asks the question the README's "25 to 40% relative" invites: with 42
# verified-absent probes and a difference of one or two false positives, how
# much of that range is real? Clopper-Pearson intervals and an exact McNemar
# test on the discordant probes, plus an image-level bootstrap.
#
# Base R only, so there is no JSON package. The reader below is deliberately
# strict: it is a check, and a check that shrugs at a malformed file is not one.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

# ------------------------------------------------------------------ JSON ----
json_parse <- function(txt) {
  pat <- '"(?:\\\\.|[^"\\\\])*"|-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|true|false|null|[][{}:,]'
  toks <- regmatches(txt, gregexpr(pat, txt, perl = TRUE))[[1]]
  i <- 1L
  unquote <- function(s) {
    s <- substr(s, 2L, nchar(s) - 1L)
    if (grepl("\\\\", s)) {
      s <- gsub('\\\\"', '"', s, fixed = TRUE)
      s <- gsub("\\\\", "\\", s, fixed = TRUE)
    }
    s
  }
  value <- function() {
    if (i > length(toks)) stop("truncated JSON")
    t <- toks[i]; i <<- i + 1L
    if (t == "{") {
      out <- list()
      if (toks[i] == "}") { i <<- i + 1L; return(out) }
      repeat {
        k <- unquote(toks[i]); i <<- i + 1L
        if (toks[i] != ":") stop("expected ':' after key ", k)
        i <<- i + 1L
        out[[k]] <- value()
        sep <- toks[i]; i <<- i + 1L
        if (sep == "}") break
        if (sep != ",") stop("expected ',' or '}'")
      }
      return(out)
    }
    if (t == "[") {
      out <- list()
      if (toks[i] == "]") { i <<- i + 1L; return(out) }
      repeat {
        out[[length(out) + 1L]] <- value()
        sep <- toks[i]; i <<- i + 1L
        if (sep == "]") break
        if (sep != ",") stop("expected ',' or ']'")
      }
      return(out)
    }
    if (substr(t, 1L, 1L) == '"') return(unquote(t))
    if (t == "true") return(TRUE)
    if (t == "false") return(FALSE)
    if (t == "null") return(NULL)
    n <- suppressWarnings(as.numeric(t))
    if (is.na(n)) stop("not a JSON value: ", t)
    n
  }
  v <- value()
  if (i <= length(toks)) stop("trailing content after the JSON value")
  v
}

read_json <- function(path) {
  if (!file.exists(path)) stop("missing file: ", path)
  json_parse(paste(readLines(path, warn = FALSE), collapse = "\n"))
}

# ------------------------------------------------------------- the data ----
ver <- read_json(file.path(root, "reports", "verification.json"))
sty <- read_json(file.path(root, "reports", "prompt_styles.json"))

zs <- lapply(ver$z_scores, function(b) unlist(b, use.names = TRUE))
cal_files <- unlist(ver$calibration_images)
test_files <- unlist(ver$test_images)
styles <- c("neutral", "leading", "presupposing")

as_frame <- function(style) {
  ps <- sty[[style]]$probes
  data.frame(
    file  = vapply(ps, function(p) p$file_name, ""),
    obj   = vapply(ps, function(p) p$object, ""),
    truth = vapply(ps, function(p) isTRUE(p$truth), TRUE),
    pred  = vapply(ps, function(p) isTRUE(p$pred), TRUE),
    stringsAsFactors = FALSE
  )
}

z_of <- function(df) {
  out <- numeric(nrow(df))
  for (k in seq_len(nrow(df))) {
    blk <- zs[[df$file[k]]]
    if (is.null(blk)) stop("no z-scores for image ", df$file[k])
    v <- blk[[df$obj[k]]]
    if (is.null(v) || !is.finite(v)) stop("no finite z-score for ", df$obj[k])
    out[k] <- v
  }
  out
}

f1_of <- function(tp, fp, fn) {
  prec <- if (tp + fp > 0) tp / (tp + fp) else 0
  rec  <- if (tp + fn > 0) tp / (tp + fn) else 0
  if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
}

clopper <- function(x, n, conf = 0.95) {
  a <- 1 - conf
  lo <- if (x == 0) 0 else qbeta(a / 2, x, n - x + 1)
  hi <- if (x == n) 1 else qbeta(1 - a / 2, x + 1, n - x)
  c(lo, hi)
}

problems <- character(0)
note <- function(...) problems <<- c(problems, paste0(...))

# --------------------------------------------- 1. refit the threshold -------
# src/vlmhall/verify.py sweeps numpy.arange(-1.0, 3.01, 0.05) and keeps the
# first strictly-better F1, so the same tie-breaking is used here.
grid <- -1.0 + (0:80) * 0.05
cat("refitting the threshold on", length(cal_files), "calibration images\n")

for (s in styles) {
  df <- as_frame(s)
  df$z <- z_of(df)
  cal <- df[df$file %in% cal_files, ]
  best_t <- 0; best_f1 <- -1
  for (t in grid) {
    acc <- cal$z >= t
    tp <- sum(cal$truth & acc); fp <- sum(!cal$truth & acc); fn <- sum(cal$truth & !acc)
    f1 <- f1_of(tp, fp, fn)
    if (f1 > best_f1) { best_t <- t; best_f1 <- f1 }
  }
  pub_t <- ver$results[[s]]$threshold
  pub_f1 <- ver$results[[s]]$calibration_f1
  cat(sprintf("  %-13s refit z >= %+.4f (published %+.4f), F1 %.10f (published %.10f)\n",
              s, best_t, pub_t, best_f1, pub_f1))
  if (abs(best_t - pub_t) > 1e-9) note(s, ": refitted threshold ", best_t, " but ", pub_t, " is published")
  if (abs(best_f1 - pub_f1) > 1e-12) note(s, ": refitted calibration F1 ", best_f1, " but ", pub_f1, " is published")
}

# ------------------------------- 2. what the reduction is worth statistically
cat("\nheld-out verified-absent probes, baseline against VLM AND CLIP\n")
set.seed(20260814)
B <- 10000
summary_rows <- list()

for (s in styles) {
  df <- as_frame(s)
  df$z <- z_of(df)
  ho <- df[df$file %in% test_files, ]
  absent <- ho[!ho$truth, ]
  thr <- ver$results[[s]]$threshold

  base_yes <- absent$pred
  ver_yes <- absent$pred & (absent$z >= thr)
  n_abs <- nrow(absent)
  fp_b <- sum(base_yes); fp_v <- sum(ver_yes)

  pub_b <- ver$results[[s]]$baseline$fp
  pub_v <- ver$results[[s]]$verified$fp
  pub_n <- ver$results[[s]]$baseline$n - (ver$results[[s]]$baseline$tp + ver$results[[s]]$baseline$fn)
  if (fp_b != pub_b) note(s, ": recomputed ", fp_b, " baseline false positives, published ", pub_b)
  if (fp_v != pub_v) note(s, ": recomputed ", fp_v, " verified false positives, published ", pub_v)
  if (n_abs != pub_n) note(s, ": recomputed ", n_abs, " absent probes, published ", pub_n)

  ci_b <- clopper(fp_b, n_abs); ci_v <- clopper(fp_v, n_abs)
  # Verification is a logical AND, so it can only ever turn a yes into a no:
  # one discordant cell is structurally zero and McNemar reduces to a sign test.
  b <- sum(base_yes & !ver_yes); c2 <- sum(!base_yes & ver_yes)
  p_mc <- if (b + c2 > 0) binom.test(b, b + c2, 0.5)$p.value else 1

  rel <- if (fp_b > 0) 1 - fp_v / fp_b else NA_real_

  # Image-level bootstrap: probes from one image share a CLIP score vector, so
  # resampling probes would understate the spread.
  imgs <- unique(absent$file)
  boot <- numeric(0)
  for (i in seq_len(B)) {
    pick <- sample(imgs, length(imgs), replace = TRUE)
    idx <- unlist(lapply(pick, function(f) which(absent$file == f)))
    bb <- sum(base_yes[idx]); vv <- sum(ver_yes[idx])
    if (bb > 0) boot <- c(boot, 1 - vv / bb)
  }
  q <- quantile(boot, c(0.025, 0.975), names = FALSE)

  cat(sprintf("  %-13s absent %d  fp %d -> %d  rate %.4f [%.4f, %.4f] -> %.4f [%.4f, %.4f]\n",
              s, n_abs, fp_b, fp_v, fp_b / n_abs, ci_b[1], ci_b[2],
              fp_v / n_abs, ci_v[1], ci_v[2]))
  cat(sprintf("                relative reduction %.3f, bootstrap 95%% [%.3f, %.3f], "
              , rel, q[1], q[2]))
  cat(sprintf("exact McNemar p = %.4f (%d discordant)\n", p_mc, b + c2))
  summary_rows[[s]] <- c(rel = rel, lo = q[1], hi = q[2], p = p_mc)
}

zero_frac <- mean(vapply(summary_rows, function(r) r["lo"] <= 0, TRUE))
cat(sprintf("\n%d of %d bootstrap intervals reach zero or below\n",
            sum(vapply(summary_rows, function(r) r["lo"] <= 0, TRUE)), length(summary_rows)))
invisible(zero_frac)

if (length(problems) > 0) {
  cat("\n", length(problems), " problems:\n", sep = "")
  for (p in problems) cat("  ", p, "\n", sep = "")
  quit(status = 1)
}
cat("the refitted threshold and every false-positive count agree with what is published\n")
