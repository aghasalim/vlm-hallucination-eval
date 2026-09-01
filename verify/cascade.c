/* Recompute the three decision rules of the verification cascade.
 *
 * reports/results.md section 3 and the METHODS table publish, for each of the
 * three question phrasings, accuracy / precision / recall / F1 / hallucination
 * rate for "baseline (VLM alone)", "CLIP alone" and "VLM AND CLIP". Those
 * numbers are read straight out of reports/verification.json, which was written
 * by src/vlmhall/verify.py. Nothing checked that the metric blocks in that file
 * follow from its own z-scores and the per-probe answers in
 * reports/prompt_styles.json.
 *
 * This rebuilds all three confusion matrices from those raw parts:
 *
 *   baseline   accept when the VLM answered yes
 *   clip_only  accept when z(image, object) >= threshold
 *   verified   accept when both
 *
 * scored over the held-out images only, and compares every published figure.
 * Objects and images are resolved by name out of the JSON, never by position,
 * so a reordered file is not a failure and a renamed key is.
 *
 * Build: cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o cascade cascade.c -lm
 * Run:   ./cascade <repo-root>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------- JSON ---
 * A small parser: enough of JSON for these two files, and strict enough that a
 * truncated or malformed file is an error rather than a partial parse.
 */
enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };

typedef struct {
    int type;
    double num;
    int bval;
    char *str;          /* JSTR: decoded value. object members: see key */
    char *key;          /* member name when the parent is an object */
    int first, next;    /* first child, next sibling; -1 for none */
} Node;

typedef struct {
    Node *n;
    size_t len, cap;
    const char *p;      /* cursor */
    const char *end;
    char **owned;       /* decoded strings, freed at the end */
    size_t nowned, cowned;
    int failed;
} Doc;

static void die(const char *msg, const char *detail)
{
    fprintf(stderr, "cascade: %s%s%s\n", msg, detail ? ": " : "", detail ? detail : "");
    exit(1);
}

static int node_new(Doc *d, int type)
{
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 1024;
        d->n = (Node *)realloc(d->n, d->cap * sizeof(Node));
        if (!d->n) die("out of memory", NULL);
    }
    Node *v = &d->n[d->len];
    memset(v, 0, sizeof(*v));
    v->type = type;
    v->first = v->next = -1;
    return (int)d->len++;
}

static void own(Doc *d, char *s)
{
    if (d->nowned == d->cowned) {
        d->cowned = d->cowned ? d->cowned * 2 : 1024;
        d->owned = (char **)realloc(d->owned, d->cowned * sizeof(char *));
        if (!d->owned) die("out of memory", NULL);
    }
    d->owned[d->nowned++] = s;
}

static void skip_ws(Doc *d)
{
    while (d->p < d->end && (*d->p == ' ' || *d->p == '\t' || *d->p == '\n' || *d->p == '\r'))
        d->p++;
}

static char *parse_string(Doc *d)
{
    if (d->p >= d->end || *d->p != '"') die("expected a string", NULL);
    d->p++;
    size_t cap = 32, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) die("out of memory", NULL);
    while (d->p < d->end && *d->p != '"') {
        char c = *d->p++;
        if (c == '\\') {
            if (d->p >= d->end) die("truncated escape", NULL);
            char e = *d->p++;
            switch (e) {
            case '"': c = '"'; break;
            case '\\': c = '\\'; break;
            case '/': c = '/'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            default: die("unsupported string escape", NULL); return NULL;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            out = (char *)realloc(out, cap);
            if (!out) die("out of memory", NULL);
        }
        out[len++] = c;
    }
    if (d->p >= d->end) die("unterminated string", NULL);
    d->p++;
    out[len] = '\0';
    own(d, out);
    return out;
}

static int parse_value(Doc *d);

static int parse_container(Doc *d, int type, char close)
{
    int me = node_new(d, type);
    d->p++;                         /* '{' or '[' */
    skip_ws(d);
    if (d->p < d->end && *d->p == close) { d->p++; return me; }
    int prev = -1;
    for (;;) {
        char *key = NULL;
        if (type == JOBJ) {
            skip_ws(d);
            key = parse_string(d);
            skip_ws(d);
            if (d->p >= d->end || *d->p != ':') die("expected ':' after a key", key);
            d->p++;
        }
        int child = parse_value(d);
        d->n[child].key = key;
        if (prev < 0) d->n[me].first = child; else d->n[prev].next = child;
        prev = child;
        skip_ws(d);
        if (d->p < d->end && *d->p == ',') { d->p++; continue; }
        if (d->p < d->end && *d->p == close) { d->p++; break; }
        die("expected ',' or a closing bracket", NULL);
    }
    return me;
}

static int parse_value(Doc *d)
{
    skip_ws(d);
    if (d->p >= d->end) die("unexpected end of input", NULL);
    char c = *d->p;
    if (c == '{') return parse_container(d, JOBJ, '}');
    if (c == '[') return parse_container(d, JARR, ']');
    if (c == '"') {
        int me = node_new(d, JSTR);
        d->n[me].str = parse_string(d);
        return me;
    }
    if ((size_t)(d->end - d->p) >= 4 && strncmp(d->p, "true", 4) == 0) {
        int me = node_new(d, JBOOL); d->n[me].bval = 1; d->p += 4; return me;
    }
    if ((size_t)(d->end - d->p) >= 5 && strncmp(d->p, "false", 5) == 0) {
        int me = node_new(d, JBOOL); d->n[me].bval = 0; d->p += 5; return me;
    }
    if ((size_t)(d->end - d->p) >= 4 && strncmp(d->p, "null", 4) == 0) {
        int me = node_new(d, JNULL); d->p += 4; return me;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *stop = NULL;
        double v = strtod(d->p, &stop);
        if (stop == d->p) die("bad number", NULL);
        int me = node_new(d, JNUM);
        d->n[me].num = v;
        d->p = stop;
        return me;
    }
    die("unexpected character in JSON", NULL);
    return -1;
}

static Doc *load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open", path);
    if (fseek(f, 0, SEEK_END) != 0) die("cannot seek", path);
    long n = ftell(f);
    if (n < 0) die("cannot size", path);
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) die("out of memory", NULL);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("short read", path);
    fclose(f);
    buf[n] = '\0';

    Doc *d = (Doc *)calloc(1, sizeof(Doc));
    if (!d) die("out of memory", NULL);
    d->p = buf;
    d->end = buf + n;
    int root = parse_value(d);
    skip_ws(d);
    if (d->p != d->end) die("trailing content after the JSON value", path);
    if (root != 0) die("internal: root is not node 0", path);
    free(buf);
    return d;
}

static int member(Doc *d, int obj, const char *key)
{
    if (obj < 0 || d->n[obj].type != JOBJ) return -1;
    for (int c = d->n[obj].first; c >= 0; c = d->n[c].next)
        if (d->n[c].key && strcmp(d->n[c].key, key) == 0) return c;
    return -1;
}

static int require(Doc *d, int obj, const char *key)
{
    int m = member(d, obj, key);
    if (m < 0) die("missing key", key);
    return m;
}

static double num_at(Doc *d, int obj, const char *key)
{
    int m = require(d, obj, key);
    if (d->n[m].type != JNUM) die("key is not a number", key);
    return d->n[m].num;
}

/* ------------------------------------------------------------- checking --- */

typedef struct { int tp, fp, tn, fn, n, n_absent; } Conf;

static int mismatches = 0, compared = 0;

static void cmp(const char *style, const char *rule, const char *field,
                double got, double want)
{
    compared++;
    double diff = fabs(got - want);
    if (diff > 1e-12) {
        printf("MISMATCH %s %s %s published=%.17g recomputed=%.17g\n",
               style, rule, field, want, got);
        mismatches++;
    }
}

static void check_rule(Doc *d, int block, const char *style, const char *rule, Conf c)
{
    double prec = (c.tp + c.fp) ? (double)c.tp / (c.tp + c.fp) : 0.0;
    double rec  = (c.tp + c.fn) ? (double)c.tp / (c.tp + c.fn) : 0.0;
    double f1   = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0.0;
    cmp(style, rule, "n",         c.n, num_at(d, block, "n"));
    cmp(style, rule, "tp",        c.tp, num_at(d, block, "tp"));
    cmp(style, rule, "fp",        c.fp, num_at(d, block, "fp"));
    cmp(style, rule, "tn",        c.tn, num_at(d, block, "tn"));
    cmp(style, rule, "fn",        c.fn, num_at(d, block, "fn"));
    cmp(style, rule, "accuracy",  (double)(c.tp + c.tn) / c.n, num_at(d, block, "accuracy"));
    cmp(style, rule, "precision", prec, num_at(d, block, "precision"));
    cmp(style, rule, "recall",    rec,  num_at(d, block, "recall"));
    cmp(style, rule, "f1",        f1,   num_at(d, block, "f1"));
    cmp(style, rule, "hallucination_rate",
        c.n_absent ? (double)c.fp / c.n_absent : 0.0,
        num_at(d, block, "hallucination_rate"));
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char pv[4096], pp[4096];
    snprintf(pv, sizeof pv, "%s/reports/verification.json", root);
    snprintf(pp, sizeof pp, "%s/reports/prompt_styles.json", root);

    Doc *V = load(pv);
    Doc *P = load(pp);

    int results = require(V, 0, "results");
    int zroot   = require(V, 0, "z_scores");
    int tests   = require(V, 0, "test_images");
    if (V->n[tests].type != JARR) die("test_images is not an array", NULL);

    int n_test = 0;
    for (int c = V->n[tests].first; c >= 0; c = V->n[c].next) n_test++;
    printf("held-out images: %d\n", n_test);

    const char *styles[3] = { "neutral", "leading", "presupposing" };
    for (int s = 0; s < 3; s++) {
        int res = require(V, results, styles[s]);
        double threshold = num_at(V, res, "threshold");
        int probes = require(P, require(P, 0, styles[s]), "probes");
        if (P->n[probes].type != JARR) die("probes is not an array", styles[s]);

        Conf base = {0, 0, 0, 0, 0, 0}, clip = base, both = base;

        for (int p = P->n[probes].first; p >= 0; p = P->n[p].next) {
            int fn_i = require(P, p, "file_name");
            const char *file = P->n[fn_i].str;
            /* held out only: the calibration images fitted the threshold */
            int held = 0;
            for (int t = V->n[tests].first; t >= 0; t = V->n[t].next)
                if (V->n[t].str && strcmp(V->n[t].str, file) == 0) { held = 1; break; }
            if (!held) continue;

            const char *obj = P->n[require(P, p, "object")].str;
            int truth_i = require(P, p, "truth");
            if (P->n[truth_i].type != JBOOL) die("truth is not a boolean", file);
            int truth = P->n[truth_i].bval;
            int pred_i = require(P, p, "pred");
            int pred = (P->n[pred_i].type == JBOOL) && P->n[pred_i].bval;

            int zimg = member(V, zroot, file);
            if (zimg < 0) die("no z-scores for image", file);
            int zobj = member(V, zimg, obj);
            if (zobj < 0) die("no z-score for object", obj);
            if (V->n[zobj].type != JNUM) die("z-score is not a number", obj);
            double z = V->n[zobj].num;
            if (!isfinite(z)) die("z-score is not finite", obj);

            int accepted[3];
            accepted[0] = pred;
            accepted[1] = z >= threshold;
            accepted[2] = pred && z >= threshold;
            Conf *cs[3] = { &base, &clip, &both };
            for (int k = 0; k < 3; k++) {
                Conf *c = cs[k];
                c->n++;
                if (!truth) c->n_absent++;
                if (truth && accepted[k]) c->tp++;
                else if (!truth && accepted[k]) c->fp++;
                else if (truth) c->fn++;
                else c->tn++;
            }
        }

        if (base.n == 0) die("no held-out probes matched", styles[s]);
        printf("%-13s threshold z >= %+.4f  probes %d  absent %d  "
               "fp base %d clip %d verified %d\n",
               styles[s], threshold, base.n, base.n_absent, base.fp, clip.fp, both.fp);

        check_rule(V, require(V, res, "baseline"),  styles[s], "baseline",  base);
        check_rule(V, require(V, res, "clip_only"), styles[s], "clip_only", clip);
        check_rule(V, require(V, res, "verified"),  styles[s], "verified",  both);
    }

    printf("compared %d published figures, %d mismatches\n", compared, mismatches);
    if (compared != 90) {
        printf("FAIL expected 90 comparisons (3 phrasings x 3 rules x 10 figures)\n");
        return 1;
    }
    return mismatches ? 1 : 0;
}
