-- Recompute every probe-level metric block in reports/prompt_styles.json from
-- the per-probe records in the same file, in SQL rather than in pandas-free
-- Python. The published tables in README.md and reports/results.md (accuracy,
-- yes-rate, recall on present objects, hallucination rate per probe split) are
-- all read out of those metric blocks, and nothing else ever checked that the
-- blocks agree with the probes they claim to summarise.
--
-- Emits one MISMATCH line per disagreeing figure, then two count lines the
-- driver checks, so an empty result set cannot be mistaken for success.
.mode list
.headers off
.separator " "

create temp table doc as
    select cast(readfile('reports/prompt_styles.json') as text) as j;

-- One row per probe, per question phrasing. truth and pred arrive as 1/0/null.
create temp table probe as
select st.key                              as style,
       json_extract(p.value, '$.file_name') as fn,
       json_extract(p.value, '$.object')    as obj,
       json_extract(p.value, '$.truth')     as truth,
       json_extract(p.value, '$.kind')      as kind,
       json_extract(p.value, '$.pred')      as pred
from doc, json_each(doc.j) st, json_each(st.value, '$.probes') p;

-- The grouping the Python uses. "present + subset": a split's block is scored
-- over the verified-present probes plus that split's absent probes, which is
-- why n is 216 for adversarial and not 44. Reproducing that exactly matters:
-- if I grouped the obvious way the accuracies would silently differ.
create temp table grp as
    select style, 'overall' as g, truth, kind, pred from probe
    union all select style, 'present',     truth, kind, pred from probe where truth = 1
    union all select style, 'adversarial', truth, kind, pred from probe where truth = 1 or kind = 'adversarial'
    union all select style, 'random',      truth, kind, pred from probe where truth = 1 or kind = 'random'
    union all select style, 'popular',     truth, kind, pred from probe where truth = 1 or kind = 'popular';

create temp table calc as
select style, g,
       count(*)                                                        as n,
       sum(case when truth = 1 and pred = 1 then 1 else 0 end)         as tp,
       sum(case when truth = 1 and coalesce(pred, 0) = 0 then 1 else 0 end) as fn,
       sum(case when truth = 0 and pred = 1 then 1 else 0 end)         as fp,
       sum(case when truth = 0 and coalesce(pred, 0) = 0 then 1 else 0 end) as tn,
       sum(case when pred = 1 then 1 else 0 end)                       as nyes,
       sum(case when pred is null then 1 else 0 end)                   as unparsed
from grp group by style, g;

-- Hallucination rate is the yes-rate on the absent probes of that split only.
create temp table hall as
    select style, kind as g,
           sum(case when pred = 1 then 1 else 0 end) * 1.0 / count(*) as hr,
           count(*) as n_absent
    from probe where truth = 0 group by style, kind
    union all
    select style, 'absent_all',
           sum(case when pred = 1 then 1 else 0 end) * 1.0 / count(*),
           count(*)
    from probe where truth = 0 group by style;

create temp table deriv as
select style, g, n, tp, fp, tn, fn, nyes, unparsed,
       (tp + tn) * 1.0 / n                                       as acc,
       case when tp + fp > 0 then tp * 1.0 / (tp + fp) else 0.0 end as prec,
       case when tp + fn > 0 then tp * 1.0 / (tp + fn) else 0.0 end as rec
from calc;

create temp table computed as
    select style, g, 'n'         as metric, n * 1.0        as v from deriv
    union all select style, g, 'tp',        tp * 1.0        from deriv
    union all select style, g, 'fp',        fp * 1.0        from deriv
    union all select style, g, 'tn',        tn * 1.0        from deriv
    union all select style, g, 'fn',        fn * 1.0        from deriv
    union all select style, g, 'unparsed',  unparsed * 1.0  from deriv
    union all select style, g, 'accuracy',  acc             from deriv
    union all select style, g, 'precision', prec            from deriv
    union all select style, g, 'recall',    rec             from deriv
    union all select style, g, 'f1',
        case when prec + rec > 0 then 2 * prec * rec / (prec + rec) else 0.0 end from deriv
    union all select style, g, 'yes_rate',  nyes * 1.0 / n  from deriv
    union all select style, g, 'hallucination_rate', hr     from hall
    union all select style, g, 'n_absent',  n_absent * 1.0  from hall where g <> 'absent_all'
    union all select style, g, 'n',         n_absent * 1.0  from hall where g =  'absent_all';

-- Everything the file publishes, key by key, so a metric I forgot to recompute
-- shows up as a missing comparison rather than passing unnoticed.
create temp table published as
select st.key as style, m.key as g, kv.key as metric, kv.value * 1.0 as v
from doc, json_each(doc.j) st, json_each(st.value, '$.metrics') m, json_each(m.value) kv;

select 'MISMATCH ' || p.style || ' ' || p.g || ' ' || p.metric
       || ' published=' || p.v || ' recomputed=' || c.v
from published p join computed c
  on p.style = c.style and p.g = c.g and p.metric = c.metric
where abs(p.v - c.v) > 1e-12;

select 'published ' || count(*) from published;
select 'compared '  || count(*) from published p join computed c
  on p.style = c.style and p.g = c.g and p.metric = c.metric;
