# `ipf_survey`

Survey-record raking via iterative proportional fitting (IPF). Given coded
survey microdata and known population marginals, `ipf_survey` produces one
weight per record so that weighted marginal distributions match the targets.

This crate sits on top of [`ipf`](https://crates.io/crates/ipf), which provides
the underlying N-dimensional IPF solver. `ipf_survey` handles everything
specific to survey weighting: cross-tabulation, target validation, mapping
cell-level adjustment factors back to per-record weights, weight trimming,
normalization, and diagnostics.

## Features

- Coded-microdata input with optional base/design weights
- Population-margin targets with validation against the survey schema
- Cell-level weight trimming with iterative re-raking
- Normalization to N, to a chosen total, or to mean one
- Diagnostics: effective sample size (ESS), design effect (DEFF), and
  weight-distribution summary (min/max/mean/median/CV)
- Optional Rayon parallelism via the `ipf` crate

## Scope

**In scope:** taking pre-coded survey records plus marginal targets and
returning per-record weights, with the trimming/normalization/diagnostics
needed to put those weights into use.

**Out of scope:** parsing raw survey files, recoding/binning categorical
variables, replicate-weight generation (BRR, jackknife), and variance
estimation. Callers are expected to arrive with integer category codes in
hand.

## Installation

```toml
[dependencies]
ipf_survey = "0.2"
```

## Quick start

```rust
use ipf_survey::{
    CodedSurvey, Variable, PopulationTargets,
    RakingConfig, Normalization, rake,
};

// 1. Define the categorical variables used for raking.
let variables = vec![
    Variable { name: "age_group".into(), levels: 3, labels: None },
    Variable { name: "gender".into(),    levels: 2, labels: None },
];

// 2. Load coded records (caller-supplied integer codes).
//    age_group: 0=young, 1=middle, 2=senior
//    gender:    0=male,  1=female
let survey = CodedSurvey::builder(variables)
    .push_record(&[0, 0])?
    .push_record(&[0, 1])?
    .push_record(&[1, 0])?
    .push_record(&[1, 0])?
    .push_record(&[2, 1])?
    .push_record(&[2, 1])?
    .build()?;

// 3. Population marginal targets.
let targets = PopulationTargets::new()
    .add("age_group", vec![2.0, 2.0, 2.0])
    .add("gender",    vec![3.0, 3.0])
    .validate(&survey)?;

// 4. Configure and rake.
let config = RakingConfig {
    weight_bounds: Some((0.5, 3.0)),
    normalization: Normalization::SumToN,
    ..Default::default()
};

let result = rake(&survey, &targets, &config)?;

for (i, w) in result.weights.iter().enumerate() {
    println!("record {i}: {w:.4}");
}
println!("ESS  = {:.1}", result.diagnostics.effective_sample_size);
println!("DEFF = {:.3}", result.diagnostics.design_effect);
# Ok::<(), ipf_survey::RakingError>(())
```

If you do not need trimming, normalization, or custom convergence settings,
`rake_simple(&survey, &targets)` is a one-line shortcut.

## How it works

1. **Cross-tabulate** the coded records (weighted by base weights, when
   present) into a dense contingency table with one axis per variable.
2. **Build IPF constraints** from the validated population targets.
3. **Run the IPF solver** from the `ipf` crate to produce a fitted matrix.
4. **Derive cell adjustment factors** as `fitted / seed`.
5. **Optionally trim** factors to `weight_bounds`: out-of-range cells are
   frozen at the bound, their mass is subtracted from the targets, and the
   remaining cells are re-raked. Cycles repeat until every factor is inside
   the bounds (or `max_trim_cycles` is exhausted).
6. **Assign per-record weights** as `base_weight * adjustment_factor[cell]`.
7. **Normalize** to the requested total, if any.
8. **Compute diagnostics** (ESS, DEFF, weight summary).

### Weight bounds

`weight_bounds` constrains the **adjustment factor**, not the final weight.
For records with a base weight of 1.0 the two are equivalent. For records
with non-uniform base weights, the final weight is
`base_weight * adjustment_factor`, and the bound applies to the factor. This
matches standard survey-methodology practice and keeps trimming compatible
with the cell-level IPF framework. If you need bounds on the final weight
itself, apply that as a post-processing step on `result.weights`. The upper
bound may be `f64::INFINITY` to trim from below only.

When trimming succeeds, the returned weights respect the bounds **and** the
weighted marginals still match the targets exactly — the mass a frozen cell
gives up is redistributed over the remaining cells. Bounds that are too
tight for the targets (e.g. a margin level that needs a 4x adjustment under
an upper bound of 2.0) are reported as `RakingError::TrimInfeasible` rather
than silently missing the targets; widen the bounds or revisit the targets.

## Errors

All fallible operations return [`RakingError`], which covers survey
construction problems (out-of-range codes, label/weight length mismatches,
non-finite weights), target problems (unknown variables, length mismatches,
negative or non-finite values, inconsistent grand totals), IPF solver
failures (wrapped from the `ipf` crate), and trimming failures
(non-convergence or infeasible bounds).

Note that `rake` returning `Ok` means the pipeline completed; whether the
IPF solver met its tolerance is reported separately in
`result.convergence.converged`. Check that flag when downstream correctness
depends on the marginals matching.

## Performance notes

- Tabulation is a single `O(n_records)` pass with sequential reads from a
  row-major code buffer.
- The contingency table is the seed for IPF and is typically small enough to
  live in L1/L2 cache for realistic survey dimensions.
- Memory is dominated by the coded records: `n_records * n_variables * 8`
  bytes. One million records by five variables is about 40 MB.
- Enable this crate's `rayon` feature to turn on the `ipf` crate's parallel
  inner loop; the `diagnostics` feature likewise enables per-iteration
  residual history (recorded when `RakingConfig::diagnostics` is set).

## License

Licensed under the [MIT License](LICENSE).
