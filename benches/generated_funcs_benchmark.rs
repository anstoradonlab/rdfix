use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use autodiff::{F1, FT};

use rdfix::forward::{DetectorForwardModelBuilder, DetectorParamsBuilder};
use rdfix::get_test_timeseries;
use rdfix::inverse::{
    fit_inverse_model, pack_state_vector, DetectorInverseModel, Gradient, InversionOptionsBuilder,
};

use rdfix_gf::generated_functions::{calc_na_nb_factors,calc_na_nb_factors_f64};


fn na_nb_factors(c: &mut Criterion) {
    let q=100.0/60.0/1000.0;
    let v = 1.5;
    let lambda_p = 1e-3;

    c.bench_function("Na,Nb factors generic version", |b| {
        b.iter(|| calc_na_nb_factors(q, v, lambda_p))
    });

    c.bench_function("Na,Nb factors f64 version", |b| {
        b.iter(|| calc_na_nb_factors_f64(q, v, lambda_p))
    });
}



//criterion_group!(benches, inv_benchmark, inv_benchmark_no_black_box);
criterion_group!(
    benches,
    na_nb_factors
);
criterion_main!(benches);
