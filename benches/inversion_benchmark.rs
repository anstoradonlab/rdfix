use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use autodifflib::{F1, FT};

use rdfix::forward::{DetectorForwardModelBuilder, DetectorParamsBuilder};
use rdfix::get_test_timeseries;
use rdfix::inverse::{
    fit_inverse_model, pack_state_vector, DetectorInverseModel, Gradient, InversionOptionsBuilder,
};

fn _inv_benchmark(c: &mut Criterion) {
    let p = DetectorParamsBuilder::default().build().unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 4;
    let mut ts = get_test_timeseries(npts);
    ts.counts[npts - 1] += 500.0;

    c.bench_function("inv black_box", |b| {
        b.iter(|| {
            fit_inverse_model(
                black_box(p.clone()),
                black_box(inv_opts.clone()),
                black_box(ts.clone()),
            )
            .expect("Failed to fit inverse model")
        })
    });
}

fn _inv_benchmark_no_black_box(c: &mut Criterion) {
    let p = DetectorParamsBuilder::default().build().unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 4;
    let mut ts = get_test_timeseries(npts);
    ts.counts[npts - 1] += 500.0;
    c.bench_function("inv black_box", |b| {
        b.iter(|| {
            fit_inverse_model(p.clone(), inv_opts.clone(), ts.clone())
                .expect("Failed to fit inverse model")
        })
    });
}

fn objective_function(c: &mut Criterion) {
    let p = DetectorParamsBuilder::default().build().unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 5;
    let ts = get_test_timeseries(npts);
    let mut radon = vec![1.0; ts.len()];
    // set this value to something higher so that gradients will be non-zero
    radon[1] = 10.0;

    let time_step = 30.0 * 60.0;

    let fwd = DetectorForwardModelBuilder::default()
        .data(ts.clone())
        .time_step(time_step)
        .radon(radon.clone())
        .build()
        .expect("Failed to build detector model");
    let cost = DetectorInverseModel {
        p: p.clone(),
        inv_opts: inv_opts,
        ts: ts.clone(),
        fwd: fwd.clone(),
    };

    let init_param = pack_state_vector(&radon[..], p.clone(), ts.clone(), inv_opts);

    //let pvec = ndarray::Array1::from_vec(init_param.clone());

    c.bench_function("Objective function, cost", |b| {
        b.iter(|| cost.generic_lnprob(init_param.clone().as_slice()))
    });

    c.bench_function("Objective function, explicit f64, cost", |b| {
        b.iter(|| cost.lnprob_f64(init_param.clone().as_slice()))
    });
}

fn objective_function_func_npts(c: &mut Criterion) {
    fn getinput(npts: usize) -> (DetectorInverseModel<f64>, Vec<f64>) {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_test_timeseries(npts);
        let mut radon = vec![1.0; ts.len()];
        // set this value to something higher so that gradients will be non-zero
        radon[1] = 10.0;

        let time_step = 30.0 * 60.0;

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(radon.clone())
            .build()
            .expect("Failed to build detector model");
        let cost = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts.clone(),
            fwd: fwd.clone(),
        };

        let init_param = pack_state_vector(&radon[..], p.clone(), ts.clone(), inv_opts);
        (cost, init_param)
    }

    let inputs: Vec<_> = (5..126).step_by(20).map(|n| getinput(n)).collect();
    let mut group = c.benchmark_group("cost_for_npts");
    for (cost, input) in inputs {
        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| cost.lnprob_f64(input.clone().as_slice()));
            },
        );
    }
    group.finish();
}

fn gradient_function(c: &mut Criterion) {
    let p = DetectorParamsBuilder::default()
        .sensitivity(1000. / (3600.0 / 2.0))
        .build()
        .unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 5;
    let ts = get_test_timeseries(npts);
    let time_step = 60.0 * 30.0; //TODO
                                 // Define initial parameter vector and cost function
    let mut radon = vec![1.0; ts.len()];
    // set this value to something higher so that gradients will be non-zero
    radon[1] = 10.0;
    let init_param = pack_state_vector(&radon, p.clone(), ts.clone(), inv_opts);

    let fwd = DetectorForwardModelBuilder::default()
        .data(ts.clone())
        .time_step(time_step)
        .radon(radon.clone())
        .build()
        .expect("Failed to build detector model");

    let cost_diff = DetectorInverseModel {
        p: p.into_inner_type::<FT<f64>>(),
        inv_opts: inv_opts,
        ts: ts,
        fwd: fwd.into_inner_type::<FT<f64>>(),
    };

    // println!("initial guess: {:#?}", init_param.values);
    let pvec = ndarray::Array1::from_vec(init_param.clone());

    c.bench_function("Objective function, gradient", |b| {
        b.iter(|| cost_diff.gradient(&pvec).unwrap())
    });
}

fn objective_function_with_autodiff_types(c: &mut Criterion) {
    let p = DetectorParamsBuilder::default()
        .sensitivity(1000. / (3600.0 / 2.0))
        .build()
        .unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 5;
    let ts = get_test_timeseries(npts);
    let time_step = 60.0 * 30.0; //TODO
                                 // Define initial parameter vector and cost function
    let mut radon = vec![1.0; ts.len()];
    // set this value to something higher so that gradients will be non-zero
    radon[1] = 10.0;
    let init_param = pack_state_vector(&radon, p.clone(), ts.clone(), inv_opts);

    let fwd = DetectorForwardModelBuilder::default()
        .data(ts.clone())
        .time_step(time_step)
        .radon(radon.clone())
        .build()
        .expect("Failed to build detector model");

    let cost_diff = DetectorInverseModel {
        p: p.into_inner_type::<FT<f64>>(),
        inv_opts: inv_opts,
        ts: ts,
        fwd: fwd.into_inner_type::<FT<f64>>(),
    };

    // println!("initial guess: {:#?}", init_param.values);
    let nums: Vec<FT<f64>> = init_param.iter().map(|&x| F1::cst(x)).collect();
    let pvec = ndarray::Array1::from_vec(nums.clone());

    c.bench_function("Objective function, cost, with autodiff types", |b| {
        b.iter(|| cost_diff.generic_lnprob(black_box(pvec.as_slice().unwrap())))
    });
}

//criterion_group!(benches, inv_benchmark, inv_benchmark_no_black_box);
criterion_group!(
    benches,
    objective_function,
    gradient_function,
    objective_function_with_autodiff_types,
    objective_function_func_npts
);
criterion_main!(benches);
