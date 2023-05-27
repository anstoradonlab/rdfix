use anyhow::Result;
use rdfix::forward::{DetectorForwardModelBuilder, DetectorParamsBuilder};
use rdfix::get_test_timeseries;
use rdfix::inverse::{fit_inverse_model, InversionOptions, InversionOptionsBuilder};

fn main() -> Result<()> {
    let npts = 48;
    let mut ts = get_test_timeseries(npts);
    let radon = {
        let mut v = vec![1.0_f64; npts];
        for x in v[5..20].iter_mut() {
            *x = 2.0;
        }
        v
    };
    let dt = ts.time[1] - ts.time[0];

    let fwd = DetectorForwardModelBuilder::<f64>::default()
        .radon(radon.clone())
        .data(ts.clone())
        .time_step(dt)
        .build()?;

    dbg!(&fwd);

    let soln1 = fwd.numerical_expected_counts()?;
    let simulated_counts: Vec<_> = soln1.iter().map(|x| x.round()).collect();
    for (x, y) in ts.counts.iter_mut().zip(simulated_counts) {
        *x = y;
    }

    let p = DetectorParamsBuilder::<f64>::default().build()?;
    let inv_opts = InversionOptionsBuilder::default().build()?;

    println!("Running inverse problem");
    let inv_results = fit_inverse_model(p, inv_opts, ts)?;

    dbg!(&inv_results);
    dbg!(soln1);

    for (x, y) in radon.iter().zip(inv_results) {
        println!("{x}, {y}");
    }

    Ok(())
}
