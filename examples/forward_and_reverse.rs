use anyhow::Result;
use itertools::Itertools;
use rdfix::forward::{DetectorForwardModelBuilder, DetectorParamsBuilder};
use rdfix::get_test_timeseries;
use rdfix::inverse::{fit_inverse_model, InversionOptionsBuilder};

fn main() -> Result<()> {
    let npts = 48;
    let mut ts = get_test_timeseries(npts);
    let radon = {
        let mut v = vec![1.0_f64; npts];
        for x in v[5..20].iter_mut() {
            *x = 1000.0;
        }
        v
    };
    let dt = ts.time[1] - ts.time[0];

    let fwd = DetectorForwardModelBuilder::default()
        .radon(radon.clone())
        .data(ts.clone())
        .time_step(dt)
        .build()?;

    dbg!(&fwd);

    let soln1 = fwd.numerical_expected_counts()?;
    let simulated_counts: Vec<_> = soln1.iter().map(|x| x.round()).collect();
    // the first "simulated_counts" value is not used
    for (x, y) in ts.counts.iter_mut().skip(1).zip_eq(simulated_counts) {
        *x = y;
    }

    let p = DetectorParamsBuilder::default().build()?;
    let inv_opts = InversionOptionsBuilder::default()
        .r_screen_sigma(1e-6)
        .exflow_sigma(1e-6)
        .build()?;

    println!("Running inverse problem");
    let inv_results = fit_inverse_model(p, inv_opts, ts)?;

    dbg!(&inv_results);
    dbg!(soln1);

    /*
    This interface no longer works
    let radon_map: Vec<_> = inv_results
        .var_ref("map_radon")
        .unwrap()
        .data
        .clone()
        .into_iter()
        .collect();
    for (x, y) in radon.iter().zip_eq(radon_map) {
        println!("{x:.2}, {y:.2}, {:.2}", x / y);
    }
    */

    Ok(())
}
