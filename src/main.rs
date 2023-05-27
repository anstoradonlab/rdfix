use autodiff::{F1, FT};

use rdfix::forward::{DetectorForwardModelBuilder, DetectorParamsBuilder};
use rdfix::get_test_timeseries;
use rdfix::inverse::{
    fit_inverse_model, pack_state_vector, DetectorInverseModel, Gradient, InversionOptionsBuilder,
};

/* MAKEITWORK
use rdfix::inverse::{
    fit_inverse_model, DetectorInverseModel, InversionOptions, InversionOptionsBuilder,
};
*/

use rdfix::{InputRecord, InputTimeSeries};

use log::{debug, error, info, log_enabled, Level};

fn main() {
    env_logger::init();
    error!("Here's an error message for tesing purposes");

    let p = DetectorParamsBuilder::<f64>::default().build().unwrap();
    println!("parameters, f64: {:#?}", p);

    let p_diff = DetectorParamsBuilder::<FT<f64>>::default().build().unwrap();
    println!("parameters, differentiable: {:#?}", p_diff);
    let trec = InputRecord {
        time: 0.0,
        /// LLD minus ULD (ULD are noise), missing values marked with NaN
        counts: 10.0 * 0.2 * 60.0 * 30.0 + 30.0, // should equal 10 Bq/m3
        background_count_rate: 1.0 / 60.0,
        sensitivity: 0.2,
        q_internal: 0.1 / 60.0,           //volumetric, m3/sec
        q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
        airt: 21.0,                       // degC
    };
    let mut ts = InputTimeSeries::new();
    for _ in 0..50 {
        ts.push(trec);
    }
    ts.counts[20] *= 2.0;
    ts.counts[21] *= 2.0;

    /*MAKEITWORK
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    fit_inverse_model(p, inv_opts, ts).unwrap();

    */

    println!("Running a simple test case...");

    let p = DetectorParamsBuilder::default().build().unwrap();
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let npts = 40;
    let mut ts = get_test_timeseries(npts);
    ts.counts[npts - 1] += 500.0;

    fit_inverse_model(p.clone(), inv_opts.clone(), ts.clone()).expect("Failed to fit inverse model")
}
