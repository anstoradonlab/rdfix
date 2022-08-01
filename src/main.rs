use rdfix::forward::DetectorParamsBuilder;

use rdfix::inverse::{
    fit_inverse_model, DetectorInverseModel, InversionOptions, InversionOptionsBuilder,
};

use rdfix::{InputRecord, InputTimeSeries};

use log::{debug, error, info, log_enabled, Level};

fn main() {
    env_logger::init();
    error!("this is printed by default");

    let p = DetectorParamsBuilder::default().build().unwrap();
    println!("Hello, world!, here's the model: {:#?}", p);

    let p = DetectorParamsBuilder::default().build().unwrap();
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
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    fit_inverse_model(p, inv_opts, ts).unwrap();
}
