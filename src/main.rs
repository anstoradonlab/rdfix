use rdfix::forward::{DetectorParamsBuilder};
use rdfix::forward::generated_functions::calc_na_nb_factors;

use rdfix::inverse::{DetectorInverseModel,InversionOptions,InversionOptionsBuilder,fit_inverse_model};

use rdfix::{InputRecord,InputTimeSeries};

use log::{debug, error, log_enabled, info, Level};


fn main() {
    env_logger::init();
    error!("this is printed by default");

    let p = DetectorParamsBuilder::default().build().unwrap();
    println!("Hello, world!, here's the model: {:#?}", p);
    println!("Here's some Na,Nb factors: {:?}", calc_na_nb_factors(1.0, 3.0, 1e-3));


    let p = DetectorParamsBuilder::default().build().unwrap();
    let trec = InputRecord{
        time: 0.0,
        /// LLD minus ULD (ULD are noise), missing values marked with NaN
        counts: 10.0*0.2*60.0*30.0+30.0,  // should equal 10 Bq/m3
        background_count_rate: 1.0/60.0,
        sensitivity: 0.2,
        q_internal: 0.1/60.0,  //volumetric, m3/sec
        q_external: 80.0/60.0/1000.0,  //volumetric, m3/sec
        airt: 21.0, // degC
    };
    let mut ts = InputTimeSeries::new();
    for _ in 0..50{
        ts.push(trec);
    }
    ts.counts[20] *= 2.0;
    ts.counts[21] *= 2.0;
    let inv_opts = InversionOptionsBuilder::default().build().unwrap();
    fit_inverse_model(p, inv_opts, ts).unwrap();


}
