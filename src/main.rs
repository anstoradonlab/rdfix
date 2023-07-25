use std::fs;
use std::fs::File;

use anyhow::Result;

use autodiff::FT;

use rdfix::appconfig::AppConfigBuilder;
use rdfix::cmdline::*;
use rdfix::forward::DetectorParamsBuilder;
use rdfix::inverse::{fit_inverse_model, InversionOptionsBuilder};
use rdfix::{get_test_timeseries, write_csv};

/* MAKEITWORK
use rdfix::inverse::{
    fit_inverse_model, DetectorInverseModel, InversionOptions, InversionOptionsBuilder,
};
*/

use rdfix::{InputRecord, InputTimeSeries};

use log::{error, info};

fn create_template(cmd_args: &TemplateArgs) -> Result<()> {
    info!("Writing template to {}", cmd_args.template_dir.display());

    let fname = cmd_args.template_dir.clone().join("raw-data.csv");
    info!("Writing example data file to {}", fname.display());
    let ts = get_test_timeseries(48 * 3);
    let mut f = File::create(&fname)?;
    write_csv(&mut f, ts)?;
    let config = AppConfigBuilder::default().build().unwrap();
    let config_str = toml::to_string(&config).unwrap();

    let config_fname = cmd_args.template_dir.clone().join("config.toml");
    info!(
        "Writing example configuration file to {}",
        config_fname.display()
    );
    fs::write(&config_fname, config_str)?;

    let output_fname = cmd_args.template_dir.clone().join("deconv-output");
    println!("Template created.  Perform a test by running:\n> rdfix-deconvolve deconv --config {} --output {} {}", config_fname.display(), output_fname.display(), fname.display());

    Ok(())
}

fn run_deconvolution(_cmd_args: &DeconvArgs) -> Result<()> {
    todo!();
}

fn main() -> Result<()> {
    env_logger::init();
    error!("Here's an error message for testing purposes");

    let program_args = parse_cmdline()?;

    match &program_args.command {
        Some(Commands::Template(cmd_args)) => {
            create_template(&cmd_args)?;
        }
        Some(Commands::Deconv(cmd_args)) => run_deconvolution(&cmd_args)?,
        None => unreachable!(),
    }
    dbg!(&program_args);

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
    let npts = 48;
    let mut ts = get_test_timeseries(npts);
    ts.counts[npts - 1] += 500.0;

    let fit_results = fit_inverse_model(p.clone(), inv_opts.clone(), ts.clone())?;
    fit_results.to_netcdf("test_results_todo.nc".into())?;

    Ok(())
}
