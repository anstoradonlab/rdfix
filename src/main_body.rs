use std::fmt::write;
/// These are the main top-level driver functions
use std::fs;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Error, Result, anyhow, bail};
use rayon::prelude::*;

use crate::appconfig::AppConfigBuilder;
use crate::forward::DetectorForwardModelBuilder;
use crate::inverse::fit_inverse_model;
use crate::postproc::{netcdf_to_csv, postproc};
use crate::{cmdline::*, read_csv, TestTimeseries, TimeExtents, TimeseriesKind};
use crate::{get_test_timeseries, write_csv};

use crate::InputTimeSeries;

use log::{error, info};

fn chunk_timeseries(
    ts: &InputTimeSeries,
    chunksize: usize,
    overlapsize: usize,
) -> Result<Vec<InputTimeSeries>> {
    let totalsize = chunksize + 2 * overlapsize;
    if ts.len() <= totalsize {
        return Ok(vec![ts.clone()]);
    }
    let mut chunks: Vec<InputTimeSeries> = vec![];
    let mut i1 = 0;
    let mut i2 = i1 + totalsize;
    while i2 < ts.len() {
        chunks.push(ts.slice(i1..i2).to_vec());
        i1 += chunksize;
        i2 += chunksize;
    }
    Ok(chunks)
}

fn create_template(cmd_args: &TemplateArgs) -> Result<()> {
    info!("Writing template to {}", cmd_args.template_dir.display());

    let fname = cmd_args.template_dir.clone().join("raw-data.csv");
    info!("Writing example data file to {}", fname.display());
    let time_step = 60.0 * 30.0;
    let mut ts = get_test_timeseries(48 * 3, time_step);
    let mut config = AppConfigBuilder::default().build().unwrap();
    let mut command_str = "deconv".to_owned();
    match cmd_args.template_kind {
        TemplateKind::Default => {}
        TemplateKind::Small => {
            config.inversion.map_search_iterations = 100;
            config.inversion.emcee.burn_in = 100;
            config.inversion.emcee.samples = 100;
        }
        TemplateKind::ConstantOneDay => {
            ts = TestTimeseries::new(48, time_step, TimeseriesKind::NoisyConstant { value: 1.0 }).ts()
        }
        TemplateKind::CalPeakOneDay => {
            ts = TestTimeseries::new(
                48, time_step,
                TimeseriesKind::CalibrationPulse {
                    low_value: 1.0,
                    high_value: 100.0,
                },
            )
            .ts()
        }
        TemplateKind::ConstantMonth => {
            ts = TestTimeseries::new(48 * 30, time_step, TimeseriesKind::NoisyConstant { value: 1.0 }).ts()
        }
        TemplateKind::CalPeakMonth => {
            ts = TestTimeseries::new(
                48 * 30, time_step,
                TimeseriesKind::CalibrationPulse {
                    low_value: 1.0,
                    high_value: 100.0,
                },
            )
            .ts()
        }
        TemplateKind::Forward => {
            // use a timestep of 60 seconds
            let time_step = 60.0;
            // use a long length (60 days)
            ts = TestTimeseries::new(
                24 * 60 * 60, 
                time_step,
                TimeseriesKind::CalibrationPulse {
                    low_value: 1.0,
                    high_value: 100.0,
                },
            )
            .ts();
            ts.counts.iter_mut().for_each(|x| *x=f64::NAN);
            command_str = "forward".to_owned();
        }
    }
    let mut f = File::create(&fname)?;
    write_csv(&mut f, ts, true)?;
    let config_str = toml::to_string(&config).unwrap();

    let config_fname = cmd_args.template_dir.clone().join("config.toml");
    info!(
        "Writing example configuration file to {}",
        config_fname.display()
    );
    fs::write(&config_fname, config_str)?;

    let output_dir = cmd_args.template_dir.clone().join("deconv-output");
    fs::create_dir_all(&output_dir)?;
    println!(
        "Template created.  Perform a test by running:\n> rdfix {} --config {} --output {} {}",
        command_str,
        config_fname.display(),
        output_dir.display(),
        fname.display()
    );

    Ok(())
}

fn run_deconvolution(cmd_args: &DeconvArgs) -> Result<()> {
    // Load configuration file
    info!("Loading configuration from {}", &cmd_args.config.display());
    let raw_toml = std::fs::read_to_string(&cmd_args.config)?;
    let config: crate::appconfig::AppConfig = toml::from_str(raw_toml.as_str())?;

    // Load raw data files
    let mut ts = InputTimeSeries::new();
    for fname in cmd_args.input_files.iter() {
        info!("Loading data from {}", fname.display());
        let f = std::fs::File::open(fname)?;
        let mut file_data = read_csv(f)?;
        ts.append(&mut file_data);
    }

    let p = config.detector.clone();
    let inv_opts = config.inversion;
    let mut chunks = vec![];
    if config.inversion.process_in_chunks {
        chunks.extend(chunk_timeseries(
            &ts,
            config.inversion.chunksize,
            config.inversion.overlapsize,
        )?);
    } else {
        chunks.push(ts.clone());
    }

    let nchunks = chunks.len();
    if nchunks > 1 {
        info!("Input data split into {} chunks.", nchunks);
    }

    // .into_par_iter() makes this parallel;
    // .panic_fuse() makes the loop stop earlier if any jobs panic
    let results_and_errors: Vec<Result<PathBuf, Error>> = chunks
        .into_par_iter()
        //        .panic_fuse()
        .map(|ts_chunk| {
            let chunk_id = ts_chunk.chunk_id();
            let output_fname = cmd_args.output.join(format!("{chunk_id}.nc"));
            if output_fname.exists() {
                info!("{} already processed, skipping to next chunk", chunk_id);
                return Ok::<PathBuf, anyhow::Error>(output_fname);
            }

            let panic_wrapper = std::panic::catch_unwind(|| {
                fit_inverse_model(p.clone(), inv_opts, ts_chunk.clone())
            });
            //let fit_result = if panic_wrapper.is_ok() {
            //    panic_wrapper.unwrap()
            //} else {
            //    // Panic occurred.  The panic message still gets printed out, so convert the error into
            //    // something we can use later and continue.
            //    let e = panic_wrapper.unwrap_err();
            //    Err(anyhow!("{:?}", e.downcast_ref::<&str>()))
            //};
            let fit_result =
                panic_wrapper.unwrap_or_else(|e| Err(anyhow!("{:?}", e.downcast_ref::<&str>())));
            match fit_result {
                Err(e) => {
                    let chunk_id = ts_chunk.chunk_id();
                    error!(
                        "Error processing {}: {}.  Continuing to next block.",
                        chunk_id, e
                    );
                    // write a copy of the chunk to an "errors" directory (unless the input data are all NaN)
                    if ts_chunk.counts.iter().any(|x| x.is_finite()) && nchunks > 1 {
                        let output_dir = cmd_args.output.join(format!("failed-chunk-{chunk_id}"));
                        std::fs::create_dir(&output_dir)?;
                        let csv_fname = output_dir.join("raw-data.csv");
                        let mut f = File::create(csv_fname)?;
                        write_csv(&mut f, ts_chunk, true)?;
                        let config_str = toml::to_string(&config).unwrap();
                        let config_fname = output_dir.join("config.toml");
                        fs::write(config_fname, config_str)?;
                        let output_dir = output_dir.join("deconv-output");
                        fs::create_dir_all(output_dir)?;
                    }
                    Err(anyhow!("Error processing {}: {}.", chunk_id, e))
                }
                Ok(fit_results) => {
                    let (t0, t1) = ts_chunk.time_extents_str();
                    let chunk_id = format!("chunk-{t0}-{t1}");

                    fit_results.to_netcdf(output_fname.clone())?;
                    info!("Finished processing {}.", chunk_id);
                    Ok::<PathBuf, anyhow::Error>(output_fname)
                }
            }
        })
        .collect();

    let processed_fnames = results_and_errors
        .iter()
        .filter(|itm| itm.is_ok())
        .map(|itm| itm.as_ref().unwrap())
        .collect::<Vec<_>>();

    let _errors = results_and_errors
        .iter()
        .filter(|itm| itm.is_err())
        .map(|itm| itm.as_ref().unwrap_err())
        .collect::<Vec<_>>();

    // postprocessing, three passes, 1) no averaging, 2) 30-min average, 3) 1-h average
    // They can run in parallel (hopefully making it more likely that they'll
    // read netCDF files from cache)
    use crossbeam_utils::thread;

    thread::scope(|s| {
        let output_fname = cmd_args.output.join("summary.nc");
        let output_csv_fname = cmd_args.output.join("summary.csv");
        let filenames = processed_fnames.clone();
        let tsc = ts.clone();
        s.spawn(move |_| {
            let _pproc = postproc(
                &tsc,
                filenames,
                config.inversion.overlapsize,
                &output_fname,
                config.inversion.radon_interpolation_option,
                None,
            );
            netcdf_to_csv(&output_fname, &output_csv_fname)?;
            anyhow::Ok(())
        });

        let output_fname = cmd_args.output.join("summary_30min_average.nc");
        let output_csv_fname = cmd_args.output.join("summary_30min_average.csv");
        let filenames = processed_fnames.clone();
        let tsc = ts.clone();
        s.spawn(move |_| {
            let _pproc = postproc(
                &tsc,
                filenames,
                config.inversion.overlapsize,
                &output_fname,
                config.inversion.radon_interpolation_option,
                Some(30 * 60),
            );
            netcdf_to_csv(&output_fname, &output_csv_fname)?;
            anyhow::Ok(())
        });

        let output_fname = cmd_args.output.join("summary_60min_average.nc");
        let output_csv_fname = cmd_args.output.join("summary_60min_average.csv");
        let filenames = processed_fnames.clone();
        let tsc = ts.clone();
        s.spawn(move |_| {
            let _pproc = postproc(
                &tsc,
                filenames,
                config.inversion.overlapsize,
                &output_fname,
                config.inversion.radon_interpolation_option,
                Some(60 * 60),
            );
            netcdf_to_csv(&output_fname, &output_csv_fname)?;
            anyhow::Ok(())
        });
    })
    .expect("Crossbeam scoped threads failure");

    Ok(())
}


fn run_forward_model(cmd_args: &DeconvArgs) -> Result<()>{
    // Load configuration file
    info!("Loading configuration from {}", &cmd_args.config.display());
    let raw_toml = std::fs::read_to_string(&cmd_args.config)?;
    let config: crate::appconfig::AppConfig = toml::from_str(raw_toml.as_str())?;

    // Load raw data files
    let mut ts = InputTimeSeries::new();
    for fname in cmd_args.input_files.iter() {
        info!("Loading data from {}", fname.display());
        let f = std::fs::File::open(fname)?;
        let mut file_data = read_csv(f)?;
        ts.append(&mut file_data);
    }
    if ts.len() < 2{
        bail!("Input timeseries is too short")
    }

    let mut chunks = vec![];
    // TODO: should take dt into account
    // Note: chunking is used because the integrator fails if it is run for more than about 30 days
    // this shouldn't happen, but in lieu of finding the root cause, we reset the integrator
    // after each day of modelled time.  This matches how the model is used in deconvolution.
    let chunksize = {
        let time_step = ts.time[1] - ts.time[0];
        let secs_per_day = 3600*24;
        secs_per_day / (time_step as usize)
    };
    let overlapsize = 0;
    chunks.extend(chunk_timeseries(
        &ts,
        chunksize,
        overlapsize,
    )?);

    let nchunks = chunks.len();
    if nchunks > 1 {
        info!("Input data split into {} chunks.", nchunks);
    }

    let mut ic = None;
    for ts_chunk in chunks.iter_mut(){
        let p = config.detector.clone();
        let radon = ts_chunk.radon_truth.clone();
        let time_step = ts_chunk.time[1] - ts_chunk.time[0];

        let fwd = DetectorForwardModelBuilder::default()
            .p(p)
            .data(ts_chunk.clone())
            .radon(radon)
            .initial_condition(ic)
            .time_step(time_step)
            .build()
            .unwrap();

        let (num_counts,final_state) = fwd.numerical_expected_counts_and_state().unwrap();
        for (nc,mnc) in ts_chunk.counts.iter_mut().zip(&num_counts){
            *nc = *mnc;
        }
        ic = Some(final_state);
    }
    
    let output_csv_fname = cmd_args.output.join("forward.csv");
    fs::create_dir_all(&cmd_args.output)?;
    let mut f = File::create(&output_csv_fname)?;
    let mut first = true;
    for ts in chunks.iter(){
        write_csv(&mut f, ts.clone(), first)?;
        first = false;
    }

    Ok(())
}

pub fn main_body(program_args: RdfixArgs) -> Result<()> {
    match &program_args.command {
        Commands::Template(cmd_args) => {
            create_template(cmd_args)?;
        }
        Commands::Deconv(cmd_args) => run_deconvolution(cmd_args)?,
        Commands::Forward(cmd_args) => run_forward_model(cmd_args)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::env;
    use tempfile::tempdir;

    /// Tests the whole program by running the top-level function as if it has been
    /// run from the commmand line.  First generate a test case from the "small" template
    /// then execute it
    #[test]
    fn run_cli() {
        if env::var("RUST_LOG").is_err() {
            env::set_var("RUST_LOG", "info")
        }
        env_logger::init();

        let dir_input = tempdir().unwrap();
        let dir_output = tempdir().unwrap();

        let cmdline = vec![
            "rdfix",
            "template",
            "-t",
            dir_input.path().to_str().unwrap(),
            "small",
        ];
        let program_args: RdfixArgs = RdfixArgs::parse_from(cmdline);
        dbg!(&program_args);
        main_body(program_args).unwrap();

        let config_fname = dir_input.path().join("config.toml");
        let input_fname = dir_input.path().join("raw-data.csv");
        let cmdline = vec![
            "rdfix",
            "deconv",
            "-c",
            config_fname.to_str().unwrap(),
            "-o",
            dir_output.path().to_str().unwrap(),
            input_fname.to_str().unwrap(),
        ];
        let program_args: RdfixArgs = RdfixArgs::parse_from(cmdline);
        main_body(program_args).unwrap();
    }
}
