/// These are the main top-level driver functions
use std::fs;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{anyhow, Error, Result};
use rayon::prelude::*;

use crate::appconfig::AppConfigBuilder;
use crate::inverse::fit_inverse_model;
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
    let mut ts = get_test_timeseries(48 * 3);
    let mut config = AppConfigBuilder::default().build().unwrap();
    match cmd_args.template_kind {
        TemplateKind::Default => {}
        TemplateKind::Small => {
            config.inversion.map_search_iterations = 100;
            config.inversion.emcee.burn_in = 100;
            config.inversion.emcee.samples = 100;
        }
        TemplateKind::ConstantOneDay => {
            ts = TestTimeseries::new(48, TimeseriesKind::NoisyConstant { value: 1.0 }).ts()
        }
        TemplateKind::CalPeakOneDay => {
            ts = TestTimeseries::new(
                48,
                TimeseriesKind::CalibrationPulse {
                    low_value: 1.0,
                    high_value: 100.0,
                },
            )
            .ts()
        }
        TemplateKind::ConstantMonth => {
            ts = TestTimeseries::new(48 * 30, TimeseriesKind::NoisyConstant { value: 1.0 }).ts()
        }
        TemplateKind::CalPeakMonth => {
            ts = TestTimeseries::new(
                48 * 30,
                TimeseriesKind::CalibrationPulse {
                    low_value: 1.0,
                    high_value: 100.0,
                },
            )
            .ts()
        }
    }
    let mut f = File::create(&fname)?;
    write_csv(&mut f, ts)?;
    let config_str = toml::to_string(&config).unwrap();
    dbg!(&config_str);

    let config_fname = cmd_args.template_dir.clone().join("config.toml");
    info!(
        "Writing example configuration file to {}",
        config_fname.display()
    );
    fs::write(&config_fname, config_str)?;

    let output_dir = cmd_args.template_dir.clone().join("deconv-output");
    fs::create_dir_all(&output_dir)?;
    println!("Template created.  Perform a test by running:\n> rdfix-deconvolve deconv --config {} --output {} {}", config_fname.display(), output_dir.display(), fname.display());

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
        chunks.push(ts);
    }

    if chunks.len() > 1 {
        info!("Input data split into {} chunks.", chunks.len());
    }

    // .into_par_iter() makes this parallel;
    // .panic_fuse() makes the loop stop earlier if any jobs panic
    let results_and_errors: Vec<Result<PathBuf, Error>> = chunks
        .into_par_iter()
        //        .panic_fuse()
        .map(|ts_chunk| {
            let chunk_id = ts_chunk.chunk_id();
            let output_fname = cmd_args.output.join(format!("{chunk_id}.nc"));
            if output_fname.exists(){
                info!("{} already processed, skipping to next chunk", chunk_id);
                return Ok::<PathBuf, anyhow::Error>(output_fname)
            }

            let panic_wrapper = std::panic::catch_unwind(||  fit_inverse_model(p.clone(), inv_opts, ts_chunk.clone()));
            let fit_result = if panic_wrapper.is_ok(){
                panic_wrapper.unwrap()
            } 
            else {
                // Panic occurred.  The panic message still gets printed out, so convert the error into 
                // something we can use later and continue. 
                let e = panic_wrapper.unwrap_err();
                Err(anyhow!("{:?}", e))
            };
            match fit_result {
                Err(e) => {
                    let chunk_id = ts_chunk.chunk_id();
                    error!(
                        "Error processing {}: {}.  Continuing to next block.",
                        chunk_id, e
                    );
                    // write a copy of the chunk to an "errors" directory (unless the input data are all NaN)
                    if ts_chunk.counts.iter().any(|x| x.is_finite()) {
                        let output_dir = cmd_args.output.join(format!("failed-chunk-{chunk_id}"));
                        std::fs::create_dir(&output_dir)?;
                        let csv_fname = output_dir.join("raw-data.csv");
                        let mut f = File::create(&csv_fname)?;
                        write_csv(&mut f, ts_chunk)?;
                        let config_str = toml::to_string(&config).unwrap();
                        let config_fname = output_dir.join("config.toml");
                        fs::write(&config_fname, config_str)?;
                        let output_dir = output_dir.join("deconv-output");
                        fs::create_dir_all(&output_dir)?;
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

    println!(
        "Processed these files: {:?}",
        processed_fnames
            .iter()
            .map(|itm| itm.display())
            .collect::<Vec<_>>()
    );

    Ok(())
}

pub fn main_body(program_args: RdfixArgs) -> Result<()> {
    match &program_args.command {
        Commands::Template(cmd_args) => {
            create_template(cmd_args)?;
        }
        Commands::Deconv(cmd_args) => run_deconvolution(cmd_args)?,
        Commands::Forward(_cmd_args) => todo!(), // run a forward model
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
            "rdfix-deconvolve",
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
            "rdfix-deconvolve",
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
