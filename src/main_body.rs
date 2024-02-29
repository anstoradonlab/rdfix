/// These are the main top-level driver functions
use std::fs;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Error, Result};
use rayon::prelude::*;

use crate::appconfig::AppConfigBuilder;
use crate::inverse::fit_inverse_model;
use crate::{cmdline::*, read_csv, TestTimeseries, TimeExtents, TimeseriesKind};
use crate::{get_test_timeseries, write_csv};

use crate::InputTimeSeries;

use log::info;

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

    // This is the same as the code block, below, but without parallel
    //for (count, ts_chunk) in chunks.into_iter().enumerate(){
    //    let fit_results = fit_inverse_model(p.clone(), inv_opts, ts_chunk.clone())?;
    //    let output_fname = cmd_args.output.join(format!("chunk-{count}.nc"));
    //    fit_results.to_netcdf(output_fname)?;
    //}

    let results: Result<Vec<PathBuf>, Error> = chunks
        .into_par_iter()
        .map(|ts_chunk| {
            let fit_results = fit_inverse_model(p.clone(), inv_opts, ts_chunk.clone())?;
            let (t0, t1) = ts_chunk.time_extents_str();
            let chunk_id = format!("chunk-{t0}-{t1}");

            let output_fname = cmd_args.output.join(format!("{chunk_id}.nc"));
            fit_results.to_netcdf(output_fname.clone())?;
            Ok::<PathBuf, anyhow::Error>(output_fname)
        })
        .collect();

    let processed_fnames = match results {
        Ok(value) => value,
        Err(e) => return Err(e),
    };

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
