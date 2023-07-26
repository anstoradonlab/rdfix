/// These are the main top-level driver functions
use std::fs;
use std::fs::File;

use anyhow::Result;

use crate::appconfig::AppConfigBuilder;
use crate::inverse::fit_inverse_model;
use crate::{cmdline::*, read_csv};
use crate::{get_test_timeseries, write_csv};

use crate::InputTimeSeries;

use log::info;

fn create_template(cmd_args: &TemplateArgs) -> Result<()> {
    info!("Writing template to {}", cmd_args.template_dir.display());

    let fname = cmd_args.template_dir.clone().join("raw-data.csv");
    info!("Writing example data file to {}", fname.display());
    let ts = get_test_timeseries(48 * 3);
    let mut f = File::create(&fname)?;
    write_csv(&mut f, ts)?;
    let mut config = AppConfigBuilder::default().build().unwrap();
    match cmd_args.template_kind {
        TemplateKind::Default => {}
        TemplateKind::Small => {
            config.inversion.map_search_iterations = 100;
            config.inversion.emcee.burn_in = 100;
            config.inversion.emcee.samples = 100;
        }
    }
    let config_str = toml::to_string(&config).unwrap();
    dbg!(&config_str);

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

    let fit_results = fit_inverse_model(p.clone(), inv_opts, ts.clone())?;
    fit_results.to_netcdf("test_results_todo.nc".into())?;

    Ok(())
}

pub fn main_body(program_args: RdfixArgs) -> Result<()> {
    match &program_args.command {
        Commands::Template(cmd_args) => {
            create_template(cmd_args)?;
        }
        Commands::Deconv(cmd_args) => run_deconvolution(cmd_args)?,
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
        dbg!(&program_args);
        main_body(program_args).unwrap();
    }
}
