use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::{env, fs, path::PathBuf};

/// Deconvolve two-filter dual-flow-loop radon detector output
/// using the method from https://doi.org/10.5194/amt-9-2689-2016
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct RdfixArgs {
    #[command(subcommand)]
    command: Option<Commands>,
}

fn get_default_dir() -> PathBuf {
    let mut path = env::current_dir().unwrap();
    // path.pop();
    path
}

fn validate_output_dir(p: &PathBuf) -> Result<()> {
    //Err(anyhow!("Output directory must be empty"))
    Ok(())
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Creates a template input and configuration file
    Template {
        /// Where to create the template
        #[arg(short, long, value_name = "DIR", default_value=get_default_dir().join("deconv-input").into_os_string())]
        template_dir: PathBuf,
    },
    /// Run deconvolution
    Deconv {
        /// Sets the config file
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        /// Sets the output directory
        #[arg(short, long, value_name = "OUTPUT_DIR", default_value=get_default_dir().join("deconv-output").into_os_string())]
        output: PathBuf,

        /// Input files
        input_files: Vec<PathBuf>,
    },
}

pub fn parse_cmdline() -> Result<RdfixArgs> {
    let args = RdfixArgs::parse_from(wild::args());

    match &args.command {
        Some(Commands::Template { template_dir }) => {
            if template_dir.exists() {
                if !template_dir.is_dir() {
                    return Err(anyhow!(
                        "Template directory path \"{0}\" is not a directory",
                        template_dir.display()
                    ));
                }
                let is_empty = template_dir.read_dir()?.next().is_none();
                if !is_empty {
                    return Err(anyhow!(
                        "Template directory \"{0}\" is not empty",
                        template_dir.display()
                    ));
                }
            } else {
                fs::create_dir(template_dir)?;
            }
        }

        Some(Commands::Deconv {
            config,
            output,
            input_files,
        }) => {
            for fname in input_files {
                if !fname.exists() {
                    return Err(anyhow!("Input file \"{0}\" not found", fname.display()));
                }
                if !config.exists() {
                    return Err(anyhow!(
                        "Configuration file \"{0}\" not found",
                        config.display()
                    ));
                }
                if output.exists() {
                    if !output.is_dir() {
                        return Err(anyhow!(
                            "Output directory path \"{0}\" is not a directory",
                            output.display()
                        ));
                    }
                    let is_empty = output.read_dir()?.next().is_none();
                    if !is_empty {
                        return Err(anyhow!(
                            "Output directory \"{0}\" is not empty",
                            output.display()
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    Ok(args)
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    RdfixArgs::command().debug_assert()
}
