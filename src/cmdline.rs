use anyhow::{anyhow, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{fs, path::PathBuf};

/// Deconvolve two-filter dual-flow-loop radon detector output
/// using the method from https://doi.org/10.5194/amt-9-2689-2016
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct RdfixArgs {
    #[command(subcommand)]
    pub command: Commands,
}

fn get_default_dir() -> PathBuf {
    //let path = env::current_dir().unwrap();
    PathBuf::from(".")
}

/* Maybe create this function?
fn validate_output_dir(_p: &PathBuf) -> Result<()> {
    //Err(anyhow!("Output directory must be empty"))
    Ok(())
}
 */

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Creates a template input and configuration file
    Template(TemplateArgs),
    /// Run deconvolution
    Deconv(DeconvArgs),
    /// Run forward model
    Forward(DeconvArgs),
}

// Currently, forward model takes the same arguments as deconvolution but we might
// want to add FwdArgs in the future.

#[derive(Args, Debug)]
pub struct DeconvArgs {
    /// Sets the config file
    #[arg(short, long, value_name = "FILE")]
    pub config: PathBuf,

    /// Sets the output directory
    #[arg(short, long, value_name = "OUTPUT_DIR", default_value=get_default_dir().join("deconv-output").into_os_string())]
    pub output: PathBuf,

    /// Input files
    pub input_files: Vec<PathBuf>,
}



/*
#[derive(Args, Debug)]
pub struct FwdArgs {
    /// Sets the output directory
    #[arg(short, long, value_name = "OUTPUT_DIR", default_value=get_default_dir().join("deconv-output").into_os_string())]
    pub output: PathBuf,

    /// Input files
    pub input_files: Vec<PathBuf>,
}
 */

/*
#[repr(transparent)]
pub struct FwdArgs(DeconvArgs);
 */

#[derive(Args, Debug)]
pub struct TemplateArgs {
    /// Where to create the template
    #[arg(short, long, value_name = "DIR", default_value=get_default_dir().join("deconv-example-input").into_os_string())]
    pub template_dir: PathBuf,

    /// The kind of template to generate
    #[arg(value_enum, default_value_t=TemplateKind::Default)]
    pub template_kind: TemplateKind,
}

#[derive(ValueEnum, Debug, Clone)]
pub enum TemplateKind {
    /// Default template; a minimal case with parameters chosen to produce valid results
    Default,
    /// Small template using very few iterations for fast execution; this template will not generate valid results
    Small,
    /// Constant radon concentration, but with Poisson noise on counts, one day of data
    ConstantOneDay,
    /// Hour-long calibration peak, one day of data
    CalPeakOneDay,
    /// Constant radon concentration, but with Poisson noise on counts, one month of data
    ConstantMonth,
    /// Hour-long calibration peak, one month of data
    CalPeakMonth,
}

pub fn parse_cmdline() -> Result<RdfixArgs> {
    let args = RdfixArgs::parse_from(wild::args());

    match &args.command {
        Commands::Template(args) => {
            if args.template_dir.exists() {
                if !args.template_dir.is_dir() {
                    return Err(anyhow!(
                        "Template directory path \"{0}\" is not a directory",
                        args.template_dir.display()
                    ));
                }
                let is_empty = args.template_dir.read_dir()?.next().is_none();
                if !is_empty {
                    return Err(anyhow!(
                        "Template directory \"{0}\" is not empty",
                        args.template_dir.display()
                    ));
                }
            } else {
                fs::create_dir(&args.template_dir)?;
            }
        }

        Commands::Deconv(args) | Commands::Forward(args) => {
            for fname in args.input_files.iter() {
                if !fname.exists() {
                    return Err(anyhow!("Input file \"{0}\" not found", fname.display()));
                }
                if !args.config.exists() {
                    return Err(anyhow!(
                        "Configuration file \"{0}\" not found",
                        args.config.display()
                    ));
                }
                if args.output.exists() {
                    if !args.output.is_dir() {
                        return Err(anyhow!(
                            "Output directory path \"{0}\" is not a directory",
                            args.output.display()
                        ));
                    }
                    let is_empty = args.output.read_dir()?.next().is_none();
                    if !is_empty {
                        return Err(anyhow!(
                            "Output directory \"{0}\" is not empty",
                            args.output.display()
                        ));
                    }
                }
            }
        }
    }

    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cli() {
        use clap::CommandFactory;
        RdfixArgs::command().debug_assert()
    }
}
