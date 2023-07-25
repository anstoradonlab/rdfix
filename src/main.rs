use anyhow::Result;

use rdfix::cmdline::parse_cmdline;
use rdfix::main_body::main_body;

fn main() -> Result<()> {
    let program_args = parse_cmdline()?;
    main_body(program_args)?;
    Ok(())
}
