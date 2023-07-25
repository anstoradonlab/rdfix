use anyhow::Result;
use rdfix::cmdline::parse_cmdline;
use rdfix::main_body::main_body;
use std::env;


fn main() -> Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info")
    }
    env_logger::init();
    let program_args = parse_cmdline()?;
    main_body(program_args)?;
    Ok(())
}
