use anyhow::Result;
use rdfix::cmdline::parse_cmdline;
use rdfix::main_body::main_body;
use std::env;

fn main() -> Result<()> {

    // TODO: switch to tracing framework and provide an option for JSON-formatted log messages
    // https://users.rust-lang.org/t/best-way-to-log-with-json/83385
    // The idea is that JSON I/O could be used from a Python process
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info")
    }
    env_logger::init();
    let program_args = parse_cmdline()?;
    main_body(program_args)?;
    Ok(())
}
