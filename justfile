build:
    cargo build --release

setup_validate:
    rm -rf validation
    cargo run --release -- template -t validation cal-peak-one-day

validate: setup_validate
    cargo run --release -- deconv --config validation/config.toml --output validation/deconv-output validation/raw-data.csv