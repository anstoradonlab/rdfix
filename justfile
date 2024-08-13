build:
    cargo build --release

build-enzyme:
    RUSTFLAGS="-Z unstable-options" cargo +enzyme build -F enzyme_ad

test-enzyme:
    RUSTFLAGS="-Z unstable-options" cargo +enzyme test -F enzyme_ad

setup_validate:
    rm -rf validation
    cargo run --release -- template -t validation cal-peak-one-day

validate: setup_validate
    cargo run --release -- deconv --config validation/config.toml --output validation/deconv-output validation/raw-data.csv

setup_validate_month:
    rm -rf validation_month
    cargo run --release -- template -t validation_month cal-peak-month

validate_month: setup_validate_month
    cargo run --release -- deconv --config validation_month/config.toml --output validation_month/deconv-output validation_month/raw-data.csv
