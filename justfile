build:
    cargo build --release

build-enzyme:
    cargo +enzyme build --release -F enzyme_ad
    #RUSTFLAGS="-Z unstable-options" cargo +enzyme build -F enzyme_ad

test-enzyme:
    cargo +enzyme test --release -F enzyme_ad
    #RUSTFLAGS="-Z unstable-options" cargo +enzyme test --release -F enzyme_ad
    #RUSTFLAGS="-Z unstable-options -Z autodiff=LooseTypes" cargo +enzyme test -F enzyme_ad
    #RUSTFLAGS="-Z unstable-options -Z autodiff=OPT" cargo +enzyme test -F enzyme_ad

setup_validate:
    rm -rf validation
    cargo run --release -- template -t validation cal-peak-one-day

setup_validate_enzyme:
    rm -rf validation_enzyme
    # TODO: switch sampler kind over to Nuts
    cargo +enzyme run --release -F enzyme_ad -- template -t validation_enzyme cal-peak-one-day

validate_enzyme: setup_validate_enzyme
    cargo +enzyme run --release -F enzyme_ad -- deconv --config validation_enzyme/config.toml --output validation_enzyme/deconv-output validation_enzyme/raw-data.csv

validate: setup_validate
    cargo run --release -- deconv --config validation/config.toml --output validation/deconv-output validation/raw-data.csv

setup_validate_month:
    rm -rf validation_month
    cargo run --release -- template -t validation_month cal-peak-month

validate_month: setup_validate_month
    cargo run --release -- deconv --config validation_month/config.toml --output validation_month/deconv-output validation_month/raw-data.csv
