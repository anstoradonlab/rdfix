
rm -r ./deconv-example-input

cargo run -- template
cargo run -- deconv --config ./deconv-example-input/config.toml --output ./deconv-example-input/deconv-output ./deconv-example-input/raw-data.csv
