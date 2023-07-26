use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::forward::{DetectorParams, DetectorParamsBuilder};
use crate::inverse::{InversionOptions, InversionOptionsBuilder};

/// Configuration options for the app
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AppConfig {
    #[builder(default = "DetectorParamsBuilder::<f64>::default().build().unwrap()")]
    pub detector: DetectorParams<f64>,

    #[builder(default = "InversionOptionsBuilder::default().build().unwrap()")]
    pub inversion: InversionOptions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_make_toml() {
        let config = AppConfigBuilder::default().build().unwrap();
        let toml = toml::to_string(&config).unwrap();
        println!("{}", toml);
    }
}
