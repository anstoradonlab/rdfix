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

    #[builder(default = "NutsOptionsBuilder::default().build().unwrap()")]
    pub nuts: NutsOptions,

    #[builder(default = "EmceeOptionsBuilder::default().build().unwrap()")]
    pub emcee: EmceeOptions,


}

/// Configuration options for No U-Turn Sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct NutsOptions {
    /// Number of NUTS samples
    #[builder(default = "1000")]
    pub samples: usize,

    /// The thinning factor, i.e. the number of samples to skip in the output
    #[builder(default = "30")]
    pub thin: usize,
    
}

/// Configuration options for EMCEE sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct EmceeOptions {
        /// Number of EMCEE samples
        #[builder(default = "1000")]
        pub samples: usize,
    
        /// Number of walkers per dimension
        #[builder(default = "3")]
        pub walkers_per_dim: usize,

        /// The thinning factor, i.e. the number of samples to skip in the output
        #[builder(default = "30")]
        pub thin: usize,


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
