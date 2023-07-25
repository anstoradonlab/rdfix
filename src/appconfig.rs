use derive_builder::Builder;
use serde::{Serialize, Deserialize};
use toml;

use crate::inverse::{InversionOptions,InversionOptionsBuilder};
use crate::forward::{DetectorParams,DetectorParamsBuilder};

/// Configuration options for the app
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AppConfig{
    #[builder(default = "DetectorParamsBuilder::<f64>::default().build().unwrap()")]
    detector: DetectorParams<f64>,

    #[builder(default = "InversionOptionsBuilder::default().build().unwrap()")]
    inversion: InversionOptions,
    
}

/// Configuration options for No U-Turn Sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct NutsOptions{
    /// Number of NUTS samples
    #[builder(default = "1000")]
    nuts_samples: usize,
}


/// Configuration options for EMCEE sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct EmceeOptions{

}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn can_make_toml(){
        let config = AppConfigBuilder::default().build().unwrap();
        let toml = toml::to_string(&config).unwrap();
        println!("{}", toml);
    }
}