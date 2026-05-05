use anyhow::Result;
use argmin::core::Gradient;
//use hammer_and_sample::Model;
use ndarray::{Array1, ArrayView1};

use nuts_rs::{
    ArrowConfig, CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Model,
    Sampler, SamplerWaitResult, Storable, Chain,
};
use nuts_rs::Settings;
use nuts_storable::{HasDims, Value};
use rand::{Rng, RngExt};
//use rand::SeedableRng;
//use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use super::forward::{DetectorForwardModelBuilder, DetectorParams, DetectorParamsBuilder};

use super::*;

use super::inverse::*;

// Define a function that computes the unnormalized posterior density
// and its gradient.
struct PosteriorDensity {
    inverse_model: DetectorInverseModel,
    dim: usize,
}

impl PosteriorDensity {
    fn new(p: DetectorParams, inv_opts: InversionOptions, ts: InputTimeSeries) -> Self {
        let time_step = 60.0 * 30.0; //TODO

        // Radon concentration, without deconvolution, non-negative
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);

        //let mean_radon =
        //    initial_radon.iter().fold(0.0, |x, y| x + y) / (initial_radon.len() as f64);

        // 1. Initialisation
        // Define initial parameter vector and cost function

        //println!("Initial radon concentration: {:?}", initial_radon);
        let _init_param = {
            let v = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
            Array1::<f64>::from_vec(v)
        };

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .p(p.clone())
            .time_step(time_step)
            .radon(initial_radon.clone())
            .build()
            .expect("Failed to build detector model");
        let inverse_model: DetectorInverseModel = DetectorInverseModel {
            p,
            inv_opts,
            ts,
            fwd,
        };

        // TODO: don't hard code the number of non-radon parameters (here, it's 2)
        let dim: usize = initial_radon.len() + 2;

        PosteriorDensity { inverse_model, dim }
    }
}

impl HasDims for PosteriorDensity{
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            // Dimension for the actual parameter vector x
            ("x".to_string(), self.dim as u64),
        ])

    }
}

/*
// Dimension definitions (MVP)
impl HasDims for PosteriorDensity {
    /// Define dimension names and sizes for storage
    ///
    /// This tells the storage system what array dimensions to expect.
    /// These dimensions will be used to structure the output data using
    /// Arrow's FixedShapeTensor extension type.
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            // Dimension for the actual parameter vector x
            ("x".to_string(), self.dim as u64),
        ])
    }

    fn coords(&self) -> HashMap<String, nuts_storable::Value> {


        let v: Vec<String> = (1..self.dim+1)
        .map(|ii| format!("x{}", ii))
        .collect();

        HashMap::from([(
            "x".to_string(),
            Value::Strings(v),
        )])
    }
}
*/

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
pub enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

/// The `Storable` derive macro automatically generates code to store this
/// struct in the trace. The `dims` attribute specifies which dimension
/// each field should use. Multi-dimensional fields will be stored as
/// FixedShapeTensor extension types in Arrow format.
#[derive(Storable)]
struct ExpandedDraw {
    /// Store the parameter values with dimension "x"
    #[storable(dims("x"))]
    prec: Vec<f64>,
}

/// NUTS sampler trait template (from documentation)
impl CpuLogpFunc for PosteriorDensity {
    type LogpError = PosteriorLogpError;
    type FlowParameters = (); // No parameter transformations needed
    type ExpandedVector = ExpandedDraw;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let logp = self.inverse_model.lnprob_nuts(position);
        let pos = ArrayView1::from(position).into_owned();
        let gradient = self.inverse_model.gradient(&pos).unwrap();

        for (g_out, g) in grad.iter_mut().zip(gradient) {
            *g_out = g;
        }
        Ok(logp)
    }

    /// This function is called for each accepted sample to compute derived
    /// quantities that should be stored in the trace. These might be
    /// transformed parameters, predictions, or other quantities of interest.
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        // Store the raw parameter values
        Ok(ExpandedDraw {
            prec: array.to_vec(),
        })
    }

    fn vector_coord(&self) -> Option<Value> {
        Some(Value::Strings(vec!["x1".to_string(), "x2".to_string()]))
    }
}


/// lnprob_nuts_wrapper helper struct
///
/// We can't (as of the version of Enzyme from 23/11/2023) pass the InversionOptions
/// struct through an enzyme #[autodiff] function, so this is a small helper struct
/// containing only the parameters required for the lnprob calculation.
#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
struct InvOptsHelper {
    pub r_screen_sigma: f64,
    pub exflow_sigma: f64,
}

impl InvOptsHelper {
    fn from_inv_opts(inv_opts: &InversionOptions) -> Self {
        InvOptsHelper {
            r_screen_sigma: inv_opts.r_screen_sigma,
            exflow_sigma: inv_opts.exflow_sigma,
        }
    }

    fn to_inv_opts(self) -> InversionOptions {
        let mut inv_opts = InversionOptionsBuilder::default().build().unwrap();
        inv_opts.r_screen_sigma = self.r_screen_sigma;
        inv_opts.exflow_sigma = self.exflow_sigma;
        inv_opts
    }
}

/// It seems that Enzyme Autodiff has problems with passing in the DetectorInverseModel
/// so instead we'll pass in the components and then rebuild it.
///
/// It turns out that the InversionOptions struct is the one giving problems (maybe because
/// it's a nested struct?  Dunno.)
///
/// There are a lot of unnecessary clone calls, fingers crossed that the compiler optimises
/// them away
 
#[cfg(feature="enzyme_ad")]
use std::autodiff::autodiff;

//Reference: https://enzyme.mit.edu/index.fcgi/rust/usage/rev.html
// `#[autodiff]` should use activities (Const|Active|Duplicated|DuplicatedNoNeed)
#[cfg_attr(
    feature = "enzyme_ad",
    autodiff(
        d_lnprob_nuts_wrapper,
        Reverse,
        Const,
        Const,
        Const,
        Const,
        Duplicated,
        Duplicated,
    )
)]
fn lnprob_nuts_wrapper(
    inv_opt: &[f64],        // Const
    p: DetectorParams,      // Const
    ts: InputRecordVec,      // Const
    fwd: forward::DetectorForwardModel,      // Const
    theta: &[f64], // Duplicated
    lnprob: &mut f64,
)  {
    //let mut inv_opts = InversionOptionsBuilder::default().build().unwrap();
    let inv_opts_default = InversionOptionsBuilder::default().build().unwrap();
    let inv_opts = InversionOptions { r_screen_sigma: inv_opt[0], exflow_sigma: inv_opt[1],
    ..inv_opts_default};
    let inv: DetectorInverseModel = DetectorInverseModel {
        p,
        inv_opts: inv_opts,
        ts: ts.clone(),
        fwd,
    };

    *lnprob = inv.lnprob_nuts(theta);
}

#[cfg(not(feature = "enzyme_ad"))]
fn d_lnprob_nuts_wrapper(
    _inv_opts: &[f64],
    _p: DetectorParams,
    _ts: InputRecordVec,
    _fwd: forward::DetectorForwardModel,
    _theta: &[f64],
    _grad: &mut [f64],
    _logp: &mut f64,
    _seed: &mut f64,
)  {
    unimplemented!();
}

impl HasDims for DetectorInverseModel{
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            // Dimension for the actual parameter vector x
            ("x".to_string(), self.dim() as u64),
        ])

    }
}



impl CpuLogpFunc for DetectorInverseModel {
    type LogpError = PosteriorLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;


    fn dim(&self) -> usize {
        self.ts.len() + NUM_VARYING_PARAMETERS
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        //let helper = InvOptsHelper::from_inv_opts(&self.inv_opts);
        let inv_opt = [self.inv_opts.r_screen_sigma, self.inv_opts.exflow_sigma];
        let inv_opt = inv_opt.as_slice();
        let mut logp = 0.0;
        lnprob_nuts_wrapper(
            &inv_opt,
            self.p.clone(),
            self.ts.clone(),
            self.fwd.clone(),
            position,
            &mut logp,
        );
        //for itm in &mut *grad {*itm=0.0};
        let mut _logp = 0.0;
        let mut seed = 1.0;
        // zero out the gradient each time this is called (as required by enzyme)
        grad.fill(0.0);
        let (_dparams, dtheta) = grad.split_at_mut(2);
        d_lnprob_nuts_wrapper(
            inv_opt,
            //dparams,
            self.p.clone(),
            self.ts.clone(),
            self.fwd.clone(),
            position,
            dtheta,
            &mut _logp,
            &mut seed,
        );
        //dbg!(&grad);
        Ok(logp)
    }
    
    
    fn expand_vector<R>(
        &mut self,
        rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, CpuMathError>
    where
        R: rand::Rng + ?Sized {
        todo!()
    }
}

impl DetectorInverseModel {
    pub fn nuts_sample(&self, _npts: usize, depth: Option<u64>) -> Result<(), anyhow::Error> {
        
        let mut settings = DiagGradNutsSettings::default();
        // and modify as we like
        settings.num_tune = 1000;
        settings.maxdepth = 3;  // small value just for testing...

        // maxdepth makes an enormous difference to runtime
        if let Some(maxdepth) = depth {
            settings.maxdepth = maxdepth;
        }

        let logp_func = self.clone();
        let dim = logp_func.dim();

        let chain = 0;
        let seed = 42;
        //let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut rng = rand::rng();
        let math = CpuMath::new(logp_func);
        let mut sampler = settings.new_chain(chain, math, &mut rng);

        // Attempt to calculate gradient
        let mut test_logp_func = self.clone();
        let test_theta = vec![0.0f64; dim];
        let mut test_gradient = vec![0.0f64; dim];
        let logp_result = test_logp_func.logp(&test_theta, &mut test_gradient);

        dbg!(&logp_result, &test_theta, &test_gradient);

        // Set to some initial position and start drawing samples.
        // Note: it's not possible to use ? here because the NUTS error isn't Sync
        sampler
            .set_position(&vec![0.0f64; dim])
            .expect("Unrecoverable error during init");
        let mut trace = vec![]; // Collection of all draws
        let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
        for iter in 0..2000 {
            let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

            // TODO: nuts_rs has changed how it reports divergence info, find out how to get this
            //if let Some(div_info) = info.divergence_info() {
            //    println!(
            //        "Divergence on iteration {:?} at position {:?}",
            //        iter, div_info.start_location
            //    );
            //}
            if iter % 100 == 0 {
                dbg!(&draw);
                dbg!(&info);
            }
            trace.push(draw);
            stats.push(info);
        }
        Ok(())
    }
}

pub fn test(npts: usize, depth: Option<u64>) -> Result<()> {

    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = 1000;
    settings.maxdepth = 3;  // small value just for testing...

    // maxdepth makes an enormous difference to runtime
    if let Some(maxdepth) = depth {
        settings.maxdepth = maxdepth; // use a small value, e.g. 3 for testing...
    }

    // We instanciate our posterior density function
    let p = DetectorParamsBuilder::default().build()?;
    let inv_opts = InversionOptionsBuilder::default().build()?;
    let ts = get_test_timeseries(npts);

    let logp_func = PosteriorDensity::new(p, inv_opts, ts);
    let dim = logp_func.dim();

    let chain = 0;
    let seed = 42;
    //let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut rng = rand::rng();
    let math = CpuMath::new(logp_func);
    let mut sampler = settings.new_chain(chain, math, &mut rng);

    // Set to some initial position and start drawing samples.
    // Note: it's not possible to use ? here because the NUTS error isn't Sync
    sampler
        .set_position(&vec![0.0f64; dim])
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    for iter in 0..2000 {
        let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

        /* TODO: fix, like above (divergence info has changed)
        if let Some(div_info) = info.divergence_info() {
            println!(
                "Divergence on iteration {:?} at position {:?}",
                iter, div_info.start_location
            );
        }
        */
        if iter % 100 == 0 {
            dbg!(&draw);
            dbg!(&info);
        }
        trace.push(draw);
        stats.push(info);
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "enzyme_ad")]
mod tests {
    use super::*;

    #[test]
    fn sample_nuts() {
        let _ = test(4, Some(3));
    }
}
