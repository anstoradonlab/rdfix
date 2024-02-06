use anyhow::Result;
use argmin::core::Gradient;
#[cfg(enzyme_ad)]
use autodiff::autodiff as enzyme_autodiff;
use ndarray::{Array1, ArrayView1};

pub use nuts_rs::{new_sampler, Chain, CpuLogpFunc, LogpError, SampleStats, SamplerArgs};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
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

        // Radon concentration, without
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
            .time_step(time_step)
            .radon(initial_radon.clone())
            .build()
            .expect("Failed to build detector model");
        let inverse_model: DetectorInverseModel = DetectorInverseModel {
            p: p,
            inv_opts,
            ts,
            fwd,
        };

        // TODO: don't hard code the number of non-radon parameters (here, it's 2)
        let dim: usize = initial_radon.len() + 2;

        PosteriorDensity { inverse_model, dim }
    }
}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
pub enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}


/// NUTS sampler trait template (from documentation)
impl CpuLogpFunc for PosteriorDensity {
    type Err = PosteriorLogpError;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
        let logp = self.inverse_model.lnprob_nuts(position);
        let pos = ArrayView1::from(position).into_owned();
        let gradient = self.inverse_model.gradient(&pos).unwrap();

        for (g_out, g) in grad.iter_mut().zip(gradient) {
            *g_out = g;
        }
        Ok(logp)
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

impl InvOptsHelper{
    fn from_inv_opts(inv_opts: &InversionOptions) -> Self
    {
        InvOptsHelper{
            r_screen_sigma: inv_opts.r_screen_sigma,
            exflow_sigma: inv_opts.exflow_sigma }
    }

    fn to_inv_opts(self) -> InversionOptions
    {
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

// `#[autodiff]` should use activities (Const|Active|Duplicated|DuplicatedNoNeed)
#[cfg_attr(enzyme_ad, enzyme_autodiff(d_lnprob_nuts_wrapper, Reverse, Active,   Const, Const, Const, Const, Duplicated))] 
fn lnprob_nuts_wrapper(helper: InvOptsHelper, p: DetectorParams, ts: InputRecordVec, fwd: forward::DetectorForwardModel, theta: &[f64]) -> f64{
    
    let inv: DetectorInverseModel = DetectorInverseModel {
        p: p,
        inv_opts: helper.to_inv_opts(),
        ts: ts.clone(),
        fwd,
    };


    inv.lnprob_nuts(theta)
}

#[cfg(not(enzyme_ad))]
fn d_lnprob_nuts_wrapper(_helper: InvOptsHelper, _p: DetectorParams, _ts: InputRecordVec, _fwd: forward::DetectorForwardModel, _theta: &[f64], _grad: &mut[f64], _tangent: f64) -> (){
    unimplemented!();
}

impl CpuLogpFunc for DetectorInverseModel {
    type Err = PosteriorLogpError;

    fn dim(&self) -> usize {
        self.ts.len() + NUM_VARYING_PARAMETERS
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
        let helper = InvOptsHelper::from_inv_opts(&self.inv_opts);
        let logp = lnprob_nuts_wrapper(helper, self.p.clone(), self.ts.clone(), self.fwd.clone(), position);
        //for itm in &mut *grad {*itm=0.0};
        d_lnprob_nuts_wrapper(helper, self.p.clone(), self.ts.clone(), self.fwd.clone(), position, grad, 1.0);
        //dbg!(&grad);
        Ok(logp)
    }
}

impl DetectorInverseModel {
    pub fn nuts_sample(&self, _npts: usize, depth: Option<u64>) -> Result<(), anyhow::Error> {
        let mut sampler_args = SamplerArgs {
            num_tune: 1000,
            ..Default::default()
        };
        // maxdepth makes an enormous difference to runtime
        if let Some(maxdepth) = depth {
            sampler_args.maxdepth = maxdepth; // use a small value, e.g. 3 for testing...
        }

        let logp_func = self.clone();
        let dim = logp_func.dim();

        let chain = 0;
        let seed = 42;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sampler = new_sampler(logp_func, sampler_args, chain, &mut rng);

        // Set to some initial position and start drawing samples.
        // Note: it's not possible to use ? here because the NUTS error isn't Sync
        sampler
            .set_position(&vec![0.0f64; dim])
            .expect("Unrecoverable error during init");
        let mut trace = vec![]; // Collection of all draws
        let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
        for iter in 0..2000 {
            let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

            if let Some(div_info) = info.divergence_info() {
                println!(
                    "Divergence on iteration {:?} at position {:?}",
                    iter, div_info.start_location
                );
            }
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
    let mut sampler_args = nuts_rs::SamplerArgs {
        num_tune: 1000,
        ..Default::default()
    };
    // maxdepth makes an enormous difference to runtime
    if let Some(maxdepth) = depth {
        sampler_args.maxdepth = maxdepth; // use a small value, e.g. 3 for testing...
    }

    // We instanciate our posterior density function
    let p = DetectorParamsBuilder::default().build()?;
    let inv_opts = InversionOptionsBuilder::default().build()?;
    let ts = get_test_timeseries(npts);

    let logp_func = PosteriorDensity::new(p, inv_opts, ts);
    let dim = logp_func.dim();

    let chain = 0;
    let seed = 42;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sampler = new_sampler(logp_func, sampler_args, chain, &mut rng);

    // Set to some initial position and start drawing samples.
    // Note: it's not possible to use ? here because the NUTS error isn't Sync
    sampler
        .set_position(&vec![0.0f64; dim])
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    for iter in 0..2000 {
        let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

        if let Some(div_info) = info.divergence_info() {
            println!(
                "Divergence on iteration {:?} at position {:?}",
                iter, div_info.start_location
            );
        }
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
#[cfg(enzyme_ad)]
mod tests {
    use super::*;

    #[test]
    fn sample_nuts() {
        let _ = test(4, Some(3));
    }
}
