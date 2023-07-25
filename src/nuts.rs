use anyhow::Result;
use argmin::core::Gradient;
use autodiff::F1;
use ndarray::{Array1, ArrayView1};

pub use nuts_rs::{new_sampler, Chain, CpuLogpFunc, LogpError, SampleStats, SamplerArgs};
use thiserror::Error;

use super::forward::{DetectorForwardModelBuilder, DetectorParams, DetectorParamsBuilder};

use super::*;

use super::inverse::*;

// Define a function that computes the unnormalized posterior density
// and its gradient.
struct PosteriorDensity {
    inverse_model: DetectorInverseModel<F1>,
    dim: usize,
}

impl PosteriorDensity {
    fn new(p: DetectorParams<f64>, inv_opts: InversionOptions, ts: InputTimeSeries) -> Self {
        let time_step = 60.0 * 30.0; //TODO
        let time_step_diff = F1::cst(time_step);

        // Radon concentration, without
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);

        // Params, as differentiable type
        let p_diff = p.into_inner_type::<F1>();
        let mean_radon =
            initial_radon.iter().fold(0.0, |x, y| x + y) / (initial_radon.len() as f64);
        let initial_radon_diff = vec![F1::cst(mean_radon); initial_radon.len()];

        // 1. Initialisation
        // Define initial parameter vector and cost function

        //println!("Initial radon concentration: {:?}", initial_radon);
        let _init_param = {
            let v = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
            Array1::<f64>::from_vec(v)
        };

        let fwd = DetectorForwardModelBuilder::<F1>::default()
            .data(ts.clone())
            .time_step(time_step_diff)
            .radon(initial_radon_diff.clone())
            .build()
            .expect("Failed to build detector model");
        let inverse_model: DetectorInverseModel<F1> = DetectorInverseModel {
            p: p_diff,
            inv_opts: inv_opts,
            ts: ts,
            fwd: fwd,
        };

        // TODO: don't hard code the number of non-radon parameters (here, it's 2)
        let dim: usize = initial_radon.len() + 2;

        PosteriorDensity {
            inverse_model: inverse_model,
            dim: dim,
        }
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

impl CpuLogpFunc for PosteriorDensity {
    type Err = PosteriorLogpError;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
        let logp = self.inverse_model.lnprob_f64(position);
        let pos = ArrayView1::from(position).into_owned();
        let gradient = self.inverse_model.gradient(&pos).unwrap();

        for (g_out, g) in grad.iter_mut().zip(gradient) {
            *g_out = g;
        }
        return Ok(logp);
    }
}

impl CpuLogpFunc for DetectorInverseModel<F1> {
    type Err = PosteriorLogpError;

    fn dim(&self) -> usize {
        self.ts.len() + NUM_VARYING_PARAMETERS
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
        let logp = self.lnprob_f64(position);
        let pos = ArrayView1::from(position).into_owned();
        let gradient = self.gradient(&pos).unwrap();

        for (g_out, g) in grad.iter_mut().zip(gradient) {
            *g_out = g;
        }
        return Ok(logp);
    }
}

impl DetectorInverseModel<F1> {
    pub fn nuts_sample(&self, _npts: usize, depth: Option<u64>) -> Result<(), anyhow::Error> {
        // We get the default sampler arguments
        let mut sampler_args = SamplerArgs::default();

        // and modify as we like
        sampler_args.num_tune = 1000;
        // maxdepth makes an enormous difference to runtime
        if let Some(maxdepth) = depth {
            sampler_args.maxdepth = maxdepth; // use a small value, e.g. 3 for testing...
        }

        let logp_func = self.clone();
        let dim = logp_func.dim();

        let chain = 0;
        let seed = 42;
        let mut sampler = new_sampler(logp_func, sampler_args, chain, seed);

        // Set to some initial position and start drawing samples.
        // Note: it's not possible to use ? here because the NUTS error isn't Sync
        sampler
            .set_position(&vec![0.0f64; dim])
            .expect("Unrecoverable error during init");
        let mut trace = vec![]; // Collection of all draws
        let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
        for iter in 0..2000 {
            let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

            let _info_vec = info.to_vec(); // We can collect the stats in a Vec
                                           // Or get more detailed information about divergences
            if let Some(div_info) = info.divergence_info() {
                println!(
                    "Divergence on iteration {:?} at position {:?}",
                    iter,
                    div_info.start_location()
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
    // We get the default sampler arguments
    let mut sampler_args = SamplerArgs::default();

    // and modify as we like
    sampler_args.num_tune = 1000;
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
    let mut sampler = new_sampler(logp_func, sampler_args, chain, seed);

    // Set to some initial position and start drawing samples.
    // Note: it's not possible to use ? here because the NUTS error isn't Sync
    sampler
        .set_position(&vec![0.0f64; dim])
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    for iter in 0..2000 {
        let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

        let _info_vec = info.to_vec(); // We can collect the stats in a Vec
                                       // Or get more detailed information about divergences
        if let Some(div_info) = info.divergence_info() {
            println!(
                "Divergence on iteration {:?} at position {:?}",
                iter,
                div_info.start_location()
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
mod tests {
    use super::*;

    #[test]
    fn sample_nuts() {
        let _ = test(4, Some(3));
    }
}
