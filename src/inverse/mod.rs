mod generic_primitives;

use anyhow::Result;
use std::collections::HashMap;
use std::time::SystemTime;
use thiserror::Error;

use crate::data::DataSet;
use crate::data::GridVariable;
use crate::LogProbContext;

use crate::inverse::generic_primitives::exp_transform;
use crate::inverse::generic_primitives::lognormal_ln_pdf;
use crate::TimeExtents;
use log::error;
use log::info;
use ndarray::Array;
use serde::{Deserialize, Serialize};
use statrs::function::logistic::checked_logit;
use statrs::function::logistic::logistic;

use self::generic_primitives::{normal_ln_pdf, poisson_ln_pmf};

use super::forward::{DetectorForwardModel, DetectorForwardModelBuilder, DetectorParams};

use cobyla::CobylaSolver;
use hammer_and_sample::auto_corr_time;

use ndarray::s;
use ndarray::Array1;

use ndarray::Array3;
use ndarray::ArrayD;
use ndarray::Axis;

use derive_builder::Builder;

/*
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::bfgs::BFGS;
use argmin::solver::trustregion::Steihaug;
use argmin::solver::trustregion::TrustRegion;
*/

pub use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};

use hammer_and_sample::{sample, MinChainLen, Model, Parallel};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

// use ndarray::{Array, Array1, Array2};

use super::InputTimeSeries;

use itertools::izip;

// Link to the BLAS C library
// extern crate blas_src;

pub const NUM_VARYING_PARAMETERS: usize = 2;

/// Errors returned by this module
#[derive(Error, Debug)]
pub enum InverseModelError {
    #[error("no valid observational data")]
    NoObservations,
    #[error("counts are zero over entire time period")]
    ZeroCounts,
    #[error("invalid value encountered when computing prior")]
    InvalidPrior { reference: String },
    #[error("unknown inverse model error")]
    Unknown,
}

/// Return a seed for PRNG
fn get_seed() -> usize {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize
}

// Transform a variable defined over [0,1] to a variable defined over [-inf, +inf]
fn transform_constrained_to_unconstrained(x: f64) -> f64 {
    let a = 0.0;
    let b = 1.0;
    #[allow(clippy::let_and_return)]
    let y = if x == a {
        f64::NEG_INFINITY
    } else if x == b {
        f64::INFINITY
    } else {
        let p = (x - a) / (b - a);
        let result = checked_logit(p);
        match result {
            Ok(y) => y,
            // TODO: attach context to the error using ThisError or Anyhow
            Err(e) => {
                panic!("{}", e);
            }
        }
    };
    y
}

// Transform a variable defined over (-inf,inf) to a variable defined over (0,1)
fn transform_unconstrained_to_constrained(y: f64) -> f64 {
    let a = 0.0;
    let b = 1.0;
    #[allow(clippy::let_and_return)]
    let x = a + (b - a) * logistic(y);
    x
}

// log transformations (for testing)
fn _disabled_transform_radon_concs(radon_conc: &mut [f64]) -> Result<()> {
    for x in radon_conc.iter_mut() {
        *x = (*x).ln();
    }
    Ok(())
}
fn _disabled_inverse_transform_radon_concs(p: &mut [f64]) -> Result<()> {
    for x in p.iter_mut() {
        *x = (*x).exp();
    }
    Ok(())
}

/// Transform radon concentrations from actual values into a simpler-to-sample form
///
/// This is the transformation described in the paper for working with EMCEE-style
/// samplers
pub fn transform_radon_concs1(radon_conc: &mut [f64]) -> Result<()> {
    let n = radon_conc.len();
    let mut rnsum = 0.0;
    for itm in radon_conc.iter() {
        rnsum += *itm;
    }

    let mut acc = rnsum;
    let mut tmp_prev = rnsum.ln();

    #[allow(clippy::needless_range_loop)]
    for ii in 0..(n - 1) {
        let tmp = radon_conc[ii];
        radon_conc[ii] = tmp_prev;
        tmp_prev = transform_constrained_to_unconstrained(tmp / acc);
        acc -= tmp;
    }
    radon_conc[n - 1] = tmp_prev;

    Ok(())
}

// Reverse transform radon concentration (from sampling form back to true values)
pub fn inverse_transform_radon_concs1(p: &mut [f64]) -> Result<()> {
    let n = p.len();
    let mut acc = p[0].exp();
    // handle out of range
    if !acc.is_finite() {
        acc = if p[0] > 0.0 { f64::MAX } else { 0.0 };
    }
    for ii in 0..(n - 1) {
        let rn = transform_unconstrained_to_constrained(p[ii + 1]) * acc;
        p[ii] = rn;
        acc -= rn;
    }
    p[n - 1] = acc;

    if p.iter().any(|x| !x.is_finite()) {
        println!("Conversion failed");
        panic!();
    };

    Ok(())
}

pub fn is_power_of_two(num: usize) -> bool {
    num & (num - 1) == 0
}

pub fn log2_usize(num: usize) -> usize {
    let mut tmp = num;
    let mut shift_count = 0;
    while tmp > 0 {
        tmp >>= 1;
        shift_count += 1;
    }
    shift_count - 1
}

/// Transform radon concentrations from actual values into a simpler-to-sample form
/// This is an experimental option
pub fn transform_radon_concs2(radon_conc: &mut [f64]) -> Result<()> {
    let n = radon_conc.len();
    assert!(is_power_of_two(n));
    let num_levels = log2_usize(n);
    let mut row = radon_conc.to_owned();
    let mut params: Vec<f64> = Vec::new();
    for _ in 0..num_levels {
        // pair elements, and then take average of consecutive elements
        params.extend(row.chunks_exact(2).map(|w| w[0] / ((w[0] + w[1]) / 2.0)));
        row = row.chunks_exact(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        //row.clear();
        //row.extend(tmp.iter());
        //println!("{:?}", row);
    }
    assert!(row.len() == 1);
    params.extend(row);
    //params.push(1.0);

    assert!(radon_conc.len() == params.len());

    #[allow(clippy::manual_memcpy)]
    for ii in 0..params.len() {
        radon_conc[ii] = params[ii]; //transform_unconstrained_to_constrained(params[ii]/2.0);
    }

    Ok(())
}

/// Reverse transform radon concentration (from sampling form back to true values)
pub fn inverse_transform_radon_concs2(p: &mut [f64]) -> Result<()> {
    let npts = p.len();
    assert!(is_power_of_two(npts));
    let num_levels = log2_usize(npts);

    //let mut params = p.iter().map(|itm| 2.0*transform_constrained_to_unconstrained(*itm)).collect::<Vec<_>>();
    let mut params = p.to_owned();

    let mut n = 1;
    let mut a: Vec<f64> = vec![params.pop().unwrap()];

    let mut rp = &params[..];

    for _ in 1..num_levels + 1 {
        // parameters for this level of the reconstruction
        let p = &rp[rp.len() - n..rp.len()];
        // remaining parameters
        rp = &rp[..rp.len() - n];
        // reconstruct this level
        assert_eq!(p.len(), a.len());
        a = izip!(a, p)
            .map(|ap| {
                let (a, p) = ap;
                [a * *p, a * (2.0 - *p)]
            })
            .flatten()
            .collect();
        n *= 2
    }
    // ensured we used up all of the parameters
    assert_eq!(rp.len(), 0);
    p[..npts].copy_from_slice(&a[..npts]);

    Ok(())
}

/* Not sure that this is needed (TODO: uncomment or delete)
pub fn counts_to_concentration<P>(net_counts_per_second: P, sensitivity: P) -> f64
where
    P: Float,
{
    net_counts_per_second / sensitivity
}
*/

/// Pack model description into a state vector
/// rs, rn0(initial radon conc), exflow
pub fn pack_state_vector(
    radon: &[f64],
    p: DetectorParams,
    _ts: InputTimeSeries,
    _opt: InversionOptions,
) -> Vec<f64> {
    let mut values = Vec::new();

    // TODO: modify this to include transforms?
    // transform_radon_concs(&mut radon_transformed).expect("Forward transform failed");

    values.push(p.r_screen_scale.ln());
    values.push(p.exflow_scale.ln());
    values.extend(radon.iter());

    values
}

// Unpack the state vector into its parts
fn unpack_state_vector<'a>(
    guess: &'a &[f64],
    _inv_opts: &InversionOptions,
) -> (f64, f64, &'a [f64]) {
    let r_screen_scale = guess[0];
    let exflow_scale = guess[1];
    let radon_transformed = &guess[2..];

    (r_screen_scale, exflow_scale, radon_transformed)
}

#[derive(Builder, Copy, Debug, Clone, Serialize, Deserialize)]
pub struct InversionOptions {
    // TODO: proper default value
    #[builder(default = "0.01")]
    pub r_screen_sigma: f64,
    // TODO: proper default value
    #[builder(default = "0.01")]
    pub exflow_sigma: f64,
    /// Constraint on the smoothness of radon timeseries
    /// log Cᵢ ∼ Normal ( μ = log(Cᵢ/Cᵢ₋₁), σ = σ_δ )
    #[builder(default = "0.42140")]
    pub sigma_delta: f64,
    /// Threshold beyond which sigma_delta has no further effect
    /// For example, if this is set to 20 then a step change of
    /// a factor of 20 is treated as a discontinuity, and is considered
    /// equally likely as any other (arbitrarily large) step change.
    /// This is a useful "escape hatch" for events such as calibration
    /// pulses.
    #[builder(default = "20.0")]
    pub sigma_delta_threshold: f64,
    /// MCMC sampling strategy
    #[builder(default = "SamplerKind::Emcee")]
    pub sampler_kind: SamplerKind,
    /// Should the MAP be reported?
    #[builder(default = "true")]
    pub report_map: bool,
    /// Maximum number of iterations when searching for MAP
    #[builder(default = "50000")]
    pub map_search_iterations: u64,
    /// Random seed for repeatable experiments
    #[builder(default = "None")]
    pub random_seed: Option<usize>,
    /// Process the timeseries in chunks
    /// (chunksize is 48 points plus 8 points of padding at each end)
    #[builder(default = "true")]
    pub process_in_chunks: bool,
    /// Size of chunks to use, not including overlapping padding
    #[builder(default = "48")]
    pub chunksize: usize,
    /// Number of points of overlap at each end of the chunks
    #[builder(default = "8")]
    pub overlapsize: usize,
    /// Options for the EMCEE sampler
    #[builder(default = "EmceeOptionsBuilder::default().build().unwrap()")]
    pub emcee: EmceeOptions,
    /// Options for NUTS
    #[builder(default = "NutsOptionsBuilder::default().build().unwrap()")]
    pub nuts: NutsOptions,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SamplerKind {
    /// Use the EMCEE MCMC sampling method
    Emcee,
    /// Use the NUTS MCMC sampling method
    Nuts,
}

/// Configuration options for No U-Turn Sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Copy)]
pub struct NutsOptions {
    /// Number of NUTS samples
    #[builder(default = "1000")]
    pub samples: usize,

    /// The thinning factor, i.e. the number of samples to skip in the output
    #[builder(default = "30")]
    pub thin: usize,
}

/// Configuration options for EMCEE sampler
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Copy)]
pub struct EmceeOptions {
    /// Number of burn-in EMCEE samples
    #[builder(default = "5000")]
    pub burn_in: usize,

    /// Number of EMCEE samples
    #[builder(default = "5000")]
    pub samples: usize,

    /// Number of walkers per dimension
    #[builder(default = "3")]
    pub walkers_per_dim: usize,

    /// The thinning factor, i.e. the number of samples to skip in the output
    #[builder(default = "50")]
    pub thin: usize,
}

#[derive(Debug, Clone)]
pub struct DetectorInverseModel {
    /// fixed detector parameters
    pub p: DetectorParams,
    /// Options which control how the inversion runs
    pub inv_opts: InversionOptions,
    /// Data
    pub ts: InputTimeSeries,
    /// Forward model
    pub fwd: DetectorForwardModel,
}

/*
   Model trait for Hammer and Sample (emcee sampler)
   ref: https://docs.rs/hammer-and-sample/latest/hammer_and_sample/
*/
impl Model for DetectorInverseModel {
    type Params = Vec<f64>;
    fn log_prob(&self, state: &Self::Params) -> f64 {
        self.lnprob_f64(state.as_slice(), LogProbContext::EmceeSample)
    }

    const SCALE: f64 = 2.;
}

impl DetectorInverseModel {
    /// Draw samples from the emcee sampler
    ///
    /// Log-probability is derived from the `Model` trait
    pub fn emcee_sample(
        &self,
        inv_opts: InversionOptions,
        map_radon: Vec<f64>,
    ) -> Result<Vec<GridVariable>, anyhow::Error> {
        let num_burn_in_samples = inv_opts.emcee.burn_in;
        let num_samples = inv_opts.emcee.samples;
        let dim = self.ts.len() + NUM_VARYING_PARAMETERS;
        let thin = inv_opts.emcee.thin;
        // num walkers -- ensure it's a multiple of 2
        let num_walkers: usize = {
            let mut x = dim * inv_opts.emcee.walkers_per_dim;
            if x % 2 != 0 {
                x += 1;
            }
            x
        };
        let seed = inv_opts.random_seed.unwrap_or(get_seed());
        let mut map_radon_transformed = map_radon.clone();
        transform_radon_concs1(map_radon_transformed.as_mut_slice())
            .expect("Forward transform failed");
        let initial_state = pack_state_vector(
            map_radon_transformed.as_slice(),
            self.p.clone(),
            self.ts.clone(),
            self.inv_opts,
        );

        // Burn in samples - starting at MAP (+/- a small factor)
        let walkers = ((seed)..(num_walkers + seed)).map(|seed| {
            let mut rng = Pcg64::seed_from_u64(seed.try_into().unwrap());
            let p = (0..dim)
                .zip(initial_state.clone())
                .map(|(_, x)| x + rng.gen_range(-1.0..=1.0) / 1e4)
                .collect();
            (p, rng)
        });

        info!("EMCEE running {} burn-in iterations", num_burn_in_samples);
        let (burn_in_chain, _accepted) = sample(
            self,
            walkers,
            MinChainLen(num_walkers * num_burn_in_samples),
            Parallel,
        );

        // Samples
        let walkers = (0..num_walkers).map(|ii_walker| {
            let seed = ii_walker + seed + num_walkers;
            let rng = Pcg64::seed_from_u64(seed.try_into().unwrap());
            let p = (0..dim)
                .map(|ii| burn_in_chain[burn_in_chain.len() - 1 - ii_walker][ii])
                .collect();
            (p, rng)
        });

        info!("EMCEE running {} sampling iterations", num_samples);
        let (chain, accepted) = sample(
            self,
            walkers,
            MinChainLen(num_walkers * num_samples),
            Parallel,
        );

        let sampled = num_walkers * num_samples;
        let acc_frac = accepted as f64 / sampled as f64;

        // per-walker autocorrelation
        let num_samples_thin = num::integer::div_ceil(num_samples, thin);
        let mut samples = Array3::<f64>::zeros((dim, num_walkers, num_samples_thin));
        for ii in (0..num_samples).step_by(thin) {
            for jj in 0..num_walkers {
                let idx = ii * num_walkers + jj;
                for kk in 0..dim {
                    samples[[kk, jj, ii / thin]] = chain[idx][kk]
                }
            }
        }

        let autocorr = samples.map_axis(Axis(2), |x| {
            let y = auto_corr_time::<_>(x.iter().cloned(), None, Some(1));
            y.unwrap_or(f64::NAN)
        });

        let autocorr_mean = autocorr.map_axis(Axis(1), |row| {
            // Calculate mean, skipping over NaN values
            let n: f64 = row
                .iter()
                .map(|x| if !x.is_finite() { 0.0 } else { 1.0 })
                .sum();
            let sum: f64 = row
                .iter()
                .map(|x| if !x.is_finite() { 0.0 } else { *x })
                .sum();
            if n == 0.0 {
                f64::NAN
            } else {
                sum / n
            }
        });
        //let autocorr_mean = autocorr.mean_axis(Axis(1)).expect("Axis length was zero");

        let n_effective = (num_samples_thin as f64) / autocorr_mean;
        let r_screen_scale_neff = n_effective.slice(s![0]);
        let exflow_scale_neff = n_effective.slice(s![1]);
        let radon_neff = n_effective.slice(s![2..]);

        // 100 iterations of 10 walkers as burn-in
        //let chain = &chain[num_walkers * num_burn_in_samples..];

        //chain.iter().map(|&[p]| p).sum::<f64>() / chain.len() as f64;

        let r_screen_scale_samples = samples.slice(s![0, .., ..]).map(|x| x.exp());
        let exflow_scale_samples = samples.slice(s![1, .., ..]).map(|x| x.exp());
        let transformed_radon_samples = samples.slice(s![2.., .., ..]);

        // inverse transform

        //let radon_ref = self.calc_radon_ref();

        //let radon_samples = transformed_radon_samples.map(|x| x.exp() * radon_ref);

        let mut radon_samples = Array::zeros(transformed_radon_samples.raw_dim());
        let shape = radon_samples.shape().to_vec();
        for ii in 0..shape[1] {
            for jj in 0..shape[2] {
                let mut work = transformed_radon_samples.slice(s![.., ii, jj]).to_owned();
                assert!(work.len() == map_radon.len());
                let work_slice = work.as_slice_mut().unwrap();
                // We need to manually ensure that the inverse transform used here is appropriate
                // and matches:
                //    - the forward transform used above
                //    - the inverse transform used in log_prob of the Model trait
                inverse_transform_radon_concs1(work_slice)?;
                radon_samples.slice_mut(s![.., ii, jj]).assign(&work);
            }
        }

        // Convert to variables with metadata
        let r_screen_scale_samples = GridVariable::new_from_parts(
            r_screen_scale_samples.into_dyn().into_owned(),
            "r_screen_scale",
            &["walker", "sample"],
            None,
        );
        let exflow_scale_samples = GridVariable::new_from_parts(
            exflow_scale_samples.into_dyn().into_owned(),
            "exflow_scale",
            &["walker", "sample"],
            None,
        );

        let radon_samples = GridVariable::new_from_parts(
            radon_samples.into_dyn().into_owned(),
            "radon",
            &["time", "walker", "sample"],
            None,
        );

        // TODO: fix timestep
        let time_step = 30.0 * 60.0;
        let model_time = Array1::range(0.0, self.ts.len() as f64, 1.0) * time_step;
        let model_time = GridVariable::new_from_parts(
            model_time.into_dyn().into_owned(),
            "model_time",
            &["time"],
            Some(HashMap::from([("units".to_owned(), "seconds".to_owned())])),
        );

        let data = vec![
            model_time,
            r_screen_scale_samples,
            exflow_scale_samples,
            radon_samples,
            GridVariable::new_from_parts(vec![acc_frac], "sampler_acceptance_fraction", &[], None),
            GridVariable::new_from_parts(
                r_screen_scale_neff.into_dyn().into_owned(),
                "r_screen_effective_samples_per_walker",
                &[],
                None,
            ),
            GridVariable::new_from_parts(
                exflow_scale_neff.into_dyn().into_owned(),
                "exflow_effective_samples_per_walker",
                &[],
                None,
            ),
            GridVariable::new_from_parts(
                radon_neff.into_dyn().into_owned(),
                "radon_effective_samples_per_walker",
                &["time"],
                None,
            ),
        ];

        Ok(data)
    }
}

impl DetectorInverseModel {
    /// Calculate a reference value for use when transforming radon timeseries
    /// Currently, this is just the mean radon concentration (skipping NaNs).
    pub fn calc_radon_ref(&self) -> f64 {
        let rn = calc_radon_without_deconvolution(&self.ts, self.fwd.time_step);
        let n = rn.iter().filter(|x| x.is_finite()).count();
        let rnavg: f64 = rn
            .iter()
            .filter(|x| x.is_finite())
            .fold(0.0, |acc, e| acc + e)
            / (n as f64);
        rnavg
    }

    pub fn lnprob_f64(&self, theta: &[f64], context: LogProbContext) -> f64 {
        let mut theta_p = Vec::<f64>::with_capacity(theta.len());
        for itm in theta {
            theta_p.push(*itm)
        }
        self.generic_lnprob(&theta_p, context)
    }

    /// Specialised version of lnprob function intended for use with NUTS sampler
    pub fn lnprob_nuts(&self, theta: &[f64]) -> f64 {
        self.lnprob_f64(theta, LogProbContext::NutsSample)
    }

    /// Generic lnprob function which can take differentiable values and is therefore
    /// usable with the autdiff crate (`P` is for "Potentially differentiable")
    #[inline(always)]
    pub fn generic_lnprob(&self, theta: &[f64], context: LogProbContext) -> f64 {
        // Note: invalid prior results are signaled by
        // returning -std::f64::INFINITY

        let mut lp = 0.0f64;
        let mut lprior = 0.0f64;

        // Parameter transformations

        let (log_r_screen_scale, log_exflow_scale, radon_transformed) =
            unpack_state_vector(&theta, &self.inv_opts);
        let (mut r_screen_scale, _lp_inc) = exp_transform(log_r_screen_scale);
        //lp = lp - lp_inc;
        let (mut exflow_scale, _lp_inc) = exp_transform(log_exflow_scale);
        //lp = lp - lp_inc;

        // println!("{:?} {:?}", log_exflow_scale, exflow_scale);

        // inverse_transform_radon_concs(&mut radon).expect("Inverse transform failure");

        // println!("{:#?}", radon);

        ////let mu = 1.0;
        ////let sigma = 0.1;
        ////lp += LogNormal::new(mu,sigma).unwrap().ln_pdf(radon_0);
        // TODO: other priors

        // println!(" ===>>> {}, {}", exflow_scale, r_screen_scale);

        // Place limits on parameters
        // (This should not be required, but...)
        if !lp.is_finite() {
            let (t0, t1) = self.fwd.data.time_extents_str();
            let chunk_id = format!("chunk-{t0}-{t1}");
            panic!("Non-finite prior encountered {}", chunk_id);
        }

        let half = 0.5;
        let two = 2.0;
        let _thousand = 1000.0;

        if r_screen_scale < half {
            // lp = lp - (r_screen_scale - half) * (r_screen_scale - half) * thousand;
            r_screen_scale = half;
        } else if r_screen_scale > 1.1 {
            //lp = lp - (r_screen_scale - f64::from(1.1).unwrap()) * (r_screen_scale - f64::from(1.1).unwrap());
            r_screen_scale = 1.1;
        }
        if exflow_scale < half {
            //lp = lp - (exflow_scale - half) * (exflow_scale - half) * thousand;
            exflow_scale = half;
        } else if exflow_scale > two {
            //  lp = lp - (exflow_scale - two) * (exflow_scale - two);
            exflow_scale = two;
        }

        if !lp.is_finite() {
            dbg!(&lp);
            dbg!(&exflow_scale);
            dbg!(&r_screen_scale);

            let (t0, t1) = self.fwd.data.time_extents_str();
            let chunk_id = format!("chunk-{t0}-{t1}");
            panic!("Non-finite prior encountered {}", chunk_id);
        }

        assert!(lp.is_finite());

        // Lognormal priors
        let r_screen_scale_mu = 1.0f64.ln();
        let r_screen_scale_sigma = self.inv_opts.r_screen_sigma;
        lprior += lognormal_ln_pdf(r_screen_scale_mu, r_screen_scale_sigma, r_screen_scale);

        // TODO: get exflow from data instead of from the parameters
        let exflow_scale_mu = 1.0f64.ln();
        let exflow_sigma = self.inv_opts.exflow_sigma;
        lprior += lognormal_ln_pdf(exflow_scale_mu, exflow_sigma, exflow_scale);

        // Normal priors on parameters
        let r_screen_scale_mu = 1.0;
        let r_screen_scale_sigma = self.inv_opts.r_screen_sigma;
        lprior += normal_ln_pdf(r_screen_scale_mu, 1.0, r_screen_scale);

        let r_screen_scale = (r_screen_scale - 1.0) * r_screen_scale_sigma + 1.0;

        let exflow_scale_mu = 1.0;
        let exflow_sigma = self.inv_opts.exflow_sigma;
        lprior += normal_ln_pdf(exflow_scale_mu, 1.0, exflow_scale);

        let exflow_scale = (exflow_scale - 1.0) * exflow_sigma + 1.0;

        // println!("{:?} {:?} {:?} {:?} || {:?} {:?} || {:?}", r_screen_scale_mu, r_screen_scale_sigma, exflow_scale_mu, exflow_sigma, r_screen_scale, exflow_scale, lprior);

        // Do the inverse transform
        let radon = match context {
            LogProbContext::EmceeSample => {
                let mut radon = radon_transformed.to_vec();
                inverse_transform_radon_concs1(radon.as_mut_slice())
                    .expect("Inverse transform failure");
                for x in &radon {
                    assert!(x.is_finite());
                }
                radon
            }
            LogProbContext::NutsSample | LogProbContext::MapSearch => {
                // Normal priors on the radon scale factor.  This is applied to keep this parameter
                // in a numerically-stable range, and should be too weak to have an impact on the
                // result.  exp(10) = 22026; exp(100) = 2.688e43
                let ten = 10.0;

                for u in radon_transformed {
                    lprior += normal_ln_pdf(0.0, ten, *u);
                }

                // for good measure, also clip this parameter to [-100, 100]
                let mut radon_transformed = radon_transformed.to_owned();
                let hundred = 100.0;
                for u in radon_transformed.iter_mut() {
                    if *u > hundred {
                        *u = hundred;
                    } else if *u < -hundred {
                        *u = -hundred;
                    }
                }

                let radon_reference_value = self.calc_radon_ref();

                let radon: Vec<_> = radon_transformed
                    .iter()
                    .map(|x| {
                        let (x_t, lp_inc) = exp_transform(*x);
                        // increment lp because of the change-of-variables
                        lp -= lp_inc;
                        x_t * radon_reference_value
                    })
                    .collect();
                for x in &radon {
                    if !x.is_finite() {
                        let (t0, t1) = self.fwd.data.time_extents_str();
                        let chunk_id = format!("chunk-{t0}-{t1}");
                        panic!(
                            "Non-finite prior encountered in radon guess.  Chunk: {}, radon reference {:?} radon {:?}, radon_transformed {:?}, counts {:?}",
                            chunk_id, radon_reference_value, &radon, &radon_transformed, &self.fwd.data.counts
                        );
                    }
                }
                radon
            }
        };

        assert!(lprior.is_finite());

        // "Numerical" prior on radon concentration (included to stop parameter values from sailing off to +oo)
        // Maxiumum reasonable radon concentration, 100 kBq, is 10000x larger than a high radon concentration
        // in the natural atmosphere
        let rn_max = 1000e3;
        let rn_max_sigma = 10e3;
        let p0 = normal_ln_pdf(rn_max, rn_max_sigma, rn_max);
        // for the PDF we'll use a uniform distribution up to Rn_max then a half-normal.  Any value
        // which is <= Rn_max leaves lp unchanged.
        for x in &radon {
            if *x > rn_max {
                let lprior_inc = normal_ln_pdf(rn_max, rn_max_sigma, *x) - p0;
                if !lprior_inc.is_finite() {
                    let (t0, t1) = self.fwd.data.time_extents_str();
                    let chunk_id = format!("chunk-{t0}-{t1}");
                    panic!(
                        "Radon value error.  Chunk: {}, radon {:?}, radon_i {:?}, lprior_inc {:?}, p0 {:?}, counts {:?}",
                        chunk_id, &radon, *x, lprior_inc, p0, &self.fwd.data.counts
                    );
                }
                lprior += normal_ln_pdf(rn_max, rn_max_sigma, *x) - p0
            }
        }

        assert!(lprior.is_finite());

        // This is the "smooth timeseries" prior on radon concentration
        let threshold = normal_ln_pdf(
            0.0,
            self.inv_opts.sigma_delta,
            self.inv_opts.sigma_delta_threshold,
        );
        if self.inv_opts.sigma_delta > 0.0 {
            for pair in radon.windows(2) {
                let mut lprior_inc;
                // Special case: one (or both) of the values is equal to zero (which might happen due
                // to underflow when exp(large_negative_value) is calculated)
                if pair[0] == 0.0 || pair[1] == 0.0 {
                    lprior_inc = threshold
                } else {
                    lprior_inc =
                        normal_ln_pdf(0.0, self.inv_opts.sigma_delta, (pair[1] / pair[0]).ln());
                    if !lprior_inc.is_finite() {
                        dbg!(&lprior_inc);
                        dbg!(pair);
                    }
                }
                // If the two points are very different (e.g. factor of 10 or more) then we should no
                // longer apply the constraint.  E.g. this is a truncated normal

                // TODO: check the sign of this comparison
                if lprior_inc < threshold {
                    lprior_inc = threshold;
                }
                lprior += lprior_inc;
            }
        }

        assert!(lprior.is_finite());

        lp += lprior;

        // Likelihood
        let mut fwd = self.fwd.clone();
        fwd.radon = radon.clone();
        fwd.p.r_screen_scale = r_screen_scale;
        fwd.p.exflow_scale = exflow_scale;

        let fwd_copy = fwd.clone();

        let expected_counts = match fwd.numerical_expected_counts() {
            Ok(counts) => counts,
            Err(e) => {
                println!("Forward model failed: {:?}, \n{:?}", e, fwd_copy);
                return -f64::INFINITY;
            }
        };

        for x in &expected_counts {
            if !x.is_finite() {
                let chunk_id = self.fwd.data.chunk_id();
                panic!(
                    "forward model failed (produced invalid value) chunk {}, model counts {:?}, fwd model: {:?}",
                    chunk_id,
                    &expected_counts,
                    &fwd_copy
                );
            }
        }

        let observed_counts: Vec<_> = self.ts.iter().map(|itm| itm.counts).collect();

        assert!(expected_counts.len() == observed_counts.len());
        assert!(radon.len() == expected_counts.len());

        // The right approach here depends on the context.  For emcee sampling, the best thing
        // to do is to return -inf, but for optimisation it might be better just to run
        // something...

        //let MIN_VALID = f64::from(1e-6).unwrap();

        for (cex, cobs) in expected_counts.iter().zip(observed_counts) {
            //let poisson_arg: f64 =  if *cex < 1e-6{
            //    println!("Expected counts out of range: {}", *cex);
            //    1e-6
            //}
            //else{
            //    *cex
            //};
            //if *cex < MIN_VALID {
            //    return -f64::from(f64::INFINITY).unwrap();
            //}
            if !cobs.is_finite() {
                continue;
            }
            let lp_inc = poisson_ln_pmf(*cex, *cobs);
            if !lp_inc.is_finite() {
                let chunk_id = self.fwd.data.chunk_id();
                panic!(
                    "encountered invalid log-probability increment in chunk {}. expected (from model) = {:?} observed = {:?} lp_increment = {:?} ln_prior = {:?} r_screen_scale = {:?} exflow_scale = {:?} radon = {:?}",
                    chunk_id, *cex, *cobs, lp_inc, lprior, r_screen_scale, exflow_scale, radon
                );
            }
            assert!(lp_inc.is_finite());
            lp += lp_inc;
            // println!(" *** expected (from model) = {:?} observed = {:?} lp_increment = {:?} {:?}", *cex, *cobs, lp_inc, radon);
        }
        assert!(lp.is_finite());
        lp
    }
}

/// 'argmin' CostFunction trait
impl CostFunction for DetectorInverseModel {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let minus_lp = -self.lnprob_f64(param.as_slice().unwrap(), LogProbContext::MapSearch);
        // TODO: if lp is -std::f64::INFINITY then we should probably
        // return an error
        //println!("{param}");
        Ok(minus_lp)
    }
}

struct CobylaDetectorInverseModel(DetectorInverseModel);

impl CostFunction for CobylaDetectorInverseModel {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let minus_lp = -self
            .0
            .lnprob_f64(param.as_slice(), LogProbContext::MapSearch);
        // TODO: if lp is -std::f64::INFINITY then we should probably
        // return an error
        Ok(vec![minus_lp])
    }
}

/// 'argmin' Gradient trait
impl Gradient for DetectorInverseModel {
    /// Type of the parameter vector
    type Param = Array1<f64>;
    /// Type of the gradient
    type Gradient = Array1<f64>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, _param: &Self::Param) -> Result<Self::Gradient, Error> {
        // Use enzyme here
        todo!();
    }
}

/*
impl Gradient for DetectorInverseModel<f64> {
    type Param = ndarray::Array1<f64>;
    type Gradient = ndarray::Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|x| self.apply(x).unwrap()))
    }
}

impl Hessian for DetectorForwardModel<f64> {
    type Param = ndarray::Array1<f64>;
    type Hessian = ndarray::Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> std::result::Result<Self::Hessian, Error> {
        Ok((*param).forward_hessian(&|x| self.gradient(x).unwrap()))
    }
}
*/

/*
impl<P, T> ArgminOp for DetectorInverseModel<P, T>
where
    P: Float,
    T: Float,
{
    // -- most of this is boilerplate from argmin docs
    // Type of the parameter vector
    type Param = ndarray::Array1<f64>;
    // Type of the return value computed by the cost function
    type Output = f64;
    // Type of the Hessian. Can be `()` if not needed.
    type Hessian = ndarray::Array2<f64>;
    // Type of the Jacobian. Can be `()` if not needed.
    type Jacobian = ();
    // Floating point precision
    type Float = f64;
    // Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // convert into emcee's Guess.  For now, just abuse .clone()
        // and hope that this either doesn't matter to runtime or that
        // perhaps the compiler will optimize it away.
        let guess = Guess {
            values: p.clone().into_raw_vec(),
        };
        let minus_lp = -self.lnprob(&guess);
        // TODO: if lp is -std::f64::INFINITY then we should probably
        // return an error
        Ok(minus_lp)
        //if lp == -std::f64::INFINITY {
        //    Err(); <-- this needs sorting out
        //}
        //else{
        //    Ok(lp)
        //}
    }
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok((*p).forward_diff(&|x| self.apply(x).unwrap()))
    }
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok((*p).forward_hessian(&|x| self.gradient(x).unwrap()))
    }
}
*/

/// Un-deconvolved calculation of radon concentration.  Negative values
/// are replaced with 0.0
pub fn calc_radon_without_deconvolution(ts: &InputTimeSeries, time_step: f64) -> Vec<f64> {
    ts.iter()
        .map(|itm| {
            let cps = itm.counts / time_step;
            (cps - itm.background_count_rate) / itm.sensitivity
        })
        .map(|itm| itm.clamp(0.0, f64::MAX))
        .collect()
}

fn can_quantise(values: &[f64], threshold: f64) -> bool {
    if values.is_empty() {
        return false;
    }
    // Unwrap: Ok, we've checked for zero-length slice
    let maxval = values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let minval = values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    if (maxval - minval) == 0.0 {
        // All values are already the same as each other - noop
        return false;
    }

    (maxval - minval) / ((maxval.abs() + minval.abs()) / 2.0) > threshold
}

fn quantise(values: &mut [f64]) {
    let mean_value = values.iter().fold(0.0, |a, b| a + b) / (values.len() as f64);
    for x in values.iter_mut() {
        *x = mean_value;
    }
}

fn quantise_timeseries(ts: InputTimeSeries) -> (InputTimeSeries, bool) {
    let mut ts = ts.clone();
    let threshold = 1e-3;
    let mut flag_changed = false;
    if can_quantise(ts.sensitivity.as_slice(), threshold) {
        quantise(ts.sensitivity.as_mut_slice());
        flag_changed = true;
    }
    if can_quantise(ts.background_count_rate.as_slice(), threshold) {
        quantise(ts.background_count_rate.as_mut_slice());
        flag_changed = true;
    }
    (ts, flag_changed)
}

pub fn fit_inverse_model(
    p: DetectorParams,
    inv_opts: InversionOptions,
    ts: InputTimeSeries,
) -> Result<DataSet> {
    let npts = ts.len();
    let time_step = 60.0 * 30.0; //TODO: calculate from input
                                 //let time_step_diff = time_step;

    // Data, which will output at the end of the function
    let mut data: Vec<GridVariable> = vec![];

    if ts.counts.iter().fold(0.0, |x, y| x + y) == 0.0 {
        return Err(InverseModelError::ZeroCounts.into());
    }

    // Radon concentration, simple calculation without deconvolution
    let initial_radon = calc_radon_without_deconvolution(&ts, time_step);

    if initial_radon.iter().all(|x| !x.is_finite()) {
        return Err(InverseModelError::NoObservations.into());
    }

    let (ts, flag_changed) = quantise_timeseries(ts);
    if flag_changed {
        let (t0, t1) = ts.time_extents_str();
        let chunk_id = format!("chunk-{t0}-{t1}");
        info!("{} Values in input timeseries were close to constant and have been replaced by constant values as an optimisation.", chunk_id)
    }

    data.push(GridVariable::new_from_parts(
        ArrayD::from_shape_vec(vec![initial_radon.len()], initial_radon.clone())?,
        "undeconvolved_radon",
        &["time"],
        None,
    ));

    // Scale by mean value and take log so that values take the range -inf,+inf
    // with most values around zero.  Mean value takes skips over NaNs in input
    // which are used as placeholders for missing observations.
    let n = initial_radon
        .iter()
        .fold(0, |x, y| x + if y.is_finite() { 1 } else { 0 });
    let mean_radon = initial_radon
        .iter()
        .fold(0.0, |x, y| if y.is_finite() { x + y } else { x })
        / (n as f64);

    assert!(mean_radon.is_finite());

    // Initial guess radon, un-deconvolved radon estimate with NaN gaps
    // filled with mean radon concentration (after scaling, mean radon
    // equals 0.0 )
    let initial_radon_scaled: Vec<_> = initial_radon
        .iter()
        .map(|x| {
            if !x.is_finite() {
                0.0
            } else {
                (x / mean_radon).ln()
            }
        })
        .collect();

    // 1. Initialisation
    // Define initial parameter vector and cost function

    //println!("Initial radon concentration: {:?}", initial_radon);
    //println!("Initial radon scaled: {:?}", initial_radon_scaled);
    let constant_radon_scaled = vec![0.0; npts];
    let init_param = {
        let v = pack_state_vector(&constant_radon_scaled, p.clone(), ts.clone(), inv_opts);
        Array1::<f64>::from_vec(v)
    };

    assert!(init_param.len() == ts.len() + 2);

    let fwd = DetectorForwardModelBuilder::default()
        .data(ts.clone())
        .time_step(time_step)
        .radon(initial_radon_scaled.clone())
        .build()
        .expect("Failed to build detector model");
    let inverse_model: DetectorInverseModel = DetectorInverseModel {
        p,
        inv_opts,
        ts: ts.clone(),
        fwd,
    };

    // 2. Optimisation (MAP)

    let map_radon = if inv_opts.report_map {
        info!("Searching for maximum a posteriori (MAP)");
        let niter = inv_opts.map_search_iterations;
        // COBYLA solver version
        let cob_inverse_model = CobylaDetectorInverseModel(inverse_model.clone());
        let solver = CobylaSolver::new(init_param.as_slice().unwrap().to_owned());
        let res = Executor::new(cob_inverse_model, solver)
            .configure(|state| {
                let mut state = state.max_iters(niter); // set to 50_000
                state.maxfun = i32::MAX;
                state
            })
            //.add_observer(SlogLogger::term(), ObserverMode::Every(100))
            .run();

        if let Err(e) = &res {
            error!("Error during MAP search: {}", e);
            None
        } else {
            let res = res.unwrap();
            info!("Finished searching for MAP.  Iterations: {}", res);
            let map = res.state.get_best_param().unwrap();
            let map_vec = map.clone().to_vec();
            let v = map_vec.as_slice();
            let (transformed_r_screen_scale, transformed_exflow_scale, transformed_map_radon) =
                unpack_state_vector(&v, &inv_opts);

            let _r_screen_scale = transformed_r_screen_scale.exp();
            let _exflow_scale = transformed_exflow_scale.exp();
            let map_radon: Vec<_> = transformed_map_radon
                .iter()
                .map(|x| x.exp() * mean_radon)
                .collect();

            data.push(GridVariable::new_from_parts(
                ArrayD::from_shape_vec(vec![map_radon.len()], map_radon.clone())?,
                "map_radon",
                &["time"],
                None,
            ));
            // TODO: log MAP radon, r_screen_scale, exflow_scale
            Some(map_radon)
        }
    } else {
        None
    };

    info!("MAP radon: {:?}", map_radon);

    match inv_opts.sampler_kind {
        SamplerKind::Emcee => {
            let sampler_output = inverse_model
                .emcee_sample(inv_opts, map_radon.expect("Emcee sampler needs MAP"))?;
            data.extend(sampler_output);
        }

        SamplerKind::Nuts => {
            //Enzyme version
            inverse_model.nuts_sample(2000, None)?;

            // Autodiff (library) version
            // let _sampler_output = inverse_model.nuts_sample(2000, None)?;

            // TODO data.extend(sampler_output);
        }
    }

    data.extend(ts.to_grid_vars());

    let ds = DataSet::new_from_variables(data);
    Ok(ds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::DetectorParamsBuilder;
    use crate::InputRecord;
    use crate::InputRecordVec;

    use assert_approx_eq::assert_approx_eq;

    fn get_timeseries(npts: usize) -> InputRecordVec {
        let trec = InputRecord {
            time: 0.0,
            // LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 1000.0 + 30.0,
            background_count_rate: 1.0 / 60.0,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
            ..Default::default()
        };
        let mut ts = InputRecordVec::new();
        for _ in 0..npts {
            ts.push(trec);
        }

        //ts.counts[npts/2] *= 5.0;
        //ts.counts[npts/2+1] *= 5.0;

        /*
        // Modify some of the timeseries
        let counts = vec![1333.0, 3473.0, 5385.0, 4935.0, 3833.0, 2828.0, 2060.0];

        assert!(npts > counts.len());
        for ii in 0..counts.len() {
            ts.counts[ii + npts - counts.len()] = counts[ii];
        }
        */

        ts
    }

    #[test]
    fn can_compute_lnprob() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(5);
        let time_step = 60.0 * 30.0; //TODO
                                     // Define initial parameter vector and cost function
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        let mut too_high_radon = initial_radon.clone();
        let rnavg = initial_radon.iter().sum::<f64>() / (initial_radon.len() as f64);
        for itm in too_high_radon.iter_mut().skip(1) {
            *itm = rnavg * 100.0;
        }

        dbg!(&too_high_radon);
        println!("Detector parameters: {:?}", p);
        println!("Inversion options: {:?}", inv_opts);

        println!("Initial radon concentration: {:?}", initial_radon);
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
        let worse_guess = pack_state_vector(&too_high_radon, p.clone(), ts.clone(), inv_opts);

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(initial_radon)
            .build()
            .expect("Failed to build detector model");
        let cost = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts.clone(),
            fwd: fwd.clone(),
        };

        // println!("initial guess: {:#?}", init_param.values);
        println!(
            "initial guess cost function evaluation: {}",
            cost.generic_lnprob(&init_param, LogProbContext::MapSearch)
        );

        // println!("const radon guess: {:#?}", worse_guess.values);
        println!(
            "Bad guess cost function evaluation: {}",
            cost.generic_lnprob(&worse_guess, LogProbContext::MapSearch)
        );
    }

    #[test]
    fn lnprob_changes_if_radon_changes() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let npts = 10;
        let ts = get_timeseries(npts);
        let time_step = 60.0 * 30.0; //TODO
                                     // Define initial parameter vector and cost function
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        // calculate lnprob reference value (it's the exact solution, so should be == lnprob_max)
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(initial_radon.clone())
            .build()
            .expect("Failed to build detector model");
        let cost = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts.clone(),
            fwd: fwd.clone(),
        };
        let lnprob_max = cost.generic_lnprob(&init_param, LogProbContext::MapSearch);
        // println!("initial guess: {:#?}", init_param.values);
        println!("cost function evaluation at MAP: {}", &lnprob_max);
        for idx in 0..npts {
            let mut too_high_radon = initial_radon.clone();
            too_high_radon[idx] += 1.0;

            // calculate lprob for perturbed radon timeseries
            let init_param = pack_state_vector(&too_high_radon, p.clone(), ts.clone(), inv_opts);
            let fwd = DetectorForwardModelBuilder::default()
                .data(ts.clone())
                .time_step(time_step)
                .radon(initial_radon.clone())
                .build()
                .expect("Failed to build detector model");
            let cost = DetectorInverseModel {
                p: p.clone(),
                inv_opts: inv_opts,
                ts: ts.clone(),
                fwd: fwd.clone(),
            };
            let lnprob_perturbed = cost.generic_lnprob(&init_param, LogProbContext::MapSearch);
            dbg!(&lnprob_perturbed, &lnprob_max);

            assert!(lnprob_perturbed < lnprob_max)
        }
    }

    #[cfg(enzyme_ad)]
    #[test]
    fn can_compute_lnprob_with_gradient() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(5);
        let time_step = 60.0 * 30.0; //TODO
                                     // Define initial parameter vector and cost function
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        let mut too_high_radon = initial_radon.clone();
        let rnavg = initial_radon.iter().sum::<f64>() / (initial_radon.len() as f64);
        for itm in too_high_radon.iter_mut().skip(1) {
            *itm = rnavg * 100.0;
        }

        dbg!(&too_high_radon);
        println!("Detector parameters: {:?}", p);
        println!("Inversion options: {:?}", inv_opts);

        println!("Initial radon concentration: {:?}", initial_radon);
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
        let worse_guess = pack_state_vector(&too_high_radon, p.clone(), ts.clone(), inv_opts);

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(initial_radon)
            .build()
            .expect("Failed to build detector model");
        let cost = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts.clone(),
            fwd: fwd.clone(),
        };

        let cost_diff = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts,
            fwd: fwd.clone(),
        };

        // println!("initial guess: {:#?}", init_param.values);
        let pvec = ndarray::Array1::from_vec(init_param.clone());
        println!(
            "initial guess cost function evaluation: {}",
            cost.generic_lnprob(&init_param, LogProbContext::MapSearch)
        );
        println!(
            "Initial guess cost function gradient: {:?}",
            cost_diff.gradient(&pvec).unwrap()
        );

        // println!("const radon guess: {:#?}", worse_guess.values);
        let worse_pvec = ndarray::Array1::from_vec(worse_guess.clone());
        println!(
            "Bad guess cost function evaluation: {}",
            cost.generic_lnprob(&worse_guess, LogProbContext::MapSearch)
        );
        println!(
            "Bad guess cost function gradient: {:?}\n       pvec: {:?}",
            cost_diff.gradient(&worse_pvec).unwrap(),
            &worse_pvec
        );
    }

    #[cfg(enzyme_ad)]
    #[test]
    fn gradient_is_sensitive_to_ambient_radon() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(2);
        let time_step = 60.0 * 30.0; //TODO
                                     // Define initial parameter vector and cost function
        let mut initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        // set this value to something higher so that gradients will be non-zero
        initial_radon[1] = 10.0;
        println!("Initial radon concentration: {:?}", initial_radon);
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(initial_radon)
            .build()
            .expect("Failed to build detector model");

        let cost_diff = DetectorInverseModel {
            p: p.clone(),
            inv_opts: inv_opts,
            ts: ts,
            fwd: fwd.clone(),
        };

        // println!("initial guess: {:#?}", init_param.values);
        let pvec = ndarray::Array1::from_vec(init_param.clone());
        let gradient = cost_diff.gradient(&pvec).unwrap();
        println!("Initial guess cost function gradient: {:?}", &gradient);

        assert!((gradient[3] - 0.0).abs() > 1e-5);
    }

    #[test]
    fn can_run_constant_inverse_problem() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let mut inv_opts = InversionOptionsBuilder::default()
            .map_search_iterations(100)
            .random_seed(Some(64))
            .build()
            .unwrap();
        inv_opts.emcee.samples = 100;
        inv_opts.emcee.burn_in = 100;
        let npts = 4;
        let ts = get_timeseries(npts);
        fit_inverse_model(p, inv_opts, ts).expect("Failed to fit inverse model");
    }

    #[test]
    fn can_run_inverse_problem() {
        // TODO: set options for very small number of iterations
        let p = DetectorParamsBuilder::default().build().unwrap();
        let mut inv_opts = InversionOptionsBuilder::default()
            .map_search_iterations(100)
            .random_seed(Some(64))
            .build()
            .unwrap();
        inv_opts.emcee.samples = 100;
        inv_opts.emcee.burn_in = 100;
        let npts = 15;
        let mut ts = get_timeseries(npts);
        ts.counts[npts - 1] += 500.0;
        fit_inverse_model(p, inv_opts, ts).expect("Failed to fit inverse model");
    }

    /*
    #[test]
    fn handles_bad_input() {
        // Input is bad since counts < background count rate
        // causes expected counts to be less than zero
        // and blows up Poisson distribution
        let p = DetectorParamsBuilder::default().build().unwrap();
        let trec = InputRecord {
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0 / 60.0,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
        };
        let mut ts = InputRecordVec::new();
        for _ in 0..10 {
            ts.push(trec);
        }

        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        fit_inverse_model(p, inv_opts, ts).expect("Failed to fit inverse model");
    }
    */

    #[test]
    fn transforms() {
        let npts = 16;
        let radon: Vec<f64> = (1..=npts).map(|itm| itm as f64).collect();
        println!("Original:      {:?}", radon);
        println!("Transformation #1");
        let mut radon_transformed = radon.clone();
        transform_radon_concs1(&mut radon_transformed).unwrap();
        println!("Transformed:   {:?}", radon_transformed);
        let mut radon_reconstructed = radon_transformed.clone();
        inverse_transform_radon_concs1(&mut radon_reconstructed).unwrap();

        println!("Reconstructed: {:?}", radon_reconstructed);
        for (r1, r2) in radon.clone().into_iter().zip(radon_reconstructed) {
            assert_approx_eq!(r1, r2);
        }

        println!("Transformation #1");
        // same, but for second pair of transforms
        let mut radon_transformed = radon.clone();
        transform_radon_concs2(&mut radon_transformed).unwrap();
        println!("Transformed:   {:?}", radon_transformed);
        let mut radon_reconstructed = radon_transformed.clone();
        inverse_transform_radon_concs2(&mut radon_reconstructed).unwrap();

        println!("Reconstructed: {:?}", radon_reconstructed);
        for (r1, r2) in radon.clone().into_iter().zip(radon_reconstructed) {
            assert_approx_eq!(r1, r2);
        }
    }

    #[test]
    fn log_funcs() {
        assert_eq!(log2_usize(2_usize.pow(12)), 12_usize);
    }

    #[test]
    fn basic_funcs() {
        assert_eq!(is_power_of_two(2_usize.pow(10)), true);
        assert_eq!(log2_usize(2_usize.pow(10)), 10)
    }
}
