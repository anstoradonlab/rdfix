use super::forward::{
    DetectorForwardModel, DetectorForwardModelBuilder, DetectorParams, DetectorParamsBuilder,
};
use argmin::solver::linesearch::HagerZhangLineSearch;
use ndarray::Array2;
use statrs::distribution::{Continuous, Discrete, LogNormal, Normal, Poisson};

use derive_builder::Builder;

/*
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::bfgs::BFGS;
use argmin::solver::trustregion::Steihaug;
use argmin::solver::trustregion::TrustRegion;
*/

use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};

use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::BFGS;
use argmin::solver::trustregion::Steihaug;
use argmin::solver::trustregion::TrustRegion;

use argmin::core::observers::{ObserverMode, SlogLogger};

use num::Float;

use hammer_and_sample::{sample, Model, Serial};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

// use ndarray::{Array, Array1, Array2};

use finitediff::FiniteDiff;
use log::{debug, error, info, log_enabled};
use statrs::function::logistic::{checked_logit, logistic, logit};

use super::{InputTimeSeries, OutputTimeSeries};

use anyhow::Result;

use itertools::izip;

use assert_approx_eq::assert_approx_eq;
use autodiff::*;

// Link to the BLAS C library
extern crate blas_src;

type FP = f64;

// Transform a variable defined over [0,1] to a variable defined over [-inf, +inf]
fn transform_constrained_to_unconstrained(x: f64) -> f64 {
    const a: f64 = 0.0;
    const b: f64 = 1.0;
    let y = if x == a {
        -f64::INFINITY
    } else if x == b {
        f64::INFINITY
    } else {
        let p = (x - a) / (b - a);
        let result = checked_logit(p);
        match result {
            Ok(y) => y,
            // TODO: attach context to the error using ThisError or Anyhow
            Err(e) => {
                println!("p={}", p);
                panic!("{}", e);
            }
        }
    };
    y
}

// Transform a variable defined over (-inf,inf) to a variable defined over (0,1)
fn transform_unconstrained_to_constrained(y: f64) -> f64 {
    const a: f64 = 0.0;
    const b: f64 = 1.0;
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

// Transform radon concentrations from actual values into a simpler-to-sample form
fn transform_radon_concs1(radon_conc: &mut [f64]) -> Result<()> {
    let N = radon_conc.len();
    let rnsum: f64 = radon_conc.iter().sum();
    let mut acc = rnsum;
    let mut tmp_prev = rnsum.ln();

    for ii in 0..(N - 1) {
        let tmp = radon_conc[ii];
        radon_conc[ii] = tmp_prev;
        tmp_prev = transform_constrained_to_unconstrained(tmp / acc);
        acc -= tmp;
    }
    radon_conc[N - 1] = tmp_prev;

    Ok(())
}

// Reverse transform radon concentration (from sampling form back to true values)
fn inverse_transform_radon_concs1(p: &mut [f64]) -> Result<()> {
    // copy so that we can report the error later
    let p_saved: Vec<_> = p.into();

    let N = p.len();
    let mut acc = p[0].exp();
    // handle out of range
    if !acc.is_finite() {
        acc = if p[0] > 0.0 { std::f64::MAX } else { 0.0 };
    }
    for ii in 0..(N - 1) {
        let rn = transform_unconstrained_to_constrained(p[ii + 1]) * acc;
        p[ii] = rn;
        acc -= rn;
    }
    p[N - 1] = acc;

    if p.iter().any(|x| !x.is_finite()) {
        println!("Conversion failed");
        println!("Input:  {:?}", p_saved);
        println!("Output: {:?}", p);
        panic!();
    };

    Ok(())
}

fn is_power_of_two(num: usize) -> bool {
    num & (num - 1) == 0
}

fn log2_usize(num: usize) -> usize {
    let mut tmp = num;
    let mut shift_count = 0;
    while tmp > 0 {
        tmp = tmp >> 1;
        shift_count += 1;
    }
    shift_count - 1
}

// Transform radon concentrations from actual values into a simpler-to-sample form
fn transform_radon_concs<P>(radon_conc: &mut [P]) -> Result<()>
where
    P: Float,
{
    let n = radon_conc.len();
    assert!(is_power_of_two(n));
    let num_levels = log2_usize(n);
    let mut row = radon_conc.to_owned();
    let mut params: Vec<P> = Vec::new();
    for _ in 0..num_levels {
        // pair elements, and then take average of consecutive elements
        params.extend(
            row.chunks_exact(2)
                .map(|w| w[0] / ((w[0] + w[1]) / P::from(2.0).unwrap())),
        );
        row = row
            .chunks_exact(2)
            .map(|w| (w[0] + w[1]) / P::from(2.0).unwrap())
            .collect();
        //row.clear();
        //row.extend(tmp.iter());
        //println!("{:?}", row);
    }
    assert!(row.len() == 1);
    params.extend(row);
    //params.push(1.0);

    assert!(radon_conc.len() == params.len());

    for ii in 0..params.len() {
        radon_conc[ii] = params[ii]; //transform_unconstrained_to_constrained(params[ii]/2.0);
    }

    Ok(())
}


/*

// Reverse transform radon concentration (from sampling form back to true values)
fn inverse_transform_radon_concs<P>(p: &mut [P]) -> Result<()>
where
    P: Float,
{
    let npts = p.len();
    assert!(is_power_of_two(npts));
    let num_levels = log2_usize(npts);

    //let mut params = p.iter().map(|itm| 2.0*transform_constrained_to_unconstrained(*itm)).collect::<Vec<_>>();
    let mut params = p.to_owned();

    let mut n = 1;
    let mut a: Vec<P> = vec![params.pop().unwrap()];

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
                [a * p, a * (P::from(2.0).unwrap() - p)]
            })
            .flatten()
            .collect();
        n *= 2
    }
    // ensured we used up all of the parameters
    assert_eq!(rp.len(), 0);
    for ii in 0..npts {
        p[ii] = a[ii];
    }

    Ok(())
}

fn counts_to_concentration<P>(net_counts_per_second: P, sensitivity: P) -> P
where
    P: Float,
{
    net_counts_per_second / sensitivity
}

/// Pack model description into a state vector
/// rs, rn0(initial radon conc), exflow
fn pack_state_vector<P>(
    radon: &[P],
    p: DetectorParams<P>,
    ts: InputTimeSeries,
    opt: InversionOptions,
) -> Vec<P>
where
    P: Float,
{
    let mut values = Vec::new();

    let mut radon_transformed = radon.to_owned();
    transform_radon_concs(&mut radon_transformed).expect("Forward transform failed");

    values.push(p.r_screen_scale);
    values.push(p.exflow_scale);
    values.extend(radon_transformed.iter());

    values
}

// Unpack the state vector into its parts
fn unpack_state_vector<'a, P>(guess: &'a &[P], _inv_opts: &InversionOptions) -> (P, P, &'a [P])
where
    P: Float,
{
    let r_screen_scale = guess.values[0];
    let exflow_scale = guess.values[1];
    let radon_transformed = &guess.values[2..];

    (r_screen_scale, exflow_scale, radon_transformed)
}

#[derive(Builder, Copy, Debug, Clone)]
pub struct InversionOptions {
    // TODO: proper default value
    #[builder(default = "0.05")]
    pub r_screen_sigma: f64,
    // TODO: proper default value
    #[builder(default = "0.05")]
    pub exflow_sigma: f64,
}

#[derive(Debug, Clone)]
pub struct DetectorInverseModel<P>
where
    P: Float,
{
    /// fixed detector parameters
    pub p: DetectorParams<P>,
    /// Options which control how the inversion runs
    pub inv_opts: InversionOptions,
    /// Data
    pub ts: InputTimeSeries,
    /// Forward model
    pub fwd: DetectorForwardModel<P>,
}

/*
   Model trait for Hammer and Sample (emcee sampler)
   ref: https://docs.rs/hammer-and-sample/latest/hammer_and_sample/
*/
impl Model for DetectorInverseModel<f64> {
    type Params = Vec<f64>;
    fn log_prob(&self, state: &Self::Params) -> f64 {
        todo!();
    }
}

impl<P: Float> DetectorInverseModel<P> {
    /*
    Generic lnprob function which can take differentable values and is therefore
    usable with the autdiff crate
    */
    fn generic_lnprob(&self, theta: &[P]) -> f64 {
        // Note: invalid prior results are signaled by
        // returning -std::f64::INFINITY

        let mut lp = P::zero();
        // Priors

        let (mut r_screen_scale, mut exflow_scale, radon_transformed) =
            unpack_state_vector(&theta, &self.inv_opts);
        let mut radon = radon_transformed.to_owned();
        inverse_transform_radon_concs(&mut radon).expect("Inverse transform failure");
        // println!("{:#?}", radon);

        ////let mu = 1.0;
        ////let sigma = 0.1;
        ////lp += LogNormal::new(mu,sigma).unwrap().ln_pdf(radon_0);
        // TODO: other priors

        // println!(" ===>>> {}, {}", exflow_scale, r_screen_scale);

        // soft limits on parameters

        if r_screen_scale < 0.5 {
            //lp += 1.0 / r_screen_scale - 1.0/0.1;
            lp = lp + (r_screen_scale - 0.5.into()) * 1e3.into();
            r_screen_scale = 0.5;
        } else if r_screen_scale > 1.1 {
            lp = lp + r_screen_scale - 1.1;
            r_screen_scale = 1.1;
        }
        if exflow_scale < 0.5 {
            lp = lp + (exflow_scale - 0.5) * 1e3;
            exflow_scale = 0.5;
        } else if exflow_scale > 2.0 {
            lp = lp + exflow_scale - 2.0;
            exflow_scale = 2.0;
        }

        let mut ln_prior = 0.0;

        let r_screen_scale_mu = 1.0.ln();
        let r_screen_scale_sigma = self.inv_opts.r_screen_sigma;
        ln_prior = ln_prior
            + LogNormal::new(r_screen_scale_mu, r_screen_scale_sigma)
                .unwrap()
                .ln_pdf(r_screen_scale);

        // TODO: get exflow from data instead of from the parameters
        let exflow_scale_mu = 1.0.ln();
        let exflow_sigma = self.inv_opts.exflow_sigma;
        ln_prior = ln_prior
            + LogNormal::new(exflow_scale_mu, exflow_sigma)
                .unwrap()
                .ln_pdf(exflow_scale);

        // println!("{} {} {} {} || {} {}", r_screen_mu, r_screen_sigma, exflow_mu, exflow_sigma, r_screen, exflow);

        lp = lp + ln_prior;

        // Likelihood
        let mut fwd = self.fwd.clone();
        fwd.radon = radon;
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

        let observed_counts: Vec<_> = self.ts.iter().map(|itm| itm.counts).collect();

        // The right apprach here depends on the context.  For emcee sampling, the best thing
        // to do is to return -inf, but for optimisation it might be better just to run
        // something...
        for (cex, cobs) in expected_counts.iter().zip(observed_counts) {
            //let poisson_arg: f64 =  if *cex < 1e-6{
            //    println!("Expected counts out of range: {}", *cex);
            //    1e-6
            //}
            //else{
            //    *cex
            //};
            if *cex < 1e-6 {
                return -f64::INFINITY;
            }
            let lp_inc = Poisson::new(*cex)
                .expect("Poisson failed")
                .ln_pmf((*cobs).round() as u64);
            //let lp_max = Poisson::new(*cobs).unwrap().ln_pmf((*cobs).round() as u64);
            lp += lp_inc;
            // println!("{} {} {} {}", *cex, *cobs, lp_inc, lp_max);
        }
        lp
    }
}

impl CostFunction for DetectorInverseModel<f64> {
    type Param = ndarray::Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let minus_lp = -self.generic_lnprob(&param);
        // TODO: if lp is -std::f64::INFINITY then we should probably
        // return an error
        Ok(minus_lp)
    }
}

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

fn calc_radon_without_deconvolution(ts: &InputTimeSeries, time_step: f64) -> Vec<f64> {
    ts.iter()
        .map(|itm| {
            let cps = itm.counts / time_step;
            (cps - itm.background_count_rate) / itm.sensitivity
        })
        .collect()
}

pub fn fit_inverse_model(
    // TODO: differentiable types for the first argument
    p: DetectorParams<f64>,
    inv_opts: InversionOptions,
    ts: InputTimeSeries,
) -> Result<(), Error> {
    let v = vec![1, 2, 3, 4];
    let s = v.as_slice();
    let time_step = 60.0 * 30.0; //TODO

    // 1. Initialisation
    // Define initial parameter vector and cost function
    let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
    println!("Initial radon concentration: {:?}", initial_radon);
    let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
    // build an identity matrix as Vec<Vec<f64>> (easier: use Array2 in the model, then Array2::eye)
    let init_inverse_hessian = ndarray::Array2::eye(init_param.values.len());
    let fwd = DetectorForwardModelBuilder::default()
        .data(ts.clone())
        .time_step(time_step)
        .radon(initial_radon.clone())
        .build()
        .expect("Failed to build detector model");
    let cost = DetectorInverseModel {
        p,
        inv_opts,
        ts,
        fwd,
    };
    let inverse_model = cost.clone();

    // 2. Optimisation (MAP)
    const ARGMIN_OPTION: u32 = 1;

    // Nelder Mead initial simplex
    // Strategy for initialisation: https://stackoverflow.com/questions/17928010/choosing-the-initial-simplex-in-the-nelder-mead-optimization-algorithm
    // Note: not working
    let mut simplex: Vec<ndarray::Array1<f64>> = vec![init_param.values.clone().into()];
    let ndims = init_param.values.len();
    // step size
    let dx = 1e-3;
    for ii in 0..ndims {
        let mut v: ndarray::Array1<f64> = init_param.values.clone().into();
        v[ii] += dx;
        simplex.push(v);
    }
    let nm_solver: NelderMead<ndarray::Array1<f64>, f64> = NelderMead::new(simplex);

    // a few different linesearch options

    let linesearch3 = MoreThuenteLineSearch::new()
        .c(1e-4, 0.9)
        .unwrap()
        .alpha(1e-10, 1e-6)
        .unwrap();
    //let linesearch4 = HagerZhangLineSearch::new().alpha(1e-8, 1e-3).unwrap();

    let map_max_iterations = 20000; //TODO: options
    let res = match ARGMIN_OPTION {
        1 => Executor::new(cost, TrustRegion::new(Steihaug::new()))
            .configure(|state| state.param(init_param.values.clone().into()).radius(0.1))
            .add_observer(SlogLogger::term(), ObserverMode::Every(100))
            .max_iters(map_max_iterations)
            .run()
            .unwrap(),

        /* disable other options for now.  This needs updating for new API.
                2 =>
                //unimplemented!(),
                {
                    Executor::new(cost, nm_solver, init_param.values.into())
                        .add_observer(SlogLogger::term(), ObserverMode::Every(100))
                        .max_iters(5000) // Probably use 20k here
                        .run()
                        .unwrap()
                }

                3 => Executor::new(
                    cost,
                    BFGS::new(init_inverse_hessian, linesearch3),
                    init_param.values.into(),
                )
                .add_observer(SlogLogger::term(), ObserverMode::Every(1))
                .max_iters(map_max_iterations)
                .run()
                .unwrap(),

                4 => Executor::new(
                    cost,
                    SteepestDescent::new(linesearch3),
                    init_param.values.into(),
                )
                .add_observer(SlogLogger::term(), ObserverMode::Every(100))
                .max_iters(map_max_iterations)
                .run()
                .unwrap(),
        */
        _ => unimplemented!(),
    };

    println!("MAP optimisation complete: {}", res);
    let map = res.state.get_best_param();
    let v: Vec<f64> = map.into_raw_vec().iter().collect();
    let (_, _, transformed_map_radon) = unpack_state_vector(&v, &inv_opts);
    let mut map_radon = &mut transformed_map_radon.to_owned();
    inverse_transform_radon_concs(&mut map_radon).unwrap();

    println!("Initial radon concentration: {:?}", initial_radon);
    println!("MAP radon concentration:     {:?}", map_radon);

    // 3. Generate initial guess around the MAP point

    let nwalkers = 6 * ndims; // TODO, add to inv_opts
    let walkers = (0..nwalkers).map(|seed| {
        let mut rng = Pcg64::seed_from_u64(seed as u64);

        let p = v
            .iter()
            .map(|p_i| p_i + rng.gen_range(-1e-6..=1.0e-6))
            .collect();

        (p, rng)
    });

    // 4. Run the emcee sampler
    let ndim = v.len();
    let niterations = 5000;

    println!("Running MCMC");
    //let mut sampler =
    //    emcee::EnsembleSampler::new(nwalkers, ndim, &inverse_model).expect("creating sampler");

    let (chain, accepted) = sample(&inverse_model, walkers, niterations * 2, &Serial);

    // half of the iterations are burn-in
    let chain = &chain[niterations * nwalkers..];

    // samples are now in chain
    // TODO: reshape into nsamples * nwalkers array
    let acceptance_fraction = accepted as f64 / niterations as f64;

    // 5. Wrangle the output and compute statistics
    println!("Complete.  Acceptance fraction: {:?}", acceptance_fraction);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputRecord;
    use crate::InputRecordVec;

    fn get_timeseries(npts: usize) -> InputRecordVec {
        let trec = InputRecord {
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 1000.0 + 30.0,
            background_count_rate: 1.0 / 60.0,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
        };
        let mut ts = InputRecordVec::new();
        for _ in 0..npts {
            ts.push(trec);
        }

        //ts.counts[npts/2] *= 5.0;
        //ts.counts[npts/2+1] *= 5.0;

        let counts = vec![1333.0, 3473.0, 5385.0, 4935.0, 3833.0, 2828.0, 2060.0];

        assert!(npts > counts.len());
        for ii in 0..counts.len() {
            ts.counts[ii + npts - counts.len()] = counts[ii];
        }

        ts
    }

    #[test]
    fn can_compute_lnprob() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(50);
        let time_step = 60.0 * 30.0; //TODO
                                     // Define initial parameter vector and cost function
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        let mut constant_radon = initial_radon.clone();
        let rnavg = initial_radon.iter().sum::<f64>() / (initial_radon.len() as f64);
        for itm in constant_radon.iter_mut() {
            *itm = rnavg * 100.0;
        }
        println!("Initial radon concentration: {:?}", initial_radon);
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
        let worse_guess = pack_state_vector(&constant_radon, p.clone(), ts.clone(), inv_opts);

        let fwd = DetectorForwardModelBuilder::default()
            .data(ts.clone())
            .time_step(time_step)
            .radon(initial_radon)
            .build()
            .expect("Failed to build detector model");
        let cost = DetectorInverseModel {
            p,
            inv_opts,
            ts,
            fwd,
        };

        // println!("initial guess: {:#?}", init_param.values);
        let pvec = ndarray::Array1::from_vec(init_param.values.clone());
        println!(
            "initial guess cost function evaluation: {}",
            cost.lnprob(&init_param)
        );
        println!(
            "Initial guess cost function gradient: {:?}",
            cost.gradient(&pvec).unwrap()
        );

        // println!("const radon guess: {:#?}", worse_guess.values);
        let worse_pvec = ndarray::Array1::from_vec(worse_guess.values.clone());
        println!(
            "const radon cost function evaluation: {}",
            cost.lnprob(&worse_guess)
        );
        println!(
            "Const radon cost function gradient: {:?}",
            cost.gradient(&worse_pvec).unwrap()
        );
    }

    #[test]
    fn can_run_inverse_problem() {
        // TODO: set options for very small number of iterations
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(64);
        fit_inverse_model(p, inv_opts, ts).expect("Failed to fit inverse model");
    }

    //#[test]
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
        fit_inverse_model(p, inv_opts, ts);
    }

    #[test]
    fn transforms() {
        let npts = 16;
        let radon: Vec<f64> = (0..npts).map(|itm| itm as f64).collect();
        let radon_avg = radon.iter().sum::<f64>() / (radon.len() as f64);
        println!("Original:      {:?}", radon);
        let mut radon_transformed = radon.clone();
        transform_radon_concs(&mut radon_transformed).unwrap();
        println!("Transformed:   {:?}", radon_transformed);
        let mut radon_reconstructed = radon_transformed.clone();
        inverse_transform_radon_concs(&mut radon_reconstructed).unwrap();

        //println!("Partial Recons:{:?}", radon_reconstructed);
        //for itm in radon_reconstructed.iter_mut(){
        //    *itm *= radon_avg;
        //}
        println!("Reconstructed: {:?}", radon_reconstructed);
        for (r1, r2) in radon.into_iter().zip(radon_reconstructed) {
            assert_approx_eq!(r1, r2);
        }
    }

    #[test]
    fn log_funcs() {
        assert_eq!(log2_usize(2_usize.pow(12)), 12_usize);
    }
}


*/