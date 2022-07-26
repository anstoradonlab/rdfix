use super::forward::{DetectorParams, DetectorParamsBuilder, DetectorForwardModel, DetectorForwardModelBuilder};
use emcee::{Guess, Prob};
use ndarray::Array2;
use statrs::distribution::{LogNormal, Poisson, Normal, Continuous, Discrete};

use derive_builder::Builder;

use argmin::prelude::*;
use argmin::solver::trustregion::TrustRegion;
use argmin::solver::trustregion::Steihaug;
use argmin::solver::quasinewton::bfgs::BFGS;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::linesearch::MoreThuenteLineSearch;

// use ndarray::{Array, Array1, Array2};

use log::{debug, error, log_enabled, info};
use finitediff::FiniteDiff;
use statrs::function::logistic::{logit, logistic};

use super::{InputTimeSeries,OutputTimeSeries};

use anyhow::Result;

use autodiff::*;


// Transform a variable defined over (0,1) to a variable defined over (-inf, +inf)
fn transform_constrained_to_unconstrained(x:f64) -> f64{
    const a: f64 = 0.0;
    const b: f64 = 1.0;
    let y = logit( (x-a)/(b-a) );
    y
}

// Transform a variable defined over (-inf,inf) to a variable defined over (0,1)
fn transform_unconstrained_to_constrained(y:f64) -> f64{
    const a: f64 = 0.0;
    const b: f64 = 1.0;
    let x = a + (b-a) * logistic(y);
    x
}

// log transformations (for testing)
fn _disabled_transform_radon_concs(radon_conc: &mut[f64]) -> Result<()>{
    for x in radon_conc.iter_mut(){
        *x = (*x).ln();
    }
    Ok(())
}
fn _disabled_inverse_transform_radon_concs(p: &mut[f64]) -> Result<()>{
    for x in p.iter_mut(){
        *x = (*x).exp();
    }
    Ok(())
}


// Transform radon concentrations from actual values into a simpler-to-sample form
fn transform_radon_concs(radon_conc: &mut[f64]) -> Result<()>{
    let N = radon_conc.len();
    let rnsum: f64 = radon_conc.iter().sum();
    let mut acc = rnsum;
    let mut tmp_prev = rnsum.ln();

    for ii in 0..(N-1){
        let tmp = radon_conc[ii];
        radon_conc[ii] = tmp_prev;
        tmp_prev = transform_constrained_to_unconstrained(tmp/acc);
        acc -= tmp;
    }
    radon_conc[N-1] = tmp_prev;

    Ok(())
}

// Reverse transform radon concentration (from sampling form back to true values)
fn inverse_transform_radon_concs(p: &mut[f64]) -> Result<()>{
    let N = p.len();
    let mut acc = p[0].exp();
    for ii in 0..(N-1){
        let rn = transform_unconstrained_to_constrained(p[ii+1]) * acc;
        p[ii] = rn;
        acc -= rn;
    }
    p[N-1] = acc;

    Ok(())
}
 


fn counts_to_concentration(net_counts_per_second: f64, sensitivity: f64) -> f64{
    net_counts_per_second / sensitivity
}


/// Pack model description into a state vector
/// rs, rn0(initial radon conc), exflow
fn pack_state_vector(radon: &[f64], p: DetectorParams, ts: InputTimeSeries, opt: InversionOptions) -> Guess{
    let mut values = Vec::new();
    
    let mut radon_transformed = radon.to_owned();
    transform_radon_concs(&mut radon_transformed).expect("Forward transform failed");
        
    values.push(p.r_screen_scale);
    values.push(p.exflow_scale);
    values.extend(radon_transformed.iter());

    Guess{values}
}

// Unpack the state vector into its parts
fn unpack_state_vector<'a>(guess: &'a Guess, _inv_opts: &InversionOptions) -> (f64,f64,&'a[f64]){
    let r_screen_scale = guess.values[0];
    let exflow_scale = guess.values[1];
    let radon_transformed = &guess.values[2..];
    
    (r_screen_scale, exflow_scale, radon_transformed)
}


#[derive(Builder,Copy,Debug,Clone)]
pub struct InversionOptions{
    // TODO: proper default value
    #[builder(default = "0.05")]
    pub r_screen_sigma: f64,
    // TODO: proper default value
    #[builder(default = "0.05")]
    pub exflow_sigma: f64,
}

pub struct DetectorInverseModel{
    /// fixed detector parameters
    pub p: DetectorParams,
    /// Options which control how the inversion runs
    pub inv_opts: InversionOptions,
    /// Data
    pub ts: InputTimeSeries,
    /// Forward model
    pub fwd: DetectorForwardModel,
}



impl Prob for DetectorInverseModel{
    fn lnprob(&self, theta: &Guess) -> f64{

        // Note: invalid prior results are signaled by 
        // returning -std::f64::INFINITY

        let mut lp = 0.0;
        // Priors
    
        let (r_screen_scale, exflow_scale, radon_transformed) =
             unpack_state_vector(&theta, &self.inv_opts);
        let mut radon = radon_transformed.to_owned();
        inverse_transform_radon_concs(&mut radon).expect("Inverse transform failure");
        // println!("{:#?}", radon);
    
        
        ////let mu = 1.0;
        ////let sigma = 0.1; 
        ////lp += LogNormal::new(mu,sigma).unwrap().ln_pdf(radon_0);
        // TODO: other priors
        
        let mut ln_prior = 0.0;
        
        let r_screen_scale_mu = 1.0.ln();
        let r_screen_scale_sigma = self.inv_opts.r_screen_sigma;
        ln_prior += LogNormal::new(r_screen_scale_mu, r_screen_scale_sigma).unwrap().ln_pdf(r_screen_scale);

        // TODO: get exflow from data instead of from the parameters
        let exflow_scale_mu = 1.0.ln();
        let exflow_sigma = self.inv_opts.exflow_sigma;
        ln_prior += LogNormal::new(exflow_scale_mu, exflow_sigma).unwrap().ln_pdf(exflow_scale);

        // println!("{} {} {} {} || {} {}", r_screen_mu, r_screen_sigma, exflow_mu, exflow_sigma, r_screen, exflow);

        lp += ln_prior;


        // Likelihood
        let mut fwd = self.fwd.clone();
        fwd.radon = radon;
        fwd.p.r_screen_scale = r_screen_scale;
        fwd.p.exflow_scale = exflow_scale;

        let expected_counts = fwd.numerical_expected_counts().expect("Forward model failure");
        let observed_counts: Vec<_> = self.ts.iter().map(|itm| itm.counts).collect();

        for (cex, cobs) in expected_counts.iter().zip(observed_counts){
            let lp_inc = Poisson::new(*cex).unwrap().ln_pmf((*cobs).round() as u64);
            let lp_max = Poisson::new(*cobs).unwrap().ln_pmf((*cobs).round() as u64);
            lp += lp_inc;
            // println!("{} {} {} {}", *cex, *cobs, lp_inc, lp_max);
        }
        lp

    }

    fn lnlike(&self, params: &Guess) -> f64 {
        unimplemented!()
    }

    fn lnprior(&self, params: &Guess) -> f64 {
        unimplemented!()
    }
}

impl ArgminOp for DetectorInverseModel{
    // -- most of this is boilerplate from argmin docs
    // Type of the parameter vector
    type Param = Vec<f64>;
    // Type of the return value computed by the cost function
    type Output = f64;
    // Type of the Hessian. Can be `()` if not needed.
    type Hessian = Vec<Vec<f64>>;
    // Type of the Jacobian. Can be `()` if not needed.
    type Jacobian = ();
    // Floating point precision
    type Float = f64;
    // Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // convert into emcee's Guess.  For now, just abuse .clone()
        // and hope that this either doesn't matter to runtime or that
        // perhaps the compiler will optimize it away.
        let guess = Guess{values:p.clone()};
        let minus_lp = - self.lnprob(&guess);
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
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>{
        Ok((*p).forward_diff(&|x| self.apply(x).unwrap()))
    }
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>{
        Ok((*p).forward_hessian(&|x| self.gradient(x).unwrap()))
    }
}

fn calc_radon_without_deconvolution(ts: & InputTimeSeries, time_step: f64) -> Vec<f64>{
    ts.iter().map(|itm|{
        let cps = itm.counts / time_step;
        (cps - itm.background_count_rate) / itm.sensitivity
    })
    .collect()
}


pub fn fit_inverse_model(p: DetectorParams, inv_opts: InversionOptions, ts: InputTimeSeries) -> Result<(),Error>{
    let v = vec![1,2,3,4];
    let s = v.as_slice();
    let time_step = 60.0*30.0; //TODO

    // 1. Initialisation
    // Define initial parameter vector and cost function
    let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
    println!("Initial radon concentration: {:?}", initial_radon);
    let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
    // build an identity matrix as Vec<Vec<f64>> (easier: use Array2 in the model, then Array2::eye)
    let mut init_inverse_hessian: Vec<Vec<f64>> = vec![vec![]];
    for ii in 0..init_param.values.len(){
        let mut tmp = vec![0.0;init_param.values.len()];
        tmp[ii] = 1.0;
        init_inverse_hessian.push(tmp);
    }
    let fwd = DetectorForwardModelBuilder::default().data(ts.clone()).time_step(time_step).radon(initial_radon.clone()).build().expect("Failed to build detector model");
    let cost = DetectorInverseModel{p, inv_opts, ts, fwd};

    // 2. Optimisation (MAP)   
    const ARGMIN_OPTION: u32 = 1;

    // Nelder Mead initial simplex
    // Strategy for initialisation: https://stackoverflow.com/questions/17928010/choosing-the-initial-simplex-in-the-nelder-mead-optimization-algorithm
    // Note: note working
    let mut simplex: Vec<_> = vec![init_param.values.clone()];
    let ndims = init_param.values.len();
    // step size
    let dx = 1e-4;
    for ii in 0..ndims{
        let mut v = init_param.values.clone();
        v[ii] += dx;
        simplex.push(v);
    }

    let map_max_iterations = 5000; //TODO: options
    let res = match ARGMIN_OPTION {
        1 =>  Executor::new(cost, TrustRegion::new(Steihaug::new()), init_param.values)
                .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
                .max_iters(map_max_iterations)
                .run().unwrap(),
        
        
        2 =>  Executor::new(cost, NelderMead::new().with_initial_params(simplex), init_param.values)
                .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
                .max_iters(map_max_iterations)
                .run().unwrap(),
        
        3 => Executor::new(cost, 
                            BFGS::new(init_inverse_hessian, MoreThuenteLineSearch::new()),
                           init_param.values)
                .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
                .max_iters(map_max_iterations)
                .run().unwrap(),

        _ => unimplemented!(),
    };
    
    println!("MAP optimisation complete: {}", res);
    let map = res.state.get_best_param();
    let map_as_guess = Guess{values:map};
    let (_,_, transformed_map_radon) = unpack_state_vector(&map_as_guess, &inv_opts);
    let mut map_radon = &mut transformed_map_radon.to_owned();
    inverse_transform_radon_concs(&mut map_radon).unwrap();

    // 3. Generate initial guess around the MAP point
    
    //convert result back from res to Guess
    let nwalkers = 100; // TODO, add to inv_opts
    let initial_positions = map_as_guess.create_initial_guess(nwalkers);

    // 4. Run the emcee sampler

    // 5. Wrangle the output and compute statistics

    println!("Initial radon concentration: {:?}", initial_radon);
    println!("MAP radon concentration:     {:?}", map_radon);

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputRecord;
    use crate::InputRecordVec;

    fn get_timeseries(npts: usize) -> InputRecordVec {
        let trec = InputRecord{
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 1000.0 + 30.0,
            background_count_rate: 1.0/60.0,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000./(3600.0/2.0),
            q_internal: 0.1/60.0,  //volumetric, m3/sec
            q_external: 80.0/60.0/1000.0,  //volumetric, m3/sec
            airt: 21.0, // degC
        };
        let mut ts = InputRecordVec::new();
        for _ in 0..npts{
            ts.push(trec);
        }

        //ts.counts[npts/2] *= 5.0;
        //ts.counts[npts/2+1] *= 5.0;
        
        let counts = vec![
            1333.0,
            3473.0,
            5385.0,
            4935.0,
            3833.0,
            2828.0,
            2060.0];
        
        assert!(npts > counts.len());
            for ii in 0..counts.len(){
            ts.counts[ii + npts - counts.len()] = counts[ii];
        }
        
        ts

    }

    #[test]
    fn can_compute_lnprob(){
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(50);
        let time_step = 60.0*30.0; //TODO
        // Define initial parameter vector and cost function
        let initial_radon = calc_radon_without_deconvolution(&ts, time_step);
        let mut constant_radon = initial_radon.clone();
        let rnavg = initial_radon.iter().sum::<f64>() / (initial_radon.len() as f64);
        for itm in constant_radon.iter_mut(){
            *itm = rnavg * 100.0;
        }
        println!("Initial radon concentration: {:?}", initial_radon);
        let init_param = pack_state_vector(&initial_radon, p.clone(), ts.clone(), inv_opts);
        let worse_guess = pack_state_vector(&constant_radon, p.clone(), ts.clone(), inv_opts);

        let fwd = DetectorForwardModelBuilder::default().data(ts.clone()).time_step(time_step).radon(initial_radon).build().expect("Failed to build detector model");
        let cost = DetectorInverseModel{p, inv_opts, ts, fwd};

        // println!("initial guess: {:#?}", init_param.values);
        println!("initial guess cost function evaluation: {}", cost.lnprob(&init_param));
        println!("Initial guess cost function gradient: {:?}", cost.gradient(&init_param.values).unwrap() );

        // println!("const radon guess: {:#?}", worse_guess.values);
        println!("const radon cost function evaluation: {}", cost.lnprob(&worse_guess));
        println!("Const radon cost function gradient: {:?}", cost.gradient(&worse_guess.values).unwrap() );


    }

    #[test]
    fn can_run_inverse_problem() {
        // TODO: set options for very small number of iterations
        let p = DetectorParamsBuilder::default().build().unwrap();
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        let ts = get_timeseries(11);
        fit_inverse_model(p, inv_opts, ts).expect("Failed to fit inverse model");
    }

    //#[test]
    fn handles_bad_input(){
        // Input is bad since counts < background count rate
        // causes expected counts to be less than zero
        // and blows up Poisson distribution
        let p = DetectorParamsBuilder::default().build().unwrap();
        let trec = InputRecord{
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0/60.0,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000./(3600.0/2.0),
            q_internal: 0.1/60.0,  //volumetric, m3/sec
            q_external: 80.0/60.0/1000.0,  //volumetric, m3/sec
            airt: 21.0, // degC
        };
        let mut ts = InputRecordVec::new();
        for _ in 0..10{
            ts.push(trec);
        }
        
        let inv_opts = InversionOptionsBuilder::default().build().unwrap();
        fit_inverse_model(p, inv_opts, ts);

    }
}