//
// This is a forward model of the radon detector i.e. given a radon timeseries
// it produces a timeseries of counts.
// 
// A key part of the model is an ODE solver.  Choice of method is discussed
// by https://scipy-lectures.org/advanced/mathematical_optimization/ (essentially, 
// this amounts to: use BGFS or BGFS-L, if you're happy to evaluate gradients, else
// use the Nelder-Mead or Powell methods).


use derive_builder::Builder;
//use ode_solvers::dop853::*;
use ode_solvers::dopri5::*;
use ode_solvers::*;

pub mod constants;
pub mod generated_functions;
pub mod quickplot;

use constants::*;
use generated_functions as gf;

use core::ops::{Add,Mul};

use super::{InputTimeSeries,InputRecordVec,InputRecord};

use anyhow::{Result};

pub enum Parameter {
    Constant(f64),
    TimeSeries(Vec<f64>),
}


#[derive(Debug, Clone, Builder)]
pub struct DetectorParams {
    /// External flow rate scale factor (default 1.0)
    /// The external flow rate is taken from the data file
    #[builder(default = "1.0")]
    pub exflow_scale: f64,   
    /// 1500 L in about 3 minutes in units of m3/s
    #[builder(default = "1.5/60.")]
    pub inflow: f64,
    /// Main radon delay volume (default 1.5 m3)
    #[builder(default = "1.5")]
    pub volume: f64,
    /// Net efficiency of detector (default of 0.2)
    #[builder(default = "0.2")]
    pub sensitivity: f64,
    /// Screen mesh capture probability (rs) (default of 0.95)
    #[builder(default = "0.95")]
    pub r_screen: f64,
    /// Scale factor for r_screen (default of 1.0)
    #[builder(default = "1.0")]
    pub r_screen_scale: f64,
    /// Overall delay time (lag) of detector (default 0.0 s)
    #[builder(default = "0.0")]
    pub delay_time: f64,
    /// Volume of external delay tank number 1 (default 0.2 m3)
    #[builder(default = "0.2")]
    pub volume_delay_1: f64,
    /// Volume of external delay tank number 2 (default 0.0 m3)
    #[builder(default = "0.0")]
    pub volume_delay_2: f64,
    /// Time constant for plateout onto detector walls (default 1/300 sec)
    #[builder(default = "1.0/300.0")]
    pub plateout_time_constant: f64,
}

impl DetectorParamsBuilder {
    /// default set of parameters for a 1500L detector
    pub fn default_1500l(&mut self) -> &mut Self {
        let mut new = self;
        new.volume = Some(1.5);
        new
    }
    /// default set of parameters for a 700L detector
    pub fn default_700l(&mut self) -> &mut Self {
        let mut new = self;
        new.volume = Some(0.7);
        new
    }
}

#[derive(Debug,Clone,Builder)]
pub struct DetectorForwardModel{
    #[builder(default = "DetectorParamsBuilder::default().build().unwrap()")]
    pub p: DetectorParams,
    pub data: InputTimeSeries,
    pub time_step: f64,
    pub radon: Vec<f64>,
    #[builder(default = "0.0")]
    pub inj_source_strength: f64,
    #[builder(default = "0.0")]
    pub inj_begin: f64,
    #[builder(default = "0.0")]
    pub inj_duration: f64,
    #[builder(default = "0.0")]
    pub cal_source_strength: f64,
    #[builder(default = "0.0")]
    pub cal_begin: f64,
    #[builder(default = "0.0")]
    pub cal_duration: f64,
}

/// interpolation utility functions
fn linear_interpolation<T>(ti: f64, y: &[T], tmax: f64) -> T
    where T: Add<Output=T> + Mul<f64,Output=T> + Copy
{    
    assert!(ti<=tmax);
    assert!(ti>=0.0);
    let time_step = tmax / (y.len() - 1) as f64;
    let p = ti/time_step;
    let idx0 = p.floor() as usize;
    let idx1 = p.ceil() as usize;
    let w1 = p - (idx0 as f64);
    let w0 = 1.0 - w1;
    y[idx0] * w0 + y[idx1] * w1
}

fn stepwise_interpolation<T>(ti: f64, y: &[T], tmax: f64) -> T
where T: Add<Output=T> + Mul<f64,Output=T> + Copy
{
    let time_step = tmax / (y.len() - 1) as f64;
    let p = ti/time_step;
    let idx1 = p.ceil() as usize;
    y[idx1]
}


/// state vector for detector
type State = SVector<f64, NUM_STATE_VARIABLES>;
//type State = DVector<f64>;

impl ode_solvers::System<State> for DetectorForwardModel {
    fn system(&self, t: f64, y: &State, dy: &mut State) {
        // TODO: enforce this earlier
        assert!(self.radon.len() == self.data.len());
        // determine values interpolated in time
        let tmax = (self.data.len() - 1) as f64 * self.time_step;
        let t_delay = self.p.delay_time;
        let ti = (t - t_delay).clamp(0.0, tmax);
        
        // interpolate inputs to current point in time
        let airt_l = linear_interpolation(ti, &self.data.airt, tmax);
        let airt_s = stepwise_interpolation(ti, &self.data.airt, tmax);

        // TODO: make radon switchable between linear and stepwise
        let radon = linear_interpolation(ti, &self.radon, tmax);
        // Extract interpolated values from linear or stepwise,
        // depending on the variable
        let q_external = stepwise_interpolation(ti, &&self.data.q_external, tmax);
        let q_internal = stepwise_interpolation(ti, &self.data.q_internal, tmax);
        let sensitivity = linear_interpolation(ti, &self.data.sensitivity, tmax);
        let background_count_rate = linear_interpolation(ti, &self.data.background_count_rate, tmax);
        // scale factors (used in inversion)
        let q_external = q_external * self.p.exflow_scale;
        let r_screen = self.p.r_screen * self.p.r_screen_scale;
        // ambient (or external) radon concentration in atoms per m3
        let n_rn_ext = radon / LAMRN;
        // limit the number of free parameters by calculating some from the others
        let (eff, recoil_prob) = calc_eff_and_recoil_prob(q_internal, r_screen, self.p.plateout_time_constant, q_external, self.p.volume_delay_1, self.p.volume_delay_2, self.p.volume, sensitivity);

        // unpack state vector
        let n_rn_d1 = y[IDX_NRND1];
        let n_rn_d2 = y[IDX_NRND2];
        let n_rn = y[IDX_NRN];
        let fa = y[IDX_FA];
        let fb = y[IDX_FB];
        let fc = y[IDX_FC];
        // The radon concentration flowing into the inlet needs to be
        // bumped up if the injection source is active
        let inj_is_active = self.inj_source_strength > 0.0
                                && (t - t_delay) > self.inj_begin
                                && (t - t_delay) <= self.inj_begin+self.inj_duration;
        let n_rn_inj = if inj_is_active
        {
            self.inj_source_strength / q_external
        }
        else
        {
            0.0
        };
        // The radon concentration flowing into the main tank needs to be
        // bumped up if the calibration source is active
        let cal_is_active = self.cal_source_strength > 0.0
                                 && (t - t_delay) > self.cal_begin
                                 && (t - t_delay) <= self.cal_begin+self.cal_duration;
        let n_rn_cal = if cal_is_active
        {
            self.cal_source_strength / q_external
        }
        else
        {
            0.0
        };
        // make sure that we can't have V_delay_2 > 0 when V_delay == 0
        let v_delay_1: f64;
        let v_delay_2: f64;
        if self.p.volume_delay_1 == 0.0 && self.p.volume_delay_2 > 0.0{
            v_delay_1 = self.p.volume_delay_2;
            v_delay_2 = 0.0;
        }
        else{
            v_delay_1 = self.p.volume_delay_1;
            v_delay_2 = self.p.volume_delay_2;
        }
        // effect of delay and tank volumes (allow V_delay to be zero)
        let d_nrn_dt: f64;
        let d_nrnd1_dt: f64;
        let d_nrnd2_dt: f64;
        if v_delay_1 == 0.0{
            // no delay tanks
            d_nrn_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                            - n_rn*LAMRN;
            // Nrnd,Nrnd2 become unimportant, but we need to do something with them
            // so just apply the same equation as for Nrn
            d_nrnd1_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                            - n_rn_d1*LAMRN;
            d_nrnd2_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                            - n_rn_d2*LAMRN;
        }
        else if v_delay_1 > 0.0 && v_delay_2 == 0.0{
            //one delay tank
            d_nrn_dt = q_external / self.p.volume * (n_rn_d1 + n_rn_cal - n_rn)
                            - n_rn*LAMRN;
            d_nrnd1_dt = q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1)
                            - n_rn_d1*LAMRN;
            //unused, but apply same eqn as delay tank 1
            d_nrnd2_dt = q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1)
                            - n_rn_d2*LAMRN;

        }
        else{
            //two delay tanks
            d_nrn_dt = q_external / self.p.volume * (n_rn_d1 + n_rn_cal - n_rn)
                            - n_rn*LAMRN;
            d_nrnd1_dt = q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1)
                            - n_rn_d1*LAMRN;
            d_nrnd2_dt = q_external / v_delay_2 * (n_rn_d1 - n_rn_d2)
                            - n_rn_d2*LAMRN;
        }

        // effect of temperature changes causing the tank to 'breathe'
        // d_nrn_dt -= n_rn_d * dTdt/T; TODO
        // Na, Nb, Nc from steady-state in tank
        // transit time assuming plug flow in the tank
        let tt = self.p.volume / q_internal;
        let (n_a, n_b) = gf::calc_na_nb_factors(q_internal, self.p.volume, self.p.plateout_time_constant);
        let n_c = 0.0;
        // compute rate of change of each state variable
        let d_fa_dt = q_internal*r_screen*n_a*n_rn*LAMRN - fa*LAMA;
        let d_fb_dt = q_internal*r_screen*n_b*n_rn*LAMRN - fb*LAMB + fa*LAMA * (1.0-recoil_prob);
        let d_fc_dt = q_internal*r_screen*n_c*n_rn*LAMRN - fc*LAMC + fb*LAMB; // TODO: why is there no recoil probability here?? i.e. * (1.0-recoil_prob);
        let d_acc_counts_dt = eff*(fa*LAMA + fc*LAMC);
        // pack into dy
        dy[IDX_NRND1] = d_nrnd1_dt;
        dy[IDX_NRND2] = d_nrnd2_dt;
        dy[IDX_NRN] = d_nrn_dt;
        dy[IDX_FA] = d_fa_dt;
        dy[IDX_FB] = d_fb_dt;
        dy[IDX_FC] = d_fc_dt;
        dy[IDX_ACC_COUNTS] = d_acc_counts_dt + background_count_rate;
    }
}


/// Calculate efficiency and recoil probability assuming that they are
/// linked (as described in paper) and with efficiency set by the net
/// detector efficiency
///
/// Arguments:
/// `Q` -- internal flow rate (m3/s)
/// `rs` -- screen efficiency (0-1)
/// `lamp` -- plateout constant (/s)
/// `Q_external` -- external flow rate (m3/s)
/// `V_delay` -- delay volume
/// `V_tank` -- tank volume (m3)
/// `total_efficiency` -- efficiency of the entire system
/// 
/// Note: some inputs are unused because the current implementation uses an
/// approximation, but the full solution needs them.
///
/// Returns:
///   alpha detection efficiency (eff), recoil probability
///
fn calc_eff_and_recoil_prob(
    q: f64,
    rs: f64,
    lamp: f64,
    q_external: f64,
    v_delay_1: f64,
    v_delay_2: f64,
    v_tank: f64,
    total_efficiency: f64,
) -> (f64, f64) {
    let recoil_prob = 0.5 * (1.0 - rs);
    let eff = 1.0;

    // account for radioactive decay in delay volumes (typical effect size: 0.3%)
    let radon0 = 1.0;
    let rn_d1 = radon0 / (LAMRN * v_delay_1 / q_external  + 1.0);
    let rn_d2 = rn_d1 / (LAMRN * v_delay_2 / q_external  + 1.0);
    let rn = rn_d2 / (LAMRN * v_tank / q_external  + 1.0);


    let ssc = gf::steady_state_count_rate(q, v_tank, eff, lamp, recoil_prob, rs) * rn/radon0;
    let eff = eff * total_efficiency / ssc;
    (eff, recoil_prob)
}

pub fn numerical_forward_model(p: &DetectorParams, ts: &InputTimeSeries) -> Vec<f64> {
    
    vec![1.0, 2.0, 3.0, 4.0]
}

pub fn analytical_forward_model(p: &DetectorParams) -> Vec<[f64; 2]> {
    vec![[1.1, 1.2], [2.1, 2.2]]
}


impl DetectorForwardModel{

    pub fn initial_state(&self, radon0: f64) -> State{
        // this is required for a DVector
        // let mut y = State::from_element(NUM_STATE_VARIABLES, 0.0);
        // this is required for a SVector
        let mut y = State::from_element(0.0);
        // Step through the delay volumes, accounting for radioactive decay in each,
        // by applying the relation 
        // N/N0 = 1 / (lambda * tt + 1)
        // where lambda is the radioactive decay constant and tt = V/Q is the transit time

        // Initial state has everything in equilibrium with first radon value
        // radon, atoms/m3
        let n_radon0 = radon0 / LAMRN;

        let q_external = self.data.q_external[0] * self.p.exflow_scale;
        let r_screen = self.p.r_screen * self.p.r_screen_scale;

        let n_rn_d1 = n_radon0 / (LAMRN * self.p.volume_delay_1 / q_external  + 1.0);
        let n_rn_d2 = n_rn_d1 / (LAMRN * self.p.volume_delay_2 / q_external  + 1.0);
        let n_rn = n_rn_d2 / (LAMRN * self.p.volume / q_external  + 1.0);
        let rn = n_rn * LAMRN;

        let (_eff, recoil_prob) = calc_eff_and_recoil_prob(self.data.q_internal[0], r_screen, self.p.plateout_time_constant, self.data.q_external[0], self.p.volume_delay_1, self.p.volume_delay_2, self.p.volume, self.data.sensitivity[0]);
        let (fa_1bq,fb_1bq,fc_1bq) = gf::num_filter_atoms_steady_state(self.data.q_internal[0], self.p.volume, self.p.plateout_time_constant, recoil_prob, r_screen);

        y[IDX_NRND1] = n_rn_d1;
        y[IDX_NRND2] = n_rn_d2;
        y[IDX_NRN] = n_rn;
        y[IDX_FA] = fa_1bq * rn;
        y[IDX_FB] = fb_1bq * rn;
        y[IDX_FC] = fc_1bq * rn;
        y[IDX_ACC_COUNTS] = 0.0;
        y
    }

    pub fn numerical_solution(self) -> Result<Vec<State>> {
        //let system = (*self).clone()
        let system = self;
        let t0 = 0.0;
        let tmax = system.time_step*(system.data.len() - 1) as f64;
        let dt = system.time_step;
        let y0 = system.initial_state(system.radon[0]);
        let rtol = 1e-3;
        let atol = 10.0;
    
        //let mut stepper = Dop853::new(system, t0, tmax, dt, y0, rtol, atol);
        let safety_factor = 0.9;
        let beta = 0.0;
        let fac_min = 0.333;
        let fac_max = 6.0;
        // keep the step size small because the boundary conditions may change on this time-scale
        let h_max = if dt<60.0 {dt} else {60.0};
        let h = 0.0;
        let n_max = 1_000_000;
        let n_stiff = 1_000;
        let out_type = dop_shared::OutputType::Dense;

        let mut stepper = Dopri5::from_param(system, t0, tmax, dt, y0, rtol, atol, safety_factor, beta,
             fac_min, fac_max, h_max, h, n_max, n_stiff, out_type);
        let stats = stepper.integrate()?;
        //println!("Integration stats: {}", stats);

        Ok(stepper.y_out().clone())
    }

    pub fn numerical_expected_counts(self) -> Result<Vec<f64>>{
        let y_out = self.numerical_solution()?;
        let ac1 = y_out.iter().map(|itm| {itm[IDX_ACC_COUNTS]});
        let mut ac2 = ac1.clone();
        let _ = ac2.next();
        let expected_counts = ac1.zip(ac2).map(|(itm1,itm2)| {itm2-itm1}).collect();
        Ok(expected_counts)
    }

    pub fn analytical_solution(&self) -> (){

    }
    pub fn radon(&mut self, radon: &[f64]) -> (){

    }
}





#[cfg(test)]
mod tests {
    use debug_plotter::plot;

    use super::*;
    use quickplot::draw_plot;

    fn get_timeseries(npts: usize) -> InputRecordVec {
        let trec = InputRecord{
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 1000.0,
            background_count_rate: 0.0, //1.0/60.0,
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
        ts.counts[npts/2] *= 5.0;
        ts.counts[npts/2+1] *= 5.0;

        ts

    }


    #[test]
    fn can_create_params() {
        let p = DetectorParamsBuilder::default()
            .delay_time(1.0)
            .build()
            .expect("Detector creation failed");
        // test a default value
        assert!(p.inflow == 1.5 / 60.);
        // test a set value
        assert!(p.delay_time == 1.0);
        // test default 700L detector
        let p = DetectorParamsBuilder::default()
            .default_700l()
            .build()
            .unwrap();
        assert!(p.volume == 0.7);
    }

    #[test]
    fn can_modify_params() {
        let mut p = DetectorParamsBuilder::default().build().unwrap();
        p.volume = 10.0;
        assert!(p.volume == 10.0);
    }

    #[test]
    fn can_integrate(){
        let p = DetectorParamsBuilder::default().build().unwrap();
        let radon = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        //let radon = vec![0.0; 11];
        let mut data = get_timeseries(11);
        let fwd = DetectorForwardModel{
            p: p,
            data: data,
            time_step: 60.0*30.0,
            radon: radon,
            inj_source_strength: 0.0,
            inj_begin: 0.0,
            inj_duration: 0.0,
            cal_source_strength: 0.0,
            cal_begin: 0.0,
            cal_duration: 0.0,
        };

        let nsoln = fwd.clone().numerical_solution().expect("integration failed");
        // diff the counts
        let mut ac1 = nsoln.iter();
        let mut ac2 = ac1.clone();
        let mut ac3 = ac1.clone();
        let _ = ac2.next();
        let nsoln_diff: Vec<_> = ac1.zip(ac2).map(|(itm1,itm2)| {itm2-itm1}).collect();
        println!("{:#?}", nsoln_diff);

        // convert from atoms to Bq
        let nsoln_conv: Vec<_> = ac3.map( |itm| {itm * LAMRN}).collect();
        println!("{:#?}", nsoln_conv);

        let nec = fwd.numerical_expected_counts().unwrap();
        println!("Numerical solution, expected counts: {:#?}", nec);
    }

    #[test]
    fn can_integrate_using_builder(){
        let mut radon = vec![];
        let trec = InputRecord{
            time: 0.0,
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 0.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000./(3600.0/2.0),
            q_internal: 0.1/60.0,  //volumetric, m3/sec
            q_external: 80.0/60.0/1000.0,  //volumetric, m3/sec
            airt: 21.0, // degC
        };
    
        let mut data = InputTimeSeries::new();
        for ii in 0..300{
            data.push(trec);
            radon.push( if ii==0 || ii>60 {0.0} else {1.0});
        }
        let time_step = 1.0*60.0;
        for (ii,itm) in data.iter_mut().enumerate(){
            *itm.time = (ii as f64) * time_step;
        }
        let fwd = DetectorForwardModelBuilder::default().data(data).radon(radon).time_step(time_step).build().unwrap();
        let num_counts = fwd.numerical_expected_counts().unwrap();
        //println!("{:#?}", num_counts);
        
        draw_plot(&num_counts[..], "test.svg").unwrap();

        for (x,y) in num_counts.iter().take(10).enumerate(){
            //println!("{} {}", x,y);
            //plot!(y where caption="response");
            debug_plotter::plot!(y);
        }
        
    }
}
