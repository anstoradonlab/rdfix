//
// This is a forward model of the radon detector i.e. given a radon timeseries
// it produces a timeseries of counts.
//
// A key part of the model is an ODE solver.  Choice of method is discussed
// by https://scipy-lectures.org/advanced/mathematical_optimization/ (essentially,
// this amounts to: use BGFS or BGFS-L, if you're happy to evaluate gradients, else
// use the Nelder-Mead or Powell methods).

/* Note: approach for generic numbers which can support autodiff

1. Use a trait bound num_traits::Float (or, equivalently, num::Float) on generic type "F"
```
use num_traits::Float

fn testfunc<F>(x:F) -> F
where F: Float
{
    x + x
}
```
2. to convert from literals, use
```
F::from(1.0).unwrap()
```
3. to convert into other types, e.g. for indexing an array or whatever, use
```
let idx = x.to_usize().unwrap();
```

*/

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
//use ode_solvers::dop853::*;
//use ode_solvers::*;
pub mod constants;
pub mod stepper;
use self::stepper::integrate;

use super::InputTimeSeries;
use crate::TimeExtents;
use anyhow::Result;
use constants::*;
use rdfix_gf::generated_functions as gf;
use std::cell::Cell;

pub enum Parameter {
    Constant(f64),
    TimeSeries(Vec<f64>),
}

/// Describe how the radon timeseries should be transformed
/// before entering the MCMC samplers
#[derive(Copy, Debug, Clone)]
pub enum TransformationKind {
    /// No transformation applied to radon timeseries
    None,
    /// Ensures that radon concentration is always positive
    /// by mapping the raw parameter from `(-inf,+inf)` to
    /// (0,inf)
    EnforcePositive,
    /// Apply a tranformation which weakens the correlation
    /// between consecutive points in the radon timeseries
    /// (details described in the paper).  This option also
    /// ensures that radon concentrations must always be > 0
    WeakCorrelation,
}

#[derive(Debug, Clone, Builder, Serialize, Deserialize)]
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
    //    /// Net efficiency of detector (default of 0.2)
    //    #[builder(default = "0.2")]
    //    pub sensitivity: f64,
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
        let new = self;
        new.volume = Some(1.5);
        new
    }
    /// default set of parameters for a 700L detector
    pub fn default_700l(&mut self) -> &mut Self {
        let new = self;
        new.volume = Some(0.7);
        new
    }
}

#[derive(Debug, Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct DetectorForwardModel {
    #[builder(default = "DetectorParamsBuilder::default().build().unwrap()")]
    pub p: DetectorParams,
    pub data: InputTimeSeries,
    #[builder(default = "self.data.as_ref().unwrap().airt.clone()")]
    pub airt_points: Vec<f64>,
    #[builder(default = "self.data.as_ref().unwrap().q_external.clone()")]
    pub q_external_points: Vec<f64>,
    #[builder(default = "self.data.as_ref().unwrap().q_internal.clone()")]
    pub q_internal_points: Vec<f64>,
    #[builder(default = "self.data.as_ref().unwrap().sensitivity.clone()")]
    pub sensitivity_points: Vec<f64>,
    #[builder(default = "self.data.as_ref().unwrap().background_count_rate.clone()")]
    pub background_count_rate_points: Vec<f64>,

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
    #[builder(default = "60")]
    pub integration_substeps: usize,

    #[builder(
        default = "Interpolator::new(self.data.as_ref().unwrap().len(), self.time_step.unwrap())"
    )]
    pub interp: Interpolator,
}

impl DetectorForwardModelBuilder {
    fn validate(&self) -> Result<(), String> {
        let l1 = match self.radon {
            Some(ref x) => x.len(),
            None => 0,
        };
        let l2 = match self.data {
            Some(ref x) => x.len(),
            None => 0,
        };

        if l1 == l2 {
            Ok(())
        } else {
            Err("radon len needs to equal data len".to_string())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Interpolator {
    pub time_step: f64,
    pub npts: usize,
    pub p: f64,
    pub tmax: f64,
    pub idx0: usize,
    pub idx1: usize,
    pub w0: f64,
    pub w1: f64,
}

trait USizeConv {
    fn to_usize(&self) -> Result<usize>;
}

impl USizeConv for f64 {
    fn to_usize(&self) -> Result<usize> {
        Ok(*self as usize)
    }
}

impl Interpolator {
    /// Create new interpolator with `npts` points on the domain [0.0, tmax]
    fn new(npts: usize, time_step: f64) -> Self {
        let tmax = time_step * f64::from(npts as u16);
        let p = 0.0_f64;
        let idx0 = p.floor() as usize;
        let idx1 = p.ceil() as usize;
        let w1 = p - f64::from(idx0 as u16);
        let w0 = 1.0 - w1;
        Interpolator {
            time_step,
            npts,
            p,
            tmax,
            idx0,
            idx1,
            w0,
            w1,
        }
    }
    /// Set the interpolation time to ti
    fn ti(&mut self, ti: f64) {
        if ti <= 0.0 {
            self.w0 = 1.0;
            self.w1 = 0.0;
            self.idx0 = 0;
            self.idx1 = 0;
        } else if ti >= self.tmax {
            self.w0 = 0.0;
            self.w1 = 1.0;
            self.idx0 = self.npts - 1;
            self.idx1 = self.npts - 1;
        } else {
            self.p = ti / self.time_step;
            self.idx0 = self.p.floor().to_usize().unwrap();
            self.idx1 = self.p.ceil().to_usize().unwrap();
            self.w1 = self.p - f64::from(self.idx0 as u16);
            self.w0 = 1.0 - self.w1;
        }
    }
    /// Linear interpolation of y at ti, values outside of [0.0, tmax] are taken from endpoints
    fn linear(&self, y: &[f64]) -> f64 {
        y[self.idx0] * self.w0 + y[self.idx1] * self.w1
    }
    /// Stepwise interpolation of y at ti, values outside of [0.0, tmax] are taken from endpoints
    fn stepwise(self, y: &[f64]) -> f64 {
        y[self.idx1]
    }
}

/*
/// state vector for detector
type FP = f64;
type State = SVector<FP, NUM_STATE_VARIABLES>;
//type State = DVector<f64>;


instead of defining State, use [P; NUM_STATE_VARIABLES]

*/

impl DetectorForwardModel {
    #[inline(always)]
    fn rate_of_change(
        &self,
        t: f64,
        y: &[f64; NUM_STATE_VARIABLES],
        dy: &mut [f64; NUM_STATE_VARIABLES],
    ) {
        self.system(t, y, dy)
    }
    //}
    //
    //impl<P> ode_solvers::System<[P; NUM_STATE_VARIABLES]> for DetectorForwardModel<P>
    //where
    //    P: Float + std::fmt::Debug,
    //
    //
    //{
    #[inline(always)]
    fn system(&self, t: f64, y: &[f64; NUM_STATE_VARIABLES], dy: &mut [f64; NUM_STATE_VARIABLES]) {
        // TODO: enforce this earlier
        assert!(self.radon.len() == self.data.len());
        // determine values interpolated in time
        let tmax = f64::from(self.data.len() as u16 - 1) * self.time_step;
        let t_delay = self.p.delay_time;
        let mut ti = t - t_delay;
        if ti < 0.0 {
            ti = 0.0
        };
        if ti > tmax {
            ti = tmax
        };

        //// Just set the gradient equal to the y value itself (attempt to get this to compile)
        //for ii in 0..dy.len(){
        //    dy[ii] = y[ii];
        //}

        let p_lamrn = LAMRN;
        let p_lama = LAMA;
        let p_lamb = LAMB;
        let p_lamc = LAMC;

        // interpolate inputs to current point in time
        let mut interp = self.interp;
        interp.ti(ti);

        //let airt_points = &self.airt_points;

        // TODO: make radon switchable between linear and stepwise
        //let radon = linear_interpolation(ti, &self.radon, tmax);
        let radon = interp.linear(&self.radon);

        // Extract interpolated values from linear or stepwise,
        // depending on the variable
        let q_external_points = &self.q_external_points;
        let q_internal_points = &self.q_internal_points;
        let sensitivity_points = &self.sensitivity_points;
        let background_count_rate_points = &self.background_count_rate_points;
        //let q_external = stepwise_interpolation(ti, &q_external_points, tmax);
        let q_external = interp.stepwise(q_external_points);
        //let q_internal = stepwise_interpolation(ti, &q_internal_points, tmax);
        let q_internal = interp.stepwise(q_internal_points);
        //let sensitivity = linear_interpolation(ti, &sensitivity_points, tmax);
        let sensitivity = interp.linear(sensitivity_points);
        //let background_count_rate = linear_interpolation(ti, &background_count_rate_points, tmax);
        let background_count_rate = interp.linear(background_count_rate_points);
        // scale factors (used in inversion)
        assert!(self.p.exflow_scale >= 0.0);
        assert!(self.p.r_screen_scale >= 0.0);
        let mut q_external = q_external * self.p.exflow_scale;
        // The smallest plausible value for q_external is with the 100L detector, where operating
        // flow rates can be typically (one volume in 45 minutes) ~2 l/min.  If the external flow
        // rate is less than this, say a threshold of 0.5 l/min, then we're in background mode.
        // The model is more-or-less invalid during backgrounds, but it would be better if it does
        // not blow up - so we'll set a threshold on q_external.
        let q_e_threshold = 0.5 / 1000.0 / 1800.0;
        q_external = q_external.clamp(q_e_threshold, f64::MAX);
        let r_screen = self.p.r_screen * self.p.r_screen_scale;
        // ambient (or external) radon concentration in atoms per m3
        let n_rn_ext = radon / p_lamrn;
        // limit the number of free parameters by calculating some from the others
        let (eff, recoil_prob) = calc_eff_and_recoil_prob(
            q_internal,
            r_screen,
            self.p.plateout_time_constant,
            q_external,
            self.p.volume_delay_1,
            self.p.volume_delay_2,
            self.p.volume,
            sensitivity,
        );

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
            && (ti) > self.inj_begin
            && (ti) <= self.inj_begin + self.inj_duration;
        let n_rn_inj = if inj_is_active {
            self.inj_source_strength / q_external
        } else {
            0.0
        };
        // The radon concentration flowing into the main tank needs to be
        // bumped up if the calibration source is active
        let cal_is_active = self.cal_source_strength > 0.0
            && (ti) > self.cal_begin
            && (ti) <= self.cal_begin + self.cal_duration;
        let n_rn_cal = if cal_is_active {
            self.cal_source_strength / q_external
        } else {
            0.0
        };
        // make sure that we can't have V_delay_2 > 0 when V_delay == 0
        let v_delay_1: f64;
        let v_delay_2: f64;
        if self.p.volume_delay_1 == 0.0 && self.p.volume_delay_2 > 0.0 {
            v_delay_1 = self.p.volume_delay_2;
            v_delay_2 = 0.0;
        } else {
            v_delay_1 = self.p.volume_delay_1;
            v_delay_2 = self.p.volume_delay_2;
        }
        // effect of delay and tank volumes (allow V_delay to be zero)
        let d_nrn_dt: f64;
        let d_nrnd1_dt: f64;
        let d_nrnd2_dt: f64;
        if v_delay_1 == 0.0 {
            // no delay tanks
            d_nrn_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn * p_lamrn;
            // Nrnd,Nrnd2 become unimportant, but we need to do something with them
            // so just apply the same equation as for Nrn
            d_nrnd1_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn_d1 * p_lamrn;
            d_nrnd2_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn_d2 * p_lamrn;
        } else if v_delay_1 > 0.0 && v_delay_2 == 0.0 {
            //one delay tank
            d_nrn_dt = q_external / self.p.volume * (n_rn_d1 + n_rn_cal - n_rn) - n_rn * p_lamrn;
            d_nrnd1_dt =
                q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1) - n_rn_d1 * p_lamrn;
            //unused, but apply same eqn as delay tank 1
            d_nrnd2_dt =
                q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1) - n_rn_d2 * p_lamrn;
        } else {
            //two delay tanks
            d_nrn_dt = q_external / self.p.volume * (n_rn_d1 + n_rn_cal - n_rn) - n_rn * p_lamrn;
            d_nrnd1_dt =
                q_external / v_delay_1 * (n_rn_ext + n_rn_inj - n_rn_d1) - n_rn_d1 * p_lamrn;
            d_nrnd2_dt = q_external / v_delay_2 * (n_rn_d1 - n_rn_d2) - n_rn_d2 * p_lamrn;
        }

        // effect of temperature changes causing the tank to 'breathe'
        // d_nrn_dt -= n_rn_d * dTdt/T; TODO
        // Na, Nb, Nc from steady-state in tank
        // transit time assuming plug flow in the tank
        //let tt = self.p.volume / q_internal;

        // This call to calc_na_nb factors takes up a large fraction of the run time
        // Thread-local storage is used to memorise the function result, avoiding the
        // need to re-evaluate the function when it is called with repeated values
        thread_local! {
            static ARG: Cell<[f64;3]> = const { Cell::new([f64::NAN; 3]) };
            static VAL: Cell<[f64;2]> = const { Cell::new([f64::NAN; 2]) };
        };
        let args = [q_internal, self.p.volume, self.p.plateout_time_constant];

        let [n_a, n_b] = if ARG.get() == args {
            // Args are the same as previous call
            VAL.get()
        } else {
            // Need to run the function because arguments have changed
            let val =
                gf::calc_na_nb_factors(q_internal, self.p.volume, self.p.plateout_time_constant)
                    .into();
            ARG.set(args);
            VAL.set(val);
            val
        };

        //let (n_a, n_b) =
        //    gf::calc_na_nb_factors(q_internal, self.p.volume, self.p.plateout_time_constant);
        let n_c = 0.0;
        // compute rate of change of each state variable
        let d_fa_dt = q_internal * r_screen * n_a * n_rn * p_lamrn - fa * p_lama;
        let d_fb_dt = q_internal * r_screen * n_b * n_rn * p_lamrn - fb * p_lamb
            + fa * p_lama * (1.0 - recoil_prob);
        let d_fc_dt = q_internal * r_screen * n_c * n_rn * p_lamrn - fc * p_lamc + fb * p_lamb; // TODO: why is there no recoil probability here?? i.e. * (1.0-recoil_prob);
        let d_acc_counts_dt = eff * (fa * p_lama + fc * p_lamc);
        // pack into dy
        dy[IDX_NRND1] = d_nrnd1_dt;
        dy[IDX_NRND2] = d_nrnd2_dt;
        dy[IDX_NRN] = d_nrn_dt;
        dy[IDX_FA] = d_fa_dt;
        dy[IDX_FB] = d_fb_dt;
        dy[IDX_FC] = d_fc_dt;
        dy[IDX_ACC_COUNTS] = d_acc_counts_dt + background_count_rate;

        if dy.iter().any(|x| !x.is_finite()) {
            panic!("{} change rate calculation failed, dy={:?}, qe={:?}, Vd1={:}, Vd2={:}, n_rn_ext={:}, n_rn_inj={:}, n_rn_d1={:}, n_rn_d2={:}, fa={}, fb={}, fc={}", 
            self.data.chunk_id(),
            &dy, &q_external, v_delay_1, v_delay_2,
            n_rn_ext,
            n_rn_inj,
            n_rn_d1,
            n_rn_d2,
            fa,
            fb,
            fc );
        }
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
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
    let lamrn = LAMRN;

    // Note: do not account for radioactive decay in delay volumes because of what happens when
    // q_external -> 0.0.  Instead, we're saying that the efficiency is determined relative to
    // the radon concentration inside the delay volume.
    // TODO: q_external here should be a "reference" flow rate

    // account for radioactive decay in delay volumes (typical effect size: 0.3%)
    let radon0 = 1.0;
    let rn_d1 = radon0 / (lamrn * v_delay_1 / q_external + 1.0);
    let rn_d2 = rn_d1 / (lamrn * v_delay_2 / q_external + 1.0);
    let rn = rn_d2 / (lamrn * v_tank / q_external + 1.0);
    //let rn = radon0; // TODO: come up with a better approach

    // This call to steady_state_count_rate takes about 20-30% of the total time spent evaluating the
    // objective function in calculations of the inverse model.  This is an approach to
    // cache the result so that repeated evaluations are fast.
    thread_local! {
        static ARG: Cell<[f64;6]> = const { Cell::new([0.0; 6]) };
        static VAL: Cell<f64> = const { Cell::new(f64::NAN) };
    };
    let args = [q, v_tank, eff, lamp, recoil_prob, rs];
    let ssc;
    let ssc = if ARG.get() == args {
        // Args are the same as previous call
        VAL.get()
    } else {
        // Need to run the function because arguments have changed
        ssc = gf::steady_state_count_rate(q, v_tank, eff, lamp, recoil_prob, rs);
        ARG.set(args);
        VAL.set(ssc);
        ssc
    };

    // let ssc = gf::steady_state_count_rate(q, v_tank, eff, lamp, recoil_prob, rs);
    let corrected_ssc = ssc * rn / radon0;
    let eff = eff * total_efficiency / corrected_ssc;
    (eff, recoil_prob)
}

//TODO: remove this once API is decided upon
//pub fn numerical_forward_model(p: &DetectorParams, ts: &InputTimeSeries) -> Vec<f64> {
//
//    vec![1.0, 2.0, 3.0, 4.0]
//}

//pub fn analytical_forward_model(p: &DetectorParams) -> Vec<[f64; 2]> {
//    vec![[1.1, 1.2], [2.1, 2.2]]
//}

impl DetectorForwardModel {
    /// Calculate the initial state, in state vector form
    ///
    /// # Arguments
    ///
    /// * `radon0` - Radon concentration at t=0 in units of Bq/m3
    pub fn initial_state(&self, radon0: f64) -> [f64; NUM_STATE_VARIABLES] {
        // this is required for a DVector
        // let mut y = State::from_element(NUM_STATE_VARIABLES, 0.0);
        // this is required for a SVector
        let mut y = [0.0; NUM_STATE_VARIABLES];
        // Step through the delay volumes, accounting for radioactive decay in each,
        // by applying the relation
        // N/N0 = 1 / (lambda * tt + 1)
        // where lambda is the radioactive decay constant and tt = V/Q is the transit time

        // Special case when radon concentration is zero
        if radon0 <= 0.0 {
            y[IDX_ACC_COUNTS] = (self.data.background_count_rate[0]) * self.time_step;
            return y;
        }

        // Generic versions of constants
        let lamrn_p = LAMRN;

        // Initial state has everything in equilibrium with first radon value
        // radon, atoms/m3
        let n_radon0 = radon0 / lamrn_p;

        // if q_external or q_internal are zero (e.g. this happens during a calibration)
        // then this calculation will fail so force them to small values
        // 1 cc/hour converted to m3/sec
        let q_fill_value = (1.0 / 1000.0) / 3600.0;

        let q_external = (self.data.q_external[0] * self.p.exflow_scale)
            .clamp(q_fill_value, f64::MAX);
        let r_screen = self.p.r_screen * self.p.r_screen_scale;

        // Note: this is Ok if volume_delay = 0
        let n_rn_d1 = n_radon0 / (lamrn_p * self.p.volume_delay_1 / q_external + 1.0);
        let n_rn_d2 = n_rn_d1 / (lamrn_p * self.p.volume_delay_2 / q_external + 1.0);
        let n_rn = n_rn_d2 / (lamrn_p * self.p.volume / q_external + 1.0);
        let rn = n_rn * lamrn_p;

        // Again, clamp flow rates, but for the purpose of calculating eff and recoil probability
        // don't include exflow_scale factor
        let qi_i = self.data.q_internal[0];
        let qi_i = qi_i.clamp(q_fill_value, f64::MAX);
        let qe_i = self.data.q_external[0];
        let qe_i = qe_i.clamp(q_fill_value, f64::MAX);

        let (_eff, recoil_prob) = calc_eff_and_recoil_prob(
            qi_i,
            r_screen,
            self.p.plateout_time_constant,
            qe_i,
            self.p.volume_delay_1,
            self.p.volume_delay_2,
            self.p.volume,
            self.data.sensitivity[0],
        );
        let (fa_1bq, fb_1bq, fc_1bq) = gf::num_filter_atoms_steady_state(
            qi_i,
            self.p.volume,
            self.p.plateout_time_constant,
            recoil_prob,
            r_screen,
        );

        let ssc = self.data.sensitivity[0] * radon0;

        // initial "accumulated counts" are equal to the steady state count rate
        // over the time interval (usually 30-minutes).  We need to add the background
        // counts in here because gf::steady_state_count_rate doesn't include BG
        let acc_counts_0 = (ssc + self.data.background_count_rate[0]) * self.time_step;

        y[IDX_NRND1] = n_rn_d1;
        y[IDX_NRND2] = n_rn_d2;
        y[IDX_NRN] = n_rn;
        y[IDX_FA] = fa_1bq * rn;
        y[IDX_FB] = fb_1bq * rn;
        y[IDX_FC] = fc_1bq * rn;
        y[IDX_ACC_COUNTS] = acc_counts_0;

        if !(y.iter().all(|x| x.is_finite())) {
            panic!(
                "{} radon0: {:?}initial condition: {:#?}",
                self.data.chunk_id(),
                radon0,
                y
            );
        };

        y
    }

    pub fn numerical_solution(&self) -> Result<Vec<[f64; NUM_STATE_VARIABLES]>> {
        //let system = (*self).clone()
        let system = self;
        let t0 = 0.0;
        let num_intervals = system.data.len() - 1;
        let _tmax = system.time_step * f64::from(num_intervals as u16);
        let dt = system.time_step;
        let mut state = system.initial_state(system.radon[0]);

        // number of small RK4 steps to take per dt
        let num_steps = 30_usize;
        let mut t = t0;
        let mut expected_counts = Vec::with_capacity(num_intervals + 1);
        let mut y_out = Vec::with_capacity(num_intervals + 1);
        // for the first output point, report the initial state
        y_out.push(state);
        for _ in 0..num_intervals {
            state[IDX_ACC_COUNTS] = 0.0;
            integrate(&mut state, self, t, t + dt, num_steps);
            // TODO: maybe just return the expected counts??
            expected_counts.push(state[IDX_ACC_COUNTS]);
            y_out.push(state);
            t += dt;
        }

        Ok(y_out)
    }

    #[inline(always)]
    pub fn numerical_expected_counts(self) -> Result<Vec<f64>> {
        let y_out = self.numerical_solution()?;
        let expected_counts = y_out.iter().map(|itm| itm[IDX_ACC_COUNTS]).collect();
        Ok(expected_counts)
    }

    pub fn analytical_solution(&self) {}
    pub fn radon(&mut self, _radon: &[f64]) {}
}

#[cfg(test)]
mod tests {
    use crate::InputRecord;

    use super::*;
    use assert_approx_eq::assert_approx_eq;

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

    /*
    #[test]
    fn can_integrate() {
        let p = DetectorParamsBuilder::default().build().unwrap();
        let radon = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        //let radon = vec![0.0; 11];
        let mut data = get_timeseries(11);
        let fwd = DetectorForwardModel {
            p: p,
            data: data,
            time_step: 60.0 * 30.0,
            radon: radon,
            inj_source_strength: 0.0,
            inj_begin: 0.0,
            inj_duration: 0.0,
            cal_source_strength: 0.0,
            cal_begin: 0.0,
            cal_duration: 0.0,
        };

        let nsoln = fwd
            .clone()
            .numerical_solution()
            .expect("integration failed");
        // diff the counts
        let mut ac1 = nsoln.iter();
        let mut ac2 = ac1.clone();
        let mut ac3 = ac1.clone();
        let _ = ac2.next();
        let nsoln_diff: Vec<_> = ac1.zip(ac2).map(|(itm1, itm2)| itm2 - itm1).collect();
        println!("{:#?}", nsoln_diff);

        // convert from atoms to Bq
        let nsoln_conv: Vec<_> = ac3.map(|itm| itm * LAMRN).collect();
        println!("{:#?}", nsoln_conv);

        let nec = fwd.numerical_expected_counts().unwrap();
        println!("Numerical solution, expected counts: {:#?}", nec);
    }
    */

    /// If the ambient radon concentration is zero, the count rate
    /// should be equal to the background count rate
    #[test]
    fn count_rate_equals_background_with_zero_radon() {
        let trec = InputRecord {
            time: 0.0,
            // LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
            radon_truth: f64::NAN,
        };
        const N: usize = 10;
        const DT: f64 = 30.0 * 60.0;
        let mut data = InputTimeSeries::from_iter(vec![trec; N]);
        let radon = vec![0.0; N];

        for (ii, itm) in data.iter_mut().enumerate() {
            *itm.time = (ii as f64) * DT;
        }

        let expected_counts_per_30min = trec.background_count_rate * DT;

        let fwd = DetectorForwardModelBuilder::default()
            .data(data)
            .radon(radon)
            .time_step(DT)
            .build()
            .unwrap();
        let num_counts = fwd.numerical_expected_counts().unwrap();
        dbg!(expected_counts_per_30min);
        println!("{:#?}", num_counts);

        for nc in num_counts {
            assert_approx_eq!(nc, expected_counts_per_30min);
        }
    }

    /// If the ambient radon concentration is constant, the count rate
    /// should be constant and easy to calculate from the senstivity
    #[test]
    fn count_rate_at_constant_radon() {
        let trec = InputRecord {
            time: 0.0,
            // LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
            radon_truth: f64::NAN,
        };
        const N: usize = 4;
        const DT: f64 = 30.0 * 60.0;
        let mut data = InputTimeSeries::from_iter(vec![trec; N]);
        let radon = vec![1.0; N];

        for (ii, itm) in data.iter_mut().enumerate() {
            *itm.time = (ii as f64) * DT;
        }

        let expected_counts_per_30min = trec.background_count_rate * DT + 1000.0;

        let fwd = DetectorForwardModelBuilder::default()
            .data(data)
            .radon(radon)
            .time_step(DT)
            .build()
            .unwrap();
        let num_counts = fwd.numerical_expected_counts().unwrap();
        dbg!(expected_counts_per_30min);
        println!("{:#?}", num_counts);

        for nc in num_counts {
            assert_approx_eq!(nc, expected_counts_per_30min);
        }
    }

    #[test]
    fn can_integrate_using_builder() {
        let mut radon = vec![];
        let trec = InputRecord {
            time: 0.0,
            // LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 0.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
            radon_truth: f64::NAN,
        };

        let mut data = InputTimeSeries::new();
        for ii in 0..600 {
            data.push(trec);
            radon.push(if ii == 0 || ii > 60 { 0.0 } else { 1.0 });
        }
        let time_step = 1.0 * 60.0 * 30.0;
        for (ii, itm) in data.iter_mut().enumerate() {
            *itm.time = (ii as f64) * time_step;
        }
        let fwd = DetectorForwardModelBuilder::default()
            .data(data.clone())
            .radon(radon.clone())
            .time_step(time_step)
            .build()
            .unwrap();
        let num_counts = fwd.numerical_expected_counts().unwrap();
        assert!(num_counts.len() == data.len());
        assert!(num_counts.len() == radon.len());
        dbg!(&num_counts);
    }
}
