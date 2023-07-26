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
use anyhow::Result;
use constants::*;
use num::{Float, ToPrimitive};
use rdfix_gf::generated_functions as gf;

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
pub struct DetectorParams<P>
where
    //P : potentially differentiably (e.g. autodiff's type) or standard float
    P: Float + std::fmt::Debug,
{
    /// External flow rate scale factor (default 1.0)
    /// The external flow rate is taken from the data file
    #[builder(default = "P::from(1.0).unwrap()")]
    pub exflow_scale: P,
    /// 1500 L in about 3 minutes in units of m3/s
    #[builder(default = "P::from(1.5/60.).unwrap()")]
    pub inflow: P,
    /// Main radon delay volume (default 1.5 m3)
    #[builder(default = "P::from(1.5).unwrap()")]
    pub volume: P,
    /// Net efficiency of detector (default of 0.2)
    #[builder(default = "P::from(0.2).unwrap()")]
    pub sensitivity: P,
    /// Screen mesh capture probability (rs) (default of 0.95)
    #[builder(default = "P::from(0.95).unwrap()")]
    pub r_screen: P,
    /// Scale factor for r_screen (default of 1.0)
    #[builder(default = "P::from(1.0).unwrap()")]
    pub r_screen_scale: P,
    /// Overall delay time (lag) of detector (default 0.0 s)
    #[builder(default = "P::from(0.0).unwrap()")]
    pub delay_time: P,
    /// Volume of external delay tank number 1 (default 0.2 m3)
    #[builder(default = "P::from(0.2).unwrap()")]
    pub volume_delay_1: P,
    /// Volume of external delay tank number 2 (default 0.0 m3)
    #[builder(default = "P::from(0.0).unwrap()")]
    pub volume_delay_2: P,
    /// Time constant for plateout onto detector walls (default 1/300 sec)
    #[builder(default = "P::from(1.0/300.0).unwrap()")]
    pub plateout_time_constant: P,
}

impl<P> DetectorParams<P>
where
    P: Float + std::fmt::Debug,
{
    pub fn into_inner_type<NP>(&self) -> DetectorParams<NP>
    where
        NP: Float + std::fmt::Debug,
    {
        DetectorParams {
            exflow_scale: NP::from(self.exflow_scale).unwrap(),
            inflow: NP::from(self.inflow).unwrap(),
            volume: NP::from(self.volume).unwrap(),
            sensitivity: NP::from(self.sensitivity).unwrap(),
            r_screen: NP::from(self.r_screen).unwrap(),
            r_screen_scale: NP::from(self.r_screen_scale).unwrap(),
            delay_time: NP::from(self.delay_time).unwrap(),
            volume_delay_1: NP::from(self.volume_delay_1).unwrap(),
            volume_delay_2: NP::from(self.volume_delay_2).unwrap(),
            plateout_time_constant: NP::from(self.plateout_time_constant).unwrap(),
        }
    }
}

impl<P> DetectorParamsBuilder<P>
where
    P: Float + std::fmt::Debug,
{
    /// default set of parameters for a 1500L detector
    pub fn default_1500l(&mut self) -> &mut Self {
        let new = self;
        new.volume = Some(P::from(1.5).unwrap());
        new
    }
    /// default set of parameters for a 700L detector
    pub fn default_700l(&mut self) -> &mut Self {
        let new = self;
        new.volume = Some(P::from(0.7).unwrap());
        new
    }
}

#[derive(Debug, Clone, Builder)]
pub struct DetectorForwardModel<P>
where
    P: Float + std::fmt::Debug,
{
    #[builder(default = "DetectorParamsBuilder::default().build().unwrap()")]
    pub p: DetectorParams<P>,
    pub data: InputTimeSeries,
    #[builder(default = "vec_as::<_, P>(&self.data.as_ref().unwrap().airt)")]
    pub airt_points: Vec<P>,
    #[builder(default = "vec_as::<_, P>(&self.data.as_ref().unwrap().q_external)")]
    pub q_external_points: Vec<P>,
    #[builder(default = "vec_as::<_, P>(&self.data.as_ref().unwrap().q_internal)")]
    pub q_internal_points: Vec<P>,
    #[builder(default = "vec_as::<_, P>(&self.data.as_ref().unwrap().sensitivity)")]
    pub sensitivity_points: Vec<P>,
    #[builder(default = "vec_as::<_, P>(&self.data.as_ref().unwrap().background_count_rate)")]
    pub background_count_rate_points: Vec<P>,

    pub time_step: P,
    pub radon: Vec<P>,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub inj_source_strength: P,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub inj_begin: P,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub inj_duration: P,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub cal_source_strength: P,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub cal_begin: P,
    #[builder(default = "P::from(0.0).unwrap()")]
    pub cal_duration: P,
    #[builder(default = "60")]
    pub integration_substeps: usize,

    #[builder(
        default = "Interpolator::<P>::new(self.data.as_ref().unwrap().len(), self.time_step.unwrap())"
    )]
    pub interp: Interpolator<P>,
}

#[derive(Copy, Clone, Debug)]
pub struct Interpolator<P: Float + std::fmt::Debug> {
    pub time_step: P,
    pub npts: usize,
    pub p: P,
    pub tmax: P,
    pub idx0: usize,
    pub idx1: usize,
    pub w0: P,
    pub w1: P,
}

impl<P: Float + std::fmt::Debug> Interpolator<P> {
    /// Create new interpolator with `npts` points on the domain [0.0, tmax]
    fn new(npts: usize, time_step: P) -> Self {
        let tmax = time_step * P::from(npts).unwrap();
        let p = P::zero();
        let idx0 = p.floor().to_usize().unwrap();
        let idx1 = p.ceil().to_usize().unwrap();
        let w1 = p - P::from(idx0).unwrap();
        let w0 = P::from(1.0).unwrap() - w1;
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
    fn ti(&mut self, ti: P) {
        if ti <= P::zero() {
            self.w0 = P::one();
            self.w1 = P::zero();
            self.idx0 = 0;
            self.idx1 = 0;
        } else if ti >= P::from(self.tmax).unwrap() {
            self.w0 = P::zero();
            self.w1 = P::one();
            self.idx0 = self.npts - 1;
            self.idx1 = self.npts - 1;
        } else {
            self.p = ti / self.time_step;
            self.idx0 = self.p.floor().to_usize().unwrap();
            self.idx1 = self.p.ceil().to_usize().unwrap();
            self.w1 = self.p - P::from(self.idx0).unwrap();
            self.w0 = P::from(1.0).unwrap() - self.w1;
        }
    }
    /// Linear interpolation of y at ti, values outside of [0.0, tmax] are taken from endpoints
    fn linear(&self, y: &[P]) -> P {
        y[self.idx0] * self.w0 + y[self.idx1] * self.w1
    }
    /// Stepwise interpolation of y at ti, values outside of [0.0, tmax] are taken from endpoints
    fn stepwise(self, y: &[P]) -> P {
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

fn vec_as<T, P>(v: &[T]) -> Vec<P>
where
    P: Float,
    T: ToPrimitive + Copy,
{
    v.iter().map(|x| P::from(*x).unwrap()).collect()
}

impl<P: Float + std::fmt::Debug> DetectorForwardModel<P> {
    #[inline(always)]
    fn rate_of_change(
        &self,
        t: P,
        y: &[P; NUM_STATE_VARIABLES],
        dy: &mut [P; NUM_STATE_VARIABLES],
    ) {
        self.system(t.to_f64().unwrap(), y, dy)
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
    fn system(&self, t: f64, y: &[P; NUM_STATE_VARIABLES], dy: &mut [P; NUM_STATE_VARIABLES]) {
        // TODO: enforce this earlier
        assert!(self.radon.len() == self.data.len());
        // determine values interpolated in time
        let tmax = P::from(self.data.len() - 1).unwrap() * self.time_step;
        let t_delay = self.p.delay_time;
        let mut ti = P::from(t).unwrap() - t_delay;
        if ti < P::zero() {
            ti = P::zero()
        };
        if ti > tmax {
            ti = tmax
        };

        //// Just set the gradient equal to the y value itself (attempt to get this to compile)
        //for ii in 0..dy.len(){
        //    dy[ii] = y[ii];
        //}

        let p_lamrn = P::from(LAMRN).unwrap();
        let p_lama = P::from(LAMA).unwrap();
        let p_lamb = P::from(LAMB).unwrap();
        let p_lamc = P::from(LAMC).unwrap();

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
        assert!(self.p.exflow_scale >= P::zero());
        assert!(self.p.r_screen_scale >= P::zero());
        let q_external = q_external * self.p.exflow_scale;
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
        let inj_is_active = self.inj_source_strength > P::zero()
            && (ti) > self.inj_begin
            && (ti) <= self.inj_begin + self.inj_duration;
        let n_rn_inj = if inj_is_active {
            self.inj_source_strength / q_external
        } else {
            P::zero()
        };
        // The radon concentration flowing into the main tank needs to be
        // bumped up if the calibration source is active
        let cal_is_active = self.cal_source_strength > P::zero()
            && (ti) > self.cal_begin
            && (ti) <= self.cal_begin + self.cal_duration;
        let n_rn_cal = if cal_is_active {
            self.cal_source_strength / q_external
        } else {
            P::zero()
        };
        // make sure that we can't have V_delay_2 > 0 when V_delay == 0
        let v_delay_1: P;
        let v_delay_2: P;
        if self.p.volume_delay_1 == P::zero() && self.p.volume_delay_2 > P::zero() {
            v_delay_1 = self.p.volume_delay_2;
            v_delay_2 = P::zero();
        } else {
            v_delay_1 = self.p.volume_delay_1;
            v_delay_2 = self.p.volume_delay_2;
        }
        // effect of delay and tank volumes (allow V_delay to be zero)
        let d_nrn_dt: P;
        let d_nrnd1_dt: P;
        let d_nrnd2_dt: P;
        if v_delay_1 == P::zero() {
            // no delay tanks
            d_nrn_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn * p_lamrn;
            // Nrnd,Nrnd2 become unimportant, but we need to do something with them
            // so just apply the same equation as for Nrn
            d_nrnd1_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn_d1 * p_lamrn;
            d_nrnd2_dt = q_external / self.p.volume * (n_rn_ext + n_rn_cal + n_rn_inj - n_rn)
                - n_rn_d2 * p_lamrn;
        } else if v_delay_1 > P::zero() && v_delay_2 == P::zero() {
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
        let (n_a, n_b) =
            gf::calc_na_nb_factors(q_internal, self.p.volume, self.p.plateout_time_constant);
        let n_c = P::zero();
        // compute rate of change of each state variable
        let d_fa_dt = q_internal * r_screen * n_a * n_rn * p_lamrn - fa * p_lama;
        let d_fb_dt = q_internal * r_screen * n_b * n_rn * p_lamrn - fb * p_lamb
            + fa * p_lama * (P::one() - recoil_prob);
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
fn calc_eff_and_recoil_prob<P: Float + std::fmt::Debug>(
    q: P,
    rs: P,
    lamp: P,
    q_external: P,
    v_delay_1: P,
    v_delay_2: P,
    v_tank: P,
    total_efficiency: P,
) -> (P, P) {
    let recoil_prob = P::from(0.5).unwrap() * (P::from(1.0).unwrap() - rs);
    let eff = P::from(1.0).unwrap();
    let lamrn = P::from(LAMRN).unwrap();

    // account for radioactive decay in delay volumes (typical effect size: 0.3%)
    let radon0 = P::from(1.0).unwrap();
    let rn_d1 = radon0 / (lamrn * v_delay_1 / q_external + P::from(1.0).unwrap());
    let rn_d2 = rn_d1 / (lamrn * v_delay_2 / q_external + P::from(1.0).unwrap());
    let rn = rn_d2 / (lamrn * v_tank / q_external + P::from(1.0).unwrap());

    // TODO: this call takes about 20-30% of the total time spent evaluating the
    // objective function in calculations of the inverse model.  Consider memorizing
    // it, probably with the help of the generic_static crate
    // Then, also do the same for gf::calc_na_nb_factors
    let ssc = gf::steady_state_count_rate(q, v_tank, eff, lamp, recoil_prob, rs) * rn / radon0;
    let eff = eff * total_efficiency / ssc;
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

impl<P> DetectorForwardModel<P>
where
    P: Float + std::fmt::Debug,
{
    pub fn into_inner_type<NP>(&self) -> DetectorForwardModel<NP>
    where
        NP: Float + std::fmt::Debug,
    {
        let radon = self.radon.iter().map(|x| NP::from(*x).unwrap()).collect();
        DetectorForwardModel::<NP> {
            p: self.p.into_inner_type::<NP>(),
            data: self.data.clone(),
            airt_points: vec_as::<_, NP>(&self.data.airt),
            q_external_points: vec_as::<_, NP>(&self.data.q_external),
            q_internal_points: vec_as::<_, NP>(&self.data.q_internal),
            sensitivity_points: vec_as::<_, NP>(&self.data.sensitivity),
            background_count_rate_points: vec_as::<_, NP>(&self.data.background_count_rate),
            time_step: NP::from(self.time_step).unwrap(),
            radon,
            integration_substeps: 60,
            inj_source_strength: NP::from(self.inj_source_strength).unwrap(),
            inj_begin: NP::from(self.inj_begin).unwrap(),
            inj_duration: NP::from(self.inj_duration).unwrap(),
            cal_source_strength: NP::from(self.cal_source_strength).unwrap(),
            cal_begin: NP::from(self.cal_begin).unwrap(),
            cal_duration: NP::from(self.cal_duration).unwrap(),
            interp: Interpolator::<NP>::new(self.data.len(), NP::from(self.time_step).unwrap()),
        }
    }

    /// Calculate the initial state, in state vector form
    ///
    /// # Arguments
    ///
    /// * `radon0` - Radon concentration at t=0 in units of Bq/m3
    pub fn initial_state(&self, radon0: P) -> [P; NUM_STATE_VARIABLES] {
        // this is required for a DVector
        // let mut y = State::from_element(NUM_STATE_VARIABLES, 0.0);
        // this is required for a SVector
        let mut y = [P::zero(); NUM_STATE_VARIABLES];
        // Step through the delay volumes, accounting for radioactive decay in each,
        // by applying the relation
        // N/N0 = 1 / (lambda * tt + 1)
        // where lambda is the radioactive decay constant and tt = V/Q is the transit time

        // Generic versions of constants
        let lamrn_p = P::from(LAMRN).unwrap();

        // Initial state has everything in equilibrium with first radon value
        // radon, atoms/m3
        let n_radon0 = radon0 / lamrn_p;

        let q_external = P::from(self.data.q_external[0]).unwrap() * self.p.exflow_scale;
        let r_screen = self.p.r_screen * self.p.r_screen_scale;

        let n_rn_d1 = n_radon0 / (lamrn_p * self.p.volume_delay_1 / q_external + P::one());
        let n_rn_d2 = n_rn_d1 / (lamrn_p * self.p.volume_delay_2 / q_external + P::one());
        let n_rn = n_rn_d2 / (lamrn_p * self.p.volume / q_external + P::one());
        let rn = n_rn * lamrn_p;

        let (_eff, recoil_prob) = calc_eff_and_recoil_prob(
            P::from(self.data.q_internal[0]).unwrap(),
            r_screen,
            self.p.plateout_time_constant,
            P::from(self.data.q_external[0]).unwrap(),
            self.p.volume_delay_1,
            self.p.volume_delay_2,
            self.p.volume,
            P::from(self.data.sensitivity[0]).unwrap(),
        );
        let (fa_1bq, fb_1bq, fc_1bq) = gf::num_filter_atoms_steady_state(
            P::from(self.data.q_internal[0]).unwrap(),
            self.p.volume,
            self.p.plateout_time_constant,
            recoil_prob,
            r_screen,
        );

        y[IDX_NRND1] = n_rn_d1;
        y[IDX_NRND2] = n_rn_d2;
        y[IDX_NRN] = n_rn;
        y[IDX_FA] = fa_1bq * rn;
        y[IDX_FB] = fb_1bq * rn;
        y[IDX_FC] = fc_1bq * rn;
        y[IDX_ACC_COUNTS] = P::zero();
        y
    }

    #[inline(always)]
    pub fn numerical_solution(&self) -> Result<Vec<[P; NUM_STATE_VARIABLES]>> {
        //let system = (*self).clone()
        let system = self;
        let t0 = P::zero();
        let num_intervals = system.data.len() - 1;
        let _tmax = system.time_step * P::from(num_intervals).unwrap();
        let dt = system.time_step;
        let mut state = system.initial_state(system.radon[0]);

        // number of small RK4 steps to take per dt
        let num_steps = 30_usize;
        let mut t = t0;
        let mut expected_counts = Vec::with_capacity(num_steps);
        let mut y_out = Vec::with_capacity(num_steps);
        for _ in 0..num_intervals {
            state[IDX_ACC_COUNTS] = P::zero();
            integrate(&mut state, self, t, t + dt, num_steps);
            // TODO: maybe just return the expected counts??
            expected_counts.push(state[IDX_ACC_COUNTS]);
            y_out.push(state);
            t = t + dt;
        }

        Ok(y_out)
    }

    #[inline(always)]
    pub fn numerical_expected_counts(self) -> Result<Vec<P>> {
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
    use quickplot::draw_plot;

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
        let p = DetectorParamsBuilder::<f64>::default()
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
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
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
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 1.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
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
            /// LLD minus ULD (ULD are noise), missing values marked with NaN
            counts: 0.0,
            background_count_rate: 0.0, // 100.0/60.0/60.,
            // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
            sensitivity: 1000. / (3600.0 / 2.0),
            q_internal: 0.1 / 60.0,           //volumetric, m3/sec
            q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
            airt: 21.0,                       // degC
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
            .data(data)
            .radon(radon)
            .time_step(time_step)
            .build()
            .unwrap();
        let num_counts = fwd.numerical_expected_counts().unwrap();
        println!("{:#?}", num_counts);

        draw_plot(&num_counts[..], "test.svg").unwrap();
    }
}
