use super::{constants::NUM_STATE_VARIABLES, DetectorForwardModel};
use itertools::izip;

/// Advance `state` from `t0` to `t1` by taking `num_steps` RK4 steps
#[inline(always)]
pub fn integrate(
    state: &mut [f64; NUM_STATE_VARIABLES],
    model: &DetectorForwardModel,
    t0: f64,
    t1: f64,
    num_steps: usize,
) {
    let mut k1 = [0.0; NUM_STATE_VARIABLES];
    let mut k2 = [0.0; NUM_STATE_VARIABLES];
    let mut k3 = [0.0; NUM_STATE_VARIABLES];
    let mut k4 = [0.0; NUM_STATE_VARIABLES];
    let mut ktmp = [0.0; NUM_STATE_VARIABLES];

    let mut state_work = *state;

    // Step definition is
    // y(n+1) = y(n) + 1/6 * (k1 + 2k2 + 2k3 + k4)
    // where
    // k1 = f(tn,     yn)
    // k2 = f(tn+h/2, yn + h*k1/2)
    // k3 = f(tn+h/2, yn + h*k2/2)
    // k4 = f(tn + h, yn + h*k3)
    // Ref: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    let mut t = t0;
    let two = 2.0;
    let half = 0.5;
    let sixth = 1.0 / 6.0;
    let h = (t1 - t0) / (num_steps as f64);
    let half_h = half * h;
    for _ in 0..num_steps {
        // k1
        model.rate_of_change(t, &state_work, &mut k1);

        // k2
        for (ktmp, yn, k1) in izip!(&mut ktmp, &state_work, &k1) {
            *ktmp = *yn + half_h * *k1;
        }
        model.rate_of_change(t + half_h, &ktmp, &mut k2);

        // k3
        for (ktmp, yn, k2) in izip!(&mut ktmp, &state_work, &k2) {
            *ktmp = *yn + half_h * *k2;
        }
        model.rate_of_change(t + half_h, &ktmp, &mut k3);

        // k4
        for (ktmp, yn, k3) in izip!(&mut ktmp, &state_work, &k3) {
            *ktmp = *yn + h * *k3;
        }
        model.rate_of_change(t + h, &ktmp, &mut k4);

        // sum result
        for (s, k1, k2, k3, k4) in izip!(&mut state_work, &k1, &k2, &k3, &k4) {
            *s += h * sixth * (*k1 + two * *k2 + two * *k3 + *k4);
        }

        t += h;
    }

    #[allow(clippy::manual_memcpy)]
    for ii in 0..state.len() {
        state[ii] = state_work[ii];
    }
}
