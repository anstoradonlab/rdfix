use super::{constants::NUM_STATE_VARIABLES, DetectorForwardModel};
use itertools::izip;
use num::Float;
use arrayvec::ArrayVec;

/// Advance `state` from `t0` to `t1` by taking `num_steps` RK4 steps
pub fn integrate<P>(state: &mut[P; NUM_STATE_VARIABLES], model: &DetectorForwardModel<P>, t0: P, t1: P, num_steps: usize) 
where P: Float + std::fmt::Debug,
{
    let mut k1 = [P::zero(); NUM_STATE_VARIABLES];
    let mut k2 = [P::zero(); NUM_STATE_VARIABLES];
    let mut k3 = [P::zero(); NUM_STATE_VARIABLES];
    let mut k4 = [P::zero(); NUM_STATE_VARIABLES];
    let mut ktmp = [P::zero(); NUM_STATE_VARIABLES];

    let mut ynew: ArrayVec<P,NUM_STATE_VARIABLES> = ArrayVec::new();

    let mut new_state = state.clone();

    // Step definition is
    // y(n+1) = y(n) + 1/6 * (k1 + 2k2 + 2k3 + k4)
    // where
    // k1 = f(tn,     yn)
    // k2 = f(tn+h/2, yn + h*k1/2)
    // k3 = f(tn+h/2, yn + h*k2/2)
    // k4 = f(tn + h, yn + h*k3)
    let mut t = t0;
    let two = P::from(2.0).unwrap();
    let half = P::from(0.5).unwrap();
    let sixth = P::from(1.0/6.0).unwrap();
    let h = (t1-t0)/P::from(num_steps).unwrap();
    let half_h = half*h;
    let mut first = true;
    for _ in 0..num_steps {

        // k1
        model.rate_of_change(t, &new_state, &mut k1);
        if first{
            first = false;
            dbg!(k1);
        }

        // k2
        for (ktmp, yn, k1) in izip!(&mut ktmp, &new_state, &k1){
            *ktmp = *yn + half_h * *k1;
        }
        model.rate_of_change(t + half_h, &ktmp, &mut k2);

        // k3
        for (ktmp, yn, k2) in izip!(&mut ktmp, &new_state, &k2){
            *ktmp = *yn + half_h * *k2;
        }
        model.rate_of_change(t + half_h, &ktmp, &mut k2);

        // k4
        for (ktmp, yn, k3) in izip!(&mut ktmp, &new_state, &k3){
            *ktmp = *yn + h * *k3;
        }
        model.rate_of_change(t + h, &ktmp, &mut k2);

        // sum result
        for (state, k1, k2, k3, k4) in izip!(&mut new_state, &k1, &k2, &k3, &k4){
            *state = *state + sixth
                * (*k1 + two * *k2 + two * *k3 + *k4);
        }
        t = t + h;
    }
    for ii in 0..state.len(){
        state[ii] = new_state[ii];
    }
}
