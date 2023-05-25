use num::Float;
use std::f64;

/// Constant value for `sqrt(2 * pi)`
const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;

/// Constant value for `ln(pi)`
const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;

/// Constant value for `ln(sqrt(2 * pi))`
const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

/// Constant value for `ln(sqrt(2 * pi * e))`
const LN_SQRT_2PIE: f64 = 1.4189385332046727417803297364056176398613974736378;

/// Constant value for `ln(2 * sqrt(e / pi))`
const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

/// Constant value for `2 * sqrt(e / pi)`
const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

/// Constant value for Euler-Masheroni constant `lim(n -> inf) { sum(k=1 -> n)
/// { 1/k - ln(n) } }`
const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310421593359399235988057672348849;

/// Targeted accuracy instantiated over `f64`
const ACC: f64 = 10e-11;

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
#[allow(non_snake_case)]
#[inline(always)]
pub fn ln_gamma<P: Float>(x: P) -> P {
    // Auxiliary variable when evaluating the `gamma_ln` function
    let GAMMA_R = P::from(10.900511).unwrap();

    // Polynomial coefficients for approximating the `gamma_ln` function
    let GAMMA_DK: &[P] = &[
        P::from(2.48574089138753565546e-5).unwrap(),
        P::from(1.05142378581721974210).unwrap(),
        P::from(-3.45687097222016235469).unwrap(),
        P::from(4.51227709466894823700).unwrap(),
        P::from(-2.98285225323576655721).unwrap(),
        P::from(1.05639711577126713077).unwrap(),
        P::from(-1.95428773191645869583e-1).unwrap(),
        P::from(1.70970543404441224307e-2).unwrap(),
        P::from(-5.71926117404305781283e-4).unwrap(),
        P::from(4.63399473359905636708e-6).unwrap(),
        P::from(-2.71994908488607703910e-9).unwrap(),
    ];

    if x < P::from(0.5).unwrap() {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + *t.1 / (P::from(t.0).unwrap() - x));

        P::from(LN_PI).unwrap()
            - (P::from(f64::consts::PI).unwrap() * x).sin().ln()
            - s.ln()
            - P::from(LN_2_SQRT_E_OVER_PI).unwrap()
            - (P::from(0.5).unwrap() - x)
                * ((P::from(0.5).unwrap() - x + GAMMA_R) / P::from(f64::consts::E).unwrap()).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| {
                s + *t.1 / (x + P::from(t.0).unwrap() - P::one())
            });

        s.ln()
            + P::from(LN_2_SQRT_E_OVER_PI).unwrap()
            + (x - P::from(0.5).unwrap())
                * ((x - P::from(0.5).unwrap() + GAMMA_R) / P::from(f64::consts::E).unwrap()).ln()
    }
}

#[inline(always)]
pub fn ln_factorial<P: Float>(x: P) -> P {
    ln_gamma(x + P::one())
}

/*
Generic version of lognormal
*/
#[inline(always)]
pub fn lognormal_ln_pdf<P: Float>(location: P, scale: P, x: P) -> P {
    // Constant value for `ln(sqrt(2 * pi))`
    const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

    if x <= P::zero() || x.is_infinite() {
        P::from(f64::NEG_INFINITY).unwrap()
    } else {
        let d = (x.ln() - location) / scale;
        (-P::from(0.5).unwrap() * d * d) - P::from(LN_SQRT_2PI).unwrap() - (x * scale).ln()
    }
}

/*
Generic version of normal
*/
#[inline(always)]
pub fn normal_ln_pdf<P: Float>(location: P, scale: P, x: P) -> P {
    // Constant value for `ln(sqrt(2 * pi))`
    const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

    let d = (x - location) / scale;
    (-P::from(0.5).unwrap() * d * d) - P::from(LN_SQRT_2PI).unwrap() - scale.ln()
}

/*
Generic version of Poisson
*/
#[inline(always)]
pub fn poisson_ln_pmf<P: Float>(lambda: P, x: P) -> P {
    -lambda + x * lambda.ln() - ln_factorial(x)
}

/// Exponential transform, including log_p increment
/// 
/// To understand this, see: 
/// 
/// "For univariate changes of variables, the resulting probability must be scaled by the absolute derivative of the transform."
///   -- https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html
///  
/// But, since we're working with log probabilities, instead of scaling
/// we instead increment by log( ... )
/// 
/// Also see this:
/// https://mc-stan.org/docs/reference-manual/change-of-variables.html
pub fn exp_transform<P: Float>(u: P) -> (P, P) {
    let maxu = (P::max_value()).ln();
    let mut u = u;
    if u>maxu {
        u = maxu;
    }
    // | d/du (exp(u)) | = exp(u)
    // log(|exp(u)|) = u
    (u.exp(), u)
}

/// Inverse exponential transform (i.e. log), this doesn't need to report a change
/// to the log_p
/// 
pub fn inverse_exp_transform<P:Float>(sigma:P) -> P{
    sigma.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use itertools::izip;
    use statrs::distribution::{Continuous, Discrete, LogNormal, Normal, Poisson};

    #[test]
    fn generic_lognormal() {
        let location = 1.0;
        let scale = 0.5;
        let x = 1.2;
        assert_eq!(
            lognormal_ln_pdf(location, scale, x),
            LogNormal::new(location, scale).unwrap().ln_pdf(x)
        );
    }

    #[test]
    fn generic_normal() {
        let location = 1.0;
        let scale = 0.5;
        let x = 1.2;
        assert_eq!(
            normal_ln_pdf(location, scale, x),
            Normal::new(location, scale).unwrap().ln_pdf(x)
        );
    }

    #[test]
    fn generic_poisson() {
        let lambda_test = [10.0, 100.0, 1000.0];
        let x_test = [11.0, 101.0, 1001.0];
        for (lambda, x) in izip!(lambda_test, x_test) {
            assert_approx_eq!(
                poisson_ln_pmf(lambda, x),
                Poisson::new(lambda)
                    .expect("Poisson failed")
                    .ln_pmf((x).round() as u64)
            );
        }
    }
}
