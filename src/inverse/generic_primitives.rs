#![allow(clippy::excessive_precision)]
use std::f64;

#[allow(unused)]
/// Constant value for `sqrt(2 * pi)`
const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;
#[allow(unused)]
/// Constant value for `ln(pi)`
const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;
#[allow(unused)]
/// Constant value for `ln(sqrt(2 * pi))`
const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;
#[allow(unused)]
/// Constant value for `ln(sqrt(2 * pi * e))`
const LN_SQRT_2PIE: f64 = 1.4189385332046727417803297364056176398613974736378;
#[allow(unused)]
/// Constant value for `ln(2 * sqrt(e / pi))`
const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;
#[allow(unused)]
/// Constant value for `2 * sqrt(e / pi)`
const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;
#[allow(unused)]
/// Constant value for Euler-Masheroni constant `lim(n -> inf) { sum(k=1 -> n)
/// { 1/k - ln(n) } }`
const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310421593359399235988057672348849;
#[allow(unused)]
/// Targeted accuracy instantiated over `f64`
const ACC: f64 = 10e-11;

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
#[allow(non_snake_case)]
#[inline(always)]
pub fn ln_gamma(x: f64) -> f64 {
    // Auxiliary variable when evaluating the `gamma_ln` function
    let GAMMA_R = f64::from(10.900511);

    // Polynomial coefficients for approximating the `gamma_ln` function
    let GAMMA_DK: &[f64] = &[
        f64::from(2.48574089138753565546e-5),
        f64::from(1.05142378581721974210),
        f64::from(-3.45687097222016235469),
        f64::from(4.51227709466894823700),
        f64::from(-2.98285225323576655721),
        f64::from(1.05639711577126713077),
        f64::from(-1.95428773191645869583e-1),
        f64::from(1.70970543404441224307e-2),
        f64::from(-5.71926117404305781283e-4),
        f64::from(4.63399473359905636708e-6),
        f64::from(-2.71994908488607703910e-9),
    ];

    if x < f64::from(0.5) {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + *t.1 / ((t.0 as f64) - x));

        f64::from(LN_PI)
            - (f64::from(f64::consts::PI) * x).sin().ln()
            - s.ln()
            - f64::from(LN_2_SQRT_E_OVER_PI)
            - (f64::from(0.5) - x)
                * ((f64::from(0.5) - x + GAMMA_R) / f64::from(f64::consts::E)).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| {
                s + *t.1 / (x + (t.0 as f64) - 1.0)
            });

        s.ln()
            + f64::from(LN_2_SQRT_E_OVER_PI)
            + (x - f64::from(0.5))
                * ((x - f64::from(0.5) + GAMMA_R) / f64::from(f64::consts::E)).ln()
    }
}

#[inline(always)]
pub fn ln_factorial(x: f64) -> f64 {
    ln_gamma(x + 1.0)
}

/*
Generic version of lognormal
*/
#[inline(always)]
pub fn lognormal_ln_pdf(location: f64, scale: f64, x: f64) -> f64 {
    // Constant value for `ln(sqrt(2 * pi))`
    const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

    if x <= 0.0 || x.is_infinite() {
        f64::from(f64::NEG_INFINITY)
    } else {
        let d = (x.ln() - location) / scale;
        (-f64::from(0.5) * d * d) - f64::from(LN_SQRT_2PI) - (x * scale).ln()
    }
}

/*
Generic version of normal
*/
#[inline(always)]
pub fn normal_ln_pdf(location: f64, scale: f64, x: f64) -> f64 {
    // Constant value for `ln(sqrt(2 * pi))`
    const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

    let d = (x - location) / scale;
    (-f64::from(0.5) * d * d) - f64::from(LN_SQRT_2PI) - scale.ln()
}

/*
Generic version of Poisson
*/
#[inline(always)]
pub fn poisson_ln_pmf(lambda: f64, x: f64) -> f64 {
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
pub fn exp_transform(u: f64) -> (f64, f64) {
    // protect against overflow
    // let maxu = (P::max_value()).ln();
    // let mut u = u;
    // if u > maxu {
    //     u = maxu;
    // }
    // | d/du (exp(u)) | = exp(u)
    // log(|exp(u)|) = u
    (u.exp(), u)
}

/// Inverse exponential transform (i.e. log), this doesn't need to report a change
/// to the log_p
///
#[allow(unused)] // TODO: remove this
pub fn inverse_exp_transform(sigma: f64) -> f64 {
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
