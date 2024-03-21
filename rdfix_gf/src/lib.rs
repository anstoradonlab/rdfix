pub mod generated_functions;

#[cfg(test)]
mod tests {
    use super::generated_functions::*;

    #[test]
    fn func_is_callable() {
        let q = 0.1;
        let v = 1.5;
        let lambda_p = 0.001;
        let (nafac, nbfac) = calc_na_nb_factors(q, v, lambda_p);
        assert!(nafac > 0.0);
        assert!(nbfac > 0.0);

        let n_e = 1e6;
        let v_d = 0.2;
        let q_e = 80.0/60.0/1000.0;
        let delta_t = 1800.0;
        let eps_d = 0.2;
        let p_r = 0.05;
        let r_s = 0.95;
        let t = 3600.0;
        let t0 = 0.0;
        let total_counts = tc_integral_square_wave(nafac, nbfac, n_e, q, q_e, v, v_d, delta_t, eps_d, p_r, r_s, t, t0);
        println!("Total counts: {}", total_counts);
        assert!(total_counts > 0.0);
    }


    #[test]
    fn generic_func_has_same_result_as_non_generic(){
        let q = 0.1;
        let v = 1.5;
        let lambda_p = 0.001;
        let (nafac1, nbfac1) = calc_na_nb_factors(q, v, lambda_p);
        let (nafac2, nbfac2) = calc_na_nb_factors_f64(q, v, lambda_p);


        assert!(nafac1 == nafac2);
        assert!(nbfac1 == nbfac2);

    }
}
