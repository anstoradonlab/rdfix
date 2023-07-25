use chrono::prelude::*;
use lazy_static::lazy_static;

// constants (despite appearances, only correct to about 2 dec places but
//            written like this to match the python version)
pub const LAMRN: f64 = 2.1001405267111005e-06;
pub const LAMA: f64 = 0.0037876895112565318;
pub const LAMB: f64 = 0.00043106167945270227;
pub const LAMC: f64 = 0.00058052527685087548;

// state vector:  Nrnd, Nrnd2, Nrn, Fa, Fb, Fc, Acc_counts
pub const NUM_STATE_VARIABLES: usize = 7;

pub const IDX_NRND1: usize = 0;
pub const IDX_NRND2: usize = 1;
pub const IDX_NRN: usize = 2;
pub const IDX_FA: usize = 3;
pub const IDX_FB: usize = 4;
pub const IDX_FC: usize = 5;
pub const IDX_ACC_COUNTS: usize = 6;

// model parameters
pub const NUM_PARAMETERS: usize = 16;

lazy_static! {
    /// reference time for internal time units
    pub static ref REFERENCE_TIME: NaiveDateTime = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
}
/// time units to use in netCDF output
pub const TIME_UNITS: &str = "seconds since 2000-01-01 00:00:00.0";
