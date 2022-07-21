//! # This is a markdown title inside the file `lib.rs`

use std::ops::{Add,Mul};
#[macro_use]
extern crate soa_derive;

#[derive(Debug,Clone,PartialEq,Copy,StructOfArray)]
#[soa_derive(Clone,Debug)]
pub struct InputRecord{
    pub time: f64,
    /// LLD minus ULD (ULD are noise), missing values marked with NaN
    pub counts: f64,
    pub background_count_rate: f64,
    pub sensitivity: f64,
    pub q_internal: f64,  //volumetric, m3/sec
    pub q_external: f64,  //volumetric, m3/sec
    pub airt: f64, // degC
}

impl Add for InputRecord {
    // Add to another InputRecord
    type Output = Self;

    fn add(self, rhs: InputRecord) -> Self {
        let mut new = self.clone();
        new.counts += rhs.counts;
        new.background_count_rate += rhs.counts;
        new.sensitivity += rhs.counts;
        new.q_internal += rhs.counts;
        new.q_external += rhs.counts;
        new.airt += rhs.counts;
        new
    }
}


impl Mul<f64> for InputRecord {
    /// Multiply by a f64
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        let mut new = self.clone();
        new.counts *= rhs;
        new.background_count_rate *= rhs;
        new.sensitivity *= rhs;
        new.q_internal *= rhs;
        new.q_external *= rhs;
        new.airt *= rhs;
        new
    }
}


#[derive(Debug,Clone,PartialEq,Copy,StructOfArray)]
#[soa_derive(Clone,Debug)]
pub struct OutputRecord{
    time: f64,
    radon: f64,

    // TODO: etc
}


//type InputTimeSeries = Vec<InputRecord>;
//type OutputTimeSeries = Vec<OutputRecord>;

pub type InputTimeSeries = InputRecordVec;
pub type OutputTimeSeries = OutputRecordVec;


/*
impl InputTimeSeries{
    pub fn time_step_seconds(&self) -> f64
    {
        // TODO: compute this from the time records
        60.0*30.0
    }
}
*/

pub mod forward;
pub mod inverse;
