//! # This is a markdown title inside the file `lib.rs`

//use std::ops::{Add, Div, Mul, Sub};

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{Write,Read};

#[macro_use]
extern crate soa_derive;

#[derive(Debug, Clone, PartialEq, Copy, StructOfArray, Default, Serialize, Deserialize)]
#[soa_derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRecord {
    pub time: f64,
    /// LLD minus ULD (ULD are noise), missing values marked with NaN
    pub counts: f64,
    pub background_count_rate: f64,
    pub sensitivity: f64,
    pub q_internal: f64, //volumetric, m3/sec
    pub q_external: f64, //volumetric, m3/sec
    pub airt: f64,       // degC
}

/*
impl Add<InputRecord> for InputRecord {
    // Add to another InputRecord
    type Output = Self;

    fn add(self, rhs: InputRecord) -> Self {
        let mut new = self.clone();
        new.counts += rhs.counts;
        new.background_count_rate += rhs.background_count_rate;
        new.sensitivity += rhs.sensitivity;
        new.q_internal += rhs.q_internal;
        new.q_external += rhs.q_external;
        new.airt += rhs.airt;
        new
    }
}


impl Add<InputRecordRef<'_>> for InputRecord {
    // Add to another InputRecord
    type Output = Self;

    fn add(self, rhs: InputRecordRef) -> Self {
        let mut new = self.clone();
        new.counts += rhs.counts;
        new.background_count_rate += rhs.background_count_rate;
        new.sensitivity += rhs.sensitivity;
        new.q_internal += rhs.q_internal;
        new.q_external += rhs.q_external;
        new.airt += rhs.airt;
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

impl Div<f64> for InputRecord{
    /// Divide by a f64
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0/rhs)
    }
}

impl InputRecordVec{
    fn mean(&self) -> InputRecord{
        let n = self.len() as f64;
        self.iter().fold(InputRecord::default(), |acc, x| acc+x ) / n
    }
}

*/

#[derive(Debug, Clone, PartialEq, Copy, StructOfArray)]
#[soa_derive(Clone, Debug)]
pub struct OutputRecord {
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

pub fn get_test_timeseries(npts: usize) -> InputRecordVec {
    let trec = InputRecord {
        time: 0.0,
        /// LLD minus ULD (ULD are noise), missing values marked with NaN
        counts: 1000.0 + 30.0,
        background_count_rate: 1.0 / 60.0,
        // sensitivity is chosen so that 1 Bq/m3 = 1000 counts / 30-min
        sensitivity: 1000. / (3600.0 / 2.0),
        q_internal: 0.1 / 60.0,           //volumetric, m3/sec
        q_external: 80.0 / 60.0 / 1000.0, //volumetric, m3/sec
        airt: 21.0,                       // degC
    };
    let mut ts = InputRecordVec::new();
    let mut t = 0.0;
    let time_step = 60.0 * 30.0;
    for _ in 0..npts {
        ts.push(trec);
        *ts.time.last_mut().unwrap() = t;
        t += time_step;
    }

    ts
}

// Generic version of this would like like this:
// pub fn write_csv<W: Write>(file: &mut W, records: impl IntoIterator<Item = impl Serialize>)
// but it's complicated by the use of SOA_Derive which forces us to use to_owned

pub fn write_csv<W: Write>(file: &mut W, records: InputTimeSeries) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_writer(file);
    for row in &records {
        wtr.serialize(row.to_owned())?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn read_csv<R: Read>(file: R) -> Result<InputTimeSeries, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_reader(file);
    let mut data = InputTimeSeries::new();
    for result in rdr.deserialize() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let row: InputRecord = result?;
        println!("{:?}", row);
        data.push(row);
    }
    Ok(data)
}


pub mod forward;
pub mod inverse;
pub mod nuts;

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn calculate_average() {
        let ts = get_test_timeseries(100);
        /*
        let avg = ts.mean();
        let val = ts.get(0).unwrap().to_owned();
        assert_approx_eq!(val.counts, avg.counts);
        assert_approx_eq!(val.background_count_rate, avg.background_count_rate);
        assert_approx_eq!(val.sensitivity, avg.sensitivity);
        assert_approx_eq!(val.q_internal, avg.q_internal);
        assert_approx_eq!(val.q_external, avg.q_external);
        assert_approx_eq!(val.airt, avg.airt);

        dbg!(&avg);
        */
    }
    #[test]
    fn csv(){
        let ts = get_test_timeseries(2);
        let mut outfile = Vec::new();
        write_csv(&mut outfile, ts).unwrap();
        let s = String::from_utf8(outfile.clone()).unwrap();
        let expected = "time,counts,background_count_rate,sensitivity,q_internal,q_external,airt\n\
        0.0,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        1800.0,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n";
        assert_eq!(s, expected);
        dbg!(&s);
        println!("{}", s);

        //let t = outfile.as_slice();
        let data = read_csv(outfile.as_slice()).unwrap();
        dbg!(&data);
    }
}
