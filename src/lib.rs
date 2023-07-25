//! # This is a markdown title inside the file `lib.rs`

pub mod data;
pub mod forward;
pub mod inverse;
pub mod nuts;
pub mod cmdline;
pub mod appconfig;

//use std::ops::{Add, Div, Mul, Sub};

use chrono::{prelude::*, Duration};
use data::{GridVariable};
use forward::constants::{REFERENCE_TIME, TIME_UNITS};
use ndarray::{ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::io::{Read, Write};
use anyhow::Result;

#[macro_use]
extern crate soa_derive;

#[derive(Debug, Clone, PartialEq, Copy, StructOfArray, Default, Serialize, Deserialize)]
#[soa_derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InputRecord {
    /// Time measured in seconds since an arbitrary reference
    pub time: f64,
    /// LLD counts minus ULD (ULD are noise), missing values marked with NaN.
    /// The counts are measured over the last interval, `(t-dt, t)`
    pub counts: f64,
    /// Background count rate measured per second at time `t`.
    pub background_count_rate: f64,
    /// Sensitivity in `(detector cps) / (ambient Bq/m3)` at time `t`
    pub sensitivity: f64,
    /// Internal flow rate, volumetric, m3/sec
    pub q_internal: f64,
    /// External flow rate, volumetric, m3/sec
    pub q_external: f64,
    /// Air temperature inside detector, degrees C
    pub airt: f64,
}

/// A version of the InputRecord which can be used for IO - this
/// one knows about dates
#[derive(Debug, Clone, PartialEq, Copy, Default, Serialize, Deserialize)]
struct IoInputRecord {
    #[serde(with = "custom_date_format")]
    pub time: NaiveDateTime,
    pub counts: f64,
    pub background_count_rate: f64,
    pub sensitivity: f64,
    pub q_internal: f64,
    pub q_external: f64,
    pub airt: f64,
}

// copy-paste from docs, https://serde.rs/custom-date-format.html, switching to NaiveDateTime objects
mod custom_date_format {
    use chrono::NaiveDateTime;
    use serde::{self, Deserialize, Deserializer, Serializer};

    const FORMAT1: &'static str = "%Y-%m-%d %H:%M:%S";
    const FORMAT2: &'static str = "%Y-%m-%dT%H:%M:%S";
    const FORMAT3: &'static str = "%Y-%m-%d %H:%M:%S%.f";
    const FORMAT4: &'static str = "%Y-%m-%dT%H:%M:%S%.f";

    // The signature of a serialize_with function must follow the pattern:
    //
    //    fn serialize<S>(&T, S) -> Result<S::Ok, S::Error>
    //    where
    //        S: Serializer
    //
    // although it may also be generic over the input types T.
    pub fn serialize<S>(date: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", date.format(FORMAT1));
        serializer.serialize_str(&s)
    }

    // The signature of a deserialize_with function must follow the pattern:
    //
    //    fn deserialize<'de, D>(D) -> Result<T, D::Error>
    //    where
    //        D: Deserializer<'de>
    //
    // although it may also be generic over the output types T.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        NaiveDateTime::parse_from_str(&s, FORMAT1)
            .or_else(|_| NaiveDateTime::parse_from_str(&s, FORMAT2))
            .or_else(|_| NaiveDateTime::parse_from_str(&s, FORMAT3))
            .or_else(|_| NaiveDateTime::parse_from_str(&s, FORMAT4))
            .map_err(serde::de::Error::custom)
    }
}

impl From<IoInputRecord> for InputRecord {
    fn from(itm: IoInputRecord) -> Self {
        // TODO: calculate properly
        let time = (itm.time - *REFERENCE_TIME).num_seconds() as f64;
        InputRecord {
            time: time,
            counts: itm.counts,
            background_count_rate: itm.background_count_rate,
            sensitivity: itm.sensitivity,
            q_internal: itm.q_internal,
            q_external: itm.q_external,
            airt: itm.airt,
        }
    }
}

impl From<InputRecord> for IoInputRecord {
    fn from(itm: InputRecord) -> Self {
        // TODO: calculate properly
        let secs = Duration::seconds(itm.time.round() as i64);
        let nanosecs = Duration::nanoseconds(((itm.time - itm.time.round()) * 1e9) as i64);
        let time: NaiveDateTime = *REFERENCE_TIME + secs + nanosecs;
        IoInputRecord {
            time: time,
            counts: itm.counts,
            background_count_rate: itm.background_count_rate,
            sensitivity: itm.sensitivity,
            q_internal: itm.q_internal,
            q_external: itm.q_external,
            airt: itm.airt,
        }
    }
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

impl InputRecordVec {
    pub fn to_grid_vars(&self) -> Vec<GridVariable> {
        let mut data = vec![];
        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.time).into_owned().into_dyn(),
            "time",
            &["time"],
            Some(HashMap::from([("units".to_owned(), TIME_UNITS.to_owned())])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.counts).into_owned().into_dyn(),
            "counts",
            &["time"],
            Some(HashMap::from([(
                "units".to_owned(),
                "Counts over the interval, `(t-dt, t)`".to_owned(),
            )])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.background_count_rate)
                .into_owned()
                .into_dyn(),
            "background_count_rate",
            &["time"],
            Some(HashMap::from([(
                "units".to_owned(),
                "second^(-1)".to_owned(),
            )])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.sensitivity).into_owned().into_dyn(),
            "sensitivity",
            &["time"],
            Some(HashMap::from([(
                "units".to_owned(),
                "(detector cps) / (ambient Bq/m3)".to_owned(),
            )])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.q_internal).into_owned().into_dyn(),
            "q_internal",
            &["time"],
            Some(HashMap::from([(
                "units".to_owned(),
                "volumetric, m3/sec".to_owned(),
            )])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.q_external).into_owned().into_dyn(),
            "q_external",
            &["time"],
            Some(HashMap::from([(
                "units".to_owned(),
                "volumetric, m3/sec".to_owned(),
            )])),
        );
        data.push(v);

        let v = GridVariable::new_from_parts(
            ArrayView1::from(&self.airt).into_owned().into_dyn(),
            "airt",
            &["time"],
            Some(HashMap::from([("units".to_owned(), "deg C".to_owned())])),
        );
        data.push(v);

        //DataSet::new_from_variables(data)
        data
    }
}

pub fn get_test_timeseries(npts: usize) -> InputRecordVec {
    let trec = InputRecord {
        time: 0.0,
        counts: 1000.0 + 30.0,
        background_count_rate: 1.0 / 60.0,
        sensitivity: 1000. / (3600.0 / 2.0),
        q_internal: 0.1 / 60.0,
        q_external: 80.0 / 60.0 / 1000.0,
        airt: 21.0,
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

pub fn write_csv<W: Write>(file: &mut W, records: InputTimeSeries) -> Result<()> {
    let mut wtr = csv::Writer::from_writer(file);
    for row in &records {
        let row: IoInputRecord = row.to_owned().into();
        wtr.serialize(row)?;
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
        let row: IoInputRecord = result?;
        let row: InputRecord = row.into();
        println!("{:?}", row);
        data.push(row);
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn calculate_average() {
        let _ts = get_test_timeseries(100);
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
    fn csv() {
        let ts = get_test_timeseries(4);
        let mut outfile = Vec::new();
        write_csv(&mut outfile, ts).unwrap();
        let s = String::from_utf8(outfile.clone()).unwrap();
        let expected = "time,counts,background_count_rate,sensitivity,q_internal,q_external,airt\n\
        2000-01-01 00:00:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01 00:30:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01 01:00:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01 01:30:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n";
        assert_eq!(s, expected);
        dbg!(&s);
        println!("{}", s);

        // Happy to read from both "T-delemited" and space delimited date strings, with/without decimal seconds
        let csvdata = "time,counts,background_count_rate,sensitivity,q_internal,q_external,airt\n\
        2000-01-01 00:00:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01T00:30:00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01 01:00:00.00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n\
        2000-01-01T01:30:00.00,1030.0,0.016666666666666666,0.5555555555555556,0.0016666666666666668,0.0013333333333333333,21.0\n";
        let parsed_data = read_csv(csvdata.as_bytes()).unwrap();

        //let t = outfile.as_slice();
        let data = read_csv(outfile.as_slice()).unwrap();
        dbg!(&data);

        assert_eq!(&data, &parsed_data);
    }
}
