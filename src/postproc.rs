use anyhow::Result;
use claim::*;
use log::{error, info};
use netcdf;
use statrs::statistics::Data;
use statrs::statistics::OrderStatistics;
use statrs::statistics::Statistics;
use std::path::PathBuf;

use crate::InputTimeSeries;

fn is_mcmc_variable(v: &netcdf::Variable) -> bool {
    v.dimensions()
        .iter()
        .any(|x| x.name() == "sample" || x.name() == "walker")
}

/// Return true if the variable is one which represents the
/// state of a variable at an instant in time (rather than a
/// time average).
/// TODO: handle this better, it's presently hard-coded
fn is_instantaneous_variable(v: &netcdf::Variable) -> bool {
    let is_mcmc = is_mcmc_variable(v);
    let has_tdim = !needs_time_dim(v);
    let inst_names = ["map_radon", "model_time", "time"];
    let vname = v.name();
    has_tdim && (is_mcmc || inst_names.into_iter().any(|x| x == vname))
}

/// Return True if this is the name of a variable which should be masked
fn is_maskable_varname(vname: &str) -> bool {
    let maskable_names = ["map_radon", "undeconvolved_radon", "radon"];
    maskable_names.into_iter().any(|x| {
        variable_suffixes().into_iter().any(|sfx| {
            vname
                .strip_suffix(sfx)
                .map_or_else(|| false, |itm| itm == x)
        })
    })
}

/// Return True for variables which should be masked
fn is_maskable_variable(v: &netcdf::Variable) -> bool {
    let vname = v.name();
    is_maskable_varname(&vname)
}

fn needs_time_dim(v: &netcdf::Variable) -> bool {
    !v.dimensions().iter().any(|x| x.name() == "time")
}

fn variable_suffixes() -> [&'static str; 6] {
    ["", "_q025", "_q160", "_q500", "_q840", "_q975"]
}

fn quantile_values() -> [f64; 5] {
    [0.025, 0.16, 0.50, 0.84, 0.975]
}

/// Copy netCDF file structure.  Assumptions:
///   1. it makes sense to create an unlimited dimension in the output file, and this
///      dimensions is called "time"
///   2. dimensions associated with sampling can be dropped.  These are currently "walker"
///      and "sample"
///   3. Variables without a time dimensions are given one (to be able to concatenate along
///      the time dimension at the end)
///   4. Variables which look like MCMC samples (i.e. have dimensions like "sample" and/or
///      "walker") are converted to summary variables of several statistics
///
///  As an example, an input variable like
///     double radon(time, walker, sample)
///  will become
///     double radon_mean(time)
///     double radon_p2p5(time)
///     double radon_p16(time)
///     double radon_p50(time)
///     double radon_p84(time)
///     double radon_p97p5(time)
fn copy_nc_structure(
    nc: &netcdf::File,
    ncout: &mut netcdf::FileMut,
    time_average: bool,
) -> Result<()> {
    // this is necessary
    ncout.add_unlimited_dimension("time")?;

    // copy structure, dimensions and globals
    for dim in nc.dimensions() {
        // Don't copy the time dimension, as it's already been added as an unlimited dimension
        // and don't copy dimensions associated with MCMC samples because we'll average over
        // those dimensions.
        if !["time", "sample", "walker"]
            .iter()
            .any(|x| *x == dim.name())
        {
            ncout.add_dimension(&dim.name(), dim.len())?;
        }
    }

    // Add time variable and other convenience variables
    let time_vnames = if time_average {
        vec!["time", "interval_start", "interval_mid", "interval_end"]
    } else {
        vec!["time"]
    };
    for vname in time_vnames {
        let v = nc.variable("time").unwrap();
        let dims = ["time"];
        let mut vout = ncout.add_variable::<f64>(vname, &dims)?;
        for att in v.attributes() {
            let attval = att.value()?;
            vout.put_attribute(att.name(), attval)?;
        }
    }

    for v in nc.variables() {
        if v.name() == "time" {
            continue;
        }
        let mut vdims = v.dimensions().iter().map(|x| x.name()).collect::<Vec<_>>();
        if needs_time_dim(&v) {
            // add time dimension in position 0
            vdims.insert(0, "time".to_string());
        }

        let variable_prefixes = if is_maskable_variable(&v) {
            vec!["unmasked_", ""]
        } else {
            vec![""]
        };
        if is_mcmc_variable(&v) {
            // MCMC variables, on output, are dimensioned by time only
            let vdims_str = vec!["time"];
            for prefix in variable_prefixes {
                for suffix in variable_suffixes() {
                    let vout_name = format!("{}{}{}", prefix, v.name(), suffix);
                    let typ = v.vartype();
                    let mut vout = ncout.add_variable_with_type(&vout_name, &vdims_str, &typ)?;
                    for att in v.attributes() {
                        let attval = att.value()?;
                        vout.put_attribute(att.name(), attval)?;
                    }
                    // TODO: append information to long_name attribute (e.g. ... 2.5th percentile)
                }
            }
        } else {
            for prefix in variable_prefixes {
                let vout_name = format!("{}{}", prefix, v.name());
                let vdims_str = vdims.iter().map(String::as_str).collect::<Vec<_>>();
                let typ = v.vartype();
                let mut vout = ncout.add_variable_with_type(&vout_name, &vdims_str, &typ)?;
                //let mut vout = ncout.add_variable::<f64>(&vout_name, &vdims_str)?;
                for att in v.attributes() {
                    let attval = att.value()?;
                    vout.put_attribute(att.name(), attval)?;
                }
            }
        }
    }

    Ok(())
}

/// Return an Extents object which can be used to slice along the time dimension
fn calc_time_dim_extents(
    v: &netcdf::Variable,
    idx0: usize,
    idx1: usize,
) -> Vec<std::ops::Range<usize>> {
    let ndim = v.dimensions().len();
    let idx_time_dim = v
        .dimensions()
        .iter()
        .position(|itm| itm.name() == "time")
        .unwrap();

    let mut extents = vec![];
    for ii in 0..ndim {
        let itm = if ii == idx_time_dim {
            idx0..idx1
        } else {
            0..v.dimensions()[ii].len()
        };
        extents.push(itm);
    }
    extents
}

fn extents_helper(v: &netcdf::Variable, num_overlap: usize) -> Vec<std::ops::Range<usize>> {
    let ndim = v.dimensions().len();
    let idx_time_dim = v
        .dimensions()
        .iter()
        .position(|itm| itm.name() == "time")
        .unwrap();
    let ntime = v.dimensions()[idx_time_dim].len();

    let mut extents = vec![];
    for ii in 0..ndim {
        let itm = if ii == idx_time_dim {
            num_overlap..ntime - num_overlap
        } else {
            0..v.dimensions()[ii].len()
        };
        extents.push(itm);
    }
    extents
}

fn calc_extents(
    v: &netcdf::Variable,
    num_overlap: usize,
    ntime: usize,
    tidx_out: usize,
) -> (Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>) {
    let ndim = v.dimensions().len();
    let idx_time_dim = v
        .dimensions()
        .iter()
        .position(|itm| itm.name() == "time")
        .unwrap();

    let mut extents = vec![];
    for ii in 0..ndim {
        let itm = if ii == idx_time_dim {
            num_overlap..ntime - num_overlap
        } else {
            0..v.dimensions()[ii].len()
        };
        extents.push(itm);
    }

    let mut extents_out = vec![];
    let n_points = ntime - num_overlap - num_overlap;
    for ii in 0..ndim {
        let itm = if ii == idx_time_dim {
            tidx_out..tidx_out + n_points
        } else {
            0..v.dimensions()[ii].len()
        };
        extents_out.push(itm);
    }
    (extents, extents_out)
}

/// Take the time average along x_in
/// x_in are the samples, measured instantaneously at time ti
/// t_in is the input sampling interval
/// t_out is the output sampling interval
fn time_average_instantaneous(x_in: &[f64], tau_in: usize, tau_out: usize) -> Vec<f64> {
    assert_ge!(tau_out, tau_in);
    assert_eq!(tau_out % tau_in, 0);
    let chunk_size = tau_out / tau_in;
    // pre-allocate extra space because of how this function is used in calling code
    let mut averaged = Vec::with_capacity(x_in.len());

    averaged.extend(
        x_in.windows(2)
            .map(|x| x.mean())
            .collect::<Vec<_>>()
            .chunks_exact(chunk_size)
            .map(|x| x.mean()),
    );

    averaged
}

/// Similar to `time_average_instantaneous`, but the input values, x_in, are assumed to be a time
/// average over the period t_in
fn time_average(x_in: &[f64], tau_in: usize, tau_out: usize) -> Vec<f64> {
    assert_ge!(tau_out, tau_in);
    assert_eq!(tau_out % tau_in, 0);
    let chunk_size = tau_out / tau_in;
    // pre-allocate extra space because of how this function is used in calling code
    let mut averaged = Vec::with_capacity(x_in.len());
    // Skip over the first value so that the output ends up on the same time grid
    // as the 'instantaneous' variables.  These time-average variables are assumed
    // to follow the datalogger timestamp convention where the timestamp relates
    // to the end of the sampling period.
    averaged.extend(x_in[1..].chunks_exact(chunk_size).map(|x| x.mean()));

    averaged
}

/// Not actaully a time average, this performs a reduction over integer timeseries
/// which is compatible with the time_average function.  The only field which is
/// treated in this way is the `flag` column.  The reduction scheme is to take the
/// max over each time window, because a flag value of 0 means "good data", so taking
/// the max gives a flag value which has a valid interpretation and avoids treating
/// a full averaging period as valid data when it contains some non-valid data
fn time_average_int(x_in: &[i32], tau_in: usize, tau_out: usize) -> Vec<i32> {
    assert_ge!(tau_out, tau_in);
    assert_eq!(tau_out % tau_in, 0);
    let chunk_size = tau_out / tau_in;
    // pre-allocate extra space because of how this function is used in calling code
    let mut averaged = Vec::with_capacity(x_in.len());
    // Skip over the first value so that the output ends up on the same time grid
    // as the 'instantaneous' variables.  These time-average variables are assumed
    // to follow the datalogger timestamp convention where the timestamp relates
    // to the end of the sampling period.
    averaged.extend(
        x_in[1..]
            .chunks_exact(chunk_size)
            .map(|x| x.iter().max().unwrap()),
    );

    averaged
}

/// Return a copy of `a` where a[mask != 0] = NAN
pub fn masked_array(
    mut a: ndarray::ArrayD<f64>,
    mask: &[i32],
    time_dim: ndarray::Axis,
) -> ndarray::ArrayD<f64> {
    let mask_array = ndarray::ArrayView1::from(mask);
    for mut x in a.lanes_mut(time_dim) {
        x.zip_mut_with(&mask_array, |xi, mi| {
            if *mi != 0 {
                *xi = f64::NAN
            }
        })
    }
    a
}

/// Return a copy of `v` where v[mask != 0] = NAN
pub fn masked_vec(v: &[f64], mask: &[i32]) -> Vec<f64> {
    let mut v = v.to_vec();
    v.iter_mut().zip(mask).for_each(|(xi, &mi)| {
        if mi != 0 {
            *xi = f64::NAN
        }
    });
    v
}

pub fn calc_tau_in(ts: &InputTimeSeries) -> usize{
    (ts.time[1] - ts.time[0]).round() as usize
}

// TODO: change this so that the output indices are calculated ahead of time (instead of file-by-file)
// The steps will be
//    - load the time dimension from the first and last file
//    - apply the averaging function
//    - extract t0 from the first file and t[end] from the last file (or, rather, from the input ts ..?)
//    - generate the time dimension, and the other aux. time variables, and write all
//    - use this information to generate a mapping from time t to index ii,
//         ii = (t-t0) / Delta_t

struct TimeInfo{
    /// The time *at* the start of the analysis period, seconds since a reference time
    t_start: f64,
    /// The timestep, in seconds
    delta_t: f64,
    /// Total number of time points
    ntime: usize,
}

impl TimeInfo{
    pub fn new_from_info(ts: &InputTimeSeries, num_overlap:usize, avg_duration: Option<usize>) -> Self{
        let tau_in = calc_tau_in(ts);
        // -1 here because the timestamp convention on the input 
        // is that t is the time at the end of the interval
        let (delta_t, idx0) = if let Some(avg) = avg_duration{
           (avg as f64, num_overlap)
        } else{
            (tau_in as f64, num_overlap - 1)
        };
        let t_start = ts.time[idx0];
        let ntime_in = ts.len();
        let ntime_out = if let Some(avg) = avg_duration {
            (ntime_in - num_overlap - num_overlap) / (avg / tau_in)
        } else {
            ntime_in - num_overlap - num_overlap
        };

        TimeInfo{t_start, delta_t, ntime: ntime_out}
    }
    /// Calculate index from time at start of interval
    pub fn idx(&self, interval_start: f64) -> usize{
        assert_ge!(interval_start, self.t_start);
        let idx = ((interval_start - self.t_start) / self.delta_t).round() as usize;
        dbg!(&idx, &interval_start, &self.t_start, &self.delta_t);
        assert_le!(idx, self.ntime);
        idx
    }

    pub fn idx_interval_end(&self, interval_end: f64) -> usize{
        let interval_start = interval_end - self.delta_t;
        self.idx(interval_start)
    }

    pub fn interval_start(&self) -> ndarray::Array1<f64> {
        ndarray::Array::range(self.t_start, 
            self.t_start + self.delta_t*(self.ntime as f64 + 0.5),
            self.delta_t)
    }

    pub fn interval_mid(&self) -> ndarray::Array1<f64> {
        ndarray::Array::range(self.t_start + (self.delta_t as f64) / 2.0, 
            self.t_start + self.delta_t*(self.ntime as f64 + 1.0),
            self.delta_t)
    }

    pub fn interval_end(&self) -> ndarray::Array1<f64> {
        ndarray::Array::range(self.t_start + (self.delta_t as f64), 
            self.t_start + self.delta_t*(self.ntime as f64 + 1.5),
            self.delta_t)
    }


}

pub fn postproc<'a, I, P>(
    ts: &InputTimeSeries,
    filenames: I,
    num_overlap: usize,
    output_fname: P,
    avg_duration: Option<usize>,
) -> Result<()>
where
    I: IntoIterator<Item = &'a PathBuf>,
    P: AsRef<std::path::Path>,
{
    // Time averaging will not work without overlap
    let num_overlap = if num_overlap < 2 && avg_duration.is_some() {
        2
    } else {
        num_overlap
    };
    // TODO: read this from the data
    let tau_in: usize = 30 * 60;

    let mut ncout = netcdf::create(output_fname)?;
    let mut first_file = true;

    let mut tidx_out: usize = 0;

    let tinfo = TimeInfo::new_from_info(ts, num_overlap, avg_duration);

    //TODO: replace with iteration
    for f in filenames {
        if let Some(tau_out) = avg_duration {
            info!("Postprocessing (avg period {} sec) {:?}", tau_out, f);
        } else {
            info!("Postprocessing {:?}", f);
        };
        let nc = netcdf::open(f)?;

        if first_file {
            copy_nc_structure(&nc, &mut ncout, avg_duration.is_some())?;
            // pre-write time variables, we write more if averaging is enabled
            if avg_duration.is_some(){
                let mut vout = ncout.variable_mut("interval_start").unwrap();
                vout.put(.., tinfo.interval_start().view())?;
                let mut vout = ncout.variable_mut("interval_mid").unwrap();
                vout.put(.., tinfo.interval_mid().view())?;
                let mut vout = ncout.variable_mut("time").unwrap();
                vout.put(.., tinfo.interval_mid().view())?;
                let mut vout = ncout.variable_mut("interval_end").unwrap();
                vout.put(.., tinfo.interval_end().view())?;
            }
            else{
                let mut vout = ncout.variable_mut("time").unwrap();
                vout.put(.., tinfo.interval_end().view())?;
            }

            first_file = false;
        } else {
            // TODO: on second file, and etc., validate structure
        }

        let ntime_in = nc.dimension("time").unwrap().len();
        let ntime_out = if let Some(avg) = avg_duration {
            (ntime_in - num_overlap - num_overlap) / (avg / tau_in)
        } else {
            ntime_in - num_overlap - num_overlap
        };

        // Calculate the starting index for output.  This contains unnecessary repetition
        // which should be refactored.
        let v = nc.variable("time").expect("No time variable");
        let tidx_out_expected = if let Some(tau_out) = avg_duration {
            let (idx0, idx1) = (num_overlap, ntime_in - num_overlap + 1);
            let extents = calc_time_dim_extents(&v, idx0, idx1);

            let data = v.get::<f64, _>(extents.clone())?;
            assert!(is_instantaneous_variable(&v));
            let interval_mid = ndarray::Array1::from(
                time_average_instantaneous(
                    data.as_slice().unwrap(),
                    tau_in,
                    tau_out,
                ));
            tinfo.idx(interval_mid[0]- ((tau_out as f64) / 2.0))
        } else {
            let (extents, _extents_out) =
            calc_extents(&v, num_overlap, ntime_in, 0);
            let data = v.get::<f64, _>(extents)?;
            tinfo.idx_interval_end(data[0])
        };

        for v in nc.variables() {
            let dims = v.dimensions();
            let has_time_dim = dims.iter().any(|itm| itm.name() == "time");
            let has_sample_dim = dims.iter().any(|itm| itm.name() == "sample");

            if has_time_dim && !has_sample_dim {
                // Copy this variable from input to output, although don't copy the overlap points
                if let Some(idx_time_dim) = dims.iter().position(|itm| itm.name() == "time") {
                    assert_eq!(v.dimensions()[idx_time_dim].len(), ntime_in);
                    // Acceptable to unwrap, we've already made sure to create this variable
                    let mut vout = ncout.variable_mut(&v.name()).unwrap();
                    let data;
                    #[allow(clippy::single_range_in_vec_init)]
                    let extents_out = vec![tidx_out..tidx_out + ntime_out];

                    if let Some(tau_out) = avg_duration {
                        // time average during copy
                        // Note, this should now be a 1-d array so let's just handle that case for simplicity
                        // and fix this later if needed.
                        if v.dimensions().len() > 1 {
                            panic!("Expected variable ({}) to be 1-d", v.name());
                        }
                        let (idx0, idx1) = (num_overlap, ntime_in - num_overlap + 1);

                        let extents = calc_time_dim_extents(&v, idx0, idx1);
                        if v.name() == "flag" {
                            // special case - flag is i32
                            let int_data = v.get::<i32, _>(extents.clone())?;
                            let values_avg =
                                time_average_int(int_data.as_slice().unwrap(), tau_in, tau_out);
                            vout.put_values(values_avg.as_slice(), extents_out.clone())?;
                        } else {
                            data = v.get::<f64, _>(extents.clone())?;
                            let values_avg = if is_instantaneous_variable(&v) {
                                time_average_instantaneous(
                                    data.as_slice().unwrap(),
                                    tau_in,
                                    tau_out,
                                )
                            } else {
                                time_average(data.as_slice().unwrap(), tau_in, tau_out)
                            };
                            assert_eq!(ntime_out, values_avg.len());
                            vout.put_values(values_avg.as_slice(), extents_out.clone())?;

                            // prevent further usage
                            drop(vout);

                            if v.name() == "time" {
                                // If this was the "time" variable then also write the convenience variables
                                let interval_mid = ndarray::Array1::from_vec(values_avg);
                                let interval_end = interval_mid.clone() + ((tau_out as f64) / 2.0);
                                let interval_start =
                                    interval_mid.clone() - ((tau_out as f64) / 2.0);
                                //tidx_out_expected = tinfo.idx(interval_start[0]);
                                let mut vout_t0 = ncout.variable_mut("interval_start").unwrap();
                                // this should have been written already.  Check for the expected value.
                                assert_eq!(vout_t0.get_value::<f64, _>([tidx_out_expected]).unwrap(), interval_start[0]);
                                vout_t0.put(extents_out.clone(), interval_start.view())?;
                                let mut vout_tm = ncout.variable_mut("interval_mid").unwrap();
                                vout_tm.put(extents_out.clone(), interval_mid.view())?;
                                let mut vout_t1 = ncout.variable_mut("interval_end").unwrap();
                                vout_t1.put(extents_out.clone(), interval_end.view())?;
                            }
                        }
                    } else {
                        // No time averaging.
                        // construct a slice which covers all except for the overlap points, ie. [num_overlap..(N-num_overlap), .., ..]
                        let (extents, extents_out) =
                            calc_extents(&v, num_overlap, ntime_in, tidx_out);
                        // data
                        let typ = v.vartype();
                        match typ {
                            netcdf::types::VariableType::Basic(netcdf::types::BasicType::Int) => {
                                let data = v.get::<i32, _>(extents)?;
                                vout.put(extents_out, data.view())?;
                            }
                            netcdf::types::VariableType::Basic(
                                netcdf::types::BasicType::Double,
                            ) => {
                                let data = v.get::<f64, _>(extents)?;
                                //if v.name() == "time"{
                                //    tidx_out_expected = tinfo.idx_interval_end(data[0]);
                                //}
                                vout.put(extents_out, data.view())?;
                            }
                            _ => unimplemented!(),
                        }
                    }
                }
            } else if has_time_dim && has_sample_dim {
                // Calculate percentiles and copy non-overlap points to output
                if v.name() != "radon" {
                    error!("Processing a variable ({}) which looks like it comes from MCMC samples and has a 'time' dimension \
                            but is not called 'radon'.  Output might be incorrect.", v.name());
                }
                if let Some(idx_time_dim) = dims.iter().position(|itm| itm.name() == "time") {
                    let data = if let Some(tau_out) = avg_duration {
                        // One extra point required for time averaging
                        let (idx0, idx1) = (num_overlap, ntime_in - num_overlap + 1);
                        let extents = calc_time_dim_extents(&v, idx0, idx1);
                        let mut data = v.get::<f64, _>(extents)?;

                        // This is a bit tricky.  Now we have an array data which is dimensioned with
                        // two or three dimensions, where we can't rely on the order
                        // If the sampler was EMCEE, it might be
                        //    data[time, walker, sample]
                        // What we want to do is apply time averaging along the "time" dimension by
                        // extracting one-dimensional arrays along that dimension.  This would be fairly
                        // simple except that time-averaging changes the length of the dimension (but the
                        // lanes_mut() iterator means that we need to write the result back to the original
                        // array, with the same length).  So, we'll pad out to the same length, write back
                        // into the data array, and the slice along the time dimension afterwards.
                        for mut x in data.lanes_mut(ndarray::Axis(idx_time_dim)) {
                            // ensure we have a contiguous slice to work with
                            let xs = x.to_vec();
                            let mut ys = time_average_instantaneous(xs.as_slice(), tau_in, tau_out);

                            assert_eq!(ntime_out, ys.len());

                            for _ in ys.len()..xs.len() {
                                // padding so that the two lengths match
                                ys.push(f64::NAN)
                            }
                            x.assign(&ndarray::Array1::from_vec(ys));
                        }

                        data.slice_axis(
                            ndarray::Axis(idx_time_dim),
                            ndarray::Slice::from(0..ntime_out),
                        )
                        .to_owned()
                    } else {
                        let extents = extents_helper(&v, num_overlap);
                        v.get::<f64, _>(extents)?
                    };

                    let var_mean: Vec<_> = data
                        .axis_iter(ndarray::Axis(idx_time_dim))
                        .map(|x| x.mean())
                        .collect();

                    let n_points = var_mean.len();
                    assert_eq!(n_points, ntime_out);
                    #[allow(clippy::single_range_in_vec_init)]
                    let extents_out = vec![tidx_out..tidx_out + ntime_out];
                    ncout
                        .variable_mut(&v.name())
                        .unwrap()
                        .put_values(&var_mean, extents_out.clone())?;

                    for (suffix, q) in variable_suffixes().iter().skip(1).zip(quantile_values()) {
                        let var_percentile: Vec<_> = data
                            .axis_iter(ndarray::Axis(idx_time_dim))
                            .map(
                                |x| {
                                    let mut x = x.to_owned();
                                    Data::new(x.as_slice_mut().unwrap()).quantile(q)
                                },
                                //data.axis_iter(ndarray::Axis(idx_time_dim)).map(|x| Data::new(x.as_slice_mut()).quantile(q)
                            )
                            .collect();
                        let vout_name = format!("{}{}", v.name(), suffix);
                        let mut vout = ncout.variable_mut(&vout_name).unwrap();
                        vout.put_values(&var_percentile, extents_out.clone())?;
                    }
                }
            } else if !has_time_dim && has_sample_dim {
                // Calculate percentiles and add an additional time dimension, expanding
                // the output along the time dimension
                let data = v.get::<f64, _>(..)?;
                let var_mean = data.clone().mean();
                let values = vec![var_mean; ntime_out];
                #[allow(clippy::single_range_in_vec_init)]
                let extents_out = vec![tidx_out..tidx_out + ntime_out];
                ncout
                    .variable_mut(&v.name())
                    .unwrap()
                    .put_values(&values, extents_out.clone())?;

                let mut x = data.to_owned();
                for (suffix, q) in variable_suffixes().iter().skip(1).zip(quantile_values()) {
                    let var_percentile = Data::new(x.as_slice_mut().unwrap()).quantile(q);
                    let values = vec![var_percentile; ntime_out];
                    let vout_name = format!("{}{}", v.name(), suffix);
                    let mut vout = ncout.variable_mut(&vout_name).unwrap();
                    vout.put_values(&values, extents_out.clone())?;
                }
            } else if dims.is_empty() {
                // This is a variable without dimenions.  It is expanded along the time
                // dimension.
                // Note: name this variable so that type inference works
                let empty_slice: &[usize; 0] = &[];
                let data = v.get_value::<f64, _>(empty_slice)?;
                let values = vec![data; ntime_out];
                let vout_name = v.name();
                let mut vout = ncout.variable_mut(&vout_name).unwrap();
                #[allow(clippy::single_range_in_vec_init)]
                let extents_out = vec![tidx_out..tidx_out + ntime_out];
                vout.put_values(&values, extents_out.clone())?;
            } else if !has_time_dim && !has_sample_dim {
                // This might be a coordinate variable, currently not expected
                // and perhaps should be handled, only once, in the "copy structure"
                // function
                todo!();
            }
        }

        assert_eq!(tidx_out , tidx_out_expected);

        tidx_out += ntime_out;
    }

    // Second pass over the output file, add the masked/unmasked versions of variables

    // Flag variable - zero means "Ok", other values should be masked
    // expect: we just created this file, so it really should have this structure
    let flag_var = ncout.variable("flag");

    let flag = match flag_var {
        Some(var) => var.get_values::<i32, _>(..).expect("Error reading flag"),
        None => {
            error!("'flag' variable missing, flagging all data as 0 (Valid)");
            vec![0; tidx_out]
        }
    };

    let vnames: Vec<_> = ncout
        .variables()
        .filter(|v| is_maskable_variable(v))
        .map(|v| v.name())
        .collect();
    for vname in vnames {
        let mut v = ncout.variable_mut(vname.as_str()).unwrap();
        let idx_time_dim = v
            .dimensions()
            .iter()
            .position(|itm| itm.name() == "time")
            .unwrap();
        let data = v.get::<f64, _>(..)?;
        let data_masked = masked_array(data.clone(), &flag, ndarray::Axis(idx_time_dim));
        v.put(.., data_masked.view())?;
        let vname_unmasked = format!("unmasked_{}", vname);
        // info!("Processing: {} and {}",&vname, &vname_unmasked);
        let mut v = ncout.variable_mut(vname_unmasked.as_str()).unwrap();
        v.put(.., data.view())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identify_maskable_variable() {
        assert_eq!(is_maskable_varname("map_radon"), true);
        assert_eq!(is_maskable_varname("map_radon_q840"), true);
        assert_eq!(is_maskable_varname("radon"), true);
        assert_eq!(is_maskable_varname("radon_q025"), true);
        assert_eq!(is_maskable_varname("undeconvolved_radon"), true);

        assert_eq!(is_maskable_varname("radon_b"), false);
        assert_eq!(is_maskable_varname("x"), false);
        assert_eq!(is_maskable_varname("unmasked_radon"), false);
    }
}
