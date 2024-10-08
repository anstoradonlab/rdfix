# rdfix

Perform a response time correction on two-filter dual-flow-loop radon detector output using the method from https://doi.org/10.5194/amt-9-2689-2016

This is a re-write of [a Python-based code](https://github.com/agriff86/rd-deconvolve) into rust and it's (hopefully) much nicer to install and use.  It's my first attempt at using rust, so the code itself is pretty
horrible.

Currently, the algorithm uses an MCMC sampler based on [hammer_and_sample](https://docs.rs/hammer-and-sample/latest/hammer_and_sample/) which itself is based on [EMCEE](https://emcee.readthedocs.io/en/stable/), the sampler used in the Python version of this code.

An option for using a No U-Turn Sampler, [NUTS](https://docs.rs/nuts-rs/latest/nuts_rs/), is in development but will probably rely on [Enzyme](https://enzyme.mit.edu/) for auto-differentiation.

## Installation

**Note:** `rdfix` binaries are built for Linux and Windows, but compiling for Mac should be straightforward.

### From binary

This is the easy option, if you just want to run the program.  Download a binary from [GitHub Releases](https://github.com/anstoradonlab/rdfix/releases) and put a copy in your `$PATH`.

### From source

Rust is required for compiling `rdfix` from source.  

First, [install rust](https://www.rust-lang.org/tools/install).  Then, execute

```bash
cargo install --locked --tag v0.3.0 --git https://github.com/anstoradonlab/rdfix.git
```

to download the clone the repository, compile the `rdfix` binary, and copy to a place in your `$PATH`.  Replace `--tag v0.3.0` with the version you wish to install, or leave it out to install from the main branch.

To do these three steps manually:

 clone the repository, 

```bash
git clone https://github.com/anstoradonlab/rdfix.git
```

compile using cargo

```bash
cargo build --release
```

then copy the binary from `target/release/rdfix` to a place in your `$PATH`.



## Quick start

### 1 Set up a test case

This command configures a run directory with simulated output from a radon detector in a subdirectory called `test`.

```bash
rdfix template -t test cal-peak-one-day
```

### 2 Run the test case

The command for running deconvolution will be printed in the terminal window.  It is:

```bash
rdfix deconv --config test/config.toml --output test/deconv-output test/raw-data.csv
```

The sampling stage of deconvolution runs in parallel and uses all available CPU cores.  It's computationally expensive, taking about three minutes to execute on my 4-core laptop.

### 3 Set up a template for your own data

Create another template.  If you have a month or more of data, use the `cal-peak-month` template, for example

```bash
rdfix template -t real_data cal-peak-month
```
### 4 Edit the configuration file

Edit the text file `real_data/config.toml`.  By default, it looks like:

```toml
[detector]
exflow_scale = 1.0
inflow = 0.025
volume = 1.5
r_screen = 0.95
r_screen_scale = 1.0
delay_time = 0.0
volume_delay_1 = 0.2
volume_delay_2 = 0.0
plateout_time_constant = 0.0033333333333333335

[inversion]
r_screen_sigma = 0.01
exflow_sigma = 0.01
sigma_delta = 0.4214
sigma_delta_threshold = 20.0
sampler_kind = "Emcee"
report_map = true
map_search_iterations = 50000
process_in_chunks = true
chunksize = 48
overlapsize = 8

[inversion.emcee]
burn_in = 5000
samples = 5000
walkers_per_dim = 3
thin = 50

[inversion.nuts]
samples = 1000
thin = 30
```

Most of these options can be left as their default values, especially for a first attempt, but pay attention to:
- `volume`, the main detector tank volume in m3, typically 1.5, 0.75, or 0.2.
- `volume_delay_1`, `volume_delay_2`, the thoron delay volumes, usually blue barrels, and typically 0.2 or 0.0.

 ### 5 Edit the data file

 The `raw-data.csv` file contains input data.  Modify this so that it contains your data.  The columns are:

  - `time`: Timestamp (at the end of interval) in YYYY-mm-dd HH:MM:SS format.  **Presently, this must be 30-minute intervals without gaps.** 
  - `counts`: Net counts over the period (t-deltaT, t).  Missing values are allowed, indicated with `NaN`
  - `background_count_rate`: Background count rate, counts/sec. If provided in counts/30-minute, then divide by 1800 (seconds/half-hour).
  - `sensitivity`: Detector calibration coefficient, (counts/sec)/(Bq/m3)
  - `q_internal`: Internal flow rate, m3/sec.  If provided as flow rate, m/s, then convert to volumetric flow by multiplying by pipe area, $A=\pi r^2$, where the pipe radius, $r$, is 50mm.
  - `q_external`: External flow rate, m3/sec.  If provided in l/min then convert to m3/sec by dividing by (1000*60).
  - `airt`: Air temperature, degC
  - `radon_truth`: Optional, default value NaN, the known, instantaneous, "True" radon concentration, Bq/m3.
  - `flag`: Optional, default value 0, data QA/QC flags.  0 implies "good measurement".  A value other than 0 forces the output to be masked out as invalid.

Typically, something like this Python code will be needed:

```python
df["background_count_rate"] = df['bg'] / (60*30)
df["q_external"] = df.exflow / 1000 / 60.0
# internal flow, convert from velocity (m/sec) to volumetric flow rate m3/sec
# the inflow parameter is in units of m/sec, pipe diameter is 100mm
pipe_area = np.pi * (100 / 1000 / 2.0) ** 2
df["q_internal"] = df.inflow * pipe_area
```

It is possible to make these changes in a spreadsheet, or use Python, R, etc. to write out a csv file.  There is some other pre-processing required to work out the background count rate and detector sensitivity.

### 6 Run deconvolution

```bash
rdfix deconv --config real_data/config.toml --output real_data/deconv-output real_data/raw-data.csv
```

### 7 View output

If deconvolution runs successfully, you'll find summary files in netCDF format

 - `deconv-output/summary_30min_average.nc`
 - `deconv-output/summary_60min_average.nc`

and csv format:

 - `deconv-output/summary_30min_average.csv`
 - `deconv-output/summary_60min_average.csv`


The most useful variables are:
 - `radon`: best single estimate of the radon concentration, the mean of the Bayesian posterior distribution also known as a Bayes estimator.
 
and these variables, which act as uncertainty bounds,
 - `radon_q025`: estimate of the radon 0.025th quantile (a.k.a 2.5th percentile)
 - `radon_q160`: estimate of the radon 0.16th quantile
 - `radon_q840`: estimate of the radon 0.84th quantile
 - `radon_q975`: estimate of the radon 0.975th quantile

A combination of `radon_q025` and `radon_q975` yeilds a Bayesian 95% credible interval, roughly analagous to quoting ±2σ error bands.
