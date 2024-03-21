# rdfix

Perform a response time correction on two-filter dual-flow-loop radon detector output using the method from https://doi.org/10.5194/amt-9-2689-2016

## Installation

**Note:** rdfix is currently linux-only.

### From binary

Download a binary from GitHub Releases and put a copy in your `$PATH`. 

### From source

Prerequisites are:

1. rust compiler,
2. openblas package

Openblas can be installed with a command like

```bash
sudo apt install -y libopenblas-dev   # Debian, Ubuntu, ...
```
or
```bash
sudo yum install -y openblas-devel.x86_64   # Fedora, Centos, ...
```


Clone the repository, 

```bash
git clone https://github.com
```

Compile using cargo

```bash
cargo build --release
```

Then copy the binary from `target/release/rdfix-deconvolve` to a place in your `$PATH`.

## Quick start rddeconv

### 1 Set up a test case

This command configures a run directory with simulated output from a radon detector in a subdirectory called `test`.

```bash
rdfix-deconvolve template -t test cal-peak-one-day
```

### 2 Run the test case

The command for running deconvolution will be printed in the terminal window.  It is:

```bash
rdfix-deconvolve deconv --config test/config.toml --output test/deconv-output test/raw-data.csv
```

The sampling stage of deconvolution runs in parallel and uses all available CPU cores.  It's computationally expensive, taking about a minute to execute this one-day test case with 20 cores available.

### 3 Set up a template for your own data

Create another template.  If you have a month or more of data, use the `cal-peak-month` template, for example

```bash
rdfix-deconvolve template -t real_data cal-peak-month
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

 The `raw-data.csv` file contains input data.  Modify your file so that it contains your data.  The columns are:

  - `time`: Timestamp (at the end of interval) in YYYY-mm-dd HH:MM:SS format.  Presently, this must be 30-minute intervals without gaps.  If you're running 
  - `counts`: Net counts over the period (t-deltaT, t).  Missing values are allowed, indicated with `NaN`
  - `background_count_rate`: Background count rate, counts/sec
  - `sensitivity`: Detector calibration coefficient, (counts/sec)/(Bq/m3)
  - `q_internal`: Internal flow rate, m3/sec
  - `q_external`: External flow rate, m3/sec
  - `airt`: Air temperature, degC
  - `radon_truth`: Optional, if present it contains the known, instantaneous, "True" radon concentration, Bq/m3.

It is possible to make these changes in a spreadsheet, or use Python, R, etc. to write out a csv file.  There is some pre-processing required to work out the background count rate and detector sensitivity.

### 6 Run deconvolution

```bash
rdfix-deconvolve deconv --config real_data/config.toml --output real_data/deconv-output real_data/raw-data.csv
```

### 7 View output

If deconvolution runs successfully, you'll find summary files in netCDF format

 - `deconv-output/summary_30min_average.nc`
 - `deconv-output/summary_60min_average.nc`

and csv format:

 - `deconv-output/summary_30min_average.csv`
 - `deconv-output/summary_60min_average.csv`


The most useful variables are:
 - `radon`: best estimate of the radon concentration
 
and these variables, which act as uncertainty bounds,
 - `radon_q160`: estimate of the radon 0.16th quantile (a.k.a 16th percentile)
 - `radon_q840`: estimate of the radon 0.84th quantile
