use anyhow::Result;
use ndarray::prelude::*;

use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum GridVarData {
    I32(Array<i32, IxDyn>),
    F64(Array<f64, IxDyn>),
}

impl GridVarData {
    pub fn shape(&self) -> &[usize] {
        match self {
            GridVarData::I32(x) => x.shape(),
            GridVarData::F64(x) => x.shape(),
        }
    }
}

impl From<Array<i32, IxDyn>> for GridVarData {
    fn from(value: Array<i32, IxDyn>) -> Self {
        GridVarData::I32(value)
    }
}

impl From<Vec<i32>> for GridVarData {
    fn from(value: Vec<i32>) -> Self {
        Array::from_vec(value).into_dyn().into()
    }
}

impl From<&[i32]> for GridVarData {
    fn from(value: &[i32]) -> Self {
        ArrayView1::from(value).into_owned().into_dyn().into()
    }
}

impl From<Array<f64, IxDyn>> for GridVarData {
    fn from(value: Array<f64, IxDyn>) -> Self {
        GridVarData::F64(value)
    }
}

impl From<Vec<f64>> for GridVarData {
    fn from(value: Vec<f64>) -> Self {
        Array::from_vec(value).into_dyn().into()
    }
}

impl From<&[f64]> for GridVarData {
    fn from(value: &[f64]) -> Self {
        ArrayView1::from(value).into_owned().into_dyn().into()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridVariable {
    pub name: String,
    pub dimensions: Vec<String>,
    //dimensions: HashMap<String, Dim>,
    //coordinates: HashMap<String, Vec<f64>>,
    pub data: GridVarData,
    pub attr: HashMap<String, String>,
}

impl GridVariable {
    pub fn new_from_parts(
        data: impl Into<GridVarData>,
        name: &str,
        dims: &[&str],
        attr: Option<HashMap<String, String>>,
    ) -> Self {
        let attr = attr.unwrap_or_default();
        let mut dimensions = vec![];
        for itm in dims {
            dimensions.push(itm.to_string());
        }
        let data = data.into();
        GridVariable {
            name: name.to_owned(),
            dimensions,
            data,
            attr,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataSet {
    vars: Vec<GridVariable>,
}

impl DataSet {
    pub fn new() -> Self {
        DataSet { vars: vec![] }
    }

    pub fn new_from_variables(vars: Vec<GridVariable>) -> Self {
        // should this be a hash map?
        // let mut map = HashMap::new();
        // for v in vars{
        //     map.insert( v.name.clone(), v);
        // }
        // extract all dimensions
        // check dimension sizes all match
        // check data matches dimensions

        // todo!();
        DataSet { vars }
    }

    pub fn var_ref(&self, vname: &str) -> Option<&GridVariable> {
        self.vars.iter().find(|&v| v.name == vname)
    }

    pub fn all_dimensions(&self) -> Vec<(String, usize)> {
        let mut dims = vec![];
        for v in self.vars.iter() {
            for (dname, dlen) in v.dimensions.iter().zip(v.shape()) {
                let itm = (dname.to_owned(), *dlen);
                if !dims.contains(&itm) {
                    dims.push(itm);
                }
            }
        }
        dims
    }

    pub fn to_netcdf(&self, f: PathBuf) -> Result<()> {
        let mut file = netcdf::create(f)?;
        // -- classic mode doesn't seem to work
        //let mut file = netcdf::create_with(&f, netcdf::Options::CLASSIC)?;
        for (dim_name, dim_len) in self.all_dimensions() {
            if &dim_name == "time" {
                file.add_unlimited_dimension(&dim_name)?;
            } else {
                file.add_dimension(&dim_name, dim_len)?;
            }
        }
        for v in self.vars.iter() {
            let dim_names: Vec<&str> = v.dimensions.iter().map(String::as_str).collect();

            let var = &mut file.add_variable::<f64>(&v.name, &dim_names)?;
            for (name, val) in v.attr.iter() {
                var.put_attribute(name.as_str(), val.as_str())?;
            }
        }
        for v in self.vars.iter() {
            let mut var = file
                .variable_mut(&v.name)
                .expect("Variable was just created in netCDF file, but now it can't be found");

            match &v.data {
                GridVarData::I32(x) => {
                    var.put_values(x.as_slice().expect("Can't convert array to slice."), ..)?
                }
                GridVarData::F64(x) => {
                    var.put_values(x.as_slice().expect("Can't convert array to slice."), ..)?
                }
            }
        }

        Ok(())
    }
}

impl Default for DataSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_var() -> GridVariable {
        let data = GridVarData::F64(ArrayD::zeros(vec![10, 20]));
        let var = GridVarData::new_from_parts(data, "test", &["x", "y"], None);
        var
    }

    fn create_ds() -> DataSet {
        let var = create_var();
        let ds = DataSet::new_from_variables(vec![var]);
        ds
    }

    #[test]
    fn can_create_var() {
        let var = create_var();
        dbg!(&var);
    }

    #[test]
    fn can_create_ds() {
        let _ds = create_ds();
    }

    #[test]
    fn can_create_netcdf() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let fname = dir.path().join("test.nc");
        let ds = create_ds();
        ds.to_netcdf(fname).unwrap();
        dir.close().unwrap();
    }
}
