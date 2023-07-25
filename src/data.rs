use anyhow::{anyhow, Result};
use ndarray::prelude::*;

use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct GridVariable {
    pub name: String,
    pub dimensions: Vec<String>,
    //dimensions: HashMap<String, Dim>,
    //coordinates: HashMap<String, Vec<f64>>,
    pub data: Array<f64, IxDyn>,
    pub attr: HashMap<String, String>,
}

impl GridVariable {
    pub fn new_from_parts(
        data: Array<f64, IxDyn>,
        name: &str,
        dims: &[&str],
        attr: Option<HashMap<String, String>>,
    ) -> Self {
        let attr = attr.unwrap_or(HashMap::new());
        let mut dimensions = vec![];
        for itm in dims {
            dimensions.push(itm.to_string());
        }
        GridVariable {
            name: name.to_owned(),
            dimensions: dimensions,
            data: data,
            attr: attr,
        }
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
        for v in self.vars.iter() {
            if v.name == vname {
                return Some(v);
            }
        }
        None
    }

    pub fn all_dimensions(&self) -> Vec<(String, usize)> {
        let mut dims = vec![];
        for v in self.vars.iter() {
            for (dname, dlen) in v.dimensions.iter().zip(v.data.shape()) {
                let itm = (dname.to_owned(), *dlen);
                if !dims.contains(&itm) {
                    dims.push(itm);
                }
            }
        }
        dims
    }

    pub fn to_netcdf(&self, f: PathBuf) -> Result<()> {
        let mut file = netcdf::create(&f)?;
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
                var.add_attribute(name.as_str(), val.as_str())?;
            }
        }
        for v in self.vars.iter() {
            let mut var = file
                .variable_mut(&v.name)
                .ok_or(anyhow!("Can't find variable"))?;
            let data = v
                .data
                .as_slice()
                .ok_or(anyhow!("Can't convert array to slice"))?;
            var.put_values(data, ..)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_var() -> GridVariable {
        let data = ArrayD::zeros(vec![10, 20]);
        let var = GridVariable::new_from_parts(data, "test", &["x", "y"], None);
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
