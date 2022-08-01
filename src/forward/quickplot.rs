use std::path::Path;

use anyhow::Result;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointMarker, PointStyle};
use plotlib::view::ContinuousView;

pub fn draw_plot<P>(x: &[f64], path: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let data = x
        .iter()
        .enumerate()
        .map(|itm| {
            let (ii, y) = itm;
            (ii as f64, *y)
        })
        .collect();

    let s1: Plot = Plot::new(data)
        .point_style(
            PointStyle::new()
                .marker(PointMarker::Circle) // setting the marker to be a square
                .colour("#DD3355"),
        ) // and a custom colour
        .line_style(LineStyle::new().width(2.0));

    let v = ContinuousView::new()
        .add(s1)
        .x_label("Some varying variable")
        .y_label("The response of something");

    // A page with a single view is then saved to an SVG file
    Page::single(&v).save(path).unwrap();

    Ok(())
}
