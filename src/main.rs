use std::{
    array,
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
};

use momtrop::{vector::Vector, Edge, Graph, SampleGenerator, TropicalSamplingSettings};
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
struct NDimRect<const N: usize> {
    minima: [f64; N],
    maxima: [f64; N],
}
#[derive(Debug, Clone, Copy)]
struct NDimPoint<const N: usize> {
    point: [f64; N],
}

impl<const N: usize> From<[f64; N]> for NDimPoint<N> {
    fn from(value: [f64; N]) -> Self {
        Self { point: value }
    }
}

impl<const N: usize> NDimRect<N> {
    fn contains(&self, point: NDimPoint<N>) -> bool {
        for i in 0..N {
            if point.point[i] > self.minima[i] && point.point[i] < self.maxima[i] {
                continue;
            } else {
                return false;
            }
        }
        true
    }
}

fn main() -> Result<(), io::Error> {
    generate_triangle_points()?;
    generate_dt_points()
}

fn build_1_loop_graph(weight: f64, is_massive: bool, n_externals: usize) -> Graph {
    let n_ext_u8 = n_externals as u8;

    let edges = (0..n_ext_u8)
        .map(|i| {
            let vertices = (i, (i + 1) % n_ext_u8);
            Edge {
                vertices,
                is_massive,
                weight,
            }
        })
        .collect();

    let externals = (0..n_ext_u8).collect();

    Graph { edges, externals }
}

fn build_1_loop_sampler<const D: usize>(
    weight: f64,
    is_massive: bool,
    n_externals: usize,
) -> Result<SampleGenerator<D>, String> {
    let loop_lines = vec![vec![1]; n_externals];
    let graph = build_1_loop_graph(weight, is_massive, n_externals);
    graph.build_sampler(loop_lines)
}

fn generate_points<const D: usize>(
    weight: f64,
    indep_externals: Vec<Vector<f64, D>>,
    mass: Option<f64>,
    region: NDimRect<D>,
    num_points: usize,
    seed: u64,
) -> Result<[Vec<f64>; D], String> {
    let n_externals = indep_externals.len() + 1;

    let is_massive = mass.is_some();
    let sampler = build_1_loop_sampler::<D>(weight, is_massive, n_externals)?;
    let edge_data = build_edge_data(mass, &indep_externals);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let settings = TropicalSamplingSettings::default();

    let points = (0..num_points)
        .filter_map(|_| {
            let sample = sampler.generate_sample_from_rng(edge_data.clone(), &settings, &mut rng);
            match sample {
                Err(_) => {
                    println!("tropical sampling failed, omitting point");
                    None
                }
                Ok(sample) => {
                    let loop_moms = sample.loop_momenta[0].get_elements();
                    if loop_moms[0].is_nan() || loop_moms[1].is_nan() {
                        return None;
                    }

                    if region.contains(loop_moms.into()) {
                        Some(loop_moms)
                    } else {
                        None
                    }
                }
            }
        })
        .collect::<Vec<_>>();

    Ok(array::from_fn(|i| {
        points.iter().map(|point| point[i]).collect::<Vec<_>>()
    }))
}

fn write_files<const D: usize>(result: [Vec<f64>; D], file_name: &str) -> Result<(), io::Error> {
    for (i, axis_res) in result.iter().enumerate() {
        let path_name = format!("{}_{}.json", file_name, i);
        let path = PathBuf::from(&path_name);

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let json_string = serde_json::to_string(axis_res)?;
        write!(file, "{}", json_string)?;
    }

    Ok(())
}

fn build_edge_data<const D: usize>(
    mass: Option<f64>,
    indep_externals: &[Vector<f64, D>],
) -> Vec<(Option<f64>, Vector<f64, D>)> {
    let mut res = Vec::with_capacity(indep_externals.len() + 1);
    res.push((mass, Vector::from_array([0.0; D])));

    for external in indep_externals {
        let cur_mom = &res.last().unwrap_or_else(|| unreachable!()).1 + external;
        res.push((mass, cur_mom));
    }

    res
}

//let indep_externals = vec![
//    Vector::from_array([3.0, 4.0]),
//    Vector::from_array([6.0, 7.0]),
//    Vector::from_array([-20.0, -20.0]),
//    Vector::from_array([15.0, 4.0]),
//    Vector::from_array([-15.0, 10.0]),
//];

//let region = NDimRect {
//    maxima: [20.0, 20.],
//    minima: [-20.0, -20.0],
//};

//let points = generate_points(weight, indep_externals, mass, region, num_points, seed).unwrap();
//write_files(points, "hexagon_2d")?;

fn generate_triangle_points() -> Result<(), io::Error> {
    let weight = 2. / 3.;

    let indep_externals = vec![
        Vector::from_array([3.0, 4.0]),
        Vector::from_array([6.0, 7.0]),
    ];

    let mass = None;
    let seed = 69;
    let num_points = 20_000_000;

    let region = NDimRect {
        maxima: [20.0, 20.],
        minima: [-20.0, -20.0],
    };

    let points = generate_points(weight, indep_externals, mass, region, num_points, seed).unwrap();
    write_files(points, "triangle_2d")?;
    Ok(())
}

fn generate_dt_points() -> Result<(), io::Error> {
    let weight = 1. / 2. - 0.2;

    let dt_graph = Graph {
        edges: vec![
            Edge {
                vertices: (0, 1),
                is_massive: false,
                weight,
            },
            Edge {
                vertices: (0, 2),
                is_massive: true,
                weight,
            },
            Edge {
                vertices: (1, 2),
                is_massive: false,
                weight,
            },
            Edge {
                vertices: (1, 3),
                is_massive: false,
                weight,
            },
            Edge {
                vertices: (2, 3),
                is_massive: true,
                weight,
            },
        ],
        externals: vec![0, 3],
    };

    let n_samples = 10_000_000;

    let loop_signature = vec![vec![1, 0], vec![1, 0], vec![1, -1], vec![0, 1], vec![0, 1]];
    let dt_sampler = dt_graph.build_sampler::<1>(loop_signature).unwrap();

    let zero_vector = Vector::from_array([0.]);
    let p = Vector::from_array([1.]);

    let mass = Some(1.);

    let edge_data = vec![
        (None, zero_vector),
        (mass, p),
        (None, zero_vector),
        (None, zero_vector),
        (mass, Vector::from_array([1.])),
    ];

    let region = NDimRect {
        minima: [-3.0, -3.0],
        maxima: [3.0, 3.0],
    };

    let settings = TropicalSamplingSettings::default();

    let mut rng = rand::rngs::StdRng::seed_from_u64(69);

    let points = (0..n_samples)
        .filter_map(|_| {
            let sample =
                dt_sampler.generate_sample_from_rng(edge_data.clone(), &settings, &mut rng);
            match sample {
                Err(err) => {
                    println!("tropical sampling failed, omitting point: {:?}", err);
                    None
                }
                Ok(sample) => {
                    let loop_moms = [
                        sample.loop_momenta[0].get_elements()[0],
                        sample.loop_momenta[1].get_elements()[0],
                    ];

                    if loop_moms[0].is_nan() || loop_moms[1].is_nan() {
                        println!("nan detected, omitting point");
                    }

                    if !region.contains(loop_moms.into()) {
                        return None;
                    }

                    Some(loop_moms)
                }
            }
        })
        .collect::<Vec<_>>();

    let points: [Vec<f64>; 2] =
        array::from_fn(|i| points.iter().map(|point| point[i]).collect::<Vec<_>>());

    write_files(points, "double_triangle")?;

    Ok(())
}
