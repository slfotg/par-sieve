use bitvec::prelude::*;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn add_mod(a: usize, b: usize, m: usize) -> usize {
    ((a % m) + (b % m)) % m
}

pub fn sub_mod(a: usize, b: usize, m: usize) -> usize {
    add_mod(a, m - (b % m), m)
}

#[pyfunction]
pub fn get_primes(n: usize) -> PyResult<Vec<usize>> {
    let n_len = ((n - 1) >> 1) + 1;
    let r = n.isqrt();
    let r_len = ((r - 1) >> 1) + 1;
    let mut small = bitvec![1; r_len];
    small.set(0, false);
    for i in 1..small.len() {
        let p = (i << 1) ^ 1;
        if p * p > n {
            break;
        }
        for j in ((p + 1) * i..small.len()).step_by(p) {
            small.set(j, false);
        }
    }
    let small_primes = [2]
        .into_iter()
        .chain(small.iter_ones().map(|x| (x << 1) ^ 1))
        .collect::<Vec<_>>();
    let chunk_size = 800_000;

    let starts = (r_len..n_len).step_by(chunk_size).collect::<Vec<_>>();

    let primes = small_primes
        .into_par_iter()
        .chain(starts.into_par_iter().flat_map(|start| {
            let len = chunk_size.min(n_len - start);
            let mut chunk = bitvec![1; len];
            let end = ((start + len - 1) << 1) ^ 1;
            for p in small.iter_ones().map(|k| (k << 1) ^ 1) {
                if p * p > end {
                    break;
                }
                let j = sub_mod((p * p - 1) >> 1, start, p);
                for j in (j..chunk.len()).step_by(p) {
                    chunk.set(j, false);
                }
            }
            chunk
                .iter_ones()
                .map(|x| ((x + start) << 1) ^ 1)
                .collect::<Vec<_>>()
        }))
        .collect::<Vec<usize>>();

    Ok(primes)
}

/// A Python module implemented in Rust.
#[pymodule]
fn par_sieve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_primes, m)?)?;
    Ok(())
}
