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

    // First, compute small primes up to sqrt(n) sequentially
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

    // If we don't need to sieve beyond sqrt(n), just return small primes
    if r_len >= n_len {
        return Ok(small_primes);
    }

    // Prepare chunk starts (just plain usize values, no Python objects)
    let chunk_starts: Vec<usize> = (r_len..n_len).step_by(chunk_size).collect();

    // Release GIL and do parallel computation
    let large_primes = Python::with_gil(|py| {
        py.allow_threads(|| {
            // Parallel computation without any GIL interaction
            chunk_starts
                .into_par_iter()
                .flat_map(|start| {
                    let len = chunk_size.min(n_len - start);
                    let mut chunk = bitvec![1; len];
                    let end = ((start + len - 1) << 1) ^ 1;

                    // Sieve this chunk using small primes
                    for &p in &small_primes[1..] {
                        // Skip 2, start from 3
                        if p * p > end {
                            break;
                        }
                        let j = sub_mod((p * p - 1) >> 1, start, p);
                        for j in (j..chunk.len()).step_by(p) {
                            chunk.set(j, false);
                        }
                    }

                    // Convert chunk indices to actual prime numbers
                    chunk
                        .iter_ones()
                        .map(|x| ((x + start) << 1) ^ 1)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<usize>>()
        })
    });

    // Combine small primes with large primes
    let mut all_primes = small_primes;
    all_primes.extend(large_primes);

    Ok(all_primes)
}

/// A Python module implemented in Rust.
#[pymodule]
fn par_sieve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_primes, m)?)?;
    Ok(())
}
