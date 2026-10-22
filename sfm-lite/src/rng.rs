//! Tiny xorshift PRNG (avoids pulling `rand`).

pub struct Rng(u64);

impl Rng {
    pub fn new() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
            | 1;
        Rng(t)
    }

    pub fn seeded(seed: u64) -> Self {
        Rng(seed | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % (n.max(1) as u64)) as usize
    }

    /// Uniform f64 in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self, sigma: f64) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * sigma
    }

    /// k distinct samples from 0..n (Fisher-Yates prefix).
    pub fn sample_k(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        let kk = k.min(n);
        let mut out = Vec::with_capacity(kk);
        for j in 0..kk {
            let r = self.below(n - j);
            out.push(idx[r]);
            idx.swap(r, n - 1 - j);
        }
        out
    }
}
