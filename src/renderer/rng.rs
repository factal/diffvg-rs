//! PCG-based RNG for CPU-side jitter.

/// PCG32 default multiplier.
const PCG_MULT: u64 = 6364136223846793005;
/// PCG32 default increment base.
const PCG_INIT: u64 = 0x853c49e6748fea9b;

/// Small PCG32 RNG for deterministic CPU-side jitter.
pub(crate) struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Create a new RNG stream using a per-sample index and global seed.
    pub(crate) fn new(idx: u64, seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: ((idx + 1) << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(PCG_INIT.wrapping_add(seed));
        rng.next_u32();
        rng
    }

    /// Generate the next 32-bit random value.
    pub(crate) fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Generate a uniform float in the half-open interval [0, 1).
    pub(crate) fn next_f32(&mut self) -> f32 {
        let u = self.next_u32();
        let bits = (u >> 9) | 0x3f800000;
        f32::from_bits(bits) - 1.0
    }
}
