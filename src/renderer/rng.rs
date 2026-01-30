//! PCG-based RNG for CPU-side jitter.

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INIT: u64 = 0x853c49e6748fea9b;

pub(crate) struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
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

    pub(crate) fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        let u = self.next_u32();
        let bits = (u >> 9) | 0x3f800000;
        f32::from_bits(bits) - 1.0
    }
}
