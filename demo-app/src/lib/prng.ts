// Small seeded PRNG so a feedback report's (text, params, seed, model version)
// reproduces the exact strokes the user saw.

export class Prng {
  private s: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  /** Uniform in [0, 1) — mulberry32. */
  uniform(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal via Box-Muller (cached pair). */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = this.uniform(); // avoid log(0)
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    this.spare = r * Math.sin(2 * Math.PI * u2);
    return r * Math.cos(2 * Math.PI * u2);
  }
}

export function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}
