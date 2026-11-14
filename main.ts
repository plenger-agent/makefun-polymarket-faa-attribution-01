// polymarket-faa-attribution-01 entry (Node 22, standard library only).
//
// Forecast: for each market and each minute h = 1..240 after the checkpoint, five
// quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that minute's Yes price.
//
// Method — weighted empirical step distribution, centred on the last price:
//   * For a grid of horizons H, take the market's own H-minute price changes over
//     the last 2,160 minutes, weighted by exponential recency (decay 120 min), and
//     read the five quantiles off that weighted empirical distribution.
//   * The median change is pure noise, so drift is shrunk fully: every row is
//     centred on the last price x, using only the empirical spread around its
//     median (q_tau - q_0.5).
//   * Spreads are rescaled by a volatility-regime ratio: the RMS 1-minute move of
//     the last 360 minutes over the RMS 1-minute move of the whole window,
//     clamped to [0.6, 1.7]. Quiet lately -> narrower; jumpy lately -> wider.
//   * Quantiles for minutes between grid horizons are interpolated linearly in
//     sqrt(h), matching random-walk spread growth.
//   * Blend 15% toward the house persistence baseline (sigma 0.0025 per
//     sqrt-minute) as a shrinkage prior against thin empirical tails.
//   * Widen the 5%/95% tails by 10% (pinball punishes a missed jump far more
//     than a slightly wide tail at quiet minutes).
//   * Clamp to [0,1], force rows non-decreasing, round to 6 decimals.
//
// Everything is deterministic: no randomness, no clock, no network. Two runs on
// the same case write the same bytes.
//
// I/O contract (kit RULES.txt): read every <id>.in in $MAKEFUN_INPUT_DIR
// (else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
// {"forecasts":[{"id","q"}]} with exactly 240 rows of 5 numbers per market.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** House persistence baseline spread, price units per square-root minute. */
const BASE_SIGMA = 0.0025;

/** How far back the empirical change distribution looks, in minutes. */
const WINDOW_MIN = 2160;
/** Exponential recency decay for change weights, in minutes. */
const DECAY_MIN = 120;
/** Length of the recent window for the volatility-regime ratio, in minutes. */
const RECENT_MIN = 360;
/** Clamp on the volatility-regime ratio. */
const RATIO_LO = 0.6;
const RATIO_HI = 1.7;
/** Weight given to the house persistence baseline in the final blend. */
const BLEND_BASE = 0.15;
/** Extra widening applied to the outer (5% / 95%) quantiles. Tuned on the
 *  practice cases: 1.10 under-covered (0.885) and scored 0.3115; 1.25 brings
 *  coverage to 0.906, near the nominal 0.90, and the best mean loss in the
 *  flat 1.20-1.35 region (0.3102 vs 0.3115 at 1.10; 1.50 turns back up). */
const TAIL_WIDEN = 1.25;
/** Horizons at which the empirical distribution is evaluated; minutes between
 *  them are interpolated in sqrt(h). */
const GRID = [1, 2, 4, 8, 16, 32, 64, 128, 240];
/** Minimum changes needed before a grid horizon is trusted. */
const MIN_CHANGES = 20;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/**
 * Weighted quantiles of (value, weight) pairs at the five contest levels.
 * Sorts by value; each level takes the smallest value whose cumulative weight
 * reaches tau of the total. Deterministic: input order is fixed.
 */
function weightedQuantiles(pairs: [number, number][]): number[] {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  let total = 0;
  for (const [, w] of sorted) total += w;
  const out: number[] = [];
  let cum = 0;
  let j = 0;
  for (const tau of QUANTILES) {
    const target = tau * total;
    while (j < sorted.length - 1 && cum + (sorted[j] as [number, number])[1] < target) {
      cum += (sorted[j] as [number, number])[1];
      j++;
    }
    out.push((sorted[j] as [number, number])[0]);
  }
  return out;
}

/** RMS of the 1-minute moves p[i+1]-p[i] for i in [lo, hi). */
function rmsMoves(p: number[], lo: number, hi: number): number {
  let sum = 0;
  let n = 0;
  for (let i = Math.max(0, lo); i < Math.min(p.length - 1, hi); i++) {
    const d = (p[i + 1] as number) - (p[i] as number);
    sum += d * d;
    n++;
  }
  return n === 0 ? 0 : Math.sqrt(sum / n);
}

/**
 * Empirical quantiles of the H-minute changes, re-centred on the last price x
 * and rescaled by the volatility-regime ratio. Returns null when the market
 * holds too few changes at this horizon.
 */
function gridQuantiles(p: number[], h: number, x: number, ratio: number, ws: number): number[] | null {
  const n = p.length;
  const pairs: [number, number][] = [];
  for (let t = Math.max(ws + h, h); t < n; t++) {
    const change = (p[t] as number) - (p[t - h] as number);
    const age = n - 1 - t; // minutes from the checkpoint
    pairs.push([change, Math.exp(-age / DECAY_MIN)]);
  }
  if (pairs.length < MIN_CHANGES) return null;
  const q = weightedQuantiles(pairs);
  const median = q[2] as number;
  // Drift fully shrunk: centre on x, keep only the spread around the median.
  return q.map((v) => x + ratio * (v - median));
}

/** The house persistence baseline row for horizon h around last price x. */
function baselineRow(x: number, h: number): number[] {
  const s = Math.sqrt(h);
  return Z.map((z) => price6(x + z * BASE_SIGMA * s));
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  const p = m.prices;
  if (p.length === 0) throw new Error(`market ${m.id} has no prices`);
  const n = p.length;
  const x = p[n - 1] as number;
  const ws = Math.max(0, n - WINDOW_MIN);

  // Volatility-regime ratio: recent RMS 1-minute move over the window's.
  const rmsRecent = rmsMoves(p, n - 1 - RECENT_MIN, n - 1);
  const rmsWindow = rmsMoves(p, ws, n - 1);
  const ratio = rmsWindow > 0 ? Math.min(RATIO_HI, Math.max(RATIO_LO, rmsRecent / rmsWindow)) : 1;

  // Empirical quantiles at each usable grid horizon.
  const grid: { h: number; q: number[] }[] = [];
  for (const h of GRID) {
    const q = gridQuantiles(p, h, x, ratio, ws);
    if (q !== null) grid.push({ h, q });
  }

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    let row: number[];
    if (grid.length === 0) {
      row = baselineRow(x, h);
    } else {
      // Interpolate between bracketing grid horizons in sqrt(h).
      const s = Math.sqrt(h);
      let lo = grid[0] as { h: number; q: number[] };
      let hi = grid[grid.length - 1] as { h: number; q: number[] };
      if (h <= lo.h) {
        row = lo.q.slice();
      } else if (h >= hi.h) {
        row = hi.q.slice();
      } else {
        for (let g = 0; g < grid.length - 1; g++) {
          const a = grid[g] as { h: number; q: number[] };
          const b = grid[g + 1] as { h: number; q: number[] };
          if (h >= a.h && h <= b.h) {
            lo = a;
            hi = b;
            break;
          }
        }
        const w = (s - Math.sqrt(lo.h)) / (Math.sqrt(hi.h) - Math.sqrt(lo.h));
        row = lo.q.map((v, j) => v + w * ((hi.q[j] as number) - v));
      }
      // Blend 15% toward the house persistence baseline.
      const base = baselineRow(x, h);
      row = row.map((v, j) => (1 - BLEND_BASE) * v + BLEND_BASE * (base[j] as number));
    }
    // Widen the outer tails by 10% around the centre price.
    row[0] = x + TAIL_WIDEN * ((row[0] as number) - x);
    row[4] = x + TAIL_WIDEN * ((row[4] as number) - x);
    // Clamp to [0,1], force non-decreasing, round to 6 decimals.
    let prev = -1;
    row = row.map((v) => {
      const c = price6(Math.max(prev, v));
      prev = c;
      return c;
    });
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
