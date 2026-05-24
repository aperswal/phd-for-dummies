// Shared gridworld MDP primitives for the reinforcement-learning visualizations.
// A small deterministic gridworld solved exactly: the on-policy transition
// matrix, exact policy evaluation by a linear solve, the discounted
// state-visitation from any starting distribution, and the optimal value
// function by value iteration. These are policy-representation agnostic (they
// take a stochastic policy pi[state][action] and a transition matrix), so both
// the policy-gradient sim and the conservative-policy-iteration sim build on
// them rather than each carrying their own copy.

import { at } from "@/lib/simulation/array";

export const GRID_W = 5;
export const GRID_H = 5;
export const N_STATES = GRID_W * GRID_H;
export const N_ACTIONS = 4;

// 0 up, 1 down, 2 left, 3 right.
export const ACTION_LABELS = ["up", "down", "left", "right"] as const;
const ACTION_DROW = [-1, 1, 0, 0];
const ACTION_DCOL = [0, 0, -1, 1];

const VI_ITERS = 1000;
const VI_TOL = 1e-12;

export interface Mdp {
  start: number;
  goal: number;
  trap: number;
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export function rowOf(state: number): number {
  return Math.floor(state / GRID_W);
}

export function colOf(state: number): number {
  return state % GRID_W;
}

export function isTerminal(mdp: Mdp, state: number): boolean {
  return state === mdp.goal || state === mdp.trap;
}

// Reward earned on entering a state: +1 at the goal, -1 at the trap, 0 anywhere
// else. The grid is deterministic and bumping the edge keeps you in place.
export function enterReward(mdp: Mdp, state: number): number {
  if (state === mdp.goal) return 1;
  if (state === mdp.trap) return -1;
  return 0;
}

export function nextState(state: number, action: number): number {
  const row = clamp(rowOf(state) + at(ACTION_DROW, action), 0, GRID_H - 1);
  const col = clamp(colOf(state) + at(ACTION_DCOL, action), 0, GRID_W - 1);
  return row * GRID_W + col;
}

// Solves the dense linear system a x = b by Gaussian elimination with partial
// pivoting. The MDP is small (25 states), so a direct solve is both faster than
// iterating the Bellman map at gamma near 1 and exact. Mutates copies, not the
// inputs.
export function solveLinear(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, at(b, i)]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(at(at(m, row), col)) > Math.abs(at(at(m, pivot), col))) {
        pivot = row;
      }
    }
    const tmp = at(m, col);
    m[col] = at(m, pivot);
    m[pivot] = tmp;
    const pivotVal = at(at(m, col), col);
    if (Math.abs(pivotVal) < 1e-14) continue;
    const pivotRow = at(m, col);
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const targetRow = at(m, row);
      const factor = at(targetRow, col) / pivotVal;
      if (factor === 0) continue;
      for (let k = col; k <= n; k++) {
        targetRow[k] = at(targetRow, k) - factor * at(pivotRow, k);
      }
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const pivotVal = at(at(m, i), i);
    x[i] = Math.abs(pivotVal) < 1e-14 ? 0 : at(at(m, i), n) / pivotVal;
  }
  return x;
}

// The on-policy transition matrix P^pi(s, s') = sum_a pi(s,a) [next(s,a)=s'] and
// the per-state expected reward, with terminal rows left at zero so terminals
// absorb mass and hold value 0.
export function policyTransition(
  mdp: Mdp,
  pi: number[][],
): { p: number[][]; reward: number[] } {
  const p: number[][] = [];
  const reward = new Array<number>(N_STATES).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    if (!isTerminal(mdp, s)) {
      const probs = at(pi, s);
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        row[ns] = at(row, ns) + at(probs, a);
        reward[s] = at(reward, s) + at(probs, a) * enterReward(mdp, ns);
      }
    }
    p.push(row);
  }
  return { p, reward };
}

// Exact policy evaluation: V solves (I - gamma P^pi) V = R^pi, then Q reads off
// from V. Terminal states hold value 0.
export function evaluatePolicy(
  mdp: Mdp,
  pi: number[][],
  p: number[][],
  reward: number[],
  gamma: number,
): { values: number[]; q: number[][] } {
  const a: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    for (let s2 = 0; s2 < N_STATES; s2++) {
      row[s2] = (s === s2 ? 1 : 0) - gamma * at(at(p, s), s2);
    }
    a.push(row);
  }
  const values = solveLinear(a, reward);

  const q: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let action = 0; action < N_ACTIONS; action++) {
      const ns = nextState(s, action);
      row.push(
        enterReward(mdp, ns) +
          (isTerminal(mdp, ns) ? 0 : gamma * at(values, ns)),
      );
    }
    q.push(row);
  }
  return { values, q };
}

// The discounted state-visitation d(s) = sum_t gamma^t P(s_t = s) starting from
// the distribution `initial`, solving (I - gamma P^pi^T) d = initial. Terminals
// receive their arrival mass once and stop propagating, so the sum stays finite.
// The returned vector sums to 1/(1-gamma); multiply by (1-gamma) for the
// normalized visitation distribution.
export function discountedVisitation(
  p: number[][],
  gamma: number,
  initial: number[],
): number[] {
  const a: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    for (let s2 = 0; s2 < N_STATES; s2++) {
      row[s2] = (s === s2 ? 1 : 0) - gamma * at(at(p, s2), s);
    }
    a.push(row);
  }
  return solveLinear(a, initial);
}

// The optimal value function V*(s) by value iteration on the known MDP. Used
// only to score how close a learned policy's return is to the best possible,
// never by the learner.
export function optimalValues(mdp: Mdp, gamma: number): number[] {
  const values = new Array<number>(N_STATES).fill(0);
  for (let iter = 0; iter < VI_ITERS; iter++) {
    let delta = 0;
    for (let s = 0; s < N_STATES; s++) {
      if (isTerminal(mdp, s)) continue;
      let best = -Infinity;
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        const r = enterReward(mdp, ns);
        best = Math.max(
          best,
          r + (isTerminal(mdp, ns) ? 0 : gamma * at(values, ns)),
        );
      }
      delta = Math.max(delta, Math.abs(best - at(values, s)));
      values[s] = best;
    }
    if (delta < VI_TOL) break;
  }
  return values;
}

export function optimalReturn(mdp: Mdp, gamma: number): number {
  return at(optimalValues(mdp, gamma), mdp.start);
}

export function oneHot(index: number): number[] {
  const e = new Array<number>(N_STATES).fill(0);
  e[index] = 1;
  return e;
}
