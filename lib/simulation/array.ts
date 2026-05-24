// Indexed access that throws on a genuine out-of-range read instead of quietly
// returning undefined. Under the strict compiler every `arr[i]` is `T |
// undefined`, and a simulation's hot loops index constantly. This keeps those
// reads typed as `T` while still failing loudly on a real bug.
export function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}
