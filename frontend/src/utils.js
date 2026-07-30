export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

export function groupExperiments(experiments = []) {
  const groups = new Map()
  for (const experiment of experiments) {
    const series = experiment.series || '未分类实验'
    if (!groups.has(series)) groups.set(series, [])
    groups.get(series).push(experiment)
  }
  return [...groups.entries()].map(([series, items]) => ({
    series,
    items: items.sort((a, b) => (a.order ?? 0) - (b.order ?? 0)),
  }))
}

export function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

export function formatPercent(value) {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

export function rotatePoint(point, angle, position) {
  const [x, y] = point
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  return [position[0] + x * cos - y * sin, position[1] + x * sin + y * cos]
}

export function chartPoints(values, width, height, padding = 8) {
  if (!values?.length) return ''
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (Math.abs(max - min) < 1e-9) {
    min -= 1
    max += 1
  }
  return values
    .map((value, index) => {
      const x = padding + (index / Math.max(1, values.length - 1)) * (width - padding * 2)
      const y = padding + (1 - (value - min) / (max - min)) * (height - padding * 2)
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

export function shortHash(hash) {
  if (!hash) return 'not available'
  return `${hash.slice(0, 10)}…${hash.slice(-8)}`
}
