import { describe, expect, it } from 'vitest'

import { chartPoints, clamp, groupExperiments, rotatePoint } from './utils.js'

describe('observer utilities', () => {
  it('groups and orders experiments by series', () => {
    const groups = groupExperiments([
      { id: 'b', series: 'S', order: 2 },
      { id: 'a', series: 'S', order: 1 },
      { id: 'c', series: 'T', order: 1 },
    ])
    expect(groups).toHaveLength(2)
    expect(groups[0].items.map((item) => item.id)).toEqual(['a', 'b'])
  })

  it('rotates local body geometry into world coordinates', () => {
    const point = rotatePoint([1, 0], Math.PI / 2, [2, 3])
    expect(point[0]).toBeCloseTo(2)
    expect(point[1]).toBeCloseTo(4)
  })

  it('creates bounded chart points', () => {
    expect(chartPoints([0, 1, 2], 100, 50)).toContain('8.00,42.00')
    expect(clamp(12, 0, 10)).toBe(10)
  })
})
