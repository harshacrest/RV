'use client'

import { useState, useMemo, useCallback } from 'react'

export type SortDirection = 'asc' | 'desc' | null

export interface SortState {
  column: string | null
  direction: SortDirection
}

export function useSortableData<T>(data: T[], initialSort: SortState = { column: null, direction: null }) {
  const [sort, setSort] = useState<SortState>(initialSort)

  const toggle = useCallback((column: string) => {
    setSort(prev => {
      if (prev.column !== column) return { column, direction: 'asc' }
      if (prev.direction === 'asc') return { column, direction: 'desc' }
      if (prev.direction === 'desc') return { column: null, direction: null }
      return { column, direction: 'asc' }
    })
  }, [])

  const sorted = useMemo(() => {
    if (!sort.column || !sort.direction) return data
    const col = sort.column
    const dir = sort.direction === 'asc' ? 1 : -1
    return [...data].sort((a, b) => {
      const av = (a as Record<string, unknown>)[col]
      const bv = (b as Record<string, unknown>)[col]
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
      if (typeof av === 'string' && typeof bv === 'string') return av.localeCompare(bv) * dir
      return 0
    })
  }, [data, sort])

  return { sorted, sort, toggle }
}

export function SortIcon({ column, sort }: { column: string; sort: SortState }) {
  const active = sort.column === column
  if (!active || !sort.direction) {
    return <span className="ml-1 text-[8px] opacity-30">⇅</span>
  }
  return (
    <span className="ml-1 text-[8px] opacity-80">
      {sort.direction === 'asc' ? '↑' : '↓'}
    </span>
  )
}

export function SortableTh({
  column, label, sort, toggle, className = '',
}: {
  column: string
  label: string
  sort: SortState
  toggle: (col: string) => void
  className?: string
}) {
  return (
    <th
      className={`cursor-pointer select-none hover:text-[var(--text-secondary)] transition-colors ${className}`}
      onClick={() => toggle(column)}
      title={`Sort by ${label}`}
    >
      {label}
      <SortIcon column={column} sort={sort} />
    </th>
  )
}
