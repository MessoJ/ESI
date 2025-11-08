import React, { useEffect, useMemo, useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_URL || ''

function withBase(path: string) {
  if (!API_BASE) return path
  return `${API_BASE.replace(/\/$/, '')}${path}`
}

type LatestPayload = {
  as_of: string
  esi: number
  esi_smoothed: number
  buckets: { financial: number; macro: number; sentiment: number }
}

type HistoryRow = {
  date: string
  ESI: number
  ESI_smoothed: number
}

const band = (x: number) => (x < -0.5 ? 'Low' : x <= 0.5 ? 'Neutral' : x <= 1.5 ? 'Elevated' : 'Acute')

function fetchJSON<T>(url: string): Promise<T> {
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  })
}

export default function App() {
  const [latest, setLatest] = useState<LatestPayload | null>(null)
  const [hist, setHist] = useState<HistoryRow[]>([])
  const [comps, setComps] = useState<Record<string, number>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [country, setCountry] = useState<'US' | 'KE'>('US')

  useEffect(() => {
    setLoading(true)
    ;(async () => {
      try {
        const qs = `?country=${country}`
        const [l, h, c] = await Promise.all([
          fetchJSON<LatestPayload>(withBase(`/api/index/latest${qs}`)),
          fetchJSON<HistoryRow[]>(withBase(`/api/index/history${qs}`)),
          fetchJSON<Record<string, number>>(withBase(`/api/index/components${qs}`)),
        ])
        setLatest(l)
        setHist(h)
        setComps(c)
        setLoading(false)
      } catch (e: any) {
        setError(e.message)
        setLoading(false)
      }
    })()
  }, [country])

  const last90 = useMemo(() => hist.slice(-90), [hist])

  if (loading) return <div className="p-4">Loading…</div>
  if (error) return <div className="p-4 text-red-600">{error}</div>
  if (!latest) return null

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto grid gap-6 md:grid-cols-3">
        <div className="md:col-span-2 bg-white rounded-xl border shadow-sm p-6">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600" htmlFor="country">Country</label>
              <select
                id="country"
                value={country}
                onChange={(e) => setCountry(e.target.value as 'US' | 'KE')}
                className="px-2 py-1 border rounded-md text-sm"
              >
                <option value="US">United States</option>
                <option value="KE">Kenya</option>
              </select>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold">Economic Stress Index</h1>
              <p className="text-gray-500 text-sm">As of {latest.as_of}</p>
            </div>
            <span
              className={
                'px-3 py-1 rounded-full text-xs font-medium border ' +
                (latest.esi_smoothed < -0.5
                  ? 'bg-green-50 border-green-200 text-green-700'
                  : latest.esi_smoothed <= 0.5
                  ? 'bg-slate-50 border-slate-200 text-slate-700'
                  : latest.esi_smoothed <= 1.5
                  ? 'bg-amber-50 border-amber-200 text-amber-700'
                  : 'bg-red-50 border-red-200 text-red-700')
              }
            >
              {band(latest.esi_smoothed)}
            </span>
          </div>

          <div className="mt-6 grid md:grid-cols-3 gap-6">
            <div className="md:col-span-1">
              <div className="p-6 rounded-2xl border bg-white">
                <div className="text-sm text-gray-500">Headline (smoothed)</div>
                <div className="text-5xl font-semibold mt-1">{latest.esi_smoothed.toFixed(2)}</div>
                <div className="mt-2 text-xs text-gray-500">Raw: {latest.esi.toFixed(2)}</div>
                <div className="mt-4 text-sm text-gray-600">
                  Financial: {latest.buckets.financial.toFixed(2)} · Macro: {latest.buckets.macro.toFixed(2)} ·
                  Sentiment: {latest.buckets.sentiment.toFixed(2)}
                </div>
                <div className="mt-4">
                  <button
                    className="px-3 py-2 rounded-md text-sm bg-blue-600 text-white hover:bg-blue-700"
                    onClick={() => {
                      const qs = new URLSearchParams({ country, format: 'csv' })
                      const url = withBase(`/api/index/export.csv?${qs.toString()}`)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `esi_export_${new Date().toISOString().slice(0,10)}.csv`
                      document.body.appendChild(a)
                      a.click()
                      a.remove()
                    }}
                  >
                    Download CSV
                  </button>
                </div>
              </div>
            </div>

            <div className="md:col-span-2 p-4 rounded-2xl border bg-white">
              <div className="text-sm text-gray-500 mb-2">90‑day trend</div>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={last90}>
                  <XAxis dataKey="date" hide />
                  <YAxis domain={["auto", "auto"]} />
                  <Tooltip />
                  <ReferenceLine y={-0.5} strokeDasharray="3 3" />
                  <ReferenceLine y={0.5} strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="ESI_smoothed" fillOpacity={0.15} stroke="#2563eb" strokeWidth={2} fill="#60a5fa" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border shadow-sm p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Component Heatmap (z‑scores)</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
            {Object.entries(comps).map(([k, v]) => {
              const abs = Math.min(Math.abs(v), 2.5)
              const bg = v >= 0 ? `rgba(239,68,68,${abs / 3})` : `rgba(34,197,94,${abs / 3})`
              return (
                <div key={k} className="p-3 rounded-xl border flex items-center justify-between" style={{ background: bg }}>
                  <span className="text-sm font-medium">{k.replace('z_', '')}</span>
                  <span className="text-sm font-mono">{v.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}


