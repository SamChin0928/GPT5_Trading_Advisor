// components/LivePrediction.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { Info } from 'lucide-react'

function prettyLabel(s) {
  if (!s) return ''
  return String(s).replace(/[_-]+/g, ' ').replace(/\b\w/g, m => m.toUpperCase())
}

function ProbBar({ label, value, active }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="flex items-center gap-2">
      <div className={`w-32 text-xs ${active ? 'text-slate-100 font-medium' : 'text-slate-400'}`}>
        {prettyLabel(label)}
      </div>
      <div className="flex-1 h-2 bg-slate-800/80 rounded-full overflow-hidden border border-white/10">
        <div className={`h-full ${active ? 'bg-emerald-500' : 'bg-slate-600'}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="w-12 text-right text-xs text-slate-400">{pct}%</div>
    </div>
  )
}

export default function LivePrediction({ sessionId }) {
  // single-image event (from capture loop)
  const [framePred, setFramePred] = useState(null)   // { label, probs, note? }
  const [imgUrl, setImgUrl] = useState(null)         // object URL for preview
  const [imgDataUrl, setImgDataUrl] = useState(null) // base64 for future feedback

  // global model + (optional) group prediction
  const [modelInfo, setModelInfo] = useState(null)   // /api/model/info
  const [groupPred, setGroupPred] = useState(null)   // { top, probs, thresholds, ... }

  // timers/refs
  const pollRef = useRef(null)
  const urlRef = useRef(null)

  /* ---------------- Live event (thumbnail + single-image pred) ---------------- */
  useEffect(() => {
    const handler = (e) => {
      const { pred, blob } = e.detail
      setFramePred(pred)

      // revoke old URL to avoid leaks
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = URL.createObjectURL(blob)
      setImgUrl(urlRef.current)

      // Keep a small data URL handy (for future feedback flow)
      const fr = new FileReader()
      fr.onload = () => setImgDataUrl(String(fr.result))
      fr.readAsDataURL(blob)
    }
    window.addEventListener('predict:update', handler)
    return () => {
      window.removeEventListener('predict:update', handler)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [])

  /* ---------------- Model info (accuracy / trained time) ---------------- */
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const info = await api.modelInfo()
        if (!cancelled) setModelInfo(info)
      } catch (_) {}
    }
    load()
    const t = setInterval(load, 60_000)
    return () => { cancelled = true; clearInterval(t) }
  }, [])

  /* ---------------- Optional: poll global group predictions ---------------- */
  useEffect(() => {
    const hasHeads = !!modelInfo?.exists && !!modelInfo?.has_heads
    if (!sessionId || !hasHeads || typeof api.predictGroup !== 'function') {
      clearTimeout(pollRef.current)
      pollRef.current = null
      return
    }

    let cancelled = false
    const tick = async () => {
      try {
        const g = await api.predictGroup(sessionId) // backend picks latest timestamp
        if (!cancelled && g && g.probs) setGroupPred(g)
      } catch (_) {
        /* ignore transient errors */
      } finally {
        pollRef.current = setTimeout(tick, 1500)
      }
    }
    tick()
    return () => { cancelled = true; clearTimeout(pollRef.current) }
  }, [sessionId, modelInfo?.exists, modelInfo?.has_heads])

  /* ---------------- What to display? Prefer groupPred when available ---------------- */
  const usingGroup = !!groupPred?.probs
  const displayProbs = usingGroup ? groupPred.probs : (framePred?.probs || {})

  const ordered = useMemo(() => {
    const entries = Object.entries(displayProbs || {})
    entries.sort((a, b) => b[1] - a[1])
    return entries
  }, [displayProbs])

  const topKey = ordered[0]?.[0]
  const displayLabel = usingGroup ? groupPred?.top : (framePred?.label || topKey || '')

  const headlineColor =
    /bull/i.test(displayLabel) ? 'text-emerald-400'
      : /bear/i.test(displayLabel) ? 'text-rose-400'
      : 'text-slate-200'

  const microF1 = modelInfo?.metrics?.overall?.micro_f1
  const trainedAt = modelInfo?.trained_at_human

  // Only show heuristic banner if there is truly no global model at all.
  const isHeuristicBanner = !modelInfo?.exists

  /* ---------------- UI ---------------- */
  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-lg font-semibold text-slate-100">Live Prediction</div>

        {/* model info pill row */}
        <div className="flex items-center gap-2 text-xs">
          {modelInfo?.exists ? (
            <>
              <span className="px-2 py-1 rounded-full bg-emerald-500/15 text-emerald-200 border border-emerald-500/30">
                Global model {modelInfo?.has_heads ? '• Heads' : modelInfo?.has_weights ? '• CNN' : ''}
              </span>
              {typeof microF1 === 'number' && (
                <span className="px-2 py-1 rounded-full bg-slate-800 text-slate-300 border border-white/10">
                  Val micro-F1: {(microF1 * 100).toFixed(1)}%
                </span>
              )}
              {trainedAt && (
                <span className="px-2 py-1 rounded-full bg-slate-800 text-slate-300 border border-white/10">
                  Trained: {trainedAt}
                </span>
              )}
            </>
          ) : (
            <span className="px-2 py-1 rounded-full bg-amber-500/15 text-amber-200 border border-amber-500/30">
              Heuristic (build a model for real predictions)
            </span>
          )}
        </div>
      </div>

      {/* preview */}
      {imgUrl ? (
        <img
          src={imgUrl}
          alt="primary"
          className="rounded-xl mb-4 border border-slate-800 w-full max-h-[320px] object-contain bg-black"
        />
      ) : (
        <div className="text-slate-400">Start the app and set a primary zone…</div>
      )}

      {/* headline */}
      {displayLabel && (
        <>
          <div className={`text-2xl font-bold ${headlineColor}`}>
            {prettyLabel(displayLabel)}
          </div>
          <div className="text-[12px] text-slate-400 mt-1 flex items-center gap-2">
            <Info className="w-3.5 h-3.5" />
            {isHeuristicBanner
              ? 'Heuristic preview. Train a model for real predictions.'
              : usingGroup
                ? 'Confidence across trained labels (global heads).'
                : modelInfo?.has_weights
                  ? 'Confidence across classes (global CNN).'
                  : 'Confidence across classes.'}
          </div>
        </>
      )}

      {/* probability bars */}
      {ordered.length > 0 && (
        <div className="mt-3 space-y-2">
          {ordered.map(([k, v]) => (
            <ProbBar key={k} label={k} value={v} active={k === topKey} />
          ))}
        </div>
      )}
    </div>
  )
}
