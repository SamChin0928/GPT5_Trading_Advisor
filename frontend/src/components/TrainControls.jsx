// components/TrainControls.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { Loader2 } from 'lucide-react'

export default function TrainControls({ sessionId }) {
  const [includeAll, setIncludeAll] = useState(true) // default to true when no global model yet
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState(null)     // object from /api/train/status
  const [jobSessions, setJobSessions] = useState(null) // sessions used for this run
  const timer = useRef(null)

  // initial fetch of status (to know if a global model exists)
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const s = await api.trainStatus(sessionId)
        if (!alive) return
        setStatus(s)
        // if a global model exists, ensure the toggle is false (hidden anyway)
        if (s?.global_model) setIncludeAll(false)
      } catch {}
    })()
    return () => { alive = false }
  }, [sessionId])

  // --- start training ---
  async function start() {
    setRunning(true)
    setStatus(prev => ({ ...(prev || {}), status: 'running', stage: 'starting' }))
    try {
      const res = await api.train(sessionId, { include_all: includeAll })
      setJobSessions(res?.sessions || null)
      poll() // begin polling loop
    } catch (e) {
      setRunning(false)
      setStatus({ status: 'error', error: e?.message || String(e) })
    }
  }

  // --- reset global model ---
  async function onReset() {
    if (!confirm('Reset the global model? This will delete the existing model.')) return
    try {
      await api.modelReset()
      // reflect state immediately in UI
      setStatus({ status: 'idle', global_model: false })
      setIncludeAll(true)
    } catch (e) {
      alert(e?.message || 'Failed to reset model')
    }
  }

  // --- poll status until done/error ---
  async function poll() {
    clearTimeout(timer.current)
    const tick = async () => {
      try {
        const s = await api.trainStatus(sessionId)
        setStatus(s)
        if (s?.status === 'running') {
          timer.current = setTimeout(tick, 1000)
        } else {
          setRunning(false)
          // one extra refresh shortly after "done" so global_model flips true
          if (s?.status === 'done') {
            setTimeout(async () => {
              try { setStatus(await api.trainStatus(sessionId)) } catch {}
            }, 1200)
          }
        }
      } catch (e) {
        setRunning(false)
        setStatus({ status: 'error', error: e?.message || String(e) })
      }
    }
    tick()
  }

  useEffect(() => () => clearTimeout(timer.current), [])

  // --- compute % only for stages where we can ---
  const percent = useMemo(() => {
    if (!status || status.status !== 'running') return null
    if (status.mode === 'legacy_cnn' && Number.isFinite(status.epoch) && Number.isFinite(status.epochs) && status.epochs > 0) {
      return Math.max(0, Math.min(100, Math.round((Number(status.epoch) / Number(status.epochs)) * 100)))
    }
    if (status.mode === 'embeddings+heads' && status.stage === 'extract' && Number.isFinite(status.count) && Number.isFinite(status.total) && status.total > 0) {
      return Math.max(0, Math.min(100, Math.round((Number(status.count) / Number(status.total)) * 100)))
    }
    return null
  }, [status])

  // --- human readable line under the bar ---
  const stageLine = useMemo(() => {
    if (!status) return null
    if (status.status === 'running') {
      if (status.stage === 'detect_pipeline') return 'Detecting best pipeline…'
      if (status.mode === 'legacy_cnn') return `Training CNN — epoch ${status.epoch ?? 0}/${status.epochs ?? '…'}`
      if (status.mode === 'embeddings+heads' && status.stage === 'extract') {
        return `Extracting group embeddings ${status.count ?? 0}/${status.total ?? '…'}`
      }
      return 'Running…'
    }
    if (status.status === 'done') return 'Training complete.'
    if (status.status === 'error') return `Error: ${status.error || 'unknown'}`
    return null
  }, [status])

  const isRunning = status?.status === 'running' || running
  const hasGlobal = !!status?.global_model

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-900/60 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold text-slate-100">Training</div>

        {/* right-side action */}
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-slate-400 hidden sm:block">
            Tip: label at least <b>6</b> folders (global labels).
          </span>

          {hasGlobal && (
            <button
              onClick={onReset}
              disabled={isRunning}
              className="px-3 py-2 rounded-lg text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 disabled:opacity-60"
              title="Delete the global model"
            >
              Reset
            </button>
          )}

          <button
            onClick={start}
            disabled={isRunning}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${
              isRunning ? 'bg-slate-700 text-slate-300 cursor-not-allowed'
                        : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            {isRunning ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Running…
              </span>
            ) : (hasGlobal ? 'Update model' : 'Build model')}
          </button>
        </div>
      </div>

      {/* options row */}
      {!hasGlobal && (
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={includeAll}
              onChange={e => setIncludeAll(e.target.checked)}
            />
            Train on all sessions
          </label>
        </div>
      )}

      {/* sessions included */}
      {(status?.sessions?.length || jobSessions?.length) ? (
        <div className="text-xs text-slate-400">
          Sessions: {(status?.sessions || jobSessions).join(', ')}
        </div>
      ) : null}

      {/* progress bar (only when we can compute %) */}
      {status?.status === 'running' && percent !== null && (
        <div className="w-full">
          <div className="h-2 w-full bg-slate-800/80 rounded-full overflow-hidden border border-white/10">
            <div
              className="h-full bg-emerald-500 transition-all"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="mt-1 text-[11px] text-slate-400">{stageLine}</div>
        </div>
      )}

      {/* fallback text when % is not available */}
      {status?.status === 'running' && percent === null && (
        <div className="text-xs text-slate-400">{stageLine}</div>
      )}

      {/* done / error */}
      {status?.status === 'done' && (
        <div className="text-xs text-emerald-300">
          Training complete{status?.report?.n_groups ? ` • ${status.report.n_groups} groups` : ''}.
        </div>
      )}
      {status?.status === 'error' && (
        <div className="text-xs text-rose-300">
          Error: {status.error || 'unknown'}
        </div>
      )}
    </div>
  )
}
