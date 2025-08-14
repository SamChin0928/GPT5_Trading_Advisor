// components/TrainControls.jsx
import React, { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

export default function TrainControls({ sessionId }) {
  const [includeAll, setIncludeAll] = useState(false)
  const [epochs, setEpochs] = useState(5)
  const [batch, setBatch] = useState(16)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState(null)
  const timer = useRef(null)

  async function start() {
    setRunning(true)
    setStatus({ status: 'running', epoch: 0, epochs: epochs })
    try {
      await api.train(sessionId, { include_all: includeAll, epochs, batch_size: batch })
      poll()
    } catch (e) {
      setRunning(false)
      setStatus({ status: 'error', error: e?.message || String(e) })
    }
  }

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
        }
      } catch (e) {
        setRunning(false)
        setStatus({ status: 'error', error: e?.message || String(e) })
      }
    }
    tick()
  }

  useEffect(() => () => clearTimeout(timer.current), [])

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-900/60 p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={includeAll}
            onChange={e => setIncludeAll(e.target.checked)}
          />
          Train on all sessions
        </label>

        <label className="text-sm text-slate-300 flex items-center gap-2">
          Epochs
          <input
            type="number"
            min={1}
            value={epochs}
            onChange={e => setEpochs(Math.max(1, Number(e.target.value || 1)))}
            className="w-16 px-2 py-1 rounded bg-slate-800/70 border border-slate-700 text-slate-100"
          />
        </label>

        <label className="text-sm text-slate-300 flex items-center gap-2">
          Batch
          <input
            type="number"
            min={1}
            value={batch}
            onChange={e => setBatch(Math.max(1, Number(e.target.value || 1)))}
            className="w-16 px-2 py-1 rounded bg-slate-800/70 border border-slate-700 text-slate-100"
          />
        </label>

        <button
          onClick={start}
          disabled={running}
          className={`px-3 py-2 rounded-lg text-sm ${
            running ? 'bg-slate-700 text-slate-300 cursor-not-allowed'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {running ? 'Training…' : 'Train'}
        </button>
      </div>

      {status && (
        <div className="text-xs text-slate-400">
          {status.status === 'running' && (
            <>Training… epoch {status.epoch ?? 0}/{status.epochs ?? epochs}</>
          )}
          {status.status === 'idle' && <>Idle</>}
          {status.status === 'error' && (
            <span className="text-rose-300">Error: {status.error || 'unknown'}</span>
          )}
        </div>
      )}
    </div>
  )
}
