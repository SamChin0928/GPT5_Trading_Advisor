// src/components/ProcessWorkspace.jsx
import React, { useEffect, useRef, useState } from 'react'
import ScreenCapture from './ScreenCapture'
import Controls from './Controls'
import LivePrediction from './LivePrediction'
import Labeler from './Labeler.jsx'
import { api } from '../lib/api'

// ---- helpers ----
function formatTodayId() {
  const d = new Date()
  const dd = String(d.getDate()).padStart(2, '0')
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  const mon = months[d.getMonth()]
  const yyyy = d.getFullYear()
  return `${dd}${mon}${yyyy}` // e.g., 13Aug2025
}

function getProcessIdFromPath() {
  const path = window.location.pathname || '/'
  return path.startsWith('/process/') ? decodeURIComponent(path.slice('/process/'.length)) : null
}

export default function ProcessWorkspace(){
  const processId = getProcessIdFromPath()

  // Tell the API which process we're in (adds process_id to all requests)
  useEffect(() => {
    api.setProcess(processId)
    return () => api.clearProcess()
  }, [processId])

  // Session ID = today's date, read-only
  const [sessionId] = useState(() => formatTodayId())

  // App-level mirrors used by Controls / LivePrediction
  const [zones, setZones] = useState([])
  const [primaryId, setPrimaryId] = useState(null)
  const [refSize, setRefSize] = useState({ w: 1920, h: 1080 })
  const captureRef = useRef(null)

  // Keep: ScreenCapture dispatches 'zones:load' (array). App stores it for Controls.
  useEffect(() => {
    const onZones = (e) => {
      setZones(e.detail)
      // auto-pick primaryId if none set and zones arrived
      if (e.detail?.length && (primaryId === null || !e.detail.some(z => z.id === primaryId))) {
        setPrimaryId(e.detail[0].id)
      }
    }
    window.addEventListener('zones:load', onZones)
    return () => window.removeEventListener('zones:load', onZones)
  }, [primaryId])

  function onCapReady(h){ captureRef.current = h; fetchRefSize() }

  async function fetchRefSize(){
    const cap = captureRef.current
    if(!cap) return
    const f = await cap.captureFrame()
    if(f) setRefSize({ w: f.w, h: f.h })
  }

  function onPredict(pred, blob){
    window.dispatchEvent(new CustomEvent('predict:update', { detail: { pred, blob } }))
  }

  return (
    <div className="max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">
          Chart Pattern Detector
          <span className="ml-3 text-xs text-slate-400 font-normal">
            Process: {processId || '—'}
          </span>
        </h1>
        <div className="flex items-center gap-6">
          {/* Session (read-only) */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-sm">Session:</span>
            <input
              className="input"
              value={sessionId}
              readOnly
              title="Session is date-based and read-only"
            />
          </div>
        </div>
      </div>

      {/* Capture + ROI overlay */}
      <div className="grid grid-cols-1">
        <ScreenCapture onReady={onCapReady} />
      </div>

      {/* Controls */}
      <Controls
        sessionId={sessionId}
        zones={zones}
        primaryId={primaryId}
        captureHandle={captureRef}
        onPredict={onPredict}
      />

      <div className="grid grid-cols-2 gap-4">
        <LivePrediction />
        <div className="card">
          <div className="font-semibold mb-2">How it works</div>
          <ol className="list-decimal ml-5 space-y-1 text-slate-300 text-sm">
            <li>Click <b>Share Screen</b> and select the chart window/tab.</li>
            <li>Draw one or more <b>Zones</b>; set a <b>Primary</b> zone.</li>
            <li>Click <b>Start</b> to begin: crops stream to backend; the primary zone shows live prediction.</li>
            <li>Open <b>Labeler</b> to label each <b>folder</b> (timestamp) or individual images. Your labels are saved to your vocab and reusable.</li>
            <li>Click <b>Train</b> when you’ve labeled enough folders.</li>
          </ol>
        </div>
      </div>

      {/* Labeler */}
      <Labeler sessionId={sessionId} />
    </div>
  )
}
