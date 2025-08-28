// src/components/ProcessWorkspace.jsx
import React, { useEffect, useRef, useState } from 'react'
import ScreenCapture from './ScreenCapture'
import Controls from './Controls'
import LivePrediction from './LivePrediction'
import Labeler from './Labeler.jsx'
import TrainControls from './TrainControls'
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
  const [initialized, setInitialized] = useState(false)

  // IMPORTANT: set the process immediately so children render with it
  api.setProcess(processId)
  console.log('[ProcessWorkspace] Mounting with process:', processId)
  
  useEffect(() => {
    // Ensure process is set and initialize if needed
    api.setProcess(processId)
    console.log('[ProcessWorkspace] Setting process context:', processId)
    
    // Initialize process directories by calling the init endpoint
    const initProcess = async () => {
      try {
        console.log('[ProcessWorkspace] Initializing process:', processId)
        // First ensure process directories exist
        if (api.initProcess) {
          const initResult = await api.initProcess()
          console.log('[ProcessWorkspace] Process init result:', initResult)
        }
        // DON'T try to load zones here - let ScreenCapture handle it
        // Just mark as initialized
      } catch (e) {
        console.log('[ProcessWorkspace] Process initialization error:', e)
      } finally {
        setInitialized(true)
        console.log('[ProcessWorkspace] Process initialized')
      }
    }
    
    initProcess()
    
    return () => {
      console.log('[ProcessWorkspace] Clearing process context')
      api.clearProcess()
    }
  }, [processId])

  // Session ID = today's date, read-only (rolls over automatically on reload)
  const [sessionId] = useState(() => formatTodayId())

  // App-level mirrors used by Controls / LivePrediction
  const [zones, setZones] = useState([])
  const [primaryId, setPrimaryId] = useState(null)
  const [refSize, setRefSize] = useState({ w: 1920, h: 1080 })
  const captureRef = useRef(null)

  // Listen for zones:load events from ScreenCapture
  useEffect(() => {
    const onZones = (e) => {
      const newZones = e.detail || []
      console.log('[ProcessWorkspace] zones:load event received:', newZones.length, 'zones')
      setZones(newZones)
      
      if (newZones.length > 0 && (primaryId === null || !newZones.some(z => z.id === primaryId))) {
        const firstPrimary = newZones.find(z => z.isPrimary)?.id ?? newZones[0].id
        setPrimaryId(firstPrimary)
        console.log('[ProcessWorkspace] Setting primary zone to:', firstPrimary)
      }
    }
    
    window.addEventListener('zones:load', onZones)
    return () => window.removeEventListener('zones:load', onZones)
  }, [primaryId])

  // Debug log zones state changes
  useEffect(() => {
    console.log('[ProcessWorkspace] Zones state updated:', zones.length, 'zones, primaryId:', primaryId)
  }, [zones, primaryId])

  function onCapReady(h){ 
    captureRef.current = h
    fetchRefSize()
  }

  async function fetchRefSize(){
    const cap = captureRef.current
    if(!cap) return
    const f = await cap.captureFrame()
    if(f) setRefSize({ w: f.w, h: f.h })
  }

  function onPredict(pred, blob){
    window.dispatchEvent(new CustomEvent('predict:update', { detail: { pred, blob } }))
  }

  function exitProcess() {
    // SPA helper if present, otherwise full reload
    if (typeof window.appNavigate === 'function') window.appNavigate('/')
    else window.location.assign('/')
  }

  // Show a loading state while initializing
  if (!initialized) {
    return (
      <div className="max-w-7xl mx-auto p-4 space-y-4">
        <div className="flex items-center justify-center h-32">
          <div className="text-slate-400">Initializing process...</div>
        </div>
      </div>
    )
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
        <div className="flex items-center gap-3">
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
          <button
            onClick={exitProcess}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm"
            title="Back to dashboard"
          >
            Exit
          </button>
        </div>
      </div>

      {/* Capture + ROI overlay */}
      <div className="grid grid-cols-1">
        {/* pass processId so ScreenCapture keys localStorage/server zones per process */}
        <ScreenCapture onReady={onCapReady} processId={processId} />
      </div>

      {/* Controls - Pass zones and log what we're passing */}
      {console.log('[ProcessWorkspace] Rendering Controls with:', { 
        zonesCount: zones.length, 
        zones: zones,
        primaryId 
      })}
      <Controls
        sessionId={sessionId}
        zones={zones}
        primaryId={primaryId}
        captureHandle={captureRef}
        onPredict={onPredict}
      />

      <div className="grid grid-cols-2 gap-4">
        <LivePrediction sessionId={sessionId} />
        <div className="card">
          <div className="font-semibold mb-2">How it works</div>
          <ol className="list-decimal ml-5 space-y-1 text-slate-300 text-sm">
            <li>Click <b>Share Screen</b> and select the chart window/tab.</li>
            <li>Draw one or more <b>Zones</b>; set a <b>Primary</b> zone.</li>
            <li>Click <b>Start</b> to begin: crops stream to backend; the primary zone shows live prediction.</li>
            <li>Open <b>Labeler</b> to label each <b>folder</b> (timestamp) or individual images.</li>
            <li>Click <b>Train</b> when you've labeled enough folders.</li>
          </ol>
        </div>
      </div>

      {/* Labeler */}
      <Labeler sessionId={sessionId} />
    </div>
  )
}