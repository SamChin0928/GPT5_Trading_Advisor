import React, { useEffect, useRef, useState } from 'react'
import ScreenCapture from './components/ScreenCapture'
import ROISelector from './components/ROISelector'
import Controls from './components/Controls'
import LivePrediction from './components/LivePrediction'
import Labeler from './components/Labeler.jsx'   // <-- regular import

export default function App(){
  const [sessionId, setSessionId] = useState(()=>localStorage.getItem('sessionId')||crypto.randomUUID())
  const [zones, setZones] = useState([])
  const [primaryId, setPrimaryId] = useState(null)
  const [refSize, setRefSize] = useState({ w: 1920, h: 1080 })
  const captureRef = useRef(null)

  useEffect(()=>{ localStorage.setItem('sessionId', sessionId) }, [sessionId])
  useEffect(()=>{
    const l = (e)=> setZones(e.detail)
    window.addEventListener('zones:load', l)
    return ()=>window.removeEventListener('zones:load', l)
  },[])

  function onCapReady(h){ captureRef.current = h; fetchRefSize() }

  async function fetchRefSize(){
    // Try to get current video size for aspect‑correct ROI canvas
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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Chart Pattern Detector</h1>
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm">Session:</span>
          <input className="input" value={sessionId} onChange={e=>setSessionId(e.target.value)} />
        </div>
      </div>

      <div className="grid grid-cols-1">
        <ScreenCapture onReady={onCapReady} />
        {/* <ROISelector refSize={refSize} zones={zones} setZones={setZones} primaryId={primaryId} setPrimaryId={setPrimaryId} /> */}
      </div>

      <Controls sessionId={sessionId} zones={zones} primaryId={primaryId} captureHandle={captureRef.current} onPredict={onPredict} />

      <div className="grid grid-cols-2 gap-4">
        <LivePrediction />
        <div className="card">
          <div className="font-semibold mb-2">How it works</div>
          <ol className="list-decimal ml-5 space-y-1 text-slate-300 text-sm">
            <li>Click <b>Share Screen</b> and select the chart window/tab.</li>
            <li>Draw one or more <b>Zones</b>; set a <b>Primary</b> zone.</li>
            <li>Click <b>Start</b> to begin: crops stream to backend; the primary zone shows live prediction.</li>
            <li>Click <b>Stop</b>, then <b>Consolidate</b> to build per‑timestamp mosaics for labeling.</li>
            <li>Open <b>Labeler</b> section (below) to tag mosaics and <b>Train</b> a model.</li>
          </ol>
        </div>
      </div>

      {/* Labeler */}
      <Labeler sessionId={sessionId} />
    </div>
  )
}