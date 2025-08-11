import React, { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

/**
 * Start/Stop run loop: grab frame -> crop per zone -> send to ingest -> predict primary
 */
export default function Controls({ sessionId, zones, primaryId, captureHandle, onPredict }) {
  const [running, setRunning] = useState(false)
  const [intervalMs, setIntervalMs] = useState(1000)
  const timerRef = useRef(null)

  useEffect(() => { return () => stop() }, [])

  async function dataUrlFromBitmap(bmp, sx, sy, sw, sh) {
    const canvas = new OffscreenCanvas(sw, sh)
    const ctx = canvas.getContext('2d')
    ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh)
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
    const buf = await blob.arrayBuffer()
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)))
    return `data:image/jpeg;base64,${b64}`
  }

  function start(){
    if(!captureHandle) return alert('Share your screen first.')
    setRunning(true)
    loop()
    timerRef.current = setInterval(loop, intervalMs)
  }

  function stop(){
    setRunning(false)
    if(timerRef.current) clearInterval(timerRef.current)
  }

  async function loop(){
    const frame = await captureHandle.captureFrame()
    if(!frame) return
    const { bmp, w, h } = frame
    const zone_ids = zones.map(z=>z.id)
    const images = []
    for(const z of zones){
      const sx = Math.round(z.x*w), sy = Math.round(z.y*h)
      const sw = Math.round(z.w*w), sh = Math.round(z.h*h)
      images.push(await dataUrlFromBitmap(bmp, sx, sy, sw, sh))
    }
    const timestamp = String(Date.now())
    if(images.length) await api.ingest({ session_id: sessionId, timestamp, zone_ids, images })

    // Predict primary zone if set
    const pz = zones.find(z=>z.id===primaryId)
    if(pz){
      const sx = Math.round(pz.x*w), sy = Math.round(pz.y*h)
      const sw = Math.round(pz.w*w), sh = Math.round(pz.h*h)
      const canvas = new OffscreenCanvas(sw, sh)
      const ctx = canvas.getContext('2d')
      ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh)
      const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
      const pred = await api.predict(sessionId, blob)
      onPredict && onPredict(pred, blob)
    }
  }

  async function saveZones(){
    await api.saveZones(sessionId, zones)
    alert('Zones saved')
  }

  async function loadZones(){
    const z = await api.loadZones(sessionId)
    window.dispatchEvent(new CustomEvent('zones:load', { detail: z }))
  }

  async function consolidate(){
    await api.consolidate(sessionId)
    alert('Consolidated mosaics created.')
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Controls</div>
        <div className="space-x-2">
          <button className="btn" onClick={start} disabled={running}>Start</button>
          <button className="btn" onClick={stop} disabled={!running}>Stop</button>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <label>Interval (ms)</label>
        <input className="input" type="number" value={intervalMs} onChange={e=>setIntervalMs(+e.target.value)} />
        <button className="btn" onClick={saveZones}>Save Zones</button>
        <button className="btn" onClick={loadZones}>Load Zones</button>
        <button className="btn" onClick={consolidate}>Consolidate</button>
      </div>
    </div>
  )
}