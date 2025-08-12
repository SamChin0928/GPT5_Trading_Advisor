import React, { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

export default function Controls({ sessionId, zones, primaryId, captureHandle, onPredict }) {
  const [running, setRunning] = useState(false)
  const [intervalMs, setIntervalMs] = useState(1000)
  const [consolidating, setConsolidating] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => { return () => stop() }, [])

  // Filter out invalid zones (with 0 width or height)
  function getValidZones() {
    return zones.filter(z => z.w > 0 && z.h > 0)
  }

  async function dataUrlFromBitmap(bmp, sx, sy, sw, sh) {
    // Validate dimensions to prevent IndexSizeError
    if(sw <= 0 || sh <= 0) {
      console.warn('Invalid zone dimensions:', { sx, sy, sw, sh })
      return null
    }
    
    // Ensure we're within bounds
    const bmpWidth = bmp.width
    const bmpHeight = bmp.height
    
    sx = Math.max(0, Math.min(sx, bmpWidth))
    sy = Math.max(0, Math.min(sy, bmpHeight))
    sw = Math.min(sw, bmpWidth - sx)
    sh = Math.min(sh, bmpHeight - sy)
    
    if(sw <= 0 || sh <= 0) {
      console.warn('Zone extends beyond bitmap bounds')
      return null
    }
    
    try {
      const canvas = new OffscreenCanvas(Math.max(1, Math.round(sw)), Math.max(1, Math.round(sh)))
      const ctx = canvas.getContext('2d')
      ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height)
      const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
      const buf = await blob.arrayBuffer()
      const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)))
      return `data:image/jpeg;base64,${b64}`
    } catch(err) {
      console.error('Error creating data URL:', err)
      return null
    }
  }

  function start() {
    const validZones = getValidZones()
    
    if(!captureHandle) {
      alert('Please share your screen first.')
      return
    }
    
    if(validZones.length === 0) {
      alert('Please draw at least one valid zone (with non-zero width and height).')
      return
    }
    
    setRunning(true)
    loop()
    timerRef.current = setInterval(loop, intervalMs)
  }

  function stop() {
    setRunning(false)
    if(timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  async function loop() {
    try {
      const frame = await captureHandle.captureFrame()
      if(!frame) return
      
      const { bmp, w, h } = frame
      if(w === 0 || h === 0) return
      
      const validZones = getValidZones()
      const zone_ids = []
      const images = []
      
      for(const z of validZones) {
        const sx = Math.round(z.x * w)
        const sy = Math.round(z.y * h)
        const sw = Math.round(z.w * w)
        const sh = Math.round(z.h * h)
        
        const dataUrl = await dataUrlFromBitmap(bmp, sx, sy, sw, sh)
        if(dataUrl) {
          zone_ids.push(z.id)
          images.push(dataUrl)
        }
      }
      
      const timestamp = String(Date.now())
      if(images.length > 0) {
        await api.ingest({ session_id: sessionId, timestamp, zone_ids, images })
      }

      // Predict primary zone if set and valid
      const pz = validZones.find(z => z.id === primaryId)
      if(pz) {
        const sx = Math.round(pz.x * w)
        const sy = Math.round(pz.y * h)
        const sw = Math.round(pz.w * w)
        const sh = Math.round(pz.h * h)
        
        if(sw > 0 && sh > 0) {
          // Ensure bounds
          const validSx = Math.max(0, Math.min(sx, w))
          const validSy = Math.max(0, Math.min(sy, h))
          const validSw = Math.min(sw, w - validSx)
          const validSh = Math.min(sh, h - validSy)
          
          if(validSw > 0 && validSh > 0) {
            const canvas = new OffscreenCanvas(validSw, validSh)
            const ctx = canvas.getContext('2d')
            ctx.drawImage(bmp, validSx, validSy, validSw, validSh, 0, 0, validSw, validSh)
            const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
            const pred = await api.predict(sessionId, blob)
            onPredict && onPredict(pred, blob)
          }
        }
      }
    } catch(err) {
      console.error('Loop error:', err)
      // Don't stop on error, just continue
    }
  }

  async function saveZones() {
    // Only save valid zones
    const validZones = getValidZones()
    await api.saveZones(sessionId, validZones)
    alert('Zones saved successfully!')
  }

  async function loadZones() {
    const z = await api.loadZones(sessionId)
    // Filter out invalid zones when loading
    const validZones = z.filter(zone => zone.w > 0 && zone.h > 0)
    window.dispatchEvent(new CustomEvent('zones:load', { detail: validZones }))
    if(z.length !== validZones.length) {
      console.warn(`Filtered out ${z.length - validZones.length} invalid zones`)
    }
  }

  async function consolidate() {
    setConsolidating(true)
    try {
      await api.consolidate(sessionId)
      alert('Consolidated mosaics created successfully!')
    } catch(err) {
      alert('Consolidation failed: ' + err.message)
    } finally {
      setConsolidating(false)
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            Capture Controls
          </h3>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-400 w-24">Interval (ms)</span>
              <input 
                type="number" 
                value={intervalMs} 
                onChange={e => setIntervalMs(Math.max(100, +e.target.value))}
                min="100"
                step="100"
                className="flex-1 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg focus:outline-none focus:border-blue-500 transition-colors text-slate-100"
              />
            </div>
            <div className="flex gap-3">
              <button 
                onClick={start} 
                disabled={running}
                className="flex-1 px-4 py-2.5 rounded-xl bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium shadow-lg shadow-green-900/20"
              >
                {running ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    Recording
                  </span>
                ) : 'Start Capture'}
              </button>
              <button 
                onClick={stop} 
                disabled={!running}
                className="flex-1 px-4 py-2.5 rounded-xl bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
              >
                Stop
              </button>
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Zone Management
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <button 
              onClick={saveZones}
              disabled={zones.length === 0}
              className="px-4 py-2.5 rounded-xl bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
            >
              Save Zones
            </button>
            <button 
              onClick={loadZones}
              className="px-4 py-2.5 rounded-xl bg-slate-700 hover:bg-slate-600 transition-all duration-200 font-medium"
            >
              Load Zones
            </button>
            <button 
              onClick={consolidate}
              disabled={consolidating}
              className="col-span-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium shadow-lg shadow-purple-900/20"
            >
              {consolidating ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Processing...
                </span>
              ) : 'Consolidate Captures'}
            </button>
          </div>
          {getValidZones().length < zones.length && (
            <p className="mt-2 text-xs text-amber-400">
              Note: {zones.length - getValidZones().length} invalid zone(s) will be ignored
            </p>
          )}
        </div>
      </div>
    </div>
  )
}