// src/components/Controls.jsx
import React, { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

export default function Controls({ sessionId, zones, primaryId, captureHandle, onPredict }) {
  const [running, setRunning] = useState(false)
  const [intervalMin, setIntervalMin] = useState(1) // minutes

  // NEW: track whether screen sharing is active
  const [sharing, setSharing] = useState(false)

  const timerRef   = useRef(null)
  const inFlight   = useRef(false)
  const stoppedRef = useRef(true)

  // listen for ScreenCapture share status
  useEffect(() => {
    const onShare = (e) => {
      const { isSharing } = e.detail || {}
      setSharing(!!isSharing)
      // auto-stop if sharing ended while recording
      if (!isSharing) stop()
    }
    window.addEventListener('screenshare:change', onShare)
    return () => window.removeEventListener('screenshare:change', onShare)
  }, [])

  // graceful unmount
  useEffect(() => () => stop(), [])

  // Resolve capture handle (ref or direct object)
  function resolveHandle() {
    if (captureHandle && typeof captureHandle === 'object' && 'current' in captureHandle) {
      return captureHandle.current
    }
    if (captureHandle && typeof captureHandle.captureFrame === 'function') {
      return captureHandle
    }
    return null
  }

  // Only zones with positive size
  function getValidZones(list = zones) {
    return (list || []).filter(z => Number(z.w) > 0 && Number(z.h) > 0)
  }

  // Canvas helpers
  function makeCanvas(w, h) {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(Math.max(1, w), Math.max(1, h))
    const c = document.createElement('canvas')
    c.width  = Math.max(1, w)
    c.height = Math.max(1, h)
    return c
  }

  async function blobFromCanvas(canvas) {
    if ('convertToBlob' in canvas) return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
    return new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92))
  }

  async function dataUrlFromBitmap(bmp, sx, sy, sw, sh) {
    if (sw <= 0 || sh <= 0) return null
    const bw = bmp.width, bh = bmp.height
    const x = Math.max(0, Math.min(sx, bw))
    const y = Math.max(0, Math.min(sy, bh))
    const w = Math.min(sw, bw - x)
    const h = Math.min(sh, bh - y)
    if (w <= 0 || h <= 0) return null

    try {
      const canvas = makeCanvas(Math.round(w), Math.round(h))
      const ctx = canvas.getContext('2d')
      ctx.drawImage(bmp, x, y, w, h, 0, 0, canvas.width, canvas.height)
      const blob = await blobFromCanvas(canvas)
      const buf = await blob.arrayBuffer()
      const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)))
      return `data:image/jpeg;base64,${b64}`
    } catch (err) {
      console.error('Error creating data URL:', err)
      return null
    }
  }

  // Map overlay-normalized zone → video pixel rect using optional view info
  function zoneToVideoRect(z, frameW, frameH, view) {
    let nx = z.x, ny = z.y, nw = z.w, nh = z.h
    if (view && view.scaleX > 0 && view.scaleY > 0) {
      nx = (z.x - view.offX) / view.scaleX
      ny = (z.y - view.offY) / view.scaleY
      nw =  z.w              / view.scaleX
      nh =  z.h              / view.scaleY
    }
    const sx = Math.round(nx * frameW)
    const sy = Math.round(ny * frameH)
    const sw = Math.round(nw * frameW)
    const sh = Math.round(nh * frameH)
    return { sx, sy, sw, sh }
  }

  // Interval in ms (min ~1s)
  function getIntervalMs() {
    const m = Number.isFinite(+intervalMin) ? Math.max(0.016, +intervalMin) : 1
    return Math.round(m * 60 * 1000)
  }

  function start() {
    // NEW GUARD: do not start if not sharing
    if (!sharing) {
      alert('Please share your screen before starting capture.')
      return
    }
    const handle = resolveHandle()
    if (!handle) {
      alert('Please share your screen first.')
      return
    }
    if (getValidZones().length === 0) {
      console.warn('No valid zones from parent; will try zones from capture frame.')
    }
    stoppedRef.current = false
    setRunning(true)
    scheduleNext(0)
  }

  function stop() {
    stoppedRef.current = true
    setRunning(false)
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  function scheduleNext(delayMs) {
    if (timerRef.current) clearTimeout(timerRef.current)
    const ms = (typeof delayMs === 'number') ? Math.max(0, delayMs) : getIntervalMs()
    timerRef.current = setTimeout(loop, ms)
  }

  async function loop() {
    if (stoppedRef.current || inFlight.current) return
    inFlight.current = true

    try {
      const handle = resolveHandle()
      if (!handle || typeof handle.captureFrame !== 'function') {
        console.warn('Capture handle missing; stopping.')
        stop()
        return
      }

      const frame = await handle.captureFrame()
      if (!frame) { scheduleNext(getIntervalMs()); return }

      const {
        bmp, w, h, view,
        zones: frameZones,
        primaryId: framePrimaryId
      } = frame

      if (!bmp || !w || !h) { scheduleNext(getIntervalMs()); return }

      // Prefer parent zones; fallback to zones captured with the frame
      let zonesForUse = getValidZones()
      if (zonesForUse.length === 0 && Array.isArray(frameZones)) {
        zonesForUse = getValidZones(frameZones)
      }

      // Build crops
      const zone_ids = []
      const images   = []
      for (const z of zonesForUse) {
        const { sx, sy, sw, sh } = zoneToVideoRect(z, w, h, view)
        const dataUrl = await dataUrlFromBitmap(bmp, sx, sy, sw, sh)
        if (dataUrl) { zone_ids.push(z.id); images.push(dataUrl) }
      }

      if (images.length > 0) {
        const timestamp = String(Date.now())
        await api.ingest({ session_id: sessionId, timestamp, zone_ids, images })
      }

      // Optional: live prediction for primary zone
      let pz = zonesForUse.find(z => z.id === primaryId)
      if (!pz && typeof framePrimaryId === 'number') {
        pz = zonesForUse.find(z => z.id === framePrimaryId)
      }
      if (pz) {
        const { sx, sy, sw, sh } = zoneToVideoRect(pz, w, h, view)
        if (sw > 0 && sh > 0) {
          const canvas = makeCanvas(sw, sh)
          const ctx = canvas.getContext('2d')
          ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh)
          const blob = await blobFromCanvas(canvas)
          const pred = await api.predict(sessionId, blob)
          onPredict && onPredict(pred, blob)
        }
      }
    } catch (err) {
      console.error('Loop error:', err)
    } finally {
      inFlight.current = false
      if (!stoppedRef.current) scheduleNext(getIntervalMs())
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Capture Controls */}
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
              <span className="text-sm text-slate-400 w-28">Interval (minutes)</span>
              <input
                type="number"
                value={intervalMin}
                onChange={e => setIntervalMin(Math.max(0.016, +e.target.value || 1))}
                min="0.016"
                step="0.25"
                className="flex-1 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg focus:outline-none focus:border-blue-500 transition-colors text-slate-100"
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={start}
                disabled={running || !sharing} // NEW: disabled unless sharing
                className="flex-1 px-4 py-2.5 rounded-xl bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium shadow-lg shadow-green-900/20"
                title={sharing ? '' : 'Share your screen to enable capture'}
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
            {!sharing && (
              <div className="text-xs text-amber-300/80">
                Share your screen in the Screen Capture panel to enable recording.
              </div>
            )}
          </div>
        </div>

        {/* Zone Management (informational only) */}
        <div>
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Zone Management
          </h3>
          <p className="text-xs text-slate-400">
            Draw and manage zones in the Screen Capture panel. Zones are stored locally and used by the capture loop automatically.
          </p>
          {getValidZones().length < (zones?.length || 0) && (
            <p className="mt-1 text-xs text-amber-400">
              Note: {(zones?.length || 0) - getValidZones().length} invalid zone(s) will be ignored
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
