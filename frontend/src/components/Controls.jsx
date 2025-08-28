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

  // SYNC-TO-MINUTE: track the next *wall-clock* capture time (ms since epoch)
  const nextAtRef  = useRef(null)

  // NEW: simple countdown UI (ms until nextAtRef)
  const [countdownMs, setCountdownMs] = useState(null)

  // listen for ScreenCapture share status
  useEffect(() => {
    const onShare = (e) => {
      const { isSharing } = (e?.detail) || {}
      setSharing(!!isSharing)
      // auto-stop if sharing ended while recording
      if (!isSharing) stop()
    }
    window.addEventListener('screenshare:change', onShare)
    return () => window.removeEventListener('screenshare:change', onShare)
  }, [])

  // graceful unmount
  useEffect(() => () => stop(), [])

  // If interval changes while running, re-align to the next grid tick
  useEffect(() => {
    if (!running) return
    alignAndScheduleNext()
  }, [intervalMin, running])

  // NEW: live countdown tick
  useEffect(() => {
    if (!running) { setCountdownMs(null); return }
    const id = setInterval(() => {
      const now = Date.now()
      const target = nextAtRef.current || computeNextAlignedTime(now)
      setCountdownMs(Math.max(0, target - now))
    }, 200)
    return () => clearInterval(id)
  }, [running, intervalMin])

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
    const validZones = (list || []).filter(z => {
      // More lenient validation - check if w and h exist and are > 0 OR are between 0-1 (normalized)
      const hasValidDimensions = (
        (Number(z.w) > 0 && Number(z.h) > 0) || 
        (z.w > 0 && z.w <= 1 && z.h > 0 && z.h <= 1)
      );
      if (!hasValidDimensions) {
        console.warn('[Controls] Invalid zone dimensions:', z);
      }
      return hasValidDimensions;
    });
    console.log('[Controls] Valid zones:', validZones.length, 'from', list?.length || 0, 'total zones');
    return validZones;
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

  // SYNC-TO-MINUTE: compute the next capture time aligned to the interval grid
  function computeNextAlignedTime(fromMs) {
    const interval = getIntervalMs()
    // Don't align to minute boundaries for intervals < 60 seconds
    // Only align to minute boundaries for intervals >= 1 minute
    if (interval >= 60000) {
      // Align to next minute boundary
      const d = new Date(fromMs)
      d.setSeconds(0, 0) // Reset seconds and milliseconds
      let target = d.getTime()
      // Move to next minute if we're past the current minute start
      if (target <= fromMs) {
        target += 60000
      }
      return target
    } else {
      // For sub-minute intervals, just use the interval from now
      return fromMs + interval
    }
  }

  function start() {
    console.log('[Controls] Starting capture with zones:', zones?.length, 'primary:', primaryId)
    
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
    const validZones = getValidZones(zones)
    if (validZones.length === 0) {
      console.warn('[Controls] No valid zones from parent; will try zones from capture frame.')
    } else {
      console.log('[Controls] Starting with', validZones.length, 'valid zones')
    }
    stoppedRef.current = false
    setRunning(true)
    alignAndScheduleNext() // SYNC-TO-MINUTE: first shot at exact boundary
  }

  function stop() {
    stoppedRef.current = true
    setRunning(false)
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    nextAtRef.current = null
    setCountdownMs(null) // NEW
  }

  // SYNC-TO-MINUTE: schedule next run based on nextAtRef
  function alignAndScheduleNext() {
    const now = Date.now()
    // If we already have a target (e.g., resume after changing interval), re-align forward
    let target = nextAtRef.current
    const interval = getIntervalMs()
    if (!target) {
      // first alignment: next grid tick (e.g., 09:52:00.000 if now is 09:51:52)
      target = computeNextAlignedTime(now)
    } else {
      // ensure target is still in the future for the (possibly) new interval
      const missed = Math.max(0, Math.ceil((now - target) / interval))
      target = target + missed * interval
    }
    nextAtRef.current = target
    setCountdownMs(Math.max(0, target - now)) // NEW
    scheduleNext(Math.max(0, target - now))
  }

  function scheduleNext(delayMs) {
    if (timerRef.current) clearTimeout(timerRef.current)
    const ms = (typeof delayMs === 'number') ? Math.max(0, delayMs) : getIntervalMs()
    timerRef.current = setTimeout(loop, ms)
  }

  async function loop() {
    if (stoppedRef.current || inFlight.current) return
    inFlight.current = true

    // Guard: if somehow triggered early/late, keep captures stamped to the planned boundary
    const plannedAt = nextAtRef.current || computeNextAlignedTime(Date.now())

    try {
      const handle = resolveHandle()
      if (!handle || typeof handle.captureFrame !== 'function') {
        console.warn('[Controls] Capture handle missing; stopping.')
        stop()
        return
      }

      const frame = await handle.captureFrame()
      if (!frame) { 
        console.warn('[Controls] No frame captured')
        finalizeAndPlanNext()
        return 
      }

      const {
        bmp, w, h, view,
        zones: frameZones,
        primaryId: framePrimaryId
      } = frame

      console.log('[Controls] Frame captured:', { w, h, hasView: !!view, frameZones: frameZones?.length, framePrimaryId })

      if (!bmp || !w || !h) { 
        console.warn('[Controls] Invalid frame data:', { hasBmp: !!bmp, w, h })
        finalizeAndPlanNext()
        return 
      }

      // CRITICAL FIX: Prefer frameZones from ScreenCapture, then fallback to parent zones
      let zonesForUse = []
      
      // First try zones from the frame (these come directly from ScreenCapture)
      if (Array.isArray(frameZones) && frameZones.length > 0) {
        console.log('[Controls] Using zones from frame:', frameZones)
        zonesForUse = frameZones.filter(z => z && Number(z.w) > 0 && Number(z.h) > 0)
      } 
      // Fallback to parent zones prop if no frame zones
      else if (Array.isArray(zones) && zones.length > 0) {
        console.log('[Controls] Using zones from props:', zones)
        zonesForUse = zones.filter(z => z && Number(z.w) > 0 && Number(z.h) > 0)
      }
      
      console.log('[Controls] Zones to use for capture:', zonesForUse.length)

      if (zonesForUse.length === 0) {
        console.warn('[Controls] No valid zones available for capture')
        finalizeAndPlanNext()
        return
      }

      // Build crops
      const zone_ids = []
      const images = []
      console.log('[Controls] Processing zones for capture:', zonesForUse)
      
      for (const z of zonesForUse) {
        // FIXED: Handle both normalized (0-1) and pixel coordinates
        let sx, sy, sw, sh
        
        // Check if normalized (all values between 0 and 1)
        if (z.x <= 1 && z.y <= 1 && z.w <= 1 && z.h <= 1) {
          // Normalized coordinates - convert to pixels
          sx = Math.round(z.x * w)
          sy = Math.round(z.y * h)
          sw = Math.round(z.w * w)
          sh = Math.round(z.h * h)
          console.log('[Controls] Zone', z.id, 'normalized:', { x: z.x, y: z.y, w: z.w, h: z.h }, '-> pixels:', { sx, sy, sw, sh })
        } else {
          // Already in pixel coordinates
          sx = Math.round(z.x)
          sy = Math.round(z.y)
          sw = Math.round(z.w)
          sh = Math.round(z.h)
          console.log('[Controls] Zone', z.id, 'pixels:', { sx, sy, sw, sh })
        }
        
        // Ensure we have valid dimensions
        sx = Math.max(0, sx)
        sy = Math.max(0, sy)
        sw = Math.min(sw, w - sx)
        sh = Math.min(sh, h - sy)
        
        if (sw > 0 && sh > 0) {
          const dataUrl = await dataUrlFromBitmap(bmp, sx, sy, sw, sh)
          if (dataUrl) { 
            zone_ids.push(z.id)
            images.push(dataUrl)
            console.log('[Controls] Successfully captured zone', z.id)
          } else {
            console.warn('[Controls] Failed to capture zone', z.id)
          }
        } else {
          console.warn('[Controls] Invalid dimensions for zone', z.id, { sx, sy, sw, sh })
        }
      }

      if (images.length > 0) {
        // EXACT-MINUTE-TS: stamp the folder with the *boundary* time, not "now"
        const timestamp = String(plannedAt)
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone
        
        console.log('[Controls] Preparing to ingest:', {
          session_id: sessionId,
          timestamp,
          zone_ids,
          num_images: images.length,
          process: api.getProcess(),
          tz
        })
        
        try {
          const result = await api.ingest({ 
            session_id: sessionId, 
            timestamp, 
            zone_ids, 
            images, 
            tz 
          })
          console.log('[Controls] Ingest success:', result)
        } catch (err) {
          console.error('[Controls] Ingest failed:', err)
          alert(`Failed to save screenshots: ${err.message || 'Unknown error'}`)
        }
        
        // Broadcast so Labeler refreshes immediately
        try { localStorage.setItem('captures_updated_at', String(Date.now())) } catch {}
        try { window.dispatchEvent(new CustomEvent('captures:updated', { detail: { session_id: sessionId, timestamp } })) } catch {}
      } else {
        console.warn('[Controls] No images captured - zones might be invalid')
      }

      // Optional: live prediction for primary zone
      let pz = zonesForUse.find(z => z.id === primaryId || z.isPrimary)
      if (!pz && typeof framePrimaryId === 'number') {
        pz = zonesForUse.find(z => z.id === framePrimaryId)
      }
      
      if (pz) {
        // Handle normalized vs pixel coordinates for prediction too
        let sx, sy, sw, sh
        
        if (pz.x <= 1 && pz.y <= 1 && pz.w <= 1 && pz.h <= 1) {
          sx = Math.round(pz.x * w)
          sy = Math.round(pz.y * h)
          sw = Math.round(pz.w * w)
          sh = Math.round(pz.h * h)
        } else {
          sx = Math.round(pz.x)
          sy = Math.round(pz.y)
          sw = Math.round(pz.w)
          sh = Math.round(pz.h)
        }
        
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
      console.error('[Controls] Loop error:', err)
    } finally {
      inFlight.current = false
      finalizeAndPlanNext()
    }
  }


  // SYNC-TO-MINUTE: move nextAt forward exactly one interval (catching up if needed)
  function finalizeAndPlanNext() {
    if (stoppedRef.current) return
    const interval = getIntervalMs()
    const now = Date.now()
    let nextAt = (nextAtRef.current || computeNextAlignedTime(now)) + interval
    if (nextAt <= now) {
      const missed = Math.floor((now - nextAt) / interval) + 1
      nextAt += missed * interval
    }
    nextAtRef.current = nextAt
    setCountdownMs(Math.max(0, nextAt - now)) // NEW
    scheduleNext(Math.max(0, nextAt - now))
  }

  function fmtDur(ms) {
    const s = Math.max(0, Math.floor(ms / 1000))
    const mm = Math.floor(s / 60).toString().padStart(2, '0')
    const ss = (s % 60).toString().padStart(2, '0')
    return `${mm}:${ss}`
  }
  const nextHuman = (() => {
    if (!countdownMs || !nextAtRef.current) return ''
    try {
      const d = new Date(nextAtRef.current)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch { return '' }
  })()

  // Progress % until next shot (for a tiny bar)
  const pct = (() => {
    if (!running || !countdownMs || !nextAtRef.current) return 0
    const interval = getIntervalMs()
    const elapsed = interval - Math.min(interval, countdownMs)
    return Math.max(0, Math.min(100, Math.round((elapsed / interval) * 100)))
  })()

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

            {/* NEW: Next-capture countdown */}
            {running && (
              <div className="rounded-lg border border-white/10 bg-slate-800/50 p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-300">Next capture</span>
                  <span className="text-slate-200 font-medium">{nextHuman}</span>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <div className="h-1.5 w-full bg-slate-700/60 rounded-full overflow-hidden border border-white/10">
                    <div className="h-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="ml-3 text-xs text-slate-300 tabular-nums">{fmtDur(countdownMs || 0)}</span>
                </div>
              </div>
            )}

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
