import React, { useEffect, useRef, useState } from 'react'

export default function ROISelector({
  refSize,
  zones,
  setZones,
  primaryId,
  setPrimaryId,
  showBackground = false,
  overlayMode = true,
  videoGeom = null,            // <-- NEW: from ScreenCapture (ref with converters)
}) {
  const canvasRef = useRef(null)
  const [drag, setDrag] = useState(null)
  const [hoveredZone, setHoveredZone] = useState(null)
  const [canvasSize, setCanvasSize] = useState({ width: 960, height: 540 })

  // Canvas size should match the overlay container exactly
  useEffect(() => {
    const updateCanvasSize = () => {
      const c = canvasRef.current
      const parent = c?.parentElement
      if (!parent) return
      const rect = parent.getBoundingClientRect()
      setCanvasSize({ width: Math.max(1, rect.width), height: Math.max(1, rect.height) })
    }
    updateCanvasSize()
    const ro = new ResizeObserver(updateCanvasSize)
    if (canvasRef.current?.parentElement) ro.observe(canvasRef.current.parentElement)
    window.addEventListener('resize', updateCanvasSize)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', updateCanvasSize)
    }
  }, [])

  useEffect(() => {
    draw()
  }, [zones, primaryId, drag, hoveredZone, showBackground, overlayMode, canvasSize])

  // ---- Mapping helpers ----
  // Zones are stored normalized to VIDEO pixels (x,y,w,h ∈ [0,1] w.r.t. videoWidth/Height).
  // Convert a stored zone to CANVAS/CSS pixels for drawing.
  function zoneVideoToCanvasPx(z) {
    const g = videoGeom?.current
    if (g) {
      const sx = g.width / g.vw
      const sy = g.height / g.vh
      const x = g.x + (z.x * g.vw) * sx
      const y = g.y + (z.y * g.vh) * sy
      const w = (z.w * g.vw) * sx
      const h = (z.h * g.vh) * sy
      return { x, y, w, h }
    }
    // Fallback (no geometry yet): assume zones were saved in canvas-normalized coords
    return {
      x: z.x * canvasSize.width,
      y: z.y * canvasSize.height,
      w: z.w * canvasSize.width,
      h: z.h * canvasSize.height,
    }
  }

  // Convert CANVAS/CSS pixel coords to normalized VIDEO coords
  function canvasToVideoNorm(x, y) {
    const g = videoGeom?.current
    if (g) {
      // clamp to displayed video rect
      const cx = Math.min(Math.max(x, g.x), g.x + g.width)
      const cy = Math.min(Math.max(y, g.y), g.y + g.height)
      const vx = g.toVideoX(cx)
      const vy = g.toVideoY(cy)
      return { x: vx / g.vw, y: vy / g.vh }
    }
    // Fallback: treat canvas as the reference
    return { x: x / canvasSize.width, y: y / canvasSize.height }
  }

  // Drawing
  function draw() {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')

    c.width = canvasSize.width
    c.height = canvasSize.height

    ctx.clearRect(0, 0, c.width, c.height)

    // Optional background (kept for parity with your original prop)
    if (!overlayMode || showBackground) {
      const gradient = ctx.createLinearGradient(0, 0, canvasSize.width, canvasSize.height)
      gradient.addColorStop(0, 'rgba(15, 23, 42, 0.9)')
      gradient.addColorStop(1, 'rgba(30, 41, 59, 0.9)')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvasSize.width, canvasSize.height)

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)'
      ctx.lineWidth = 1
      const gridSize = 40
      for (let i = 0; i <= canvasSize.width; i += gridSize) {
        ctx.beginPath()
        ctx.moveTo(i, 0)
        ctx.lineTo(i, canvasSize.height)
        ctx.stroke()
      }
      for (let i = 0; i <= canvasSize.height; i += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, i)
        ctx.lineTo(canvasSize.width, i)
        ctx.stroke()
      }
    }

    // Draw zones (mapped to displayed video area)
    zones.forEach(z => {
      if (z.w <= 0 || z.h <= 0) return
      const p = zoneVideoToCanvasPx(z)
      const isPrimary = z.id === primaryId
      const isHovered = z.id === hoveredZone

      ctx.fillStyle = isPrimary
        ? 'rgba(34, 197, 94, 0.15)'
        : isHovered
        ? 'rgba(96, 165, 250, 0.2)'
        : 'rgba(96, 165, 250, 0.1)'
      ctx.fillRect(p.x, p.y, p.w, p.h)

      if (isPrimary) {
        ctx.shadowBlur = 15
        ctx.shadowColor = '#22c55e'
      } else if (isHovered) {
        ctx.shadowBlur = 8
        ctx.shadowColor = '#60a5fa'
      }
      ctx.strokeStyle = isPrimary ? '#22c55e' : '#60a5fa'
      ctx.lineWidth = isPrimary ? 3 : 2
      ctx.strokeRect(p.x, p.y, p.w, p.h)
      ctx.shadowBlur = 0

      // Corner handles
      const handleSize = 8
      ctx.fillStyle = '#ffffff'
      ctx.strokeStyle = isPrimary ? '#22c55e' : '#60a5fa'
      ctx.lineWidth = 2
      const drawHandle = (x, y) => {
        ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize)
        ctx.strokeRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize)
      }
      drawHandle(p.x, p.y)
      drawHandle(p.x + p.w, p.y)
      drawHandle(p.x, p.y + p.h)
      drawHandle(p.x + p.w, p.y + p.h)

      // Zone label
      const label = `Zone ${z.id}${isPrimary ? ' • PRIMARY' : ''}`
      ctx.font = 'bold 12px system-ui'
      const metrics = ctx.measureText(label)
      const labelPadding = 6
      const labelHeight = 20
      const labelWidth = metrics.width + labelPadding * 2
      const labelY = p.y - 8

      ctx.fillStyle = isPrimary ? 'rgba(34, 197, 94, 0.9)' : 'rgba(15, 23, 42, 0.9)'
      ctx.fillRect(p.x, labelY - labelHeight, labelWidth, labelHeight)
      ctx.fillStyle = isPrimary ? '#ffffff' : '#e2e8f0'
      ctx.fillText(label, p.x + labelPadding, labelY - 6)

      // Dimensions (canvas pixels)
      const dimText = `${Math.round(p.w)}×${Math.round(p.h)}px`
      ctx.font = '11px system-ui'
      const dimMetrics = ctx.measureText(dimText)
      const dimPadding = 4
      const dimBgWidth = dimMetrics.width + dimPadding * 2
      const dimBgHeight = 18

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(p.x + p.w - dimBgWidth - 4, p.y + p.h - dimBgHeight - 4, dimBgWidth, dimBgHeight)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(dimText, p.x + p.w - dimBgWidth + dimPadding - 4, p.y + p.h - 8)
    })

    // Draw current drag (in canvas coordinates)
    if (drag && Math.abs(drag.x - drag.x0) > 5 && Math.abs(drag.y - drag.y0) > 5) {
      const x = Math.min(drag.x0, drag.x)
      const y = Math.min(drag.y0, drag.y)
      const w = Math.abs(drag.x - drag.x0)
      const h = Math.abs(drag.y - drag.y0)

      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
      ctx.fillRect(x, y, w, h)
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.setLineDash([8, 4])
      ctx.strokeRect(x, y, w, h)
      ctx.setLineDash([])

      const dimText = `${Math.round(w)}×${Math.round(h)}px`
      ctx.font = '12px system-ui'
      const textWidth = ctx.measureText(dimText).width
      ctx.fillStyle = 'rgba(15, 23, 42, 0.9)'
      ctx.fillRect(x + w / 2 - textWidth / 2 - 5, y + h / 2 - 10, textWidth + 10, 20)
      ctx.fillStyle = '#3b82f6'
      ctx.fillText(dimText, x + w / 2 - textWidth / 2, y + h / 2 + 2)
    }
  }

  // Mouse helpers
  function getCanvasXY(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    // Because we set canvas width/height to its CSS size, 1:1 mapping is fine.
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    return { x, y }
  }

  function onDown(e) {
    const { x, y } = getCanvasXY(e)
    const g = videoGeom?.current
    // Only start a drag if inside the displayed video rectangle (prevents drift)
    if (g) {
      if (x < g.x || x > g.x + g.width || y < g.y || y > g.y + g.height) return
    }
    setDrag({ x0: x, y0: y, x, y })
  }

  function onMove(e) {
    const { x, y } = getCanvasXY(e)
    if (drag) {
      setDrag(d => ({ ...d, x, y }))
      return
    }
    const hovered = zones.find(z => {
      if (z.w <= 0 || z.h <= 0) return false
      const p = zoneVideoToCanvasPx(z)
      return x >= p.x && x <= p.x + p.w && y >= p.y && y <= p.y + p.h
    })
    setHoveredZone(hovered?.id ?? null)
  }

  function onUp() {
    if (!drag) return

    const x = Math.min(drag.x0, drag.x)
    const y = Math.min(drag.y0, drag.y)
    const w = Math.abs(drag.x - drag.x0)
    const h = Math.abs(drag.y - drag.y0)

    if (w < 20 || h < 20) {
      setDrag(null)
      return
    }

    // Convert both corners to normalized VIDEO coords, clamp handled inside
    const tl = canvasToVideoNorm(x, y)
    const br = canvasToVideoNorm(x + w, y + h)
    const nz = {
      x: Math.min(tl.x, br.x),
      y: Math.min(tl.y, br.y),
      w: Math.abs(br.x - tl.x),
      h: Math.abs(br.y - tl.y),
    }

    const id = zones.length ? Math.max(...zones.map(z => z.id)) + 1 : 0
    setZones([...zones, { id, ...nz }])
    if (zones.length === 0) setPrimaryId(id)

    setDrag(null)
  }

  // Overlay mode (used over the <video/>)
  if (overlayMode) {
    return (
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => {
          setHoveredZone(null)
          if (drag) setDrag(null)
        }}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          cursor: 'crosshair',
          pointerEvents: 'auto',
          zIndex: 10,
        }}
      />
    )
  }

  // Standalone mode (kept intact)
  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          Detection Zones
        </h3>
        <button
          onClick={() => { setZones([]); setPrimaryId(null) }}
          disabled={zones.length === 0}
          className="px-4 py-2 rounded-xl bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
        >
          Clear All
        </button>
      </div>

      <div className="relative rounded-xl overflow-hidden mb-4 border border-slate-700/50">
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => {
            setHoveredZone(null)
            if (drag) setDrag(null)
          }}
          className="w-full cursor-crosshair"
          style={{ display: 'block' }}
        />
      </div>
      {/* your existing zone list UI stays as-is */}
    </div>
  )
}
