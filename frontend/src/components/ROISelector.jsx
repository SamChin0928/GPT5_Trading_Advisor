import React, { useEffect, useRef, useState } from 'react'

export default function ROISelector({ 
  refSize, 
  zones, 
  setZones, 
  primaryId, 
  setPrimaryId,
  showBackground = false,
  overlayMode = true
}) {
  const canvasRef = useRef(null)
  const [drag, setDrag] = useState(null)
  const [hoveredZone, setHoveredZone] = useState(null)
  const [canvasSize, setCanvasSize] = useState({ width: 960, height: 540 })

  // Calculate canvas dimensions based on container
  useEffect(() => {
    const updateCanvasSize = () => {
      if (canvasRef.current && canvasRef.current.parentElement) {
        const parent = canvasRef.current.parentElement
        const rect = parent.getBoundingClientRect()
        // Maintain aspect ratio
        const aspectRatio = refSize.h / Math.max(1, refSize.w)
        const width = rect.width
        const height = width * aspectRatio
        setCanvasSize({ width, height })
      }
    }

    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)
    return () => window.removeEventListener('resize', updateCanvasSize)
  }, [refSize])

  useEffect(() => { 
    draw() 
  }, [zones, primaryId, refSize, drag, hoveredZone, showBackground, overlayMode, canvasSize])

  function toPx(z) { 
    return ({ 
      x: z.x * canvasSize.width, 
      y: z.y * canvasSize.height, 
      w: z.w * canvasSize.width, 
      h: z.h * canvasSize.height 
    }) 
  }
  
  function toNorm(x, y, w, h) { 
    return ({ 
      x: x / canvasSize.width, 
      y: y / canvasSize.height, 
      w: w / canvasSize.width, 
      h: h / canvasSize.height 
    }) 
  }

  function draw() {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')
    
    // Set canvas resolution
    c.width = canvasSize.width
    c.height = canvasSize.height
    
    ctx.clearRect(0, 0, c.width, c.height)
    
    // Only draw background if not in overlay mode or explicitly requested
    if (!overlayMode || showBackground) {
      // Background gradient
      const gradient = ctx.createLinearGradient(0, 0, canvasSize.width, canvasSize.height)
      gradient.addColorStop(0, 'rgba(15, 23, 42, 0.9)')
      gradient.addColorStop(1, 'rgba(30, 41, 59, 0.9)')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvasSize.width, canvasSize.height)
      
      // Grid pattern
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)'
      ctx.lineWidth = 1
      const gridSize = 40
      for(let i = 0; i <= canvasSize.width; i += gridSize) {
        ctx.beginPath()
        ctx.moveTo(i, 0)
        ctx.lineTo(i, canvasSize.height)
        ctx.stroke()
      }
      for(let i = 0; i <= canvasSize.height; i += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, i)
        ctx.lineTo(canvasSize.width, i)
        ctx.stroke()
      }
    }
    
    // Draw zones
    zones.forEach(z => {
      if(z.w <= 0 || z.h <= 0) return
      
      const p = toPx(z)
      const isPrimary = z.id === primaryId
      const isHovered = z.id === hoveredZone
      
      // Zone fill
      ctx.fillStyle = isPrimary 
        ? 'rgba(34, 197, 94, 0.15)' 
        : isHovered 
        ? 'rgba(96, 165, 250, 0.2)'
        : 'rgba(96, 165, 250, 0.1)'
      ctx.fillRect(p.x, p.y, p.w, p.h)
      
      // Zone border with glow
      if(isPrimary) {
        ctx.shadowBlur = 15
        ctx.shadowColor = '#22c55e'
      } else if(isHovered) {
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
        ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize)
        ctx.strokeRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize)
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
      
      // Dimensions
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
    
    // Draw current drag
    if(drag && Math.abs(drag.x - drag.x0) > 5 && Math.abs(drag.y - drag.y0) > 5) {
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
      ctx.fillStyle = '#3b82f6'
      const textWidth = ctx.measureText(dimText).width
      ctx.fillStyle = 'rgba(15, 23, 42, 0.9)'
      ctx.fillRect(x + w/2 - textWidth/2 - 5, y + h/2 - 10, textWidth + 10, 20)
      ctx.fillStyle = '#3b82f6'
      ctx.fillText(dimText, x + w/2 - textWidth/2, y + h/2 + 2)
    }
  }

  function onDown(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    const scaleX = canvasSize.width / rect.width
    const scaleY = canvasSize.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    setDrag({ x0: x, y0: y, x, y })
  }

  function onMove(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    const scaleX = canvasSize.width / rect.width
    const scaleY = canvasSize.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    
    if(drag) {
      setDrag(d => ({...d, x, y}))
    } else {
      const hovered = zones.find(z => {
        if(z.w <= 0 || z.h <= 0) return false
        const p = toPx(z)
        return x >= p.x && x <= p.x + p.w && y >= p.y && y <= p.y + p.h
      })
      setHoveredZone(hovered?.id ?? null)
    }
  }

  function onUp() {
    if(!drag) return
    
    const x = Math.min(drag.x0, drag.x)
    const y = Math.min(drag.y0, drag.y)
    const w = Math.abs(drag.x - drag.x0)
    const h = Math.abs(drag.y - drag.y0)
    
    if(w < 20 || h < 20) {
      setDrag(null)
      return
    }
    
    const nz = toNorm(x, y, w, h)
    const id = zones.length ? Math.max(...zones.map(z => z.id)) + 1 : 0
    setZones([...zones, { id, ...nz }])
    
    if(zones.length === 0) {
      setPrimaryId(id)
    }
    
    setDrag(null)
  }

  // Return just the canvas overlay if in overlay mode
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
          if(drag) setDrag(null)
        }}
        style={{ 
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          cursor: 'crosshair',
          pointerEvents: 'auto',
          zIndex: 10
        }}
      />
    )
  }

  // Original standalone component layout
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
          onClick={() => {
            setZones([])
            setPrimaryId(null)
          }}
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
            if(drag) setDrag(null)
          }}
          className="w-full cursor-crosshair"
          style={{ display: 'block' }}
        />
      </div>
      
      {/* Zone list UI remains the same... */}
    </div>
  )
}