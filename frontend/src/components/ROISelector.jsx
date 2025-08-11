import React, { useEffect, useRef, useState } from 'react'

/**
 * Draw/select multiple rectangular ROIs over a reference size (video frame size)
 * Stores normalized coordinates [0..1] for scalability across resolutions.
 */
export default function ROISelector({ refSize, zones, setZones, primaryId, setPrimaryId }) {
  const canvasRef = useRef(null)
  const [drag, setDrag] = useState(null)

  const width = 960
  const height = Math.round(width * (refSize.h / Math.max(1, refSize.w)))

  useEffect(() => { draw() }, [zones, primaryId, refSize])

  function toPx(z) { return ({ x: z.x*width, y: z.y*height, w: z.w*width, h: z.h*height }) }
  function toNorm(x,y,w,h) { return ({ x: x/width, y: y/height, w: w/width, h: h/height }) }

  function draw() {
    const c = canvasRef.current; if (!c) return
    const ctx = c.getContext('2d')
    ctx.clearRect(0,0,c.width,c.height)
    // Dim background
    ctx.fillStyle = 'rgba(20,20,30,0.6)'; ctx.fillRect(0,0,c.width,c.height)
    // Draw zones
    zones.forEach(z => {
      const p = toPx(z)
      ctx.strokeStyle = z.id===primaryId ? '#22c55e' : '#60a5fa'
      ctx.lineWidth = 3
      ctx.strokeRect(p.x, p.y, p.w, p.h)
      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(p.x, p.y-24, 120, 24)
      ctx.fillStyle = 'white'
      ctx.fillText(`Zone ${z.id}${z.id===primaryId?' (primary)':''}`, p.x+6, p.y-8)
    })
  }

  function onDown(e){
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX-rect.left, y = e.clientY-rect.top
    setDrag({ x0:x, y0:y, x, y })
  }
  function onMove(e){
    if(!drag) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX-rect.left, y = e.clientY-rect.top
    setDrag(d => ({...d, x, y}))
  }
  function onUp(){
    if(!drag) return
    const x = Math.min(drag.x0, drag.x), y = Math.min(drag.y0, drag.y)
    const w = Math.abs(drag.x-drag.x0), h = Math.abs(drag.y-drag.y0)
    const nz = toNorm(x,y,w,h)
    const id = zones.length ? Math.max(...zones.map(z=>z.id))+1 : 0
    setZones([...zones, { id, ...nz }])
    setDrag(null)
  }

  function clearAll(){ setZones([]) }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Screenshot Zones (draw rectangles)</div>
        <div className="space-x-2">
          <button className="btn" onClick={clearAll}>Clear</button>
        </div>
      </div>
      <div className="relative">
        <canvas ref={canvasRef} width={width} height={height}
          onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp}
          className="rounded-xl border border-slate-800" />
      </div>
      <div className="mt-3 space-y-2">
        {zones.map(z => (
          <div key={z.id} className="flex items-center justify-between bg-slate-900 p-2 rounded-xl">
            <div>Zone {z.id}</div>
            <div className="space-x-2">
              <button className="btn" onClick={()=>setPrimaryId(z.id)}>Set Primary</button>
              <button className="btn" onClick={()=>setZones(zones.filter(q=>q.id!==z.id))}>Delete</button>
            </div>
          </div>
        ))}
        {zones.length>0 && (
          <div className="text-sm text-slate-400">Primary zone is used for real‑time prediction overlay.</div>
        )}
      </div>
    </div>
  )
}