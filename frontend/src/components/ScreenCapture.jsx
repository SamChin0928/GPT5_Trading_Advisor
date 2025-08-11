import React, { useEffect, useRef, useState } from 'react'

/**
 * Displays the selected screen via getDisplayMedia in a <video>,
 * and exposes a capture() method that returns an ImageBitmap of the current frame.
 */
export default function ScreenCapture({ onReady }) {
  const videoRef = useRef(null)
  const [streaming, setStreaming] = useState(false)

  useEffect(() => { onReady && onReady({ captureFrame }) }, [])

  async function start() {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30 }, audio: false
    })
    videoRef.current.srcObject = stream
    await videoRef.current.play()
    setStreaming(true)
  }

  function stop() {
    const stream = videoRef.current?.srcObject
    stream?.getTracks().forEach(t => t.stop())
    setStreaming(false)
  }

  async function captureFrame() {
    if (!videoRef.current) return null
    const video = videoRef.current
    const w = video.videoWidth, h = video.videoHeight
    const canvas = new OffscreenCanvas(w, h)
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, w, h)
    const bmp = await createImageBitmap(canvas)
    return { bmp, w, h }
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Screen Capture</div>
        <div className="space-x-2">
          <button className="btn" onClick={start} disabled={streaming}>Share Screen</button>
          <button className="btn" onClick={stop} disabled={!streaming}>Stop Share</button>
        </div>
      </div>
      <video ref={videoRef} autoPlay muted className="w-full rounded-xl border border-slate-800" />
    </div>
  )
}