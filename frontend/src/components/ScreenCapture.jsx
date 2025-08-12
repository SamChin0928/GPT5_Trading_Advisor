import React, { useState, useRef, useEffect } from 'react'
import ROISelector from './ROISelector'
import { Monitor, Share2, StopCircle, Settings, AlertTriangle } from 'lucide-react'

export default function ScreenCapture({ onReady }) {
  const [zones, setZones] = useState([])
  const [primaryId, setPrimaryId] = useState(null)
  const [isSharing, setIsSharing] = useState(false)
  const videoRef = useRef(null)
  const containerRef = useRef(null)
  const [screenSize, setScreenSize] = useState({ w: 1920, h: 1080 })

  // Expose captureFrame method to parent component
  useEffect(() => {
    if (onReady) {
      onReady({ captureFrame })
    }
  }, [onReady])

  // Capture current frame - from your working code
  async function captureFrame() {
    const v = videoRef.current
    if (!v || v.videoWidth === 0 || v.videoHeight === 0) return null
    
    const w = v.videoWidth
    const h = v.videoHeight
    
    // Create canvas (use OffscreenCanvas if available)
    const canvas = (typeof OffscreenCanvas !== 'undefined') 
      ? new OffscreenCanvas(w, h) 
      : document.createElement('canvas')
    
    canvas.width = w
    canvas.height = h
    
    const ctx = canvas.getContext('2d')
    ctx.drawImage(v, 0, 0, w, h)
    
    // Try to create ImageBitmap if available
    if (typeof createImageBitmap !== 'undefined') {
      try {
        const bmp = 'transferToImageBitmap' in canvas 
          ? canvas.transferToImageBitmap() 
          : await createImageBitmap(canvas)
        return { bmp, w, h, zones, primaryId }
      } catch (e) {
        // Fall through to canvas return
      }
    }
    
    return { bmp: canvas, w, h, zones, primaryId }
  }

  // Start screen sharing - using working logic
  const startScreenShare = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: false
      })

      const v = videoRef.current
      if (!v) return

      v.srcObject = stream
      
      // Use await and ensure video is playing properly
      await v.play()
      
      // Wait for video dimensions to be available
      await new Promise(resolve => {
        if (v.videoWidth > 0 && v.videoHeight > 0) {
          return resolve()
        }
        v.onloadedmetadata = () => resolve()
      })

      // Set screen size after we know dimensions are ready
      setScreenSize({
        w: v.videoWidth || 1920,
        h: v.videoHeight || 1080
      })

      setIsSharing(true)

      // Listen for stream end
      const [track] = stream.getVideoTracks()
      if (track) {
        track.onended = () => stopScreenShare()
      }
    } catch (err) {
      console.error('Error sharing screen:', err)
      setIsSharing(false)
    }
  }

  // Stop screen sharing
  const stopScreenShare = () => {
    const stream = videoRef.current?.srcObject
    if (stream) {
      stream.getTracks().forEach(t => t.stop())
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsSharing(false)
  }

  // Zone management functions
  const deleteZone = (zoneId) => {
    setZones(zones.filter(z => z.id !== zoneId))
    if(zoneId === primaryId) {
      const remaining = zones.filter(z => z.id !== zoneId)
      if(remaining.length > 0) {
        setPrimaryId(remaining[0].id)
      } else {
        setPrimaryId(null)
      }
    }
  }

  const clearAllZones = () => {
    setZones([])
    setPrimaryId(null)
  }

  return (
    <div className="flex gap-6 p-6 bg-gradient-to-br from-slate-950">
      {/* Main Screen Capture Area */}
      <div className="flex-1">
        <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-3">
              <div className="p-2 bg-blue-600/20 rounded-lg">
                <Monitor className="w-6 h-6 text-blue-400" />
              </div>
              Screen Capture
            </h2>
            <div className="flex items-center gap-3">
              {isSharing && (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-green-600/20 rounded-lg border border-green-600/30">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-sm text-green-400 font-medium">Live</span>
                </div>
              )}
              <button 
                onClick={isSharing ? stopScreenShare : startScreenShare}
                className={`px-5 py-2.5 rounded-xl font-medium transition-all duration-200 flex items-center gap-2 ${
                  isSharing 
                    ? 'bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30' 
                    : 'bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-600/30'
                }`}
              >
                {isSharing ? (
                  <>
                    <StopCircle className="w-4 h-4" />
                    Stop Share
                  </>
                ) : (
                  <>
                    <Share2 className="w-4 h-4" />
                    Share Screen
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Screen Preview with ROI Overlay */}
          <div
            ref={containerRef}
            className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-700/50"
            style={{
              aspectRatio: '16/9',
              minHeight: '320px'
            }}
          >
            {/* Video element - simplified like working version */}
            <video 
              ref={videoRef} 
              autoPlay 
              muted
              playsInline
              className="w-full h-full object-contain"
              style={{ 
                display: isSharing ? 'block' : 'none',
                maxWidth: '100%',
                maxHeight: '100%'
              }}
            />

            {/* Placeholder when not sharing */}
            {!isSharing && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="p-6 bg-slate-800/50 rounded-2xl">
                    <Monitor className="w-20 h-20 mx-auto mb-4 text-slate-600" />
                    <p className="text-slate-400 text-lg mb-2">No screen shared</p>
                    <p className="text-slate-500 text-sm">Click "Share Screen" to begin</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* ROI Selector Overlay - only show when sharing */}
            {isSharing && (
              <div
                className="absolute inset-0"
                style={{ zIndex: 10 }}
              >
                <ROISelector 
                  refSize={screenSize}
                  zones={zones}
                  setZones={setZones}
                  primaryId={primaryId}
                  setPrimaryId={setPrimaryId}
                  overlayMode={true}
                  showBackground={false}
                />
              </div>
            )}
            
            {/* Status overlay */}
            {isSharing && zones.length > 0 && (
              <div 
                className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 text-sm text-white pointer-events-none"
                style={{ zIndex: 20 }}
              >
                <div className="flex items-center gap-4">
                  <span className="text-slate-400">Active Zones:</span>
                  <span className="font-medium">{zones.length}</span>
                  {primaryId !== null && (
                    <>
                      <span className="text-slate-400">Primary:</span>
                      <span className="text-green-400 font-medium">Zone {primaryId}</span>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sidebar - Zone Management (same as before) */}
      <div className="w-96">
        <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Zone Management
            </h3>
            <button 
              onClick={clearAllZones}
              disabled={zones.length === 0}
              className="px-4 py-2 rounded-xl bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 text-sm font-medium"
            >
              Clear All
            </button>
          </div>

          {zones.length > 0 ? (
            <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
              {zones.map(zone => (
                <div 
                  key={zone.id}
                  className={`p-4 rounded-xl transition-all duration-200 ${
                    zone.id === primaryId 
                      ? 'bg-gradient-to-r from-green-900/30 to-emerald-900/20 border border-green-600/30' 
                      : 'bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`w-2.5 h-2.5 rounded-full ${
                        zone.id === primaryId ? 'bg-green-500 animate-pulse' : 'bg-blue-500'
                      }`} />
                      <span className="font-medium text-slate-200">Zone {zone.id}</span>
                      {zone.id === primaryId && (
                        <span className="text-xs px-2 py-1 bg-green-600/20 text-green-400 rounded-full font-semibold">
                          PRIMARY
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-xs text-slate-400 mb-3">
                    Size: {Math.round(zone.w * screenSize.w)}×{Math.round(zone.h * screenSize.h)}px
                  </div>
                  
                  <div className="flex gap-2">
                    {zone.id !== primaryId && (
                      <button 
                        onClick={() => setPrimaryId(zone.id)}
                        className="flex-1 px-3 py-1.5 text-sm rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-600/30 transition-all duration-200"
                      >
                        Set Primary
                      </button>
                    )}
                    <button 
                      onClick={() => deleteZone(zone.id)}
                      className="px-3 py-1.5 text-sm rounded-lg bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 transition-all duration-200"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="inline-flex p-4 bg-slate-800/30 rounded-full mb-4">
                <svg className="w-12 h-12 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                        d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                </svg>
              </div>
              <p className="text-slate-400 mb-2">No detection zones</p>
              <p className="text-sm text-slate-500">Click and drag on the screen to create zones</p>
            </div>
          )}
          
          {zones.length > 0 && primaryId === null && (
            <div className="mt-4 p-4 bg-amber-900/20 border border-amber-600/30 rounded-xl">
              <div className="flex gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-amber-400 font-medium mb-1">No Primary Zone</p>
                  <p className="text-xs text-amber-400/70">
                    Set a primary zone to enable live predictions and pattern detection.
                  </p>
                </div>
              </div>
            </div>
          )}
          
          <div className="mt-6 p-4 bg-slate-800/30 rounded-xl">
            <h4 className="text-sm font-medium text-slate-300 mb-3">Quick Guide</h4>
            <div className="space-y-2 text-xs text-slate-500">
              <div className="flex items-start gap-2">
                <span className="text-blue-400">•</span>
                <span>Click and drag on the screen to create detection zones</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-400">•</span>
                <span>Set a primary zone for main pattern detection</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-400">•</span>
                <span>Zones require minimum 20×20px size</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-400">•</span>
                <span>Drag zone borders to resize after creation</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}