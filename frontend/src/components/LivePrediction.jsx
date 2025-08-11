import React, { useEffect, useState } from 'react'

export default function LivePrediction(){
  const [pred, setPred] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)

  useEffect(()=>{
    const handler = (e)=>{
      const { pred, blob } = e.detail
      setPred(pred)
      setImgUrl(URL.createObjectURL(blob))
    }
    window.addEventListener('predict:update', handler)
    return ()=>window.removeEventListener('predict:update', handler)
  },[])

  const color = pred?.label==='bullish' ? 'text-emerald-400' : pred?.label==='bearish' ? 'text-rose-400' : 'text-slate-300'

  return (
    <div className="card">
      <div className="font-semibold mb-2">Live Prediction (Primary Zone)</div>
      {imgUrl ? <img src={imgUrl} alt="primary" className="rounded-xl mb-3 border border-slate-800" /> : (
        <div className="text-slate-400">Start the app and set a primary zone…</div>
      )}
      {pred && (
        <div className={`text-xl font-bold ${color}`}>{pred.label.toUpperCase()}</div>
      )}
      {pred && (
        <div className="text-sm text-slate-400 mt-1">{Object.entries(pred.probs).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' • ')}</div>
      )}
      {pred?.note && <div className="text-xs text-amber-400 mt-2">{pred.note}</div>}
    </div>
  )
}