import React, { useEffect, useState } from 'react'
import { api } from '../lib/api'

export default function Labeler({ sessionId }){
  const [mosaics, setMosaics] = useState([])
  const [labels, setLabels] = useState({})
  const [trainStatus, setTrainStatus] = useState(null)
  const [params, setParams] = useState({ epochs: 5, lr: 1e-3, batch_size: 16 })

  async function refresh(){
    const mos = await api.mosaics(sessionId)
    setMosaics(mos)
  }

  useEffect(()=>{ refresh() },[])

  async function saveAll(){
    for(const m of Object.keys(labels)){
      await api.label(sessionId, m, labels[m])
    }
    alert('Labels saved')
  }

  async function train(){
    await api.train(sessionId, params)
    poll()
  }

  async function poll(){
    const s = await api.trainStatus(sessionId)
    setTrainStatus(s)
    if(s.status==='running') setTimeout(poll, 1000)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Labeler & Training</div>
        <button className="btn" onClick={refresh}>Refresh</button>
      </div>

      <div className="grid grid-cols-2 gap-3 max-h-[420px] overflow-auto">
        {mosaics.map(m => (
          <div key={m} className="bg-slate-900 p-2 rounded-xl border border-slate-800">
            <img src={`${import.meta.env.VITE_API_BASE.replace('/api','')}/data/sessions/${sessionId}/${m}`} alt="mosaic" className="rounded-lg" />
            <div className="mt-2 flex gap-2 text-sm">
              {['bullish','bearish','neutral'].map(l => (
                <label key={l} className="flex items-center gap-1">
                  <input type="radio" name={m} onChange={()=>setLabels({...labels, [m]:l})} /> {l}
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 mt-3">
        <button className="btn" onClick={saveAll}>Save Labels</button>
        <div className="text-sm text-slate-400">Training params:</div>
        <input className="input w-24" type="number" step="1" value={params.epochs} onChange={e=>setParams({...params, epochs:+e.target.value})} />
        <input className="input w-28" type="number" step="0.0001" value={params.lr} onChange={e=>setParams({...params, lr:+e.target.value})} />
        <input className="input w-28" type="number" step="1" value={params.batch_size} onChange={e=>setParams({...params, batch_size:+e.target.value})} />
        <button className="btn" onClick={train}>Train</button>
      </div>

      {trainStatus && (
        <div className="mt-2 text-sm text-slate-300">
          Status: {trainStatus.status} {trainStatus.epoch?`(epoch ${trainStatus.epoch}/${trainStatus.epochs})`:''}
          {trainStatus.train_acc!==undefined && ` • train_acc ${(trainStatus.train_acc*100).toFixed(1)}%`}
          {trainStatus.val_acc!==undefined && ` • val_acc ${(trainStatus.val_acc*100).toFixed(1)}%`}
          {trainStatus.error && <div className="text-rose-400">{trainStatus.error}</div>}
        </div>
      )}
    </div>
  )
}