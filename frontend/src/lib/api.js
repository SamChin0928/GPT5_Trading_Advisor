const BASE = import.meta.env.VITE_API_BASE

export const api = {
  health: () => fetch(`${BASE}/api/health`).then(r=>r.json()),

  saveZones: (session_id, zones) => fetch(`${BASE}/api/zones/save`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, zones })
  }).then(r=>r.json()),

  loadZones: (session_id) => fetch(`${BASE}/api/zones/load?session_id=${session_id}`).then(r=>r.json()),

  ingest: (payload) => fetch(`${BASE}/api/ingest`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(r=>r.json()),

  consolidate: (session_id) => fetch(`${BASE}/api/consolidate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id })
  }).then(r=>r.json()),

  mosaics: (session_id) => fetch(`${BASE}/api/mosaics?session_id=${session_id}`).then(r=>r.json()),

  label: (session_id, mosaic_rel_path, label) => fetch(`${BASE}/api/label`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, mosaic_rel_path, label })
  }).then(r=>r.json()),

  train: (session_id, params) => fetch(`${BASE}/api/train`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, ...params })
  }).then(r=>r.json()),

  trainStatus: (session_id) => fetch(`${BASE}/api/train/status?session_id=${session_id}`).then(r=>r.json()),

  predict: async (session_id, blob) => {
    const fd = new FormData()
    fd.append('session_id', session_id)
    fd.append('file', blob, 'crop.jpg')
    const res = await fetch(`${BASE}/api/predict?session_id=${session_id}`, { method: 'POST', body: fd })
    return res.json()
  }
}