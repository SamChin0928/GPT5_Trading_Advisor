// Resolve base: env (VITE_API_BASE) or '' for relative (use Vite proxy)
const RAW  = import.meta?.env?.VITE_API_BASE?.trim?.()
export const BASE = RAW ? RAW.replace(/\/+$/, '') : ''

const DEFAULT_TIMEOUT = 20000

const join = (base, path) => `${base}${path.startsWith('/') ? path : `/${path}`}`

const toQuery = (obj = {}) => {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined || v === null) continue
    sp.append(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

async function handle(res) {
  const text = await res.text().catch(() => '')
  const ct = res.headers.get('content-type') || ''
  const data = ct.includes('application/json') && text ? JSON.parse(text) : text || null

  if (!res.ok) {
    const message =
      (data && data.detail) ||
      (typeof data === 'string' ? data : '') ||
      res.statusText ||
      'Request failed'

    const err = new Error(message)
    err.status = res.status
    err.data = data
    throw err
  }
  return data
}

function req(path, { method = 'GET', query, json, form, headers, signal, timeout = DEFAULT_TIMEOUT } = {}) {
  const url = join(BASE, path) + toQuery(query)
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(new DOMException('Request timeout', 'AbortError')), timeout)

  const init = {
    method,
    headers: { Accept: 'application/json', ...(headers || {}) },
    signal: signal || ctrl.signal
  }

  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(json)
  } else if (form instanceof FormData) {
    init.body = form // let browser set content-type boundary
  }

  return fetch(url, init).then(handle).finally(() => clearTimeout(timer))
}

export const api = {
  // ---- health ----
  health: () => req('/api/health'),

  // ---- zones ----
  saveZones: (session_id, zones) =>
    req('/api/zones/save', { method: 'POST', json: { session_id, zones } }),

  loadZones: (session_id) =>
    req('/api/zones/load', { query: { session_id } }),

  // ---- ingest / consolidate / mosaics ----
  ingest: (payload) =>
    req('/api/ingest', { method: 'POST', json: payload }),

  consolidate: (session_id) =>
    req('/api/consolidate', { method: 'POST', json: { session_id } }),

  mosaics: (session_id) =>
    req('/api/mosaics', { query: { session_id } }),

  // ---- legacy labels & training ----
  label: (session_id, mosaic_rel_path, label) =>
    req('/api/label', { method: 'POST', json: { session_id, mosaic_rel_path, label } }),

  train: (session_id, params) =>
    req('/api/train', { method: 'POST', json: { session_id, ...params } }),

  trainStatus: (session_id) =>
    req('/api/train/status', { query: { session_id } }),

  predict: (session_id, blob) => {
    const fd = new FormData()
    fd.append('session_id', session_id)
    fd.append('file', blob, 'crop.jpg')
    return req('/api/predict', { method: 'POST', query: { session_id }, form: fd })
  },

  // ---- NEW: vocab / groups / annotations ----
  getVocab: () => req('/api/labels/vocab'),
  createVocabLabel: (name, parent) =>
    req('/api/labels/vocab', { method: 'POST', json: { name, parent } }),

  groups: (session_id) =>
    req('/api/groups', { query: { session_id } }),

  annotate: (session_id, { timestamp, global_labels, by_role, notes }) =>
    req('/api/annotate', { method: 'POST', json: { session_id, timestamp, global_labels, by_role, notes } }),

  // Helper: load existing annotations (served statically)
  annotations: async (session_id) => {
    const url = `${BASE}/data/sessions/${encodeURIComponent(session_id)}/annotations.json`
    try {
      const res = await fetch(url, { headers: { 'Accept': 'application/json' } })
      if (!res.ok) return {}
      return await res.json()
    } catch {
      return {}
    }
  }
}
