// lib/api.js

// Resolve base: env (VITE_API_BASE) or '' for relative (use Vite proxy)
const RAW = import.meta?.env?.VITE_API_BASE?.trim?.()
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

// ---- current process context (forward-compatible) ----
let _currentProcessId = null
export const setProcess = (pid) => { _currentProcessId = pid ? String(pid) : null }
export const clearProcess = () => { _currentProcessId = null }
export const getProcess = () => _currentProcessId

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

function req(
  path,
  { method = 'GET', query, json, form, headers, signal, timeout = DEFAULT_TIMEOUT } = {}
) {
  // Merge process_id into query/body/form so backend can scope per-process when ready.
  const queryWithProcess = { ...(query || {}) }
  if (_currentProcessId && queryWithProcess.process_id == null) {
    queryWithProcess.process_id = _currentProcessId
  }

  const url = join(BASE, path) + toQuery(queryWithProcess)
  const ctrl = new AbortController()
  const timer = setTimeout(
    () => ctrl.abort(new DOMException('Request timeout', 'AbortError')),
    timeout
  )

  const init = {
    method,
    headers: { Accept: 'application/json', ...(headers || {}) },
    signal: signal || ctrl.signal,
  }

  if (json !== undefined) {
    const body = { ...json }
    if (_currentProcessId && body.process_id == null) {
      body.process_id = _currentProcessId
    }
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  } else if (form instanceof FormData) {
    if (_currentProcessId && !form.has('process_id')) {
      try { form.append('process_id', _currentProcessId) } catch {}
    }
    init.body = form // let browser set content-type boundary
  }

  return fetch(url, init).then(handle).finally(() => clearTimeout(timer))
}

// ---- Realtime helpers (SSE + compatibility pings) ----
const notifyUpdate = (type, detail = {}) => {
  try {
    window.dispatchEvent(
      new CustomEvent('captures:updated', { detail: { type, ...detail, at: Date.now() } })
    )
  } catch {}
  try {
    localStorage.setItem('captures_updated_at', String(Date.now()))
  } catch {}
}

let _es = null
export const realtime = {
  start() {
    if (typeof window === 'undefined' || _es) return _es
    const url = join(BASE, '/api/events') + toQuery(
      _currentProcessId ? { process_id: _currentProcessId } : {}
    )
    try {
      _es = new EventSource(url)
    } catch (e) {
      console.warn('EventSource init failed:', e)
      return null
    }
    // Default "message" handler (server sends JSON in data)
    _es.onmessage = (e) => {
      if (!e?.data) return
      try {
        const evt = JSON.parse(e.data)
        const t = evt?.type
        if (t === 'capture' ||
            t === 'consolidated' ||
            t === 'annotation_saved' ||
            t === 'group_deleted' ||
            t === 'image_deleted') {
          notifyUpdate(t, evt)
        }
      } catch {}
    }
    _es.addEventListener('hello', () => {})
    _es.onerror = () => { /* auto-reconnect */ }
    return _es
  },
  stop() {
    try { _es?.close() } catch {}
    _es = null
  },
}

// Build a safe image URL for a file relative to a session folder
// NOTE: For now we keep the existing structure so nothing breaks.
// When backend supports per-process data roots, we can add process_id to the path.
const sessionImageUrl = (session_id, relPath) => {
  const safe = String(relPath).split('/').map(encodeURIComponent).join('/')
  const DATA_PREFIX = BASE ? '/data' : '/api/data'
  return join(BASE, `${DATA_PREFIX}/sessions/${encodeURIComponent(session_id)}/${safe}`)
}

// Wrap single-image delete helper so it also pings listeners
const deleteGroupImage = (session_id, timestamp, pathOrFilename) => {
  const s = String(pathOrFilename || '')
  const hasSlash = s.includes('/')
  const payload = hasSlash
    ? { session_id, timestamp, path: s }
    : { session_id, timestamp, filename: s }
  return req('/api/group/delete_image', { method: 'POST', json: payload })
    .then((r) => { notifyUpdate('image_deleted', { session_id, timestamp }); return r })
}

export const api = {
  // ---- process context helpers ----
  setProcess,
  getProcess,
  clearProcess,

  // ---- health ----
  health: () => req('/api/health'),

  // ---- zones ----
  saveZones: (session_id, zones) =>
    req('/api/zones/save', { method: 'POST', json: { session_id, zones } }),

  loadZones: (session_id) =>
    req('/api/zones/load', { query: { session_id } }),

  // ---- ingest / consolidate / mosaics ----
  ingest: (payload) =>
    req('/api/ingest', { method: 'POST', json: payload })
      .then((r) => { notifyUpdate('capture', { session_id: payload.session_id, timestamp: payload.timestamp }); return r }),

  consolidate: (session_id) =>
    req('/api/consolidate', { method: 'POST', json: { session_id } })
      .then((r) => { notifyUpdate('consolidated', { session_id }); return r }),

  mosaics: (session_id) =>
    req('/api/mosaics', { query: { session_id } }),

  // ---- legacy labels & training ----
  label: (session_id, mosaic_rel_path, label) =>
    req('/api/label', { method: 'POST', json: { session_id, mosaic_rel_path, label } }),

  train: (session_id, params) =>
    req('/api/train', { method: 'POST', json: { session_id, ...params } }),

  trainStatus: (session_id) =>
    req('/api/train/status', { query: { session_id } }),

  // reset global model
  modelReset: () => req('/api/model/reset', { method: 'POST' }),

  // ---- prediction ----
  predict: (session_id, blob) => {
    const fd = new FormData()
    fd.append('session_id', session_id)
    fd.append('file', blob, 'crop.jpg')
    return req('/api/predict', { method: 'POST', query: { session_id }, form: fd })
  },

  // Global heads/group prediction (timestamp optional; backend picks latest if omitted)
  predictGroup: (session_id, timestamp) =>
    req('/api/predict_group', { query: { session_id, ...(timestamp ? { timestamp } : {}) } }),

  // ---- vocab / groups / annotations ----
  getVocab: () => req('/api/labels/vocab'),
  createVocabLabel: (name, parent) =>
    req('/api/labels/vocab', { method: 'POST', json: { name, parent } }),

  // Single-session groups (include ann + labeled from backend)
  groups: (session_id, includePred = true) =>
    req('/api/groups', { query: { session_id, include_pred: includePred ? 1 : 0 } }),

  // All sessions (set onlyUnlabeled=true to fetch only unlabeled)
  groupsAll: (onlyUnlabeled = false, includePred = true) =>
    req('/api/groups_all', {
      query: { only_unlabeled: onlyUnlabeled ? 1 : 0, include_pred: includePred ? 1 : 0 },
    }),

  // Allow model_feedback (and future fields) to pass through unchanged
  annotate: (session_id, payload) =>
    req('/api/annotate', { method: 'POST', json: { session_id, ...payload } })
      .then((r) => { notifyUpdate('annotation_saved', { session_id, timestamp: payload.timestamp }); return r }),

  // Centralized annotations slice for a session
  annotations: async (session_id) => {
    try {
      const data = await req('/api/annotations', { query: { session_id }, timeout: 10000 })
      return (data && typeof data === 'object') ? data : {}
    } catch (e) {
      console.warn('annotations API error:', e)
      return {}
    }
  },

  // ---- deletion helpers ----
  deleteGroup: (session_id, timestamp) =>
    req('/api/group/delete', { method: 'POST', json: { session_id, timestamp } })
      .then((r) => { notifyUpdate('group_deleted', { session_id, timestamp }); return r }),

  deleteGroupImageByPath: (session_id, timestamp, path) =>
    req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, path } })
      .then((r) => { notifyUpdate('image_deleted', { session_id, timestamp }); return r }),

  deleteGroupImageByFilename: (session_id, timestamp, filename) =>
    req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, filename } })
      .then((r) => { notifyUpdate('image_deleted', { session_id, timestamp }); return r }),

  deleteGroupImage,

  deleteSession: (session_id) =>
    req('/api/session/delete', { method: 'POST', json: { session_id } })
      .then((r) => { notifyUpdate('session_deleted', { session_id }); return r }),

  // ---- model info & (optional) suggestion upsert ----
  modelInfo: () => req('/api/model/info'),

  saveModelSuggestion: (session_id, timestamp, suggestion) =>
    req('/api/annotations/model_suggestion', {
      method: 'POST',
      json: { session_id, timestamp, suggestion },
    }),

  // Expose the image URL builder
  imageUrl: sessionImageUrl,
}
