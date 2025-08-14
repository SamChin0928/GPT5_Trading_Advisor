// lib/api.js

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

// NEW: no-store by default to avoid cached GETs (e.g., /api/train/status)
function req(path, { method = 'GET', query, json, form, headers, signal, timeout = DEFAULT_TIMEOUT, cache = 'no-store' } = {}) {
  const url = join(BASE, path) + toQuery(query)
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(new DOMException('Request timeout', 'AbortError')), timeout)

  const init = {
    method,
    headers: { Accept: 'application/json', ...(headers || {}) },
    signal: signal || ctrl.signal,
    cache, // <--- important for status polling
  }

  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(json)
  } else if (form instanceof FormData) {
    init.body = form // let browser set content-type boundary
  }

  // Add explicit header for proxies that ignore Request.cache
  if (method === 'GET') init.headers['Cache-Control'] = 'no-store'

  return fetch(url, init).then(handle).finally(() => clearTimeout(timer))
}

// Build a safe image URL for a file relative to a session folder
// Use /api/data when BASE is empty (dev); use /data when BASE is set (prod).
const sessionImageUrl = (session_id, relPath) => {
  const safe = String(relPath).split('/').map(encodeURIComponent).join('/')
  const DATA_PREFIX = BASE ? '/data' : '/api/data'
  return join(BASE, `${DATA_PREFIX}/sessions/${encodeURIComponent(session_id)}/${safe}`)
}

const deleteGroupImage = (session_id, timestamp, pathOrFilename) => {
  const s = String(pathOrFilename || '')
  const hasSlash = s.includes('/')
  return hasSlash
    ? req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, path: s } })
    : req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, filename: s } })
}

// Export the API functions
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

  // NEW: add cache-buster for status polling to be extra safe
  trainStatus: (session_id) =>
    req('/api/train/status', {
      query: { session_id, t: Date.now() }, // bust any intermediary caches
      timeout: 15000
    }),

  predict: (session_id, blob) => {
    const fd = new FormData()
    fd.append('session_id', session_id)
    fd.append('file', blob, 'crop.jpg')
    return req('/api/predict', { method: 'POST', query: { session_id }, form: fd })
  },

  // ---- vocab / groups / annotations ----
  getVocab: () => req('/api/labels/vocab'),
  createVocabLabel: (name, parent) =>
    req('/api/labels/vocab', { method: 'POST', json: { name, parent } }),

  // Single-session groups (include ann + labeled from backend)
  groups: (session_id) =>
    req('/api/groups', { query: { session_id } }),

  // All sessions (set onlyUnlabeled=true to fetch only unlabeled)
  groupsAll: (onlyUnlabeled = false) =>
    req('/api/groups_all', { query: { only_unlabeled: onlyUnlabeled ? 1 : 0 } }),

  annotate: (session_id, { timestamp, global_labels, by_role, notes }) =>
    req('/api/annotate', { method: 'POST', json: { session_id, timestamp, global_labels, by_role, notes } }),

  // ---- annotations (API only, centralized) ----
  annotations: async (session_id) => {
    try {
      const data = await req('/api/annotations', { query: { session_id }, timeout: 10000 })
      return (data && typeof data === 'object') ? data : {}
    } catch (e) {
      console.warn('annotations API error:', e)
      return {}
    }
  },

  // ---- deletion helpers (NEW) ----
  /**
   * Delete an entire timestamp folder (captures + mosaic) and prune indexes.
   */
  deleteGroup: (session_id, timestamp) =>
    req('/api/group/delete', { method: 'POST', json: { session_id, timestamp } }),

  /**
   * Delete a single image inside captures/<timestamp>.
   * Pass either a full relative path (preferred) or just the filename.
   * Example path: "captures/1723567890000/zone_1.jpg"
   */
  deleteGroupImageByPath: (session_id, timestamp, path) =>
    req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, path } }),

  deleteGroupImageByFilename: (session_id, timestamp, filename) =>
    req('/api/group/delete_image', { method: 'POST', json: { session_id, timestamp, filename } }),

  deleteGroupImage, // <-- ensure this is exported

  deleteSession: (session_id) =>
    req('/api/session/delete', { method: 'POST', json: { session_id } }),

  // ---- evaluation & group prediction (NEW) ----
  evalHeads: (session_id) =>
    req('/api/eval', { query: { session_id } }),

  predictGroup: (session_id, timestamp) =>
    req('/api/predict_group', { query: { session_id, ...(timestamp ? { timestamp } : {}) } }),

  // Expose the image URL builder
  imageUrl: sessionImageUrl,
}
