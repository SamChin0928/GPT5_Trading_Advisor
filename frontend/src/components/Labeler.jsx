// components/Labeler.jsx
import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { Trash2, Info, CheckCircle2, Clock, Calendar, ChevronDown, ChevronRight, X } from 'lucide-react'
import TrainControls from './TrainControls'

// --- small helpers ---
function fmtTS(ts, tz) {
  const n = Number(ts)
  if (!Number.isFinite(n)) return String(ts)
  const tzOpt = (!tz || tz.toUpperCase() === 'UTC' || tz === 'Etc/UTC') ? undefined : tz
  const d = new Date(n)
  const opt = { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: tzOpt }
  let time = new Intl.DateTimeFormat('en-US', opt).format(d)
  time = time.toLowerCase().replace(' ', '')
  const day  = new Intl.DateTimeFormat('en-GB', { day: 'numeric', timeZone: tzOpt }).format(d)
  const mon  = new Intl.DateTimeFormat('en-US', { month: 'short',  timeZone: tzOpt }).format(d)
  const year = new Intl.DateTimeFormat('en-GB', { year: 'numeric', timeZone: tzOpt }).format(d)
  return `${time} ${day}${mon}${year}`
}
const asSet = a => new Set(a || [])
const eqArr = (a = [], b = []) => a.length === b.length && a.every(x => asSet(b).has(x))
const eqMapOfArrays = (a = {}, b = {}) => {
  const ka = Object.keys(a), kb = Object.keys(b)
  if (ka.length !== kb.length) return false
  return ka.every(k => eqArr(a[k] || [], b[k] || []))
}
const safeId = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9_-]/gi, '_')

// date helpers
const localYMD = (d) => {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}
const dateKey = (ts, tz) => {
  const d = new Date(Number(ts))
  const tzOpt = (!tz || tz.toUpperCase() === 'UTC' || tz === 'Etc/UTC') ? undefined : tz
  const y = new Intl.DateTimeFormat('en-CA', { timeZone: tzOpt, year: 'numeric' }).format(d)
  const m = new Intl.DateTimeFormat('en-CA', { timeZone: tzOpt, month: '2-digit'  }).format(d)
  const day = new Intl.DateTimeFormat('en-CA', { timeZone: tzOpt, day: '2-digit' }).format(d)
  return `${y}-${m}-${day}`
}
const dateLabel = (ymd) => {
  const [y, m, d] = ymd.split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1, d))
  return new Intl.DateTimeFormat('en-US', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }).format(dt)
}

const PRESETS = [
  'Bull Breakout', 'Bear Breakout', 'Neutral', 'Consolidation',
  'Bull Continuation', 'Bear Continuation',
]

// ---------- Image-path utils (force correct folder + try .jpg first) ----------
const EXT_ORDER = ['jpg', 'jpeg', 'png', 'webp']

function stripToSessionRel(sessionId, raw) {
  if (!raw) return ''
  let p = String(raw).replace(/^\/+/, '')
  // If absolute /api/data or /data — trim to ".../sessions/<sid>/..."
  if (p.startsWith('api/data/')) p = p.slice('api/data/'.length)
  if (p.startsWith('data/'))     p = p.slice('data/'.length)
  if (p.startsWith('sessions/')) {
    const parts = p.split('/')
    if (parts[1] === String(sessionId)) return parts.slice(2).join('/') // drop "sessions/<sid>/"
  }
  return p // already session-relative (or unknown; we'll normalize below)
}

function zoneNumericId(z) {
  if (Number.isFinite(z?.id)) return z.id
  const m = String(z?.role ?? '').match(/\d+/)
  return m ? Number(m[0]) : 0
}

function relCandidates(sessionId, timestamp, z) {
  const id = zoneNumericId(z)
  const providedRel = stripToSessionRel(sessionId, z?.path || z?.rel || z?.filename)
  const providedName = (providedRel.split('/').pop() || `zone_${id}`).replace(/\.(jpg|jpeg|png|webp)$/i, '')
  const base1 = `captures/${timestamp}/zone_${id}`       // our deterministic base
  const base2 = `captures/${timestamp}/${providedName}`  // backend-provided basename (normalized)
  const ordered = [
    // Always try deterministic .jpg first
    `${base1}.jpg`,
    // Then use provided basename in our captures/<ts>/...
    ...EXT_ORDER.map(ext => `${base2}.${ext}`),
  ]
  // If backend already gave full rel path under captures, try its basename variants too
  if (providedRel.startsWith('captures/')) {
    const base3 = providedRel.replace(/\.(jpg|jpeg|png|webp)$/i, '')
    for (const ext of EXT_ORDER) ordered.push(`${base3}.${ext}`)
  }
  // De-dupe
  return Array.from(new Set(ordered))
}

function imageUrlWithProcess(sessionId, rel, bust) {
  const base = api.imageUrl(sessionId, rel) // -> /data|/api/data/<process>/sessions/<sid>/<rel>
  return base + (base.includes('?') ? '&' : '?') + 'v=' + encodeURIComponent(bust)
}

// Tries each candidate URL in order until one loads.
function SmartImg({ sessionId, candidates, bust, className, alt = '', style }) {
  const [idx, setIdx] = useState(0)
  const src = imageUrlWithProcess(sessionId, candidates[idx], bust)
  return (
    <img
      src={src}
      alt={alt}
      className={className}
      style={style}
      loading="lazy"
      onError={() => {
        if (idx < candidates.length - 1) setIdx(idx + 1)
      }}
    />
  )
}

// composite key helper (session::timestamp)
const gkey = (sid, ts) => `${sid}::${String(ts)}`

function Chip({ children, onRemove }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full bg-slate-700/80 text-slate-200">
      {children}
      {onRemove && (
        <button className="text-slate-300 hover:text-white" onClick={onRemove} title="Remove">×</button>
      )}
    </span>
  )
}

function HelpModal({ open, onClose }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-[200]">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-slate-900/95 shadow-2xl p-6">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-5 h-5 text-blue-400" />
            <h3 className="text-slate-100 font-semibold">How to use training & labels</h3>
          </div>
          <ol className="list-decimal ml-5 space-y-2 text-sm text-slate-300">
            <li>Capture your screen — folders (by timestamp) are created automatically at the start of each minute.</li>
            <li>In <b>Labeler</b>, enable <b>Show unlabeled only</b> to focus on new data. Click a folder to open it.</li>
            <li>Use <b>Folder labels</b> (presets or custom) and optionally per-image labels (Zone 0 / Zone 1…).</li>
            <li>Click <b>Save</b>. The badge turns green (“Labeled”).</li>
            <li>After a batch, go to <b>Training</b> and click <b>Train</b>.</li>
          </ol>
          <div className="mt-4 flex justify-end">
            <button onClick={onClose} className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm">Close</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusPill({ labeled, className = '' }) {
  return (
    <span
      className={[
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium',
        'shadow-sm ring-1 backdrop-blur-[1px]',
        labeled
          ? 'bg-gradient-to-b from-emerald-500/20 to-emerald-500/10 text-emerald-100 ring-emerald-400/30'
          : 'bg-gradient-to-b from-amber-500/20 to-amber-500/10 text-amber-100 ring-amber-400/30',
        className
      ].join(' ')}
      aria-label={labeled ? 'Labeled' : 'Unlabeled'}
      title={labeled ? 'Labeled' : 'Unlabeled'}
    >
      {labeled ? <CheckCircle2 className="w-3.5 h-3.5" /> : <Clock className="w-3.5 h-3.5" />}
      <span className="leading-none">{labeled ? 'Labeled' : 'Unlabeled'}</span>
    </span>
  )
}

/* ---------- Main Labeler ---------- */
export default function Labeler({ sessionId }) {
  const [loading, setLoading] = useState(true)
  const [groups, setGroups] = useState([])
  const [vocab, setVocab]   = useState([])   // array of strings
  const [ann, setAnn]       = useState({})
  const [onlyUnlabeled, setOnlyUnlabeled] = useState(true)
  const [includeAll, setIncludeAll] = useState(true)
  const [savingKey, setSavingKey] = useState(null)

  const [open, setOpen] = useState(null)
  const [sel, setSel] = useState({})
  const [helpOpen, setHelpOpen] = useState(false)

  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [collapsedDays, setCollapsedDays] = useState(() => new Set())

  const [cacheBust, setCacheBust] = useState(0)

  // ------- data loading -------
  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      try {
        const [vocabResp, groupsResp, annResp] = await Promise.all([
          api.getVocab(),
          includeAll ? api.groupsAll(onlyUnlabeled) : api.groups(sessionId),
          includeAll ? Promise.resolve({}) : api.annotations(sessionId)
        ])
        if (cancelled) return
        const vocabList = Array.isArray(vocabResp) ? vocabResp : (vocabResp?.labels || [])
        const groupsList = groupsResp?.groups || []
        const annMap     = annResp || {}
        setVocab(vocabList)
        setGroups(groupsList)
        setAnn(includeAll ? {} : annMap)
        setSel(seedSel(groupsList, includeAll ? {} : annMap))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [sessionId, includeAll, onlyUnlabeled])

  // SOFT auto refresh (push + gentle poll)
  useEffect(() => {
    let stop = false
    let t = null

    const softReload = async () => {
      if (stop) return
      try {
        const gResp = includeAll ? await api.groupsAll(onlyUnlabeled) : await api.groups(sessionId)
        const groupsList = gResp?.groups || []
        const a = includeAll ? {} : await api.annotations(sessionId)
        const annMap = a || {}
        setGroups(groupsList)
        setAnn(includeAll ? {} : annMap)
        setSel(seedSel(groupsList, includeAll ? {} : annMap))
        setCacheBust(Date.now())
      } catch {}
    }

    const onUpdated = () => softReload()
    const onVisible = () => { if (!document.hidden) softReload() }

    window.addEventListener('captures:updated', onUpdated)
    document.addEventListener('visibilitychange', onVisible)

    const tick = async () => {
      if (stop) return
      if (!document.hidden) await softReload()
      t = setTimeout(tick, 2000)
    }
    t = setTimeout(tick, 2000)

    return () => {
      stop = true
      clearTimeout(t)
      window.removeEventListener('captures:updated', onUpdated)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [sessionId, includeAll, onlyUnlabeled])

  function seedSel(groupsList, annMap) {
    const seeded = {}
    for (const grp of groupsList) {
      const sid = grp.session_id || sessionId
      const ts = String(grp.timestamp)
      const base = (grp.ann && includeAll) ? grp.ann : (annMap[ts] || {})
      const draft = { global: '' }
      const roleMap = { ...(base.by_role || {}) }
      for (const z of grp.zones || []) {
        if (!(z.role in roleMap)) roleMap[z.role] = []
        if (!(z.role in draft)) draft[z.role] = ''
      }
      seeded[gkey(sid, ts)] = {
        global: [...(base.global || [])],  // array of strings
        by_role: roleMap,                  // map role -> array of strings
        drafts: draft
      }
    }
    return seeded
  }

  // Vocab helpers (strings)
  const vocabLowerSet = useMemo(() => new Set((vocab || []).map(s => String(s).trim().toLowerCase())), [vocab])

  const selected = (sid, ts) => sel[gkey(sid, ts)] || { global: [], by_role: {}, drafts: { global: '' } }
  const setSelected = (sid, ts, updater) => {
    const key = gkey(sid, ts)
    setSel(prev => {
      const cur = prev[key] || { global: [], by_role: {}, drafts: { global: '' } }
      const nxt = typeof updater === 'function' ? updater(cur) : updater
      return { ...prev, [key]: nxt }
    })
  }

  async function ensureLabel(name) {
    const val = (name || '').trim()
    if (!val) return null
    if (!vocabLowerSet.has(val.toLowerCase())) {
      const out = await api.createVocabLabel(val)
      const nextList = Array.isArray(out?.labels) ? out.labels : (Array.isArray(out) ? out : [])
      if (Array.isArray(nextList)) setVocab(nextList)
    }
    return val
  }

  async function addGlobal(sid, ts, name) {
    const label = await ensureLabel(name); if (!label) return
    setSelected(sid, ts, cur => {
      if (cur.global.includes(label)) return cur
      return { ...cur, global: [...cur.global, label], drafts: { ...cur.drafts, global: '' } }
    })
  }
  const removeGlobal = (sid, ts, lid) =>
    setSelected(sid, ts, cur => ({ ...cur, global: cur.global.filter(x => x !== lid) }))

  async function addRole(sid, ts, role, name) {
    const label = await ensureLabel(name); if (!label) return
    setSelected(sid, ts, cur => {
      const curList = cur.by_role[role] || []
      if (curList.includes(label)) return cur
      return {
        ...cur,
        by_role: { ...cur.by_role, [role]: [...curList, label] },
        drafts: { ...cur.drafts, [role]: '' }
      }
    })
  }
  const removeRole = (sid, ts, role, lid) =>
    setSelected(sid, ts, cur => ({
      ...cur,
      by_role: { ...cur.by_role, [role]: (cur.by_role[role] || []).filter(x => x !== lid) }
    }))

  async function saveGroup(sid, ts) {
    const data = selected(sid, ts)
    const key = gkey(sid, ts)
    setSavingKey(key)
    try {
      await api.annotate(sid, { timestamp: ts, global_labels: data.global, by_role: data.by_role })
      if (includeAll) {
        setGroups(gs => gs.map(g =>
          (String(g.timestamp) === String(ts) && (g.session_id || sessionId) === sid)
            ? { ...g, ann: { global: data.global, by_role: data.by_role }, labeled: Array.isArray(data.global) && data.global.length > 0 }
            : g
        ))
        if (onlyUnlabeled) setOpen(null)
      } else {
        setAnn(a => ({ ...a, [String(ts)]: { ...(a[String(ts)] || {}), global: data.global, by_role: data.by_role } }))
        if (onlyUnlabeled) setOpen(null)
      }
    } finally {
      setSavingKey(null)
    }
  }

  async function deleteFolder(sid, ts, tz) {
    if (!confirm(`Delete entire folder ${fmtTS(ts, tz)} from session ${sid}? This cannot be undone.`)) return
    try {
      await api.deleteGroup(sid, ts)
      setGroups(gs => gs.filter(g => !(String(g.timestamp) === String(ts) && (g.session_id || sessionId) === sid)))
      setSel(prev => { const k = gkey(sid, ts); if (!(k in prev)) return prev; const cp = { ...prev }; delete cp[k]; return cp })
      if (open && open.sid === sid && String(open.ts) === String(ts)) setOpen(null)
    } catch (e) {
      alert(e?.message || 'Failed to delete folder')
    }
  }

  async function deleteImage(sid, ts, tz, z) {
    // Try to delete with a precise session-relative path (assume .jpg first)
    const id = zoneNumericId(z)
    const rel = stripToSessionRel(sid, z?.path || z?.rel || z?.filename)
    const hasExt = /\.(jpg|jpeg|png|webp)$/i.test(rel)
    const path = rel ? (rel.startsWith('captures/') ? rel : `captures/${ts}/${rel.replace(/^captures\//, '')}`) : `captures/${ts}/zone_${id}.jpg`
    const jpgPath = hasExt ? rel.replace(/\.(jpg|jpeg|png|webp)$/i, '.jpg') : path
    try {
      await api.deleteGroupImageByPath(sid, ts, jpgPath)
      setGroups(gs => {
        const out = []
        for (const g of gs) {
          if ((g.session_id || sessionId) === sid && String(g.timestamp) === String(ts)) {
            const nz = (g.zones || []).filter(zz => {
              const r = stripToSessionRel(sid, zz.path || zz.rel || zz.filename)
              return (r !== rel) && (r !== jpgPath)
            })
            if (nz.length > 0) out.push({ ...g, zones: nz })
          } else out.push(g)
        }
        return out
      })
      setSelected(sid, ts, cur => {
        const { [z.role]: _drop, ...restRoles } = cur.by_role || {}
        const { [z.role]: _d2, ...restDrafts } = cur.drafts || {}
        return { ...cur, by_role: restRoles, drafts: restDrafts }
      })
    } catch (e) {
      alert(e?.message || 'Failed to delete image')
    }
  }

  // filtering & grouping
  const filteredGroups = useMemo(() => {
    let list = groups || []
    if (onlyUnlabeled) {
      if (includeAll) list = list.filter(g => !g.labeled)
      else {
        list = list.filter(g => {
          const rec = ann[String(g.timestamp)]
          const hasGlobal = !!(rec && Array.isArray(rec.global) && rec.global.length)
          return !hasGlobal
        })
      }
    }
    if (dateFrom || dateTo) {
      list = list.filter(g => {
        const ymd = dateKey(g.timestamp, g.tz)
        if (dateFrom && ymd < dateFrom) return false
        if (dateTo && ymd > dateTo) return false
        return true
      })
    }
    return [...list].sort((a, b) => Number(a.timestamp) - Number(b.timestamp))
  }, [groups, ann, onlyUnlabeled, includeAll, dateFrom, dateTo])

  const groupedByDay = useMemo(() => {
    const map = new Map()
    for (const g of filteredGroups) {
      const key = dateKey(g.timestamp, g.tz)
      if (!map.has(key)) map.set(key, [])
      map.get(key).push(g)
    }
    const keys = Array.from(map.keys()).sort((a, b) => (a < b ? 1 : -1))
    return keys.map(k => ({ day: k, label: dateLabel(k), groups: map.get(k) }))
  }, [filteredGroups])

  function applyPreset(preset) {
    const today = new Date()
    if (preset === 'all') { setDateFrom(''); setDateTo(''); return }
    if (preset === 'today') {
      const ymd = localYMD(today)
      setDateFrom(ymd); setDateTo(ymd); return
    }
    const days = preset === '7d' ? 6 : 29
    const start = new Date(today); start.setDate(start.getDate() - days)
    setDateFrom(localYMD(start)); setDateTo(localYMD(today))
  }

  function toggleDay(dayKey) { setCollapsedDays(prev => { const n = new Set(prev); if (n.has(dayKey)) n.delete(dayKey); else n.add(dayKey); return n }) }

  const reload = async () => {
    setLoading(true)
    try {
      const gResp = includeAll ? await api.groupsAll(onlyUnlabeled) : await api.groups(sessionId)
      const groupsList = gResp?.groups || []
      const a = includeAll ? {} : await api.annotations(sessionId)
      const annMap = a || {}
      setGroups(groupsList)
      setAnn(includeAll ? {} : annMap)
      setSel(seedSel(groupsList, includeAll ? {} : annMap))
      setCacheBust(Date.now())
    } finally {
      setLoading(false)
    }
  }

  // ------- UI -------
  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-slate-100">Labeler</h2>
          <button
            onClick={() => setHelpOpen(true)}
            className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600/60 text-xs text-slate-200 border border-white/10"
            title="How to use the model"
          >
            <Info className="w-3.5 h-3.5" />
            Help
          </button>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Calendar className="w-4 h-4 opacity-80" />
            <span>From</span>
            <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} className="px-2 py-1 rounded bg-slate-800 border border-white/10 text-slate-100" />
            <span>to</span>
            <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} className="px-2 py-1 rounded bg-slate-800 border border-white/10 text-slate-100" />
            {(dateFrom || dateTo) && (
              <button onClick={() => { setDateFrom(''); setDateTo('') }} className="ml-1 p-1 rounded bg-slate-700 hover:bg-slate-600 border border-white/10 text-slate-200" title="Clear dates">
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-1">
            {[
              { k: 'today', label: 'Today' },
              { k: '7d',    label: '7d' },
              { k: '30d',   label: '30d' },
              { k: 'all',   label: 'All' },
            ].map(btn => (
              <button key={btn.k} onClick={() => applyPreset(btn.k)} className="px-2 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-600/60 text-xs text-slate-200 border border-white/10">
                {btn.label}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-2 text-sm text-slate-300 ml-2">
            <input type="checkbox" checked={includeAll} onChange={e => setIncludeAll(e.target.checked)} />
            Include previous sessions
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={onlyUnlabeled} onChange={e => setOnlyUnlabeled(e.target.checked)} />
            Show unlabeled only
          </label>

          <button onClick={reload} className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm">Refresh</button>
        </div>
      </div>

      <TrainControls sessionId={sessionId} />

      {loading && <div className="text-slate-400 text-sm">Loading…</div>}
      {!loading && groupedByDay.length === 0 && (
        <div className="text-slate-400 text-sm m-4">
          {onlyUnlabeled ? 'No unlabeled folders in this range.' : 'No folders found in this range.'}
        </div>
      )}

      <div className="space-y-6 mt-4">
        {groupedByDay.map(({ day, label, groups: dayGroups }) => {
          const collapsed = collapsedDays.has(day)
          return (
            <section key={day} className="bg-black/10 border border-white/10 rounded-2xl overflow-hidden">
              <button onClick={() => toggleDay(day)} className="w-full flex items-center justify-between px-4 py-3 bg-white/[0.03] border-b border-white/10">
                <div className="flex items-center gap-3">
                  {collapsed ? <ChevronRight className="w-4 h-4 text-slate-300" /> : <ChevronDown className="w-4 h-4 text-slate-300" />}
                  <div className="text-slate-100 font-medium">{label}</div>
                  <div className="text-slate-400 text-xs">{dayGroups.length} folder(s)</div>
                </div>
                <div className="text-slate-400 text-xs">Click to {collapsed ? 'expand' : 'collapse'}</div>
              </button>

              {!collapsed && (
                <div className="p-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                  {dayGroups.map(g => {
                    const ts = String(g.timestamp)
                    const sid = g.session_id || sessionId
                    const zones = (g.zones || []).slice(0, 4)
                    const isLabeled = includeAll ? !!g.labeled : !!(ann[ts]?.global?.length)

                    let predText = null
                    if (g.pred && g.pred.top) {
                      const top = g.pred.top
                      const p = Number(g.pred?.probs?.[top] ?? 0)
                      const pct = Math.round(p * 100)
                      predText = `${top} (${pct}%)`
                    }

                    return (
                      <div
                        key={`${sid}:${ts}`}
                        className={`relative group rounded-2xl p-3 transition-colors backdrop-blur w-full text-left
                                    border shadow-2xl
                                    ${isLabeled ? 'bg-emerald-500/5 border-emerald-500/30'
                                                : 'bg-white/5 border-white/10'}`}
                      >
                        <div className="pointer-events-none absolute inset-0 z-20">
                          <StatusPill labeled={isLabeled} className="absolute top-3 left-3 pointer-events-auto" />
                          <button
                            onClick={() => deleteFolder(sid, ts, g.tz)}
                            className="absolute top-2 right-2 p-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-200 border border-red-400/30 pointer-events-auto opacity-0 group-hover:opacity-100 transition"
                            title="Delete folder"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>

                        <button onClick={() => setOpen({ sid, ts })} className="block w-full text-left relative z-10">
                          <div className="relative h-32">
                            {zones.map((z, i) => (
                              <SmartImg
                                key={i}
                                sessionId={sid}
                                candidates={relCandidates(sid, ts, z)}
                                bust={cacheBust}
                                className={`absolute inset-0 object-contain rounded-xl bg-black/60 ring-1 ring-white/10 shadow-xl`}
                                style={{
                                  zIndex: 5 - i,
                                  width: '100%',
                                  height: '100%',
                                  transform:
                                    i === 0 ? '' :
                                    i === 1 ? 'translateX(0.5rem) translateY(-0.5rem) scale(0.95)' :
                                    i === 2 ? 'translateX(1rem) translateY(-1rem) scale(0.90)' :
                                              'translateX(1.5rem) translateY(-1.5rem) scale(0.85)'
                                }}
                              />
                            ))}
                          </div>
                          <div className="mt-3 flex items-center justify-between">
                            <div>
                              <div className="text-slate-100 text-sm font-medium">{fmtTS(ts, g.tz)}</div>
                              <div className="text-slate-400 text-xs">{g.zones?.length || 0} image(s)</div>
                              {predText && (
                                <div className="text-slate-300 text-xs mt-1">
                                  <span className="px-1.5 py-0.5 rounded bg-black/30 border border-white/10">Pred: {predText}</span>
                                </div>
                              )}
                            </div>
                            <span className="text-slate-300 text-xs px-2 py-1 rounded-lg bg-black/30 border border-white/10">
                              Open
                            </span>
                          </div>
                        </button>
                      </div>
                    )
                  })}
                </div>
              )}
            </section>
          )
        })}
      </div>

      {open && (() => {
        const grp = groups.find(x =>
          String(x.timestamp) === String(open.ts) &&
          (x.session_id || sessionId) === open.sid
        )
        if (!grp) return null
        const cur = selected(open.sid, open.ts)
        const appliedRec = includeAll ? (grp.ann || { global: [], by_role: {} }) : (ann[open.ts] || { global: [], by_role: {} })
        const changed = !eqArr(cur.global, appliedRec.global) || !eqMapOfArrays(cur.by_role, appliedRec.by_role)
        const isLabeled = (appliedRec.global && appliedRec.global.length > 0)

        // ---- group-level prediction (from backend) ----
        const predTop = grp?.pred?.top
        const predProbs = grp?.pred?.probs || {}
        const topPct = predTop ? Math.round(Number(predProbs[predTop] || 0) * 100) : null
        const ranked = Object.entries(predProbs)
          .map(([k, v]) => [k, Number(v)])
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)

        return (
          <div className="fixed inset-0 z-[180]">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(null)} />
            <div className="absolute inset-0 flex items-center justify-center p-4">
              <div className="w-full max-w-5xl rounded-2xl border border-white/10 bg-slate-900/90 shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-white/10">
                  <div className="flex items-center gap-4">
                    <div>
                      <div className="text-slate-100 font-semibold">Folder • {fmtTS(open.ts, grp.tz)}</div>
                      <div className="text-slate-400 text-xs">{grp.tz || 'UTC'} • {grp.zones?.length || 0} image(s)</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="hidden md:flex flex-wrap gap-2 mr-2">
                      {PRESETS.map(q => (
                        <button key={q} onClick={() => addGlobal(open.sid, open.ts, q)}
                          className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[12px] text-slate-200 border border-white/10">
                          {q}
                        </button>
                      ))}
                    </div>
                    {predTop && (
                      <div className="px-2 py-1 rounded-lg bg-indigo-500/15 text-indigo-200 border border-indigo-400/30 text-xs">
                        Model prediction: <b>{predTop}</b>{topPct != null ? ` (${topPct}%)` : ''}
                      </div>
                    )}
                    <StatusPill labeled={isLabeled} />
                    <button
                      onClick={() => saveGroup(open.sid, open.ts)}
                      disabled={!changed || savingKey === gkey(open.sid, open.ts)}
                      className={`px-3 py-2 rounded-lg text-sm ${
                        changed && savingKey !== gkey(open.sid, open.ts)
                          ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                          : 'bg-slate-700 text-slate-300 cursor-not-allowed'
                      }`}
                    >
                      {savingKey === gkey(open.sid, open.ts) ? 'Saving…' : changed ? 'Save' : 'Saved'}
                    </button>
                    <button onClick={() => setOpen(null)} className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm">
                      Close
                    </button>
                  </div>
                </div>

                {ranked.length > 0 && (
                  <div className="px-4 pt-3 text-xs text-slate-300">
                    <span className="text-slate-400 mr-2">Probabilities:</span>
                    {ranked.map(([k, v], i) => (
                      <span key={k} className={`inline-block mr-2 mb-2 px-1.5 py-0.5 rounded border ${i === 0 ? 'border-indigo-400/40 bg-indigo-400/10' : 'border-white/10 bg-white/5'}`}>
                        {k}: {Math.round(v * 100)}%
                      </span>
                    ))}
                  </div>
                )}

                <div className="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {(grp.zones || []).map(z => {
                    const listId = `label-suggestions-${open.ts}-${safeId(z.role)}`
                    return (
                      <div key={z.id} className="relative bg-slate-800/60 border border-white/10 rounded-xl overflow-hidden">
                        <button
                          onClick={() => deleteImage(open.sid, open.ts, grp.tz, z)}
                          className="absolute top-2 right-2 p-1.5 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-200 border border-red-400/30 z-20"
                          title="Delete image"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>

                        <div className="p-2 text-xs text-slate-400 flex items-center justify-between">
                          <span>Zone {zoneNumericId(z)}</span>
                          <span className="text-slate-500">{z.role || `zone_${zoneNumericId(z)}`}</span>
                        </div>

                        <SmartImg
                          sessionId={open.sid}
                          candidates={relCandidates(open.sid, open.ts, z)}
                          bust={cacheBust}
                          className="w-full h-40 object-contain bg-black"
                        />

                        <div className="p-2 space-y-2">
                          <div className="flex flex-wrap gap-1">
                            {(cur.by_role[z.role] || []).map(lab => (
                              <Chip key={lab} onRemove={() => removeRole(open.sid, open.ts, z.role, lab)}>
                                {lab}
                              </Chip>
                            ))}
                            {(cur.by_role[z.role] || []).length === 0 && (
                              <span className="text-[11px] text-slate-500">No labels</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              list={listId}
                              placeholder="Add label to this image…"
                              className="flex-1 px-2 py-1 bg-slate-900/60 border border-white/10 rounded text-xs text-slate-100"
                              value={cur.drafts[z.role] || ''}
                              onChange={e => setSelected(open.sid, open.ts, s => ({ ...s, drafts: { ...s.drafts, [z.role]: e.target.value } }))}
                              onKeyDown={e => {
                                if (e.key === 'Enter') {
                                  e.preventDefault()
                                  addRole(open.sid, open.ts, z.role, cur.drafts[z.role])
                                }
                              }}
                            />
                            <button
                              onClick={() => addRole(open.sid, open.ts, z.role, cur.drafts[z.role])}
                              className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-xs border border-white/10"
                            >
                              Add
                            </button>
                            <datalist id={listId}>
                              {vocab.map(name => (<option key={name} value={name} />))}
                            </datalist>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                <div className="p-4 border-t border-white/10 space-y-2">
                  <div className="text-sm text-slate-200 font-medium">Folder labels</div>
                  <div className="flex flex-wrap gap-2">
                    {cur.global.map(lab => (
                      <Chip key={lab} onRemove={() => removeGlobal(open.sid, open.ts, lab)}>
                        {lab}
                      </Chip>
                    ))}
                    {cur.global.length === 0 && <span className="text-xs text-slate-500">No labels selected</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      list={`label-suggestions-global-${open.ts}`}
                      placeholder="Type a label (e.g., Bull Breakout) and press Enter"
                      className="flex-1 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-lg text-sm text-slate-100"
                      value={cur.drafts.global || ''}
                      onChange={e => setSelected(open.sid, open.ts, s => ({ ...s, drafts: { ...s.drafts, global: e.target.value } }))}
                      onKeyDown={e => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          addGlobal(open.sid, open.ts, cur.drafts.global)
                        }
                      }}
                    />
                    <button
                      onClick={() => addGlobal(open.sid, open.ts, cur.drafts.global)}
                      className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm border border-white/10"
                    >
                      Add
                    </button>
                    <datalist id={`label-suggestions-global-${open.ts}`}>
                      {[...PRESETS, ...vocab].map((name, i) => (
                        <option key={`${name}-${i}`} value={name} />
                      ))}
                    </datalist>
                  </div>

                  <div style={{ marginTop: '32px' }}>
                    <button
                      onClick={() => deleteFolder(open.sid, open.ts, grp.tz)}
                      className="px-3 py-2 rounded-lg bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-600/30 text-sm flex items-center gap-2"
                      title="Delete entire folder"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Folder
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      })()}

      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  )
}
