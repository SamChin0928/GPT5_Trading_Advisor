// components/Labeler.jsx
import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { Trash2, Info, CheckCircle2, Clock, Calendar, ChevronDown, ChevronRight, X } from 'lucide-react'
import TrainControls from './TrainControls' // <-- shared component

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

// New: date helpers
const localYMD = (d) => {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}` // YYYY-MM-DD
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

// use API helper so dev uses /api/data and prod uses /data
const imgSrcFrom = (sessionId, rel) => api.imageUrl(sessionId, rel)
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

/* ---------- Help modal ---------- */
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

          {/* Steps */}
          <ol className="list-decimal ml-5 space-y-2 text-sm text-slate-300">
            <li>
              Capture your screen — folders (by timestamp) are created automatically at the
              start of each minute.
            </li>
            <li>
              In <b>Labeler</b>, enable <b>Show unlabeled only</b> to focus on new data. Click a
              folder to open it.
            </li>
            <li>
              At the top you may see <b>Model prediction (folder)</b> with confidence
              percentages. These are suggestions and are <i>not</i> saved automatically.
            </li>
            <li>
              Apply the <b>correct Folder labels</b> (use the presets for speed, or type your own)
              and optionally add per-image labels (Zone 0 / Zone 1…).
            </li>
            <li>
              Click <b>Save</b>. The badge turns green (“Labeled”) and your feedback is stored.
              The model updates <i>incrementally</i> in the background to improve future
              predictions.
            </li>
            <li>
              After you’ve labeled a batch, go to <b>Training</b> and click <b>Train</b>.
              Training rebuilds the global heads (epochs, LR, batch, thresholds) from your data.
              Use <b>Train on all sessions</b> to include past dates. New predictions will then
              use the latest model.
            </li>
            <li>
              Use the date filters to browse by day, and the trash button to delete bad folders
              or images. The list refreshes automatically after changes.
            </li>
          </ol>

          {/* How the learning loop works */}
          <div className="mt-4 rounded-lg bg-slate-800/60 border border-white/10 p-3 text-[13px] text-slate-300">
            <div className="font-medium text-slate-200 mb-1">How the model learns</div>
            <ul className="list-disc ml-5 space-y-1">
              <li>
                The card shows the current model’s top guess (e.g. <code>Pred: Consolidation (57%)</code>).
              </li>
              <li>
                Your saved labels are the ground truth. Each save gives the model a small
                incremental update; full <b>Train</b> consolidates everything for best accuracy.
              </li>
              <li>
                If the model is wrong, simply apply the correct label(s) and save — that’s how you
                teach it.
              </li>
            </ul>
          </div>

          <div className="mt-4 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm"
            >
              Close
            </button>
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
  const [groups, setGroups] = useState([])     // [{session_id?, timestamp, zones, tz?, ann?, labeled?, pred?}]
  const [vocab, setVocab]   = useState([])     // [{id,name,slug,parent}]
  const [ann, setAnn]       = useState({})     // single-session annotations map
  const [onlyUnlabeled, setOnlyUnlabeled] = useState(true)
  const [includeAll, setIncludeAll] = useState(true)
  const [savingKey, setSavingKey] = useState(null)

  const [open, setOpen] = useState(null) // { sid, ts } | null
  const [sel, setSel] = useState({})     // edits buffer
  const [helpOpen, setHelpOpen] = useState(false) // help modal

  // NEW: date range filters
  const [dateFrom, setDateFrom] = useState('') // YYYY-MM-DD (local)
  const [dateTo, setDateTo] = useState('')     // YYYY-MM-DD (local)
  const [collapsedDays, setCollapsedDays] = useState(() => new Set())

  // CACHE-BUST for thumbnails on refresh
  const [cacheBust, setCacheBust] = useState(0) // increments on reload()

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
        const vocabList  = vocabResp?.labels || []
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

  // AUTO-REFRESH: custom event + storage ping + visibility change
  useEffect(() => {
    const onUpdated = () => reload()
    const onStorage = (e) => {
      if (e.key === 'captures_updated_at') reload()
    }
    const onVisible = () => {
      if (!document.hidden) reload()
    }
    window.addEventListener('captures:updated', onUpdated)
    window.addEventListener('storage', onStorage)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.removeEventListener('captures:updated', onUpdated)
      window.removeEventListener('storage', onStorage)
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
        global: [...(base.global || [])],
        by_role: roleMap,
        drafts: draft
      }
    }
    return seeded
  }

  const vocabByName = useMemo(() => {
    const m = new Map()
    for (const e of vocab) m.set((e.name || '').trim().toLowerCase(), e)
    return m
  }, [vocab])

  const vocabById = useMemo(() => {
    const m = new Map()
    for (const e of vocab) m.set(e.id, e)
    return m
  }, [vocab])

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
    const key = (name || '').trim().toLowerCase()
    if (!key) return null
    let entry = vocabByName.get(key)
    if (!entry) {
      const out = await api.createVocabLabel(name.trim())
      entry = out?.label
      setVocab(v => v.find(x => x.id === entry.id) ? v : [...v, entry])
    }
    return entry
  }

  // global (folder) labels
  async function addGlobal(sid, ts, name) {
    const entry = await ensureLabel(name); if (!entry) return
    setSelected(sid, ts, cur => {
      if (cur.global.includes(entry.id)) return cur
      return { ...cur, global: [...cur.global, entry.id], drafts: { ...cur.drafts, global: '' } }
    })
  }
  const removeGlobal = (sid, ts, lid) =>
    setSelected(sid, ts, cur => ({ ...cur, global: cur.global.filter(x => x !== lid) }))

  // per-role image labels
  async function addRole(sid, ts, role, name) {
    const entry = await ensureLabel(name); if (!entry) return
    setSelected(sid, ts, cur => {
      const curList = cur.by_role[role] || []
      if (curList.includes(entry.id)) return cur
      return {
        ...cur,
        by_role: { ...cur.by_role, [role]: [...curList, entry.id] },
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
      const grp = groups.find(g => (g.session_id || sessionId) === sid && String(g.timestamp) === String(ts))
      let model_feedback = undefined
      if (grp?.pred?.model) {
        const predPairs = (grp.pred.global || []).map(p => [p.id, p.conf])
        model_feedback = { pred: predPairs, meta: grp.pred.model }
      }

      await api.annotate(sid, { timestamp: ts, global_labels: data.global, by_role: data.by_role, model_feedback })
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

  // --- delete a whole folder ---
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

  // --- delete single image inside a folder ---
  async function deleteImage(sid, ts, tz, z) {
    const name = z?.path?.split('/').pop() || `zone_${z.id}`
    if (!confirm(`Delete image "${name}" from ${fmtTS(ts, tz)}?`)) return
    try {
      await api.deleteGroupImage(sid, ts, z.path || name)
      setGroups(gs => {
        const out = []
        for (const g of gs) {
          if ((g.session_id || sessionId) === sid && String(g.timestamp) === String(ts)) {
            const nz = (g.zones || []).filter(zz => (zz.path || '').toString() !== (z.path || '').toString())
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

  // ------ filtering & grouping ------
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

  // ------- UI controls: date quick presets -------
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

  function toggleDay(dayKey) {
    setCollapsedDays(prev => {
      const n = new Set(prev)
      if (n.has(dayKey)) n.delete(dayKey); else n.add(dayKey)
      return n
    })
  }

  // shared reload helper
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
      setCacheBust(Date.now()) // CACHE-BUST thumbnails on refresh
    } finally {
      setLoading(false)
    }
  }

  // ------- UI -------
  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      {/* header row */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-slate-100">Labeler</h2>
          <button
            onClick={() => setHelpOpen(true)}
            className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600/60
                       text-xs text-slate-200 border border-white/10"
            title="How to use the model"
          >
            <Info className="w-3.5 h-3.5" />
            Help
          </button>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Calendar className="w-4 h-4 opacity-80" />
            <span>From</span>
            <input
              type="date"
              value={dateFrom}
              onChange={e => setDateFrom(e.target.value)}
              className="px-2 py-1 rounded bg-slate-800 border border-white/10 text-slate-100"
            />
            <span>to</span>
            <input
              type="date"
              value={dateTo}
              onChange={e => setDateTo(e.target.value)}
              className="px-2 py-1 rounded bg-slate-800 border border-white/10 text-slate-100"
            />
            {(dateFrom || dateTo) && (
              <button
                onClick={() => { setDateFrom(''); setDateTo('') }}
                className="ml-1 p-1 rounded bg-slate-700 hover:bg-slate-600 border border-white/10 text-slate-200"
                title="Clear dates"
              >
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
              <button
                key={btn.k}
                onClick={() => applyPreset(btn.k)}
                className="px-2 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-600/60 text-xs text-slate-200 border border-white/10"
              >
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

          <button
            onClick={reload}
            className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Training controls */}
      <TrainControls sessionId={sessionId} />

      {loading && <div className="text-slate-400 text-sm">Loading…</div>}
      {!loading && groupedByDay.length === 0 && (
        <div className="text-slate-400 text-sm m-4">
          {onlyUnlabeled ? 'No unlabeled folders in this range.' : 'No folders found in this range.'}
        </div>
      )}

      {/* Grouped-by-day sections */}
      <div className="space-y-6 mt-4">
        {groupedByDay.map(({ day, label, groups: dayGroups }) => {
          const collapsed = collapsedDays.has(day)
          return (
            <section key={day} className="bg-black/10 border border-white/10 rounded-2xl overflow-hidden">
              {/* Day header */}
              <button
                onClick={() => toggleDay(day)}
                className="w-full flex items-center justify-between px-4 py-3 bg-white/[0.03] border-b border-white/10"
              >
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
                    // CACHE-BUST on previews
                    const previews = (g.zones || []).slice(0, 4).map(z => `${imgSrcFrom(sid, z.path)}${cacheBust ? `?v=${cacheBust}` : ''}`)
                    const isLabeled = includeAll ? !!g.labeled : !!(ann[ts]?.global?.length)

                    // top prediction (if any)
                    let predText = null
                    if (g.pred && Array.isArray(g.pred.global) && g.pred.global.length > 0) {
                      const top = g.pred.global[0]
                      const name = vocabById.get(top.id)?.name || top.id
                      const pct = Math.round((Number(top.conf) || 0) * 100)
                      predText = `${name} (${pct}%)`
                    }

                    return (
                      <div
                        key={`${sid}:${ts}`}
                        className={`relative group rounded-2xl p-3 transition-colors backdrop-blur w-full text-left
                                    border shadow-2xl
                                    ${isLabeled ? 'bg-emerald-500/5 border-emerald-500/30'
                                                : 'bg-white/5 border-white/10'}`}
                      >
                        {/* overlay controls */}
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

                        {/* clickable area */}
                        <button onClick={() => setOpen({ sid, ts })} className="block w-full text-left relative z-10">
                          <div className="relative h-32">
                            {previews.map((src, i) => (
                              <img
                                key={i}
                                src={src}
                                alt=""
                                className={`absolute inset-0 w-full h-full object-contain rounded-xl bg-black/60
                                           ring-1 ring-white/10 shadow-xl
                                           ${i === 0 ? '' : i === 1 ? 'translate-x-2 -translate-y-2 scale-95' :
                                              i === 2 ? 'translate-x-4 -translate-y-4 scale-90' : 'translate-x-6 -translate-y-6 scale-[.85]'}`}
                                style={{ zIndex: 5 - i }}
                                loading="lazy"
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

      {/* Modal (expanded folder) */}
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

        return (
          <div className="fixed inset-0 z-[180]">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(null)} />
            <div className="absolute inset-0 flex items-center justify-center p-4">
              <div className="w-full max-w-5xl rounded-2xl border border-white/10 bg-slate-900/90 shadow-2xl overflow-hidden">
                {/* header */}
                <div className="flex items-center justify-between p-4 border-b border-white/10">
                  <div>
                    <div className="text-slate-100 font-semibold">Folder • {fmtTS(open.ts, grp.tz)}</div>
                    <div className="text-slate-400 text-xs">{grp.tz || 'UTC'} • {grp.zones?.length || 0} image(s)</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* compact presets row */}
                    <div className="hidden md:flex flex-wrap gap-2 mr-2">
                      {PRESETS.map(q => (
                        <button key={q} onClick={() => addGlobal(open.sid, open.ts, q)}
                          className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[12px] text-slate-200 border border-white/10">
                          {q}
                        </button>
                      ))}
                    </div>

                    {/* status pill sits next to Save */}
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
                    <button onClick={() => setOpen(null)}
                      className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm">
                      Close
                    </button>
                  </div>
                </div>

                {/* Optional: show model predictions row if present */}
                {grp.pred && Array.isArray(grp.pred.global) && grp.pred.global.length > 0 && (
                  <div className="px-4 pt-4">
                    <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Model prediction (folder)</div>
                    <div className="flex flex-wrap gap-2">
                      {grp.pred.global.slice(0, 5).map(p => (
                        <span key={p.id}
                          className="px-2 py-1 rounded-full bg-slate-700/60 border border-white/10 text-slate-200 text-xs">
                          {(vocabById.get(p.id)?.name || p.id)} ({Math.round((Number(p.conf)||0)*100)}%)
                        </span>
                      ))}
                      <span className="px-2 py-1 rounded-lg bg-slate-800 border border-white/10 text-slate-300 text-xs">
                        Apply all
                      </span>
                    </div>
                  </div>
                )}

                {/* images grid */}
                <div className="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {(grp.zones || []).map(z => {
                    const listId = `label-suggestions-${open.ts}-${safeId(z.role)}`
                    return (
                      <div key={z.id} className="relative bg-slate-800/60 border border-white/10 rounded-xl overflow-hidden">
                        {/* delete image (in front) */}
                        <button
                          onClick={() => deleteImage(open.sid, open.ts, grp.tz, z)}
                          className="absolute top-2 right-2 p-1.5 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-200 border border-red-400/30 z-20"
                          title="Delete image"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>

                        <div className="p-2 text-xs text-slate-400 flex items-center justify-between">
                          <span>Zone {z.id}</span>
                          <span className="text-slate-500">{z.role || `zone_${z.id}`}</span>
                        </div>
                        <img
                          src={`${imgSrcFrom(open.sid, z.path)}${cacheBust ? `?v=${cacheBust}` : ''}`}
                          alt=""
                          className="w-full h-40 object-contain bg-black"
                          loading="lazy"
                        />
                        {/* per-role labels */}
                        <div className="p-2 space-y-2">
                          <div className="flex flex-wrap gap-1">
                            {(cur.by_role[z.role] || []).map(lid => (
                              <Chip key={lid} onRemove={() => removeRole(open.sid, open.ts, z.role, lid)}>
                                {vocabById.get(lid)?.name || lid}
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
                              {vocab.map(v => (
                                <option key={v.id} value={v.name} />
                              ))}
                            </datalist>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Folder/global labels */}
                <div className="p-4 border-t border-white/10 space-y-2">
                  <div className="text-sm text-slate-200 font-medium">Folder labels</div>
                  <div className="flex flex-wrap gap-2">
                    {cur.global.map(lid => (
                      <Chip key={lid} onRemove={() => removeGlobal(open.sid, open.ts, lid)}>
                        {vocabById.get(lid)?.name || lid}
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
                      {[...PRESETS, ...vocab.map(v => v.name)].map((name, i) => (
                        <option key={`${name}-${i}`} value={name} />
                      ))}
                    </datalist>
                  </div>

                  {/* bottom-left delete button */}
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

      {/* Header Help modal */}
      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  )
}
