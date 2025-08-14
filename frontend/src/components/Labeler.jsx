// components/Labeler.jsx
import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { Trash2, Info, Loader2, Space } from 'lucide-react'

// --- small helpers ---
function fmtTS(ts, tz) {
  const n = Number(ts)
  if (!Number.isFinite(n)) return String(ts)
  const d = new Date(n)
  const opt = { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: tz || undefined }
  let time = new Intl.DateTimeFormat('en-US', opt).format(d)
  time = time.toLowerCase().replace(' ', '')
  const day  = new Intl.DateTimeFormat('en-GB', { day: 'numeric', timeZone: tz || undefined }).format(d)
  const mon  = new Intl.DateTimeFormat('en-US', { month: 'short',  timeZone: tz || undefined }).format(d)
  const year = new Intl.DateTimeFormat('en-GB', { year: 'numeric', timeZone: tz || undefined }).format(d)
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
          <ol className="list-decimal ml-5 space-y-2 text-sm text-slate-300">
            <li>Capture screen and create folders of images automatically.</li>
            <li>Open a folder and add <b>Folder labels</b> (top right presets help).</li>
            <li>Optionally add labels per image (Zone 0 / Zone 1 etc.).</li>
            <li>Save the folder. “Labeled” badge turns green.</li>
            <li>In <b>Training</b>, set epochs / LR / batch and check <b>Train on all sessions</b> if you want to include past dates.</li>
            <li>Click <b>Train</b>. Status shows <i>running / epoch</i>. Once done, predictions will use the new weights.</li>
          </ol>
          <div className="mt-4 flex justify-end">
            <button onClick={onClose}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ---------- Training controls ---------- */
function TrainControls({ sessionId }) {
  const [epochs, setEpochs] = useState(5)
  const [lr, setLR] = useState(1e-3)
  const [batch, setBatch] = useState(16)
  const [includeAll, setIncludeAll] = useState(false)
  const [status, setStatus] = useState('idle')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let t = null, alive = true
    async function poll() {
      try {
        const s = await api.trainStatus(sessionId)
        if (!alive) return
        setStatus(typeof s?.status === 'string' ? s.status : 'idle')
      } catch {}
      t = setTimeout(poll, 1200)
    }
    poll()
    return () => { alive = false; if (t) clearTimeout(t) }
  }, [sessionId])

  async function start() {
    setBusy(true)
    try {
      await api.train(sessionId, {
        epochs: Number(epochs) || 5,
        lr: Number(lr) || 1e-3,
        batch_size: Number(batch) || 16,
        include_all: !!includeAll
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="mb-6 rounded-2xl border border-slate-700/50 bg-gradient-to-b from-slate-900 to-slate-900/70 shadow-2xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-lg font-semibold text-slate-100">Training</div>
          <span className="text-xs px-2 py-1 rounded-full bg-slate-800 text-slate-300 border border-white/10">
            {status || 'idle'}
          </span>
        </div>

        <div className="grid md:grid-cols-12 gap-3">
          <label className="md:col-span-2 text-[13px] text-slate-300">Epochs
            <input type="number" min="1" step="1" value={epochs}
              onChange={e => setEpochs(Math.max(1, +e.target.value || 5))}
              className="mt-1 w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-white/10 text-slate-100" />
          </label>

          <label className="md:col-span-3 text-[13px] text-slate-300">Learning Rate
            <input type="number" step="0.0001" value={lr}
              onChange={e => setLR(+e.target.value || 0.001)}
              className="mt-1 w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-white/10 text-slate-100" />
          </label>

          <label className="md:col-span-2 text-[13px] text-slate-300">Batch Size
            <input type="number" min="1" step="1" value={batch}
              onChange={e => setBatch(Math.max(1, +e.target.value || 16))}
              className="mt-1 w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-white/10 text-slate-100" />
          </label>

          <label className="md:col-span-3 flex items-end gap-2 text-[13px] text-slate-300">
            <input type="checkbox" className="accent-emerald-500" checked={includeAll}
                   onChange={e => setIncludeAll(e.target.checked)} />
            Train on all sessions
          </label>

          <div className="md:col-span-2 flex items-end justify-end gap-2">
            <button
              onClick={start}
              disabled={busy}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium disabled:opacity-60"
            >
              {busy ? <span className="inline-flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" /> Starting…
              </span> : 'Train'}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

/* ---------- Main Labeler ---------- */
export default function Labeler({ sessionId }) {
  const [loading, setLoading] = useState(true)
  const [groups, setGroups] = useState([])     // [{session_id?, timestamp, zones, tz?, ann?, labeled?}]
  const [vocab, setVocab]   = useState([])     // [{id,name,slug,parent}]
  const [ann, setAnn]       = useState({})     // single-session annotations map
  const [onlyUnlabeled, setOnlyUnlabeled] = useState(true)
  const [includeAll, setIncludeAll] = useState(true)
  const [savingKey, setSavingKey] = useState(null)

  const [open, setOpen] = useState(null) // { sid, ts } | null
  const [sel, setSel] = useState({})     // edits buffer
  const [helpOpen, setHelpOpen] = useState(false) // NEW: help modal trigger at header

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

  // --- delete a whole folder ---
  async function deleteFolder(sid, ts) {
    if (!confirm(`Delete entire folder ${fmtTS(ts)} from session ${sid}? This cannot be undone.`)) return
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
  async function deleteImage(sid, ts, z) {
    const name = z?.path?.split('/').pop() || `zone_${z.id}`
    if (!confirm(`Delete image "${name}" from ${fmtTS(ts)}?`)) return
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

  // ------ folder grid ordering + optional unlabeled filtering ------
  const visibleGroups = useMemo(() => {
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
    return [...list].sort((a, b) => Number(a.timestamp) - Number(b.timestamp))
  }, [groups, ann, onlyUnlabeled, includeAll])

  // ------- UI -------
  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      {/* header row */}
      <div className="flex items-center justify-between mb-4">
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

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={includeAll} onChange={e => setIncludeAll(e.target.checked)} />
            Include previous sessions
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={onlyUnlabeled} onChange={e => setOnlyUnlabeled(e.target.checked)} />
            Show unlabeled only
          </label>
          <button
            onClick={async () => {
              setLoading(true)
              try {
                const gResp = includeAll ? await api.groupsAll(onlyUnlabeled) : await api.groups(sessionId)
                const groupsList = gResp?.groups || []
                const a = includeAll ? {} : await api.annotations(sessionId)
                const annMap = a || {}
                setGroups(groupsList)
                setAnn(includeAll ? {} : annMap)
                setSel(seedSel(groupsList, includeAll ? {} : annMap))
              } finally {
                setLoading(false)
              }
            }}
            className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Training controls (nice, compact) */}
      <TrainControls sessionId={sessionId} />

      {loading && <div className="text-slate-400 text-sm">Loading…</div>}
      {!loading && visibleGroups.length === 0 && (
        <div className="text-slate-400 text-sm">
          {onlyUnlabeled ? 'No unlabeled folders.' : 'No folders found.'}
        </div>
      )}

      {/* Folder grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        {visibleGroups.map(g => {
          const ts = String(g.timestamp)
          const sid = g.session_id || sessionId
          const previews = (g.zones || []).slice(0, 4).map(z => imgSrcFrom(sid, z.path))
          const isLabeled = includeAll ? !!g.labeled : !!(ann[ts]?.global?.length)

          return (
            <div
              key={`${sid}:${ts}`}
              className={`relative group rounded-2xl p-3 transition-colors backdrop-blur w-full text-left
                          border shadow-2xl
                          ${isLabeled ? 'bg-emerald-500/5 border-emerald-500/30'
                                      : 'bg-white/5 border-white/10'}`}
            >
              {/* make controls sit above everything */}
              <div className="pointer-events-none absolute inset-0 z-20">
                <span
                  className={`absolute top-2 left-2 px-2 py-0.5 rounded-full text-[10px] font-medium border pointer-events-auto
                             ${isLabeled
                               ? 'bg-emerald-500/15 text-emerald-200 border-emerald-400/30'
                               : 'bg-amber-500/10 text-amber-200 border-amber-400/20'}`}
                >
                  {isLabeled ? '✓ Labeled' : 'Unlabeled'}
                </span>
                <button
                  onClick={() => deleteFolder(sid, ts)}
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
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-medium border
                                  ${isLabeled
                                    ? 'bg-emerald-500/15 text-emerald-200 border-emerald-400/30'
                                    : 'bg-amber-500/10 text-amber-200 border-amber-400/20'}`}
                    >
                      {isLabeled ? '✓ Labeled' : 'Unlabeled'}
                    </span>

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

                {/* images grid */}
                <div className="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {(grp.zones || []).map(z => {
                    const listId = `label-suggestions-${open.ts}-${safeId(z.role)}`
                    return (
                      <div key={z.id} className="relative bg-slate-800/60 border border-white/10 rounded-xl overflow-hidden">
                        {/* delete image (in front) */}
                        <button
                          onClick={() => deleteImage(open.sid, open.ts, z)}
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
                          src={imgSrcFrom(open.sid, z.path)}
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

                  {/* bottom-left delete button (properly spaced) */}
                  <div style={{ marginTop: '32px' }}>
                    <button
                      onClick={() => deleteFolder(open.sid, open.ts)}
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
