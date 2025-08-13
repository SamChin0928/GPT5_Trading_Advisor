// components/Labeler.jsx
import React, { useEffect, useMemo, useState } from 'react'
import { api, BASE } from '../lib/api'

function fmtTS(ts) {
  const n = Number(ts)
  if (!Number.isFinite(n)) return String(ts)
  const d = new Date(n)
  return d.toLocaleString()
}

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

export default function Labeler({ sessionId }) {
  const [loading, setLoading] = useState(true)
  const [groups, setGroups] = useState([])     // [{timestamp, zones:[{id,role,path}], tz?}]
  const [vocab, setVocab]   = useState([])     // [{id,name,slug,parent}]
  const [ann, setAnn]       = useState({})     // { [ts]: {global:[ids], by_role:{role:[ids]}, notes } }
  const [filter, setFilter] = useState('')
  const [onlyUnlabeled, setOnlyUnlabeled] = useState(true)

  // modal
  const [openTs, setOpenTs] = useState(null)

  // local editable selection (buffer before save)
  // sel[ts] = { global:[ids], by_role:{role:[ids]}, drafts:{global:'', [role]:''] }
  const [sel, setSel] = useState({})

  // ------- data loading -------
  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      try {
        const [v, g, a] = await Promise.all([
          api.getVocab(),
          api.groups(sessionId),
          api.annotations(sessionId)
        ])
        if (cancelled) return
        const vocabList  = v?.labels || []
        const groupsList = g?.groups || []
        const annMap     = a || {}

        setVocab(vocabList)
        setGroups(groupsList)
        setAnn(annMap)

        const seeded = {}
        for (const grp of groupsList) {
          const ts = String(grp.timestamp)
          const base = annMap[ts] || {}
          const draft = { global: '' }
          const roleMap = { ...(base.by_role || {}) }
          for (const z of grp.zones || []) {
            if (!(z.role in roleMap)) roleMap[z.role] = []
            if (!(z.role in draft)) draft[z.role] = ''
          }
          seeded[ts] = {
            global: [...(base.global || [])],
            by_role: roleMap,
            drafts: draft
          }
        }
        setSel(seeded)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [sessionId])

  const vocabByName = useMemo(() => {
    const m = new Map()
    for (const e of vocab) m.set(e.name.toLowerCase(), e)
    return m
  }, [vocab])
  const vocabById = useMemo(() => {
    const m = new Map()
    for (const e of vocab) m.set(e.id, e)
    return m
  }, [vocab])

  const selected = (ts) => sel[String(ts)] || { global: [], by_role: {}, drafts: { global: '' } }
  const setSelected = (ts, updater) => {
    const key = String(ts)
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
      const out = await api.createVocabLabel(name)
      entry = out?.label
      setVocab(v => v.find(x => x.id === entry.id) ? v : [...v, entry])
    }
    return entry
  }

  // global (folder) labels
  async function addGlobal(ts, name) {
    const entry = await ensureLabel(name); if (!entry) return
    setSelected(ts, cur => {
      if (cur.global.includes(entry.id)) return cur
      return { ...cur, global: [...cur.global, entry.id], drafts: { ...cur.drafts, global: '' } }
    })
  }
  const removeGlobal = (ts, lid) =>
    setSelected(ts, cur => ({ ...cur, global: cur.global.filter(x => x !== lid) }))

  // per-role image labels
  async function addRole(ts, role, name) {
    const entry = await ensureLabel(name); if (!entry) return
    setSelected(ts, cur => {
      const curList = cur.by_role[role] || []
      if (curList.includes(entry.id)) return cur
      return {
        ...cur,
        by_role: { ...cur.by_role, [role]: [...curList, entry.id] },
        drafts: { ...cur.drafts, [role]: '' }
      }
    })
  }
  const removeRole = (ts, role, lid) =>
    setSelected(ts, cur => ({
      ...cur,
      by_role: { ...cur.by_role, [role]: (cur.by_role[role] || []).filter(x => x !== lid) }
    }))

  async function saveGroup(ts) {
    const data = selected(ts)
    await api.annotate(sessionId, { timestamp: ts, global_labels: data.global, by_role: data.by_role })
    setAnn(a => ({ ...a, [String(ts)]: { ...(a[String(ts)] || {}), global: data.global, by_role: data.by_role } }))
  }

  const imgSrc = (rel) => `${BASE}/data/sessions/${encodeURIComponent(sessionId)}/${rel}`

  // ------ folder grid filtering ------
  const visibleGroups = useMemo(() => {
    let list = groups || []
    if (filter.trim()) {
      const f = filter.trim().toLowerCase()
      list = list.filter(g =>
        String(g.timestamp).includes(f) ||
        (g?.zones || []).some(z => (z.role || '').toLowerCase().includes(f))
      )
    }
    if (onlyUnlabeled) {
      list = list.filter(g => {
        const rec = ann[String(g.timestamp)]
        const hasGlobal = !!(rec && Array.isArray(rec.global) && rec.global.length)
        return !hasGlobal
      })
    }
    return list
  }, [groups, ann, filter, onlyUnlabeled])

  // ------- UI -------
  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 shadow-2xl border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-100">Labeler</h2>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={onlyUnlabeled} onChange={e => setOnlyUnlabeled(e.target.checked)} />
            Show unlabeled only
          </label>
          <input
            placeholder="Filter by timestamp or role…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-100"
          />
          <button
            onClick={async () => {
              setLoading(true)
              try {
                const [g, a] = await Promise.all([api.groups(sessionId), api.annotations(sessionId)])
                setGroups(g?.groups || [])
                setAnn(a || {})
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

      {loading && <div className="text-slate-400 text-sm">Loading…</div>}

      {!loading && visibleGroups.length === 0 && (
        <div className="text-slate-400 text-sm">
          {onlyUnlabeled ? 'No unlabeled folders.' : 'No folders found.'}
        </div>
      )}

      {/* Folder grid (iOS-style translucent stack) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        {visibleGroups.map(g => {
          const ts = String(g.timestamp)
          // take up to 4 preview images
          const previews = (g.zones || []).slice(0, 4).map(z => imgSrc(z.path))
          return (
            <button
              key={ts}
              onClick={() => setOpenTs(ts)}
              className="relative group rounded-2xl p-3 bg-white/5 hover:bg-white/10 transition-colors
                         backdrop-blur border border-white/10 shadow-2xl w-full text-left"
            >
              {/* stacked thumbs */}
              <div className="relative h-32">
                {previews.map((src, i) => (
                  <img
                    key={i}
                    src={src}
                    alt=""
                    className={`absolute inset-0 w-full h-full object-contain rounded-xl bg-black/60
                               ring-1 ring-white/10 shadow-xl
                               ${i === 0 ? '' : i === 1 ? 'translate-x-2 -translate-y-2 scale-95' :
                                  i === 2 ? 'translate-x-4 -translate-y-4 scale-90' : 'translate-x-6 -translate-y-6 scale-85'}`}
                    style={{ zIndex: 10 - i }}
                    loading="lazy"
                  />
                ))}
              </div>
              {/* footer info */}
              <div className="mt-3 flex items-center justify-between">
                <div>
                  <div className="text-slate-100 text-sm font-medium">{fmtTS(ts)}</div>
                  <div className="text-slate-400 text-xs">{g.zones?.length || 0} image(s)</div>
                </div>
                <span className="text-slate-300 text-xs px-2 py-1 rounded-lg bg-black/30 border border-white/10">
                  Open
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {/* Modal (expanded folder) */}
      {openTs && (() => {
        const grp = groups.find(x => String(x.timestamp) === String(openTs))
        if (!grp) return null
        const cur = selected(openTs)
        const applied = ann[openTs]?.global || []
        const changed = JSON.stringify(applied) !== JSON.stringify(cur.global)

        return (
          <div className="fixed inset-0 z-50">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpenTs(null)} />
            <div className="absolute inset-0 flex items-center justify-center p-4">
              <div className="w-full max-w-5xl rounded-2xl border border-white/10 bg-slate-900/90 shadow-2xl">
                <div className="flex items-center justify-between p-4 border-b border-white/10">
                  <div>
                    <div className="text-slate-100 font-semibold">Folder • {fmtTS(openTs)}</div>
                    <div className="text-slate-400 text-xs">{grp.tz || 'UTC'} • {grp.zones?.length || 0} image(s)</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {['Bull Breakout','Bear Breakout','Neutral','Consolidation'].map(q => (
                      <button key={q} onClick={() => addGlobal(openTs, q)}
                        className="hidden md:inline px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-white/10">
                        {q}
                      </button>
                    ))}
                    <button
                      onClick={() => saveGroup(openTs)}
                      disabled={!changed}
                      className={`px-3 py-2 rounded-lg text-sm ${
                        changed ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-slate-700 text-slate-300 cursor-not-allowed'
                      }`}
                    >
                      {changed ? 'Save' : 'Saved'}
                    </button>
                    <button onClick={() => setOpenTs(null)}
                      className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10 text-sm">
                      Close
                    </button>
                  </div>
                </div>

                <div className="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {(grp.zones || []).map(z => (
                    <div key={z.id} className="bg-slate-800/60 border border-white/10 rounded-xl overflow-hidden">
                      <div className="p-2 text-xs text-slate-400 flex items-center justify-between">
                        <span>Zone {z.id}</span>
                        <span className="text-slate-500">{z.role || `zone_${z.id}`}</span>
                      </div>
                      <img
                        src={imgSrc(z.path)}
                        alt=""
                        className="w-full h-40 object-contain bg-black"
                        loading="lazy"
                      />
                      {/* per-role labels */}
                      <div className="p-2 space-y-2">
                        <div className="flex flex-wrap gap-1">
                          {(cur.by_role[z.role] || []).map(lid => (
                            <Chip key={lid} onRemove={() => removeRole(openTs, z.role, lid)}>
                              {vocabById.get(lid)?.name || lid}
                            </Chip>
                          ))}
                          {(cur.by_role[z.role] || []).length === 0 && (
                            <span className="text-[11px] text-slate-500">No labels</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            list={`label-suggestions-${openTs}-${z.role}`}
                            placeholder="Add label to this image…"
                            className="flex-1 px-2 py-1 bg-slate-900/60 border border-white/10 rounded text-xs text-slate-100"
                            value={cur.drafts[z.role] || ''}
                            onChange={e => setSelected(openTs, s => ({ ...s, drafts: { ...s.drafts, [z.role]: e.target.value } }))}
                            onKeyDown={e => {
                              if (e.key === 'Enter') {
                                e.preventDefault()
                                addRole(openTs, z.role, cur.drafts[z.role])
                              }
                            }}
                          />
                          <button
                            onClick={() => addRole(openTs, z.role, cur.drafts[z.role])}
                            className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-xs border border-white/10"
                          >
                            Add
                          </button>
                          <datalist id={`label-suggestions-${openTs}-${z.role}`}>
                            {vocab.map(v => (
                              <option key={v.id} value={v.name} />
                            ))}
                          </datalist>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Folder/global labels */}
                <div className="p-4 border-t border-white/10 space-y-2">
                  <div className="text-sm text-slate-200 font-medium">Folder labels</div>
                  <div className="flex flex-wrap gap-2">
                    {cur.global.map(lid => (
                      <Chip key={lid} onRemove={() => removeGlobal(openTs, lid)}>
                        {vocabById.get(lid)?.name || lid}
                      </Chip>
                    ))}
                    {cur.global.length === 0 && <span className="text-xs text-slate-500">No labels selected</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      list={`label-suggestions-global-${openTs}`}
                      placeholder="Type a label (e.g., Bull Breakout) and press Enter"
                      className="flex-1 px-3 py-2 bg-slate-900/60 border border-white/10 rounded-lg text-sm text-slate-100"
                      value={cur.drafts.global || ''}
                      onChange={e => setSelected(openTs, s => ({ ...s, drafts: { ...s.drafts, global: e.target.value } }))}
                      onKeyDown={e => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          addGlobal(openTs, cur.drafts.global)
                        }
                      }}
                    />
                    <button
                      onClick={() => addGlobal(openTs, cur.drafts.global)}
                      className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm border border-white/10"
                    >
                      Add
                    </button>
                    <datalist id={`label-suggestions-global-${openTs}`}>
                      {vocab.map(v => (
                        <option key={v.id} value={v.name} />
                      ))}
                    </datalist>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
