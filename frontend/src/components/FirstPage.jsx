// FirstPage.jsx — now supports MOCK MODE (no Firebase required) for points (1) Login + (2) Process List
// - Use `?mock=1` in the URL or set VITE_USE_MOCK=1 to run without Firebase.
// - In mock mode, data is saved to localStorage under `mock:users/<uid>:processes`.
// - When you're ready to use Firebase, remove the `?mock=1` (or set VITE_USE_MOCK=0) and
//   fill in the firebaseConfig below. The code lazy‑loads Firebase so the bundle
//   still works even if the package isn’t installed during mock usage.
//
// Quick setup:
//   npm i lucide-react   // (no firebase needed for mock)
//   (optional later) npm i firebase
//   Ensure Tailwind is configured
//
// Security (when switching to Firebase):
// Firestore rules to scope data by user:
// rules_version = '2';
// service cloud.firestore {
//   match /databases/{database}/documents {
//     match /users/{userId}/processes/{processId} {
//       allow read, write: if request.auth != null && request.auth.uid == userId;
//     }
//   }
// }

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Search, Plus, LogOut, ChevronRight, Loader2 } from 'lucide-react'

// -------------------------------------------------------------------------------------
// 0) Mode detection
// -------------------------------------------------------------------------------------
const USE_MOCK = (() => {
  try {
    if (typeof window !== 'undefined') {
      const p = new URLSearchParams(window.location.search)
      if (p.has('mock')) return true
    }
    // Vite-style env check (guarded)
    // @ts-ignore
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_USE_MOCK === '1') return true
  } catch {}
  return false
})()

// -------------------------------------------------------------------------------------
// 1) Firebase (lazy) — only loaded if USE_MOCK is false
// -------------------------------------------------------------------------------------
const firebaseConfig = {
  apiKey: 'YOUR_API_KEY',
  authDomain: 'YOUR_PROJECT_ID.firebaseapp.com',
  projectId: 'YOUR_PROJECT_ID',
  appId: 'YOUR_APP_ID',
}

let _fb = null
async function fb() {
  if (_fb) return _fb
  const appMod = await import('firebase/app')
  const authMod = await import('firebase/auth')
  const fsMod   = await import('firebase/firestore')

  if (appMod.getApps().length === 0) appMod.initializeApp(firebaseConfig)

  _fb = {
    app: appMod,
    auth: authMod.getAuth(),
    fs: fsMod.getFirestore(),
    GoogleAuthProvider: authMod.GoogleAuthProvider,
    signInWithPopup: authMod.signInWithPopup,
    onAuthStateChanged: authMod.onAuthStateChanged,
    signOut: authMod.signOut,
    collection: fsMod.collection,
    addDoc: fsMod.addDoc,
    serverTimestamp: fsMod.serverTimestamp,
    query: fsMod.query,
    orderBy: fsMod.orderBy,
    onSnapshot: fsMod.onSnapshot,
  }
  return _fb
}

// -------------------------------------------------------------------------------------
// 2) Mock adapters
// -------------------------------------------------------------------------------------
function useAuthMock() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(false)
  function signIn() {
    setLoading(true)
    setTimeout(() => {
      setUser({ uid: 'demo-user', email: 'demo@example.com', displayName: 'Demo User' })
      setLoading(false)
    }, 300)
  }
  function signOut() { setUser(null) }
  return { user, loading, signIn, signOut, mode: 'mock' }
}

function loadMockProcesses(uid) {
  try {
    const raw = localStorage.getItem(`mock:users/${uid}:processes`)
    const arr = raw ? JSON.parse(raw) : []
    return Array.isArray(arr) ? arr : []
  } catch { return [] }
}
function saveMockProcesses(uid, rows) {
  try { localStorage.setItem(`mock:users/${uid}:processes`, JSON.stringify(rows)) } catch {}
}

// -------------------------------------------------------------------------------------
// 3) Firebase auth hook
// -------------------------------------------------------------------------------------
function useAuthFirebase() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let unsub = () => {}
    ;(async () => {
      const { onAuthStateChanged } = await fb()
      unsub = onAuthStateChanged((await fb()).auth, (u) => { setUser(u); setLoading(false) })
    })()
    return () => unsub()
  }, [])

  async function signIn() {
    const { auth, GoogleAuthProvider, signInWithPopup } = await fb()
    const provider = new GoogleAuthProvider()
    await signInWithPopup(auth, provider)
  }
  async function signOutFn() { const { auth, signOut } = await fb(); await signOut(auth) }

  return { user, loading, signIn, signOut: signOutFn, mode: 'firebase' }
}

// -------------------------------------------------------------------------------------
// 4) Main Page
// -------------------------------------------------------------------------------------
export default function FirstPage() {
  const authApi = USE_MOCK ? useAuthMock() : useAuthFirebase()

  if (authApi.loading) {
    return (
      <div className="min-h-screen grid place-items-center bg-neutral-950 text-neutral-200">
        <div className="flex items-center gap-3 text-neutral-300">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Loading…</span>
        </div>
      </div>
    )
  }

  return authApi.user ? (
    <ProcessesView authApi={authApi} />
  ) : (
    <LoginView authApi={authApi} />
  )
}

// -------------------------------------------------------------------------------------
// 5) Login View (supports mock + firebase)
// -------------------------------------------------------------------------------------
function LoginView({ authApi }) {
  const isMock = authApi.mode === 'mock'

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="max-w-5xl mx-auto px-6 py-20">
        {isMock && (
          <div className="mb-4 text-xs inline-flex items-center gap-2 rounded-full border border-yellow-700/40 bg-yellow-900/20 px-3 py-1 text-yellow-200">
            <span className="font-medium">Mock mode</span>
            <span className="text-yellow-300/80">(no Firebase required)</span>
          </div>
        )}
        <header className="mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">GPT‑5 Trading — Console</h1>
          <p className="text-neutral-400 mt-2">Sign in to access your processes and continue where you left off.</p>
        </header>

        <div className="grid md:grid-cols-2 gap-8 items-start">
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6 shadow-lg">
            <h2 className="text-xl font-semibold">Welcome</h2>
            <p className="text-neutral-400 mt-1">Choose how to continue.</p>
            <div className="mt-6 flex flex-col gap-3">
              <button
                onClick={authApi.signIn}
                className="inline-flex items-center gap-2 rounded-xl px-4 py-2 bg-white text-neutral-900 hover:bg-neutral-200 transition"
              >
                {isMock ? (
                  <>
                    <span>Continue in Demo Mode</span>
                  </>
                ) : (
                  <>
                    {/* Simple G icon */}
                    <svg viewBox="0 0 24 24" className="w-5 h-5" aria-hidden>
                      <path fill="#EA4335" d="M12 10.2v3.8h5.4c-.2 1.4-1.6 4.1-5.4 4.1A6.2 6.2 0 1 1 12 5.8c1.8 0 3 .8 3.7 1.5l2.5-2.4C16.8 3.3 14.6 2.4 12 2.4 6.9 2.4 2.8 6.5 2.8 11.6S6.9 20.8 12 20.8c3.6 0 6-1.2 7.3-3.4 1.1-1.6 1.3-3.6 1.3-4.5 0-.5 0-1 0-1.4H12z"/>
                    </svg>
                    <span>Sign in with Google</span>
                  </>
                )}
              </button>

              {!isMock && (
                <p className="text-xs text-neutral-500">
                  Tip: You can also enable Email/Password or other providers later in Firebase Auth.
                </p>
              )}

              {isMock && (
                <div className="text-xs text-neutral-500">
                  Running with localStorage. To switch to Firebase later, remove <code>?mock=1</code> from the URL
                  (or set <code>VITE_USE_MOCK=0</code>), install <code>firebase</code>, and fill your config.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/40 p-6">
            <h3 className="font-semibold text-neutral-200">What happens next?</h3>
            <ul className="mt-3 space-y-2 text-neutral-400 text-sm">
              <li>• After sign‑in, you’ll be redirected to your personal process list.</li>
              <li>• Create new processes or search existing ones.</li>
              <li>• Clicking a process will route you to its workspace.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// -------------------------------------------------------------------------------------
// 6) Processes View — works with mock or firebase
// -------------------------------------------------------------------------------------
function ProcessesView({ authApi }) {
  const user = authApi.user
  const isMock = authApi.mode === 'mock'

  const [q, setQ] = useState('')
  const [adding, setAdding] = useState(false)
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const [processes, setProcesses] = useState([])
  const [loading, setLoading] = useState(true)

  // Subscribe to processes (mock or firebase)
  useEffect(() => {
    let unsub = () => {}
    setLoading(true)

    if (isMock) {
      const rows = loadMockProcesses(user.uid)
      setProcesses(rows)
      setLoading(false)

      // also listen to storage changes (if multiple tabs)
      const onStorage = (e) => {
        if (e.key === `mock:users/${user.uid}:processes`) {
          setProcesses(loadMockProcesses(user.uid))
        }
      }
      window.addEventListener('storage', onStorage)
      unsub = () => window.removeEventListener('storage', onStorage)
    } else {
      ;(async () => {
        try {
          const { fs, collection, query, orderBy, onSnapshot } = await fb()
          const colRef = collection(fs, 'users', user.uid, 'processes')
          const qy = query(colRef, orderBy('createdAt', 'desc'))
          unsub = onSnapshot(qy, (snap) => {
            const rows = snap.docs.map((d) => ({ id: d.id, ...d.data() }))
            setProcesses(rows)
            setLoading(false)
          }, (e) => {
            console.error(e)
            setError(e.message || 'Failed to load processes')
            setLoading(false)
          })
        } catch (e) {
          console.error(e)
          setError(e.message || 'Failed to initialize Firebase')
          setLoading(false)
        }
      })()
    }

    return () => { try { unsub() } catch {} }
  }, [isMock, user?.uid])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    if (!s) return processes
    return processes.filter(p => (p.name || '').toLowerCase().includes(s))
  }, [q, processes])

  async function createProcess() {
    setError('')
    const name = newName.trim()
    if (!name) { setError('Please enter a process name.'); return }
    setBusy(true)

    try {
      if (isMock) {
        const id = (crypto?.randomUUID && crypto.randomUUID()) || String(Date.now())
        const doc = { id, name, slug: slugify(name), createdAt: { seconds: Math.floor(Date.now()/1000) } }
        const rows = [{ ...doc }, ...loadMockProcesses(user.uid)]
        saveMockProcesses(user.uid, rows)
        setProcesses(rows)
        setAdding(false)
        setNewName('')
        goToProcess(id)
      } else {
        const { fs, collection, addDoc, serverTimestamp } = await fb()
        const colRef = collection(fs, 'users', user.uid, 'processes')
        const ref = await addDoc(colRef, { name, slug: slugify(name), createdAt: serverTimestamp() })
        setAdding(false)
        setNewName('')
        goToProcess(ref.id)
      }
    } catch (e) {
      console.error(e)
      setError(e.message || 'Failed to create process')
    } finally {
      setBusy(false)
    }
  }

  function goToProcess(id) {
    // If using react-router, prefer: navigate(`/process/${id}`)
    window.location.assign(`/process/${id}`)
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Your Processes</h1>
            <p className="text-neutral-400 text-sm mt-1">
              Signed in as <span className="font-medium text-neutral-200">{user.email}</span>
              {isMock && <span className="ml-2 rounded-full px-2 py-0.5 text-[10px] border border-yellow-700/40 bg-yellow-900/20 text-yellow-200">mock</span>}
            </p>
          </div>
          <button
            onClick={authApi.signOut}
            className="inline-flex items-center gap-2 rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-2 hover:bg-neutral-800"
            title="Sign out"
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline">Sign out</span>
          </button>
        </div>

        {/* Search + Add */}
        <div className="mt-6 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search your processes…"
              className="w-full rounded-xl bg-neutral-900 border border-neutral-800 pl-10 pr-3 py-2 outline-none focus:ring-2 focus:ring-neutral-700"
            />
          </div>
          <button
            onClick={() => { setAdding(true); setError(''); setNewName('') }}
            className="inline-flex items-center gap-2 rounded-xl bg-white text-neutral-900 px-4 py-2 hover:bg-neutral-200"
          >
            <Plus className="w-4 h-4" />
            New Process
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-lg border border-red-700/30 bg-red-900/20 text-red-200 px-4 py-2 text-sm">
            {error}
          </div>
        )}

        {/* List */}
        <div className="mt-6 rounded-2xl border border-neutral-800 bg-neutral-900/40">
          {loading ? (
            <div className="p-8 flex items-center gap-3 text-neutral-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Loading your processes…</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="p-8 text-neutral-400">{q ? 'No matches found.' : 'No processes yet. Create your first one!'}</div>
          ) : (
            <ul className="divide-y divide-neutral-800">
              {filtered.map((p) => (
                <li key={p.id}>
                  <button
                    onClick={() => goToProcess(p.id)}
                    className="w-full text-left px-4 py-3 hover:bg-neutral-800/60 flex items-center justify-between"
                  >
                    <div>
                      <div className="font-medium text-neutral-100">{p.name}</div>
                      <div className="text-xs text-neutral-400">
                        {p.slug ? <span className="mr-2">/{p.slug}</span> : null}
                        {p.createdAt?.seconds ? new Date(p.createdAt.seconds * 1000).toLocaleString() : '—'}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-neutral-500" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Add Dialog */}
      {adding && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4" onClick={() => setAdding(false)}>
          <div
            className="w-full max-w-md rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold">Create a new process</h3>
            <p className="text-neutral-400 text-sm mt-1">Give your process a clear name. You can rename it later.</p>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="e.g., 5‑min Chart, Scalping v2, HK Session"
              className="mt-4 w-full rounded-xl bg-neutral-950 border border-neutral-800 px-3 py-2 outline-none focus:ring-2 focus:ring-neutral-700"
              autoFocus
            />
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setAdding(false)}
                className="rounded-xl px-4 py-2 border border-neutral-800 hover:bg-neutral-800"
              >
                Cancel
              </button>
              <button
                disabled={busy}
                onClick={createProcess}
                className={[
                  'inline-flex items-center gap-2 rounded-xl px-4 py-2',
                  busy ? 'bg-neutral-400 text-neutral-800' : 'bg-white text-neutral-900 hover:bg-neutral-200',
                ].join(' ')}
              >
                {busy && <Loader2 className="w-4 h-4 animate-spin" />}
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// -------------------------------------------------------------------------------------
// 7) Utils
// -------------------------------------------------------------------------------------
function slugify(name) {
  return String(name || '')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)+/g, '')
}
