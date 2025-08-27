// src/App.jsx
import React, { useEffect, useState } from 'react'
import FirstPage from './components/FirstPage'
import ProcessWorkspace from './components/ProcessWorkspace'

// Tiny route parser
function parseRoute() {
  const path = window.location.pathname || '/'
  if (path.startsWith('/process/')) {
    const processId = decodeURIComponent(path.slice('/process/'.length))
    return { name: 'process', processId }
  }
  return { name: 'home' }
}

// Optional SPA navigate helper (so we can switch pages without reload)
function navigate(path) {
  if (window.location.pathname === path) return
  window.history.pushState({}, '', path)
  window.dispatchEvent(new Event('popstate'))
}

export default function App() {
  const [route, setRoute] = useState(parseRoute())

  useEffect(() => {
    const onPop = () => setRoute(parseRoute())
    window.addEventListener('popstate', onPop)
    // Expose SPA nav so FirstPage can call window.appNavigate(`/process/<id>`)
    window.appNavigate = navigate
    return () => {
      window.removeEventListener('popstate', onPop)
      delete window.appNavigate
    }
  }, [])

  if (route.name === 'process') {
    return <ProcessWorkspace key={route.processId} />
  }
  return <FirstPage />
}
