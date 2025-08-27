// src/lib/firebase.js
import { initializeApp, getApps } from 'firebase/app'
import { getAuth, GoogleAuthProvider, signInWithPopup, onAuthStateChanged, signOut } from 'firebase/auth'
import { getFirestore } from 'firebase/firestore'
import { getStorage } from 'firebase/storage'

// Your existing config (moved here):
const firebaseConfig = {
  apiKey: "AIzaSyC55H0aXYePqiYCHtdhlA_ygxkua6DZp58",
  authDomain: "trade-pred-analyser.firebaseapp.com",
  projectId: "trade-pred-analyser",
  storageBucket: "trade-pred-analyser.firebasestorage.app",
  messagingSenderId: "891971986053",
  appId: "1:891971986053:web:bfd8edbb4cfe6d89627fec",
  measurementId: "G-SVCNHWT2XX"
}

const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig)

export const auth = getAuth(app)
export const db = getFirestore(app)
export const storage = getStorage(app)

// Re-export auth helpers so FirstPage can import from one place
export { GoogleAuthProvider, signInWithPopup, onAuthStateChanged, signOut }
