import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Latin subsets only; see fonts.css.
import '@fontsource/newsreader/latin-400.css'
import './fonts.css'
import './styles.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
