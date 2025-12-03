import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Train from './pages/Train'
import Inference from './pages/Inference'
import Models from './pages/Models'
import Experiments from './pages/Experiments'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="train" element={<Train />} />
        <Route path="inference" element={<Inference />} />
        <Route path="models" element={<Models />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
