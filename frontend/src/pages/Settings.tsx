import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Settings as SettingsIcon,
  Palette,
  Cpu,
  Database,
  Save,
  RefreshCw,
  Check,
  ExternalLink
} from 'lucide-react'
import { getHardwareInfo } from '../api'
import { useStore } from '../store'
import { cn } from '../lib/utils'

export default function Settings() {
  const { sidebarCollapsed, toggleSidebar } = useStore()
  
  // Settings state
  const [theme, setTheme] = useState<'dark' | 'light' | 'system'>('dark')
  const [aimUrl, setAimUrl] = useState('http://localhost:53800')
  const [defaultModel, setDefaultModel] = useState('gpt2')
  const [autoSave, setAutoSave] = useState(true)
  const [notifications, setNotifications] = useState(true)
  const [saved, setSaved] = useState(false)
  
  // Fetch hardware
  const { data: hw, refetch: refetchHardware, isRefetching } = useQuery({
    queryKey: ['hardware'],
    queryFn: getHardwareInfo,
  })
  
  const handleSave = () => {
    // Save settings to localStorage
    localStorage.setItem('onn-settings', JSON.stringify({
      theme,
      aimUrl,
      defaultModel,
      autoSave,
      notifications,
    }))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }
  
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Appearance */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Palette className="w-5 h-5 text-primary-400" />
          Appearance
        </h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-surface-400 mb-2">Theme</label>
            <div className="flex gap-3">
              {[
                { id: 'dark', label: 'Dark' },
                { id: 'light', label: 'Light' },
                { id: 'system', label: 'System' },
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id as typeof theme)}
                  className={cn(
                    "px-4 py-2 rounded-lg border transition-colors",
                    theme === t.id
                      ? "border-primary-500 bg-primary-500/10 text-primary-400"
                      : "border-surface-700 text-surface-300 hover:border-surface-600"
                  )}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center justify-between py-3 border-t border-surface-700">
            <div>
              <p className="text-surface-100">Collapsed Sidebar</p>
              <p className="text-sm text-surface-400">Show compact navigation</p>
            </div>
            <button
              onClick={toggleSidebar}
              className={cn(
                "w-12 h-6 rounded-full transition-colors relative",
                sidebarCollapsed ? "bg-primary-500" : "bg-surface-700"
              )}
            >
              <span className={cn(
                "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                sidebarCollapsed ? "left-7" : "left-1"
              )} />
            </button>
          </div>
        </div>
      </section>
      
      {/* Hardware */}
      <section className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Cpu className="w-5 h-5 text-accent-400" />
            Hardware
          </h2>
          <button
            onClick={() => refetchHardware()}
            disabled={isRefetching}
            className="btn-ghost text-sm"
          >
            <RefreshCw className={cn("w-4 h-4", isRefetching && "animate-spin")} />
            Refresh
          </button>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-surface-800 rounded-lg p-4">
            <p className="text-sm text-surface-400 mb-1">GPU</p>
            <p className="font-medium text-surface-100">
              {hw?.gpu?.name || 'None'}
            </p>
            {hw?.gpu && (
              <p className="text-xs text-surface-500 mt-1">
                {hw.gpu.vram_gb} GB VRAM
              </p>
            )}
          </div>
          <div className="bg-surface-800 rounded-lg p-4">
            <p className="text-sm text-surface-400 mb-1">CPU</p>
            <p className="font-medium text-surface-100 truncate">
              {hw?.cpu?.name || 'Unknown'}
            </p>
            <p className="text-xs text-surface-500 mt-1">
              {hw?.cpu?.threads || '?'} threads
            </p>
          </div>
          <div className="bg-surface-800 rounded-lg p-4">
            <p className="text-sm text-surface-400 mb-1">RAM</p>
            <p className="font-medium text-surface-100">
              {hw?.ram_gb || '?'} GB
            </p>
          </div>
          <div className="bg-surface-800 rounded-lg p-4">
            <p className="text-sm text-surface-400 mb-1">Storage</p>
            <p className="font-medium text-surface-100">
              {hw?.storage_gb || '?'} GB free
            </p>
          </div>
        </div>
      </section>
      
      {/* Integrations */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-green-400" />
          Integrations
        </h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-surface-400 mb-2">Aim Server URL</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={aimUrl}
                onChange={(e) => setAimUrl(e.target.value)}
                className="input flex-1"
              />
              <a
                href={aimUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
          
          <div>
            <label className="block text-sm text-surface-400 mb-2">Default Model</label>
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="input"
            >
              <option value="gpt2">GPT-2</option>
              <option value="distilgpt2">DistilGPT-2</option>
              <option value="facebook/opt-125m">OPT-125M</option>
              <option value="microsoft/DialoGPT-small">DialoGPT-Small</option>
            </select>
          </div>
        </div>
      </section>
      
      {/* Preferences */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <SettingsIcon className="w-5 h-5 text-yellow-400" />
          Preferences
        </h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-surface-700">
            <div>
              <p className="text-surface-100">Auto-save conversations</p>
              <p className="text-sm text-surface-400">Save chats automatically</p>
            </div>
            <button
              onClick={() => setAutoSave(!autoSave)}
              className={cn(
                "w-12 h-6 rounded-full transition-colors relative",
                autoSave ? "bg-primary-500" : "bg-surface-700"
              )}
            >
              <span className={cn(
                "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                autoSave ? "left-7" : "left-1"
              )} />
            </button>
          </div>
          
          <div className="flex items-center justify-between py-3">
            <div>
              <p className="text-surface-100">Notifications</p>
              <p className="text-sm text-surface-400">Training completion alerts</p>
            </div>
            <button
              onClick={() => setNotifications(!notifications)}
              className={cn(
                "w-12 h-6 rounded-full transition-colors relative",
                notifications ? "bg-primary-500" : "bg-surface-700"
              )}
            >
              <span className={cn(
                "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                notifications ? "left-7" : "left-1"
              )} />
            </button>
          </div>
        </div>
      </section>
      
      {/* Save button */}
      <div className="flex justify-end">
        <button onClick={handleSave} className="btn-primary">
          {saved ? (
            <>
              <Check className="w-4 h-4" />
              Saved!
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              Save Settings
            </>
          )}
        </button>
      </div>
      
      {/* About */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4">About</h2>
        <div className="text-sm text-surface-400 space-y-2">
          <p><strong className="text-surface-100">OpenNeuralEngine</strong> v2.0.0</p>
          <p>Production-Grade Democratic AI Framework</p>
          <p>Train any model on any data with automatic hardware-aware configuration.</p>
          <div className="flex gap-4 mt-4">
            <a href="https://github.com/CaptnHyuga/OpenNeuralEngine" target="_blank" rel="noopener noreferrer" className="text-primary-400 hover:text-primary-300">
              GitHub
            </a>
            <a href="#" className="text-primary-400 hover:text-primary-300">
              Documentation
            </a>
            <a href="#" className="text-primary-400 hover:text-primary-300">
              Report Issue
            </a>
          </div>
        </div>
      </section>
    </div>
  )
}
