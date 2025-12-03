import { useLocation } from 'react-router-dom'
import { Bell, Search, ExternalLink, Cpu } from 'lucide-react'
import { useStore } from '../store'
import { cn } from '../lib/utils'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/train': 'Training',
  '/inference': 'Inference',
  '/models': 'Models',
  '/experiments': 'Experiments',
  '/settings': 'Settings',
}

export default function Header() {
  const location = useLocation()
  const { hardware, activeRuns } = useStore()
  const title = pageTitles[location.pathname] || 'OpenNeuralEngine'
  
  const runningCount = activeRuns.filter(r => r.status === 'running').length
  
  return (
    <header className="h-16 border-b border-surface-700 bg-surface-900/50 backdrop-blur-xl px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Title */}
      <div>
        <h1 className="text-xl font-semibold text-surface-100">{title}</h1>
      </div>
      
      {/* Right side */}
      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
          <input
            type="text"
            placeholder="Search..."
            className="pl-9 pr-4 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm w-64 focus:outline-none focus:border-primary-500"
          />
        </div>
        
        {/* Hardware status */}
        {hardware && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-800 border border-surface-700">
            <Cpu className="w-4 h-4 text-primary-400" />
            <span className="text-sm text-surface-300">
              {hardware.gpu ? hardware.gpu.name : 'CPU'}
            </span>
          </div>
        )}
        
        {/* Active runs indicator */}
        {runningCount > 0 && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20">
            <div className="status-running" />
            <span className="text-sm text-green-400">{runningCount} running</span>
          </div>
        )}
        
        {/* Notifications */}
        <button className="relative p-2 rounded-lg hover:bg-surface-800 text-surface-400 hover:text-surface-100 transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-accent-500 rounded-full" />
        </button>
        
        {/* Aim link */}
        <a
          href="http://localhost:53800"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg",
            "bg-surface-800 border border-surface-700",
            "text-sm text-surface-300 hover:text-surface-100 hover:border-surface-600",
            "transition-colors"
          )}
        >
          <span>Aim</span>
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </header>
  )
}
