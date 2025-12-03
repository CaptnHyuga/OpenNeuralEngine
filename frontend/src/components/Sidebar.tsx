import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Play, 
  MessageSquare, 
  Box, 
  FlaskConical,
  Settings,
  ChevronLeft,
  ChevronRight,
  Sparkles
} from 'lucide-react'
import { useStore } from '../store'
import { cn } from '../lib/utils'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/train', icon: Play, label: 'Train' },
  { to: '/inference', icon: MessageSquare, label: 'Inference' },
  { to: '/models', icon: Box, label: 'Models' },
  { to: '/experiments', icon: FlaskConical, label: 'Experiments' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar() {
  const { sidebarCollapsed, toggleSidebar } = useStore()
  
  return (
    <aside 
      className={cn(
        "fixed left-0 top-0 h-full bg-surface-900 border-r border-surface-700",
        "flex flex-col transition-all duration-300 z-50",
        sidebarCollapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-surface-700">
        <div className={cn(
          "flex items-center gap-3 transition-opacity duration-200",
          sidebarCollapsed && "opacity-0"
        )}>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <span className="font-semibold text-lg gradient-text">ONN</span>
        </div>
        <button
          onClick={toggleSidebar}
          className="p-1.5 rounded-lg hover:bg-surface-800 text-surface-400 hover:text-surface-100 transition-colors"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <ChevronLeft className="w-5 h-5" />
          )}
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
              "text-surface-300 hover:text-surface-100 hover:bg-surface-800",
              isActive && "bg-primary-500/10 text-primary-400 border border-primary-500/20"
            )}
          >
            <Icon className="w-5 h-5 shrink-0" />
            <span className={cn(
              "font-medium transition-opacity duration-200",
              sidebarCollapsed && "opacity-0 w-0"
            )}>
              {label}
            </span>
          </NavLink>
        ))}
      </nav>
      
      {/* Version */}
      <div className={cn(
        "p-4 border-t border-surface-700 text-xs text-surface-500",
        "transition-opacity duration-200",
        sidebarCollapsed && "opacity-0"
      )}>
        <p>OpenNeuralEngine</p>
        <p className="text-surface-600">v2.0.0</p>
      </div>
    </aside>
  )
}
