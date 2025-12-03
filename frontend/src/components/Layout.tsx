import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'
import { useStore } from '../store'
import { cn } from '../lib/utils'

export default function Layout() {
  const { sidebarCollapsed } = useStore()
  
  return (
    <div className="flex h-screen overflow-hidden bg-surface-950">
      <Sidebar />
      <div 
        className={cn(
          "flex-1 flex flex-col transition-all duration-300",
          sidebarCollapsed ? "ml-16" : "ml-64"
        )}
      >
        <Header />
        <main className="flex-1 overflow-auto p-6 bg-grid">
          <div className="max-w-7xl mx-auto animate-in">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
