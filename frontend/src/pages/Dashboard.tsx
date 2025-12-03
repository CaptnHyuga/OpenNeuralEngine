import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Cpu, 
  HardDrive, 
  Zap, 
  Play, 
  MessageSquare, 
  Box,
  TrendingUp,
  Clock,
  ArrowRight
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { getHardwareInfo, listTrainRuns, listModels } from '../api'
import { useStore } from '../store'

export default function Dashboard() {
  const { hardware, setHardware, activeRuns } = useStore()
  
  // Fetch hardware info
  const { data: hwData } = useQuery({
    queryKey: ['hardware'],
    queryFn: getHardwareInfo,
  })
  
  // Update store when hardware data changes
  useEffect(() => {
    if (hwData) {
      setHardware(hwData)
    }
  }, [hwData, setHardware])
  
  // Fetch recent runs
  const { data: runs } = useQuery({
    queryKey: ['train-runs'],
    queryFn: listTrainRuns,
  })
  
  // Fetch models
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })
  
  const hw = hwData || hardware
  
  return (
    <div className="space-y-8">
      {/* Welcome banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary-600/20 via-surface-900 to-accent-600/20 border border-surface-700 p-8">
        <div className="relative z-10">
          <h1 className="text-3xl font-bold mb-2">
            Welcome to <span className="gradient-text">OpenNeuralEngine</span>
          </h1>
          <p className="text-surface-300 max-w-xl">
            Train any model on any data with automatic hardware-aware configuration.
            Get started by training a model or running inference.
          </p>
          <div className="flex gap-4 mt-6">
            <Link to="/train" className="btn-primary">
              <Play className="w-4 h-4" />
              Start Training
            </Link>
            <Link to="/inference" className="btn-secondary">
              <MessageSquare className="w-4 h-4" />
              Run Inference
            </Link>
          </div>
        </div>
        <div className="absolute right-0 top-0 w-96 h-96 bg-gradient-to-br from-primary-500/10 to-accent-500/10 rounded-full blur-3xl" />
      </div>
      
      {/* Hardware Overview */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-primary-400" />
          Hardware Profile
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* GPU */}
          <div className="card-hover">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary-500/10">
                <Zap className="w-5 h-5 text-primary-400" />
              </div>
              <span className="text-surface-400 text-sm">GPU</span>
            </div>
            {hw?.gpu ? (
              <>
                <p className="font-semibold text-surface-100">{hw.gpu.name}</p>
                <p className="text-sm text-surface-400">
                  {hw.gpu.vram_gb} GB VRAM
                  {hw.gpu.cuda_version && ` • CUDA ${hw.gpu.cuda_version}`}
                </p>
              </>
            ) : (
              <>
                <p className="font-semibold text-surface-100">No GPU</p>
                <p className="text-sm text-surface-400">CPU mode available</p>
              </>
            )}
          </div>
          
          {/* CPU */}
          <div className="card-hover">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-accent-500/10">
                <Cpu className="w-5 h-5 text-accent-400" />
              </div>
              <span className="text-surface-400 text-sm">CPU</span>
            </div>
            <p className="font-semibold text-surface-100">{hw?.cpu?.name || 'Unknown'}</p>
            <p className="text-sm text-surface-400">
              {hw?.cpu?.cores || '?'} cores • {hw?.cpu?.threads || '?'} threads
            </p>
          </div>
          
          {/* RAM */}
          <div className="card-hover">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <HardDrive className="w-5 h-5 text-green-400" />
              </div>
              <span className="text-surface-400 text-sm">Memory</span>
            </div>
            <p className="font-semibold text-surface-100">{hw?.ram_gb || '?'} GB</p>
            <p className="text-sm text-surface-400">System RAM</p>
          </div>
          
          {/* Storage */}
          <div className="card-hover">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-yellow-500/10">
                <HardDrive className="w-5 h-5 text-yellow-400" />
              </div>
              <span className="text-surface-400 text-sm">Storage</span>
            </div>
            <p className="font-semibold text-surface-100">{hw?.storage_gb || '?'} GB</p>
            <p className="text-sm text-surface-400">Available space</p>
          </div>
        </div>
      </section>
      
      {/* Quick Stats */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Active Runs */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-surface-300">Active Training</h3>
            <TrendingUp className="w-5 h-5 text-primary-400" />
          </div>
          <p className="text-3xl font-bold text-surface-100">
            {activeRuns.filter(r => r.status === 'running').length}
          </p>
          <p className="text-sm text-surface-400 mt-1">
            {activeRuns.length} total runs
          </p>
        </div>
        
        {/* Models */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-surface-300">Loaded Models</h3>
            <Box className="w-5 h-5 text-accent-400" />
          </div>
          <p className="text-3xl font-bold text-surface-100">
            {models?.filter(m => m.loaded).length || 0}
          </p>
          <p className="text-sm text-surface-400 mt-1">
            {models?.length || 0} available
          </p>
        </div>
        
        {/* Recent Activity */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-surface-300">Last Training</h3>
            <Clock className="w-5 h-5 text-green-400" />
          </div>
          {runs && runs.length > 0 ? (
            <>
              <p className="text-lg font-semibold text-surface-100 truncate">
                {runs[0].id}
              </p>
              <p className="text-sm text-surface-400 mt-1">
                Status: {runs[0].status}
              </p>
            </>
          ) : (
            <p className="text-surface-400">No recent runs</p>
          )}
        </div>
      </section>
      
      {/* Quick Actions */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <Link 
            to="/train" 
            className="card-hover group flex items-center justify-between"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-xl bg-gradient-to-br from-primary-500/20 to-primary-600/20 border border-primary-500/20">
                <Play className="w-6 h-6 text-primary-400" />
              </div>
              <div>
                <p className="font-medium text-surface-100">Train GPT-2</p>
                <p className="text-sm text-surface-400">Fine-tune on your data</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-surface-500 group-hover:text-surface-300 group-hover:translate-x-1 transition-all" />
          </Link>
          
          <Link 
            to="/inference" 
            className="card-hover group flex items-center justify-between"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-xl bg-gradient-to-br from-accent-500/20 to-accent-600/20 border border-accent-500/20">
                <MessageSquare className="w-6 h-6 text-accent-400" />
              </div>
              <div>
                <p className="font-medium text-surface-100">Chat with Models</p>
                <p className="text-sm text-surface-400">Interactive inference</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-surface-500 group-hover:text-surface-300 group-hover:translate-x-1 transition-all" />
          </Link>
          
          <Link 
            to="/models" 
            className="card-hover group flex items-center justify-between"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-xl bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/20">
                <Box className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <p className="font-medium text-surface-100">Browse Models</p>
                <p className="text-sm text-surface-400">HuggingFace & local</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-surface-500 group-hover:text-surface-300 group-hover:translate-x-1 transition-all" />
          </Link>
        </div>
      </section>
    </div>
  )
}
