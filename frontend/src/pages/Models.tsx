import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Search, 
  Download, 
  Trash2, 
  ExternalLink,
  Box,
  Loader2,
  Check,
  HardDrive
} from 'lucide-react'
import { listModels, loadModel, unloadModel, searchHuggingFace, ModelInfo } from '../api'
import { useStore } from '../store'
import { cn, formatBytes, formatNumber } from '../lib/utils'

const taskFilters = [
  { id: 'all', label: 'All' },
  { id: 'text-generation', label: 'Text Generation' },
  { id: 'conversational', label: 'Conversational' },
  { id: 'fill-mask', label: 'Fill Mask' },
  { id: 'text-classification', label: 'Classification' },
  { id: 'image-classification', label: 'Image' },
]

export default function Models() {
  const queryClient = useQueryClient()
  const { setActiveModel } = useStore()
  
  const [searchQuery, setSearchQuery] = useState('')
  const [taskFilter, setTaskFilter] = useState('all')
  const [view, setView] = useState<'local' | 'huggingface'>('local')
  
  // Fetch local models
  const { data: localModels, isLoading: loadingLocal } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })
  
  // Search HuggingFace
  const { data: hfModels, isLoading: loadingHF } = useQuery({
    queryKey: ['hf-models', searchQuery],
    queryFn: () => searchHuggingFace(searchQuery, taskFilter !== 'all' ? taskFilter : undefined),
    enabled: view === 'huggingface' && searchQuery.length > 2,
  })
  
  // Load model
  const loadMutation = useMutation({
    mutationFn: loadModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
  
  // Unload model
  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
  
  const models = view === 'local' ? localModels : hfModels
  const filteredModels = models?.filter(m => 
    taskFilter === 'all' || m.task === taskFilter
  )
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-surface-100">Models</h1>
          <p className="text-surface-400">Browse and manage your models</p>
        </div>
        
        {/* View toggle */}
        <div className="flex bg-surface-800 rounded-lg p-1">
          <button
            onClick={() => setView('local')}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              view === 'local'
                ? "bg-surface-700 text-surface-100"
                : "text-surface-400 hover:text-surface-100"
            )}
          >
            <HardDrive className="w-4 h-4 inline mr-2" />
            Local
          </button>
          <button
            onClick={() => setView('huggingface')}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors",
              view === 'huggingface'
                ? "bg-surface-700 text-surface-100"
                : "text-surface-400 hover:text-surface-100"
            )}
          >
            🤗 HuggingFace
          </button>
        </div>
      </div>
      
      {/* Search and filters */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={view === 'local' ? "Search local models..." : "Search HuggingFace models..."}
            className="input pl-10"
          />
        </div>
        
        <select
          value={taskFilter}
          onChange={(e) => setTaskFilter(e.target.value)}
          className="input w-48"
        >
          {taskFilters.map(f => (
            <option key={f.id} value={f.id}>{f.label}</option>
          ))}
        </select>
      </div>
      
      {/* Models grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {(loadingLocal || loadingHF) && (
          <div className="col-span-full flex justify-center py-12">
            <Loader2 className="w-8 h-8 text-primary-400 animate-spin" />
          </div>
        )}
        
        {filteredModels?.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            onLoad={() => loadMutation.mutate(model.id)}
            onUnload={() => unloadMutation.mutate(model.id)}
            onSelect={() => setActiveModel(model.id)}
            isLoading={loadMutation.isPending || unloadMutation.isPending}
          />
        ))}
        
        {filteredModels?.length === 0 && !loadingLocal && !loadingHF && (
          <div className="col-span-full text-center py-12">
            <Box className="w-12 h-12 text-surface-500 mx-auto mb-4" />
            <p className="text-surface-400">No models found</p>
            {view === 'huggingface' && searchQuery.length < 3 && (
              <p className="text-sm text-surface-500 mt-2">
                Enter at least 3 characters to search
              </p>
            )}
          </div>
        )}
      </div>
      
      {/* Popular models section (for HuggingFace view) */}
      {view === 'huggingface' && !searchQuery && (
        <section>
          <h2 className="text-lg font-semibold mb-4">Popular Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { id: 'gpt2', name: 'GPT-2', task: 'text-generation', params: 124_000_000, size: 500 },
              { id: 'distilgpt2', name: 'DistilGPT-2', task: 'text-generation', params: 82_000_000, size: 330 },
              { id: 'facebook/opt-125m', name: 'OPT-125M', task: 'text-generation', params: 125_000_000, size: 500 },
              { id: 'EleutherAI/gpt-neo-125m', name: 'GPT-Neo-125M', task: 'text-generation', params: 125_000_000, size: 500 },
              { id: 'microsoft/DialoGPT-small', name: 'DialoGPT-Small', task: 'conversational', params: 124_000_000, size: 500 },
              { id: 'bert-base-uncased', name: 'BERT Base', task: 'fill-mask', params: 110_000_000, size: 440 },
            ].map((model) => (
              <ModelCard
                key={model.id}
                model={{
                  id: model.id,
                  name: model.name,
                  source: 'huggingface',
                  task: model.task,
                  parameters: model.params,
                  size_mb: model.size,
                  loaded: false,
                }}
                onLoad={() => loadMutation.mutate(model.id)}
                onUnload={() => {}}
                onSelect={() => setActiveModel(model.id)}
                isLoading={loadMutation.isPending}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

function ModelCard({
  model,
  onLoad,
  onUnload,
  onSelect,
  isLoading,
}: {
  model: ModelInfo
  onLoad: () => void
  onUnload: () => void
  onSelect: () => void
  isLoading: boolean
}) {
  return (
    <div className="card-hover group">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center",
            model.source === 'huggingface' ? "bg-yellow-500/10" : "bg-primary-500/10"
          )}>
            {model.source === 'huggingface' ? (
              <span className="text-lg">🤗</span>
            ) : (
              <Box className="w-5 h-5 text-primary-400" />
            )}
          </div>
          <div>
            <h3 className="font-medium text-surface-100">{model.name}</h3>
            <p className="text-xs text-surface-500">{model.id}</p>
          </div>
        </div>
        
        {model.loaded && (
          <span className="px-2 py-0.5 rounded-full bg-green-500/10 text-green-400 text-xs">
            Loaded
          </span>
        )}
      </div>
      
      <div className="flex flex-wrap gap-2 mb-4">
        <span className="px-2 py-1 rounded bg-surface-800 text-xs text-surface-300">
          {model.task}
        </span>
        <span className="px-2 py-1 rounded bg-surface-800 text-xs text-surface-300">
          {formatNumber(model.parameters)} params
        </span>
        <span className="px-2 py-1 rounded bg-surface-800 text-xs text-surface-300">
          {formatBytes(model.size_mb * 1024 * 1024)}
        </span>
      </div>
      
      <div className="flex gap-2">
        {model.loaded ? (
          <>
            <button onClick={onSelect} className="btn-primary flex-1 text-sm py-2">
              <Check className="w-4 h-4" />
              Use Model
            </button>
            <button onClick={onUnload} className="btn-ghost px-3">
              <Trash2 className="w-4 h-4 text-red-400" />
            </button>
          </>
        ) : (
          <button 
            onClick={onLoad} 
            disabled={isLoading}
            className="btn-secondary flex-1 text-sm py-2"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            Load Model
          </button>
        )}
        
        {model.source === 'huggingface' && (
          <a
            href={`https://huggingface.co/${model.id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-ghost px-3"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>
    </div>
  )
}
