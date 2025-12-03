import { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { 
  Play, 
  Pause, 
  Upload, 
  Settings, 
  ChevronDown,
  Loader2,
  Folder,
  Database
} from 'lucide-react'
import { startTraining, getTrainStatus, stopTraining, listDatasets, getRecommendedConfig } from '../api'
import { useStore } from '../store'
import { cn } from '../lib/utils'

// Popular models
const popularModels = [
  { id: 'gpt2', name: 'GPT-2', params: '124M', task: 'text-generation' },
  { id: 'distilgpt2', name: 'DistilGPT-2', params: '82M', task: 'text-generation' },
  { id: 'facebook/opt-125m', name: 'OPT-125M', params: '125M', task: 'text-generation' },
  { id: 'EleutherAI/gpt-neo-125m', name: 'GPT-Neo-125M', params: '125M', task: 'text-generation' },
  { id: 'microsoft/DialoGPT-small', name: 'DialoGPT-Small', params: '124M', task: 'conversational' },
  { id: 'bert-base-uncased', name: 'BERT Base', params: '110M', task: 'fill-mask' },
]

export default function Train() {
  const { hardware, addRun, updateRun } = useStore()
  
  // Form state
  const [model, setModel] = useState('gpt2')
  const [customModel, setCustomModel] = useState('')
  const [dataSource, setDataSource] = useState<'upload' | 'path' | 'hf'>('path')
  const [dataPath, setDataPath] = useState('')
  const [hfDataset, setHfDataset] = useState('')
  const [outputDir, setOutputDir] = useState('./output')
  
  // Training config
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatchSize] = useState(4)
  const [learningRate, setLearningRate] = useState(2e-4)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [precision, setPrecision] = useState('auto')
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true)
  
  // Training state
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  
  // Get recommended config
  const { data: recommended } = useQuery({
    queryKey: ['recommended-config', model],
    queryFn: () => getRecommendedConfig(124_000_000, 10000),
    enabled: !!hardware,
  })
  
  // List datasets
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })
  
  // Start training mutation
  const trainMutation = useMutation({
    mutationFn: startTraining,
    onSuccess: (data) => {
      setActiveRunId(data.run_id)
      addRun({
        id: data.run_id,
        name: `Training ${model}`,
        model,
        dataset: dataPath || hfDataset,
        status: 'running',
        progress: 0,
        metrics: { loss: 0, epoch: 0, step: 0 },
        started_at: new Date().toISOString(),
      })
    },
  })
  
  // Poll training status
  const { data: statusData } = useQuery({
    queryKey: ['train-status', activeRunId],
    queryFn: () => getTrainStatus(activeRunId!),
    enabled: !!activeRunId,
    refetchInterval: 1000,
  })
  
  // Handle status updates
  useEffect(() => {
    if (statusData && activeRunId) {
      updateRun(activeRunId, {
        status: statusData.status,
        progress: statusData.progress,
        metrics: statusData.metrics,
      })
      if (statusData.status === 'completed' || statusData.status === 'failed') {
        setActiveRunId(null)
      }
    }
  }, [statusData, activeRunId, updateRun])
  
  // Stop training
  const stopMutation = useMutation({
    mutationFn: () => stopTraining(activeRunId!),
    onSuccess: () => {
      updateRun(activeRunId!, { status: 'failed' })
      setActiveRunId(null)
    },
  })
  
  const handleStartTraining = () => {
    const selectedModel = customModel || model
    const dataset = dataSource === 'hf' ? hfDataset : dataPath
    
    trainMutation.mutate({
      model: selectedModel,
      dataset,
      output_dir: outputDir,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      precision: precision === 'auto' ? undefined : precision,
      gradient_checkpointing: gradientCheckpointing,
    })
  }
  
  const isTraining = !!activeRunId
  
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Model Selection */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-primary-400" />
          Select Model
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          {popularModels.map((m) => (
            <button
              key={m.id}
              onClick={() => { setModel(m.id); setCustomModel('') }}
              className={cn(
                "p-3 rounded-lg border text-left transition-all",
                model === m.id && !customModel
                  ? "border-primary-500 bg-primary-500/10"
                  : "border-surface-700 hover:border-surface-600"
              )}
            >
              <p className="font-medium text-surface-100">{m.name}</p>
              <p className="text-xs text-surface-400">{m.params} params</p>
            </button>
          ))}
        </div>
        
        <div className="relative">
          <input
            type="text"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            placeholder="Or enter custom HuggingFace model ID..."
            className="input"
          />
        </div>
      </section>
      
      {/* Data Source */}
      <section className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Folder className="w-5 h-5 text-accent-400" />
          Training Data
        </h2>
        
        {/* Data source tabs */}
        <div className="flex gap-2 mb-4">
          {[
            { id: 'path', label: 'Local Path', icon: Folder },
            { id: 'upload', label: 'Upload', icon: Upload },
            { id: 'hf', label: 'HuggingFace', icon: Database },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setDataSource(id as typeof dataSource)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg transition-colors",
                dataSource === id
                  ? "bg-primary-500/10 text-primary-400 border border-primary-500/20"
                  : "text-surface-400 hover:text-surface-100 hover:bg-surface-800"
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
        
        {/* Data input based on source */}
        {dataSource === 'path' && (
          <input
            type="text"
            value={dataPath}
            onChange={(e) => setDataPath(e.target.value)}
            placeholder="./data/my_dataset.jsonl"
            className="input"
          />
        )}
        
        {dataSource === 'upload' && (
          <div className="border-2 border-dashed border-surface-600 rounded-lg p-8 text-center hover:border-surface-500 transition-colors cursor-pointer">
            <Upload className="w-8 h-8 text-surface-400 mx-auto mb-2" />
            <p className="text-surface-300">Drop files here or click to upload</p>
            <p className="text-sm text-surface-500 mt-1">
              Supports JSON, JSONL, CSV, Parquet
            </p>
          </div>
        )}
        
        {dataSource === 'hf' && (
          <input
            type="text"
            value={hfDataset}
            onChange={(e) => setHfDataset(e.target.value)}
            placeholder="wikitext, openwebtext, squad..."
            className="input"
          />
        )}
        
        {/* Recent datasets */}
        {datasets && datasets.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-surface-400 mb-2">Recent datasets:</p>
            <div className="flex flex-wrap gap-2">
              {datasets.slice(0, 5).map((ds) => (
                <button
                  key={ds.id}
                  onClick={() => { setDataSource('path'); setDataPath(ds.path) }}
                  className="px-3 py-1 rounded-full bg-surface-800 text-sm text-surface-300 hover:bg-surface-700 transition-colors"
                >
                  {ds.name}
                </button>
              ))}
            </div>
          </div>
        )}
      </section>
      
      {/* Training Config */}
      <section className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Settings className="w-5 h-5 text-green-400" />
            Training Configuration
          </h2>
          {recommended && (
            <span className="text-xs text-surface-400 bg-surface-800 px-2 py-1 rounded">
              Auto-configured for your hardware
            </span>
          )}
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-surface-400 mb-1">Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min={1}
              max={100}
              className="input"
            />
          </div>
          <div>
            <label className="block text-sm text-surface-400 mb-1">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min={1}
              max={128}
              className="input"
            />
          </div>
          <div>
            <label className="block text-sm text-surface-400 mb-1">Learning Rate</label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              step={0.0001}
              min={0}
              className="input"
            />
          </div>
          <div>
            <label className="block text-sm text-surface-400 mb-1">Output Dir</label>
            <input
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              className="input"
            />
          </div>
        </div>
        
        {/* Advanced settings */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 mt-4 text-sm text-surface-400 hover:text-surface-100 transition-colors"
        >
          <ChevronDown className={cn("w-4 h-4 transition-transform", showAdvanced && "rotate-180")} />
          Advanced Settings
        </button>
        
        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4 pt-4 border-t border-surface-700">
            <div>
              <label className="block text-sm text-surface-400 mb-1">Precision</label>
              <select
                value={precision}
                onChange={(e) => setPrecision(e.target.value)}
                className="input"
              >
                <option value="auto">Auto</option>
                <option value="fp32">FP32</option>
                <option value="fp16">FP16</option>
                <option value="bf16">BF16</option>
                <option value="int8">INT8</option>
              </select>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="gradient-checkpointing"
                checked={gradientCheckpointing}
                onChange={(e) => setGradientCheckpointing(e.target.checked)}
                className="w-4 h-4 rounded border-surface-600 bg-surface-800"
              />
              <label htmlFor="gradient-checkpointing" className="text-sm text-surface-300">
                Gradient Checkpointing
              </label>
            </div>
          </div>
        )}
      </section>
      
      {/* Start/Stop Button */}
      <div className="flex gap-4">
        {!isTraining ? (
          <button
            onClick={handleStartTraining}
            disabled={trainMutation.isPending || (!dataPath && !hfDataset)}
            className="btn-primary flex-1 py-3 text-lg"
          >
            {trainMutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Training
              </>
            )}
          </button>
        ) : (
          <button
            onClick={() => stopMutation.mutate()}
            disabled={stopMutation.isPending}
            className="btn-danger flex-1 py-3 text-lg"
          >
            <Pause className="w-5 h-5" />
            Stop Training
          </button>
        )}
      </div>
      
      {/* Training Progress */}
      {isTraining && (
        <section className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Training Progress</h3>
            <div className="status-running" />
          </div>
          
          <div className="space-y-4">
            {/* Progress bar */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-surface-400">Progress</span>
                <span className="text-surface-100">
                  {Math.round((trainMutation.data ? 50 : 0))}%
                </span>
              </div>
              <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-primary-500 to-accent-500 transition-all duration-500"
                  style={{ width: `${trainMutation.data ? 50 : 0}%` }}
                />
              </div>
            </div>
            
            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-sm text-surface-400">Loss</p>
                <p className="text-xl font-semibold text-surface-100">--</p>
              </div>
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-sm text-surface-400">Epoch</p>
                <p className="text-xl font-semibold text-surface-100">-- / {epochs}</p>
              </div>
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-sm text-surface-400">Step</p>
                <p className="text-xl font-semibold text-surface-100">--</p>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
