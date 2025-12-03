import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  FlaskConical, 
  GitCompare, 
  ExternalLink, 
  ChevronDown,
  Clock,
  CheckCircle,
  XCircle,
  Loader2
} from 'lucide-react'
import { listExperiments, getExperiment, compareRuns, ExperimentRun } from '../api'
import { cn, formatRelativeTime, formatDuration } from '../lib/utils'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export default function Experiments() {
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)
  const [selectedRuns, setSelectedRuns] = useState<string[]>([])
  const [showComparison, setShowComparison] = useState(false)
  
  // Fetch experiments
  const { data: experiments, isLoading: loadingExperiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: listExperiments,
  })
  
  // Fetch runs for selected experiment
  const { data: runs, isLoading: loadingRuns } = useQuery({
    queryKey: ['experiment-runs', selectedExperiment],
    queryFn: () => getExperiment(selectedExperiment!),
    enabled: !!selectedExperiment,
  })
  
  // Compare runs
  const { data: comparison, isLoading: loadingComparison } = useQuery({
    queryKey: ['run-comparison', selectedRuns],
    queryFn: () => compareRuns(selectedRuns),
    enabled: selectedRuns.length >= 2 && showComparison,
  })
  
  const toggleRunSelection = (runId: string) => {
    setSelectedRuns(prev => 
      prev.includes(runId) 
        ? prev.filter(id => id !== runId)
        : [...prev, runId]
    )
  }
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-surface-100">Experiments</h1>
          <p className="text-surface-400">Track and compare your training runs</p>
        </div>
        
        <div className="flex gap-3">
          {selectedRuns.length >= 2 && (
            <button
              onClick={() => setShowComparison(!showComparison)}
              className={cn(
                "btn-secondary",
                showComparison && "bg-primary-500/10 text-primary-400 border-primary-500/20"
              )}
            >
              <GitCompare className="w-4 h-4" />
              Compare ({selectedRuns.length})
            </button>
          )}
          <a
            href="http://localhost:53800"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary"
          >
            Open Aim
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Experiments list */}
        <div className="lg:col-span-1">
          <div className="card">
            <h2 className="font-semibold mb-4 flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-primary-400" />
              Experiments
            </h2>
            
            {loadingExperiments ? (
              <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 text-primary-400 animate-spin" />
              </div>
            ) : experiments && experiments.length > 0 ? (
              <div className="space-y-2">
                {experiments.map((exp) => (
                  <button
                    key={exp.id}
                    onClick={() => setSelectedExperiment(exp.id)}
                    className={cn(
                      "w-full text-left p-3 rounded-lg transition-colors",
                      selectedExperiment === exp.id
                        ? "bg-primary-500/10 border border-primary-500/20"
                        : "hover:bg-surface-800"
                    )}
                  >
                    <p className="font-medium text-surface-100">{exp.name}</p>
                    <div className="flex items-center gap-2 mt-1 text-xs text-surface-400">
                      <span>{exp.runs} runs</span>
                      <span>•</span>
                      <span>{formatRelativeTime(exp.last_run_at)}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-surface-400 text-center py-8">
                No experiments yet
              </p>
            )}
          </div>
        </div>
        
        {/* Runs list */}
        <div className="lg:col-span-3">
          {selectedExperiment ? (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-semibold">Training Runs</h2>
                {selectedRuns.length > 0 && (
                  <button
                    onClick={() => setSelectedRuns([])}
                    className="text-sm text-surface-400 hover:text-surface-100"
                  >
                    Clear selection
                  </button>
                )}
              </div>
              
              {loadingRuns ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-6 h-6 text-primary-400 animate-spin" />
                </div>
              ) : runs && runs.length > 0 ? (
                <div className="space-y-3">
                  {runs.map((run) => (
                    <RunCard
                      key={run.id}
                      run={run}
                      selected={selectedRuns.includes(run.id)}
                      onToggle={() => toggleRunSelection(run.id)}
                    />
                  ))}
                </div>
              ) : (
                <p className="text-surface-400 text-center py-8">
                  No runs in this experiment
                </p>
              )}
            </div>
          ) : (
            <div className="card flex flex-col items-center justify-center py-16">
              <FlaskConical className="w-12 h-12 text-surface-500 mb-4" />
              <p className="text-surface-400">
                Select an experiment to view runs
              </p>
            </div>
          )}
        </div>
      </div>
      
      {/* Comparison view */}
      {showComparison && selectedRuns.length >= 2 && (
        <div className="card">
          <h2 className="font-semibold mb-4 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-accent-400" />
            Run Comparison
          </h2>
          
          {loadingComparison ? (
            <div className="flex justify-center py-8">
              <Loader2 className="w-6 h-6 text-primary-400 animate-spin" />
            </div>
          ) : comparison ? (
            <div className="space-y-6">
              {/* Loss chart */}
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart>
                    <XAxis 
                      dataKey="step" 
                      stroke="#71717a"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#71717a"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#27272a', 
                        border: '1px solid #3f3f46',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend />
                    {comparison.runs.map((run, i) => (
                      <Line
                        key={run.id}
                        data={run.metrics.loss?.map((v, idx) => ({ step: idx, loss: v })) || []}
                        dataKey="loss"
                        name={run.name}
                        stroke={['#0ea5e9', '#d946ef', '#22c55e', '#f59e0b'][i % 4]}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              {/* Metrics table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-surface-700">
                      <th className="text-left py-2 px-4 text-surface-400 font-medium">Metric</th>
                      {comparison.runs.map((run) => (
                        <th key={run.id} className="text-left py-2 px-4 text-surface-400 font-medium">
                          {run.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(comparison.comparison).map((metric) => (
                      <tr key={metric} className="border-b border-surface-800">
                        <td className="py-2 px-4 text-surface-300">{metric}</td>
                        {comparison.runs.map((run) => (
                          <td key={run.id} className="py-2 px-4 text-surface-100 font-mono">
                            {comparison.comparison[metric]?.[run.id]?.toFixed(4) || '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p className="text-surface-400 text-center py-8">
              Unable to load comparison
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function RunCard({ 
  run, 
  selected, 
  onToggle 
}: { 
  run: ExperimentRun
  selected: boolean
  onToggle: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  
  const statusIcon = {
    completed: <CheckCircle className="w-4 h-4 text-green-400" />,
    failed: <XCircle className="w-4 h-4 text-red-400" />,
    running: <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />,
  }[run.status] || <Clock className="w-4 h-4 text-surface-400" />
  
  return (
    <div 
      className={cn(
        "border rounded-lg transition-colors",
        selected ? "border-primary-500 bg-primary-500/5" : "border-surface-700"
      )}
    >
      <div className="flex items-center gap-4 p-4">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          className="w-4 h-4 rounded border-surface-600 bg-surface-800"
        />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {statusIcon}
            <span className="font-medium text-surface-100">{run.name}</span>
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-surface-400">
            <span>{formatRelativeTime(run.created_at)}</span>
            <span>•</span>
            <span>{formatDuration(run.duration_s * 1000)}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {run.metrics.loss && (
            <div className="text-right">
              <p className="text-xs text-surface-400">Final Loss</p>
              <p className="font-mono text-surface-100">
                {run.metrics.loss[run.metrics.loss.length - 1]?.toFixed(4)}
              </p>
            </div>
          )}
          
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 rounded hover:bg-surface-800 transition-colors"
          >
            <ChevronDown className={cn(
              "w-4 h-4 text-surface-400 transition-transform",
              expanded && "rotate-180"
            )} />
          </button>
        </div>
      </div>
      
      {expanded && (
        <div className="px-4 pb-4 border-t border-surface-700 mt-2 pt-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(run.params).map(([key, value]) => (
              <div key={key}>
                <p className="text-xs text-surface-400">{key}</p>
                <p className="text-sm text-surface-100 font-mono">
                  {typeof value === 'number' ? value.toFixed(6) : String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
