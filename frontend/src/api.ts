const API_BASE = '/api'

// Generic fetch wrapper
async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `API Error: ${response.status}`)
  }

  return response.json()
}

// Hardware
export const getHardwareInfo = () => 
  fetchAPI<{
    gpu: { name: string; vram_gb: number; cuda_version: string | null } | null
    cpu: { name: string; cores: number; threads: number }
    ram_gb: number
    storage_gb: number
  }>('/hardware')

export const getRecommendedConfig = (modelSize: number, datasetSize: number) =>
  fetchAPI<{
    batch_size: number
    precision: string
    gradient_checkpointing: boolean
    quantization: string | null
  }>(`/hardware/recommend?model_size=${modelSize}&dataset_size=${datasetSize}`)

// Models
export interface ModelInfo {
  id: string
  name: string
  source: string
  size_mb: number
  parameters: number
  task: string
  loaded: boolean
}

export const listModels = () => fetchAPI<ModelInfo[]>('/models')

export const loadModel = (modelId: string) =>
  fetchAPI<{ success: boolean; model: ModelInfo }>('/models/load', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  })

export const unloadModel = (modelId: string) =>
  fetchAPI<{ success: boolean }>(`/models/${modelId}/unload`, {
    method: 'POST',
  })

export const searchHuggingFace = (query: string, task?: string) =>
  fetchAPI<ModelInfo[]>(`/models/search?query=${encodeURIComponent(query)}${task ? `&task=${task}` : ''}`)

// Training
export interface TrainConfig {
  model: string
  dataset: string
  output_dir: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  precision?: string
  gradient_checkpointing?: boolean
}

export interface TrainStatus {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  metrics: {
    loss: number
    accuracy?: number
    epoch: number
    step: number
  }
  logs: string[]
}

export const startTraining = (config: TrainConfig) =>
  fetchAPI<{ run_id: string }>('/train/start', {
    method: 'POST',
    body: JSON.stringify(config),
  })

export const getTrainStatus = (runId: string) =>
  fetchAPI<TrainStatus>(`/train/${runId}/status`)

export const stopTraining = (runId: string) =>
  fetchAPI<{ success: boolean }>(`/train/${runId}/stop`, {
    method: 'POST',
  })

export const listTrainRuns = () => fetchAPI<TrainStatus[]>('/train/runs')

// Inference
export interface InferenceRequest {
  model: string
  prompt: string
  max_tokens?: number
  temperature?: number
  top_p?: number
  stream?: boolean
}

export interface InferenceResponse {
  id: string
  model: string
  output: string
  tokens_generated: number
  time_ms: number
}

export const runInference = (request: InferenceRequest) =>
  fetchAPI<InferenceResponse>('/inference/generate', {
    method: 'POST',
    body: JSON.stringify(request),
  })

// Streaming inference
export async function* streamInference(request: InferenceRequest): AsyncGenerator<string> {
  const response = await fetch(`${API_BASE}/inference/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, stream: true }),
  })

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    
    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data === '[DONE]') return
        try {
          const parsed = JSON.parse(data)
          if (parsed.token) yield parsed.token
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }
}

// Datasets
export interface DatasetInfo {
  id: string
  name: string
  path: string
  format: string
  size_mb: number
  num_samples: number
  columns?: string[]
}

export const listDatasets = () => fetchAPI<DatasetInfo[]>('/datasets')

export const uploadDataset = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch(`${API_BASE}/datasets/upload`, {
    method: 'POST',
    body: formData,
  })
  
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`)
  }
  
  return response.json() as Promise<DatasetInfo>
}

export const analyzeDataset = (path: string) =>
  fetchAPI<DatasetInfo>(`/datasets/analyze?path=${encodeURIComponent(path)}`)

// Experiments (Aim)
export interface Experiment {
  id: string
  name: string
  runs: number
  created_at: string
  last_run_at: string
}

export interface ExperimentRun {
  id: string
  experiment: string
  name: string
  status: string
  metrics: Record<string, number[]>
  params: Record<string, unknown>
  created_at: string
  duration_s: number
}

export const listExperiments = () => fetchAPI<Experiment[]>('/experiments')

export const getExperiment = (id: string) => fetchAPI<ExperimentRun[]>(`/experiments/${id}/runs`)

export const compareRuns = (runIds: string[]) =>
  fetchAPI<{ runs: ExperimentRun[]; comparison: Record<string, Record<string, number>> }>(
    `/experiments/compare?runs=${runIds.join(',')}`
  )

// Export
export interface ExportConfig {
  model_path: string
  format: 'onnx' | 'torchscript' | 'safetensors'
  output_path: string
  optimize?: boolean
  quantize?: 'int8' | 'int4' | null
}

export const exportModel = (config: ExportConfig) =>
  fetchAPI<{ success: boolean; output_path: string; size_mb: number }>('/export', {
    method: 'POST',
    body: JSON.stringify(config),
  })
