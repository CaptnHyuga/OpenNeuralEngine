import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// Types
export interface HardwareInfo {
  gpu: {
    name: string
    vram_gb: number
    cuda_version: string | null
  } | null
  cpu: {
    name: string
    cores: number
    threads: number
  }
  ram_gb: number
  storage_gb: number
}

export interface TrainingRun {
  id: string
  name: string
  model: string
  dataset: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  metrics: {
    loss: number
    accuracy?: number
    epoch: number
    step: number
  }
  started_at: string
  completed_at?: string
}

export interface Model {
  id: string
  name: string
  source: 'huggingface' | 'local' | 'timm'
  size_mb: number
  parameters: number
  task: string
  loaded: boolean
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  model?: string
}

export interface Conversation {
  id: string
  title: string
  model: string
  messages: Message[]
  created_at: string
  updated_at: string
}

// Global Store
interface AppState {
  // Hardware
  hardware: HardwareInfo | null
  setHardware: (hw: HardwareInfo) => void
  
  // Training
  activeRuns: TrainingRun[]
  addRun: (run: TrainingRun) => void
  updateRun: (id: string, updates: Partial<TrainingRun>) => void
  removeRun: (id: string) => void
  
  // Models
  models: Model[]
  setModels: (models: Model[]) => void
  activeModel: string | null
  setActiveModel: (id: string | null) => void
  
  // Conversations
  conversations: Conversation[]
  activeConversation: string | null
  createConversation: (model: string) => Conversation
  setActiveConversation: (id: string | null) => void
  addMessage: (conversationId: string, message: Omit<Message, 'id' | 'timestamp'>) => void
  
  // UI
  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useStore = create<AppState>()(
  persist(
    (set: (fn: (state: AppState) => Partial<AppState>) => void, _get: () => AppState) => ({
      // Hardware
      hardware: null,
      setHardware: (hw: HardwareInfo) => set(() => ({ hardware: hw })),
      
      // Training
      activeRuns: [],
      addRun: (run: TrainingRun) => set((s: AppState) => ({ activeRuns: [...s.activeRuns, run] })),
      updateRun: (id: string, updates: Partial<TrainingRun>) => set((s: AppState) => ({
        activeRuns: s.activeRuns.map((r: TrainingRun) => 
          r.id === id ? { ...r, ...updates } : r
        )
      })),
      removeRun: (id: string) => set((s: AppState) => ({
        activeRuns: s.activeRuns.filter((r: TrainingRun) => r.id !== id)
      })),
      
      // Models
      models: [],
      setModels: (models: Model[]) => set(() => ({ models })),
      activeModel: null,
      setActiveModel: (id: string | null) => set(() => ({ activeModel: id })),
      
      // Conversations
      conversations: [],
      activeConversation: null,
      createConversation: (model: string) => {
        const conv: Conversation = {
          id: crypto.randomUUID(),
          title: 'New Chat',
          model,
          messages: [],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
        set((s: AppState) => ({ 
          conversations: [conv, ...s.conversations],
          activeConversation: conv.id
        }))
        return conv
      },
      setActiveConversation: (id: string | null) => set(() => ({ activeConversation: id })),
      addMessage: (conversationId: string, message: Omit<Message, 'id' | 'timestamp'>) => {
        const fullMessage: Message = {
          ...message,
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
        }
        set((s: AppState) => ({
          conversations: s.conversations.map((c: Conversation) =>
            c.id === conversationId
              ? { 
                  ...c, 
                  messages: [...c.messages, fullMessage],
                  updated_at: new Date().toISOString(),
                  title: c.messages.length === 0 && message.role === 'user' 
                    ? message.content.slice(0, 30) + '...'
                    : c.title
                }
              : c
          )
        }))
      },
      
      // UI
      sidebarCollapsed: false,
      toggleSidebar: () => set((s: AppState) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    {
      name: 'onn-storage',
      partialize: (state: AppState) => ({
        conversations: state.conversations,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
)
