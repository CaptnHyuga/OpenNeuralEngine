import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Send, 
  Loader2, 
  Bot, 
  User, 
  Settings,
  Copy,
  Check,
  Sparkles
} from 'lucide-react'
import { listModels, streamInference } from '../api'
import { useStore, Message } from '../store'
import { cn, formatRelativeTime } from '../lib/utils'

export default function Inference() {
  const { 
    conversations, 
    activeConversation, 
    createConversation,
    setActiveConversation,
    addMessage
  } = useStore()
  
  const [input, setInput] = useState('')
  const [selectedModel, setSelectedModel] = useState('gpt2')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const [copied, setCopied] = useState<string | null>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  
  // Get models
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })
  
  // Get current conversation
  const conversation = conversations.find(c => c.id === activeConversation)
  
  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages, streamingContent])
  
  // Focus input
  useEffect(() => {
    inputRef.current?.focus()
  }, [activeConversation])
  
  // Send message
  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return
    
    // Ensure conversation exists
    let convId = activeConversation
    if (!convId) {
      const conv = createConversation(selectedModel)
      convId = conv.id
    }
    
    // Add user message
    addMessage(convId, { role: 'user', content: input.trim() })
    const userInput = input.trim()
    setInput('')
    
    // Stream response
    setIsStreaming(true)
    setStreamingContent('')
    
    try {
      let fullResponse = ''
      
      for await (const token of streamInference({
        model: selectedModel,
        prompt: userInput,
        max_tokens: maxTokens,
        temperature,
      })) {
        fullResponse += token
        setStreamingContent(fullResponse)
      }
      
      // Add assistant message
      addMessage(convId, { role: 'assistant', content: fullResponse, model: selectedModel })
    } catch (error) {
      console.error('Inference error:', error)
      addMessage(convId, { 
        role: 'assistant', 
        content: 'Sorry, there was an error generating a response.',
        model: selectedModel
      })
    } finally {
      setIsStreaming(false)
      setStreamingContent('')
    }
  }
  
  // Handle enter key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }
  
  // Copy message
  const copyMessage = (content: string, id: string) => {
    navigator.clipboard.writeText(content)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }
  
  // New chat
  const startNewChat = () => {
    const conv = createConversation(selectedModel)
    setActiveConversation(conv.id)
  }
  
  return (
    <div className="flex h-[calc(100vh-8rem)] gap-6">
      {/* Sidebar - Conversations */}
      <div className="w-64 flex flex-col bg-surface-900 rounded-xl border border-surface-700 overflow-hidden">
        <div className="p-4 border-b border-surface-700">
          <button
            onClick={startNewChat}
            className="btn-primary w-full"
          >
            <Sparkles className="w-4 h-4" />
            New Chat
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => setActiveConversation(conv.id)}
              className={cn(
                "w-full text-left px-3 py-2 rounded-lg transition-colors",
                conv.id === activeConversation
                  ? "bg-primary-500/10 text-primary-400"
                  : "text-surface-300 hover:bg-surface-800"
              )}
            >
              <p className="truncate text-sm font-medium">{conv.title}</p>
              <p className="text-xs text-surface-500 truncate">
                {conv.model} • {formatRelativeTime(conv.updated_at)}
              </p>
            </button>
          ))}
          
          {conversations.length === 0 && (
            <p className="text-center text-surface-500 text-sm py-8">
              No conversations yet
            </p>
          )}
        </div>
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-surface-900 rounded-xl border border-surface-700 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-surface-700">
          <div className="flex items-center gap-3">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-primary-500"
            >
              {models?.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              )) || (
                <>
                  <option value="gpt2">GPT-2</option>
                  <option value="distilgpt2">DistilGPT-2</option>
                  <option value="microsoft/DialoGPT-small">DialoGPT</option>
                </>
              )}
            </select>
            {conversation && (
              <span className="text-sm text-surface-400">
                {conversation.messages.length} messages
              </span>
            )}
          </div>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={cn(
              "p-2 rounded-lg transition-colors",
              showSettings
                ? "bg-primary-500/10 text-primary-400"
                : "text-surface-400 hover:text-surface-100 hover:bg-surface-800"
            )}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
        
        {/* Settings panel */}
        {showSettings && (
          <div className="px-4 py-3 border-b border-surface-700 bg-surface-800/50">
            <div className="flex gap-6">
              <div>
                <label className="block text-xs text-surface-400 mb-1">Temperature</label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-32"
                />
                <span className="ml-2 text-sm text-surface-300">{temperature}</span>
              </div>
              <div>
                <label className="block text-xs text-surface-400 mb-1">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  min={1}
                  max={2048}
                  className="w-20 bg-surface-700 border border-surface-600 rounded px-2 py-1 text-sm"
                />
              </div>
            </div>
          </div>
        )}
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {conversation?.messages.map((message) => (
            <MessageBubble 
              key={message.id} 
              message={message}
              copied={copied === message.id}
              onCopy={() => copyMessage(message.content, message.id)}
            />
          ))}
          
          {/* Streaming message */}
          {isStreaming && streamingContent && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="flex-1 bg-surface-800 rounded-lg p-4">
                <p className="text-surface-100 whitespace-pre-wrap">{streamingContent}</p>
                <span className="inline-block w-2 h-4 bg-primary-400 animate-pulse ml-0.5" />
              </div>
            </div>
          )}
          
          {/* Loading indicator */}
          {isStreaming && !streamingContent && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-surface-800 rounded-lg p-4">
                <Loader2 className="w-5 h-5 text-surface-400 animate-spin" />
              </div>
            </div>
          )}
          
          {/* Empty state */}
          {!conversation?.messages.length && !isStreaming && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500/20 to-accent-500/20 border border-primary-500/20 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-primary-400" />
              </div>
              <h3 className="text-lg font-medium text-surface-100 mb-2">
                Start a conversation
              </h3>
              <p className="text-surface-400 max-w-sm">
                Select a model and type a message to begin chatting.
                Your conversations are saved locally.
              </p>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input */}
        <div className="p-4 border-t border-surface-700">
          <div className="flex gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              rows={1}
              className="flex-1 resize-none bg-surface-800 border border-surface-700 rounded-lg px-4 py-3 focus:outline-none focus:border-primary-500 transition-colors"
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isStreaming}
              className="btn-primary px-4"
            >
              {isStreaming ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="text-xs text-surface-500 mt-2 text-center">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  )
}

// Message bubble component
function MessageBubble({ 
  message, 
  copied, 
  onCopy 
}: { 
  message: Message
  copied: boolean
  onCopy: () => void
}) {
  const isUser = message.role === 'user'
  
  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div className={cn(
        "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
        isUser 
          ? "bg-surface-700"
          : "bg-gradient-to-br from-primary-500 to-accent-500"
      )}>
        {isUser ? (
          <User className="w-4 h-4 text-surface-300" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>
      <div className={cn(
        "flex-1 rounded-lg p-4 group relative",
        isUser ? "bg-primary-500/10" : "bg-surface-800"
      )}>
        <p className="text-surface-100 whitespace-pre-wrap">{message.content}</p>
        {message.model && (
          <p className="text-xs text-surface-500 mt-2">{message.model}</p>
        )}
        
        {/* Copy button */}
        <button
          onClick={onCopy}
          className="absolute top-2 right-2 p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-surface-700 transition-all"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-400" />
          ) : (
            <Copy className="w-4 h-4 text-surface-400" />
          )}
        </button>
      </div>
    </div>
  )
}
