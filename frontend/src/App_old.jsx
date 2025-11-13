import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Send, Brain, Zap, Database, DollarSign,
  Upload, FileText, Trash2, X, Copy, Check, Globe
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  // Agent selector enabled
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const [showUpload, setShowUpload] = useState(false)
  const [showDocumentsModal, setShowDocumentsModal] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAgent, setSelectedAgent] = useState('advanced') // 'advanced' or 'basic'

  // Fetch initial documents
  useEffect(() => {
    fetchDocuments()
  }, [])

  // Handle paste to upload files (Ctrl+V)
  useEffect(() => {
    const handlePaste = (e) => {
      // Only handle paste when upload modal is open
      if (!showUpload) return

      const items = e.clipboardData?.items
      if (!items) return

      const files = []
      for (let i = 0; i < items.length; i++) {
        const item = items[i]
        if (item.kind === 'file') {
          const file = item.getAsFile()
          if (file) files.push(file)
        }
      }

      if (files.length > 0) {
        e.preventDefault()
        handleUpload(files)
      }
    }

    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [showUpload])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim() || loading) return

    const userMessage = { role: 'user', content: question }
    setMessages(prev => [...prev, userMessage])
    const currentQuestion = question // Save question before clearing
    setQuestion('')
    setLoading(true)

    try {
      // Start timing
      const startTime = Date.now()

      // Use fetch for streaming support
      const response = await fetch(`${API_URL}/api/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          session_id: `session-${Date.now()}`,
          agent_type: selectedAgent
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Read the stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullContent = ''
      let metadata = {}

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'token') {
                fullContent += data.content
                // Create or update the streaming message
                setMessages(prev => {
                  const newMessages = [...prev]
                  const lastIndex = newMessages.length - 1

                  // Check if last message is streaming
                  const hasStreamingMessage = lastIndex >= 0 &&
                    newMessages[lastIndex].role === 'assistant' &&
                    newMessages[lastIndex].isStreaming

                  if (!hasStreamingMessage) {
                    // Create new message on first token
                    newMessages.push({
                      role: 'assistant',
                      content: fullContent,
                      isStreaming: true,
                      metadata: {}
                    })
                  } else {
                    // Update existing message
                    newMessages[lastIndex] = {
                      ...newMessages[lastIndex],
                      content: fullContent
                    }
                  }
                  return newMessages
                })
              } else if (data.type === 'complete') {
                metadata = data.metadata || {}
                const latency = (Date.now() - startTime) / 1000

                // Update the final message with complete content and metadata
                setMessages(prev => {
                  const newMessages = [...prev]
                  const lastIndex = newMessages.length - 1
                  if (lastIndex >= 0) {
                    newMessages[lastIndex] = {
                      role: 'assistant',
                      content: fullContent || data.content,
                      isStreaming: false,
                      metadata: {
                        method: 'quality_rag_agent',
                        complexity: 'high',
                        latency: latency,
                        cost: 0.002,
                        fromCache: false,
                        fallbackUsed: metadata.used_retrieval === false
                      }
                    }
                  }
                  return newMessages
                })
              } else if (data.type === 'error') {
                throw new Error(data.error)
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      // Add error message
      setMessages(prev => {
        const newMessages = [...prev]
        const lastIndex = newMessages.length - 1

        if (lastIndex >= 0 && newMessages[lastIndex].isStreaming) {
          // Update existing streaming message with error
          newMessages[lastIndex] = {
            role: 'assistant',
            content: `Désolé, une erreur est survenue: ${error.message}`,
            error: true,
            isStreaming: false
          }
        } else {
          // Add new error message
          newMessages.push({
            role: 'assistant',
            content: `Désolé, une erreur est survenue: ${error.message}`,
            error: true
          })
        }
        return newMessages
      })
    } finally {
      setLoading(false)
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/documents`)
      setDocuments(response.data.documents || [])
    } catch (error) {
      console.error('Error fetching documents:', error)
    }
  }

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadProgress({ total: files.length, current: 0, status: 'Uploading...' })

    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }

    try {
      const response = await axios.post(`${API_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      // Analyze results to provide detailed feedback
      const results = response.data.results || []
      const successCount = results.filter(r => r.status === 'success').length
      const errorCount = results.filter(r => r.status === 'error').length
      const totalChunks = response.data.total_chunks

      let status = ''
      let isError = false

      if (errorCount === 0 && successCount > 0) {
        // All files succeeded
        status = `✅ ${totalChunks} chunks added from ${successCount} file${successCount > 1 ? 's' : ''}`
      } else if (successCount > 0 && errorCount > 0) {
        // Partial success
        status = `⚠️ ${successCount} succeeded, ${errorCount} failed`
        isError = true
      } else if (errorCount > 0) {
        // All failed
        const firstError = results.find(r => r.status === 'error')
        status = `❌ Upload failed: ${firstError?.message || 'Unknown error'}`
        isError = true
      } else {
        // No chunks added (shouldn't happen but handle it)
        status = `❌ No documents were added`
        isError = true
      }

      setUploadProgress({
        total: response.data.total_files,
        current: response.data.total_files,
        status: status,
        results: results,
        error: isError
      })

      // Refresh documents list if any succeeded
      if (successCount > 0) {
        fetchDocuments()
      }

      // Clear after 5 seconds (increased to allow reading detailed errors)
      setTimeout(() => {
        setUploadProgress(null)
        if (successCount > 0 && errorCount === 0) {
          setShowUpload(false)
        }
      }, 5000)

    } catch (error) {
      setUploadProgress({
        total: files.length,
        current: 0,
        status: `❌ Network error: ${error.message}`,
        error: true
      })
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (source) => {
    if (!window.confirm(`Delete "${source}"?`)) return

    try {
      await axios.delete(`${API_URL}/api/documents`, {
        params: { source: source }
      })
      fetchDocuments()
    } catch (error) {
      console.error('Error deleting document:', error)
      alert('Failed to delete document')
    }
  }

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    handleUpload(files)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    handleUpload(files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const getMethodColor = (method) => {
    const colors = {
      'basic_rag': '#2ecc71',
      'advanced_rag': '#f39c12',
      'agent_rag': '#e74c3c'
    }
    return colors[method] || '#95a5a6'
  }

  const getMethodLabel = (method) => {
    const labels = {
      'basic_rag': 'Basic',
      'advanced_rag': 'Advanced',
      'agent_rag': 'Agent'
    }
    return labels[method] || method
  }

  return (
    <div className="app">
      <div className="main-container">
        {/* Sidebar - Agent Selection & Documents */}
        <aside className="sidebar">
          {/* Agent Selection */}
          <div style={{ marginBottom: '2rem' }}>
            <h2 className="sidebar-title" style={{ marginBottom: '1rem' }}>
              <Brain size={20} />
              RAG Agent
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <button
                onClick={() => setSelectedAgent('basic')}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  background: selectedAgent === 'basic' ? 'var(--accent-blue)' : 'var(--bg-dark)',
                  color: selectedAgent === 'basic' ? 'white' : 'var(--text-primary)',
                  border: `1px solid ${selectedAgent === 'basic' ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                  borderRadius: '0.5rem',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  transition: 'all 0.2s',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem'
                }}
              >
                <span>Basic RAG</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '400', opacity: 0.8 }}>
                  Standard retrieval
                </span>
              </button>
              <button
                onClick={() => setSelectedAgent('advanced')}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  background: selectedAgent === 'advanced' ? 'var(--accent-blue)' : 'var(--bg-dark)',
                  color: selectedAgent === 'advanced' ? 'white' : 'var(--text-primary)',
                  border: `1px solid ${selectedAgent === 'advanced' ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                  borderRadius: '0.5rem',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  transition: 'all 0.2s',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem'
                }}
              >
                <span>Advanced RAG</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '400', opacity: 0.8 }}>
                  With grading & rewriting
                </span>
              </button>
            </div>
          </div>

          <h2 className="sidebar-title">
            <FileText size={20} />
            Documents
          </h2>

          <button
            onClick={() => setShowUpload(!showUpload)}
            style={{
              width: '100%',
              padding: '0.75rem',
              background: 'var(--accent-blue)',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              marginBottom: '1rem',
              fontSize: '0.9rem',
              fontWeight: '600'
            }}
          >
            <Upload size={18} />
            Upload
          </button>

          {showUpload && (
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '0.75rem',
                padding: '1.5rem',
                textAlign: 'center',
                marginBottom: '1rem',
                background: 'var(--bg-dark)',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onClick={() => document.getElementById('file-upload').click()}
            >
              <Upload size={32} style={{margin: '0 auto 0.5rem', color: 'var(--accent-blue)'}} />
              <p style={{fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.5rem'}}>
                Drag & drop files or click to browse
              </p>
              <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)'}}>
                PDF, TXT, MD, DOCX
              </p>
              <input
                id="file-upload"
                type="file"
                multiple
                accept=".pdf,.txt,.md,.docx,.doc"
                onChange={handleFileSelect}
                style={{display: 'none'}}
              />
            </div>
          )}

          {uploadProgress && (
            <div style={{
              padding: '0.75rem',
              background: uploadProgress.error ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
              border: `1px solid ${uploadProgress.error ? 'var(--accent-red)' : 'var(--accent-green)'}`,
              borderRadius: '0.5rem',
              marginBottom: '1rem',
              fontSize: '0.875rem'
            }}>
              <div style={{
                fontWeight: '600',
                marginBottom: uploadProgress.results?.length > 0 ? '0.5rem' : '0',
                color: uploadProgress.error ? 'var(--accent-red)' : 'var(--accent-green)'
              }}>
                {uploadProgress.status}
              </div>
              {uploadProgress.results && uploadProgress.results.length > 0 && (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem',
                  marginTop: '0.5rem',
                  fontSize: '0.75rem'
                }}>
                  {uploadProgress.results.map((result, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.5rem',
                        padding: '0.25rem 0',
                        color: result.status === 'success' ? 'var(--accent-green)' : 'var(--accent-red)'
                      }}
                    >
                      <span style={{ flexShrink: 0 }}>
                        {result.status === 'success' ? '✓' : '✗'}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: '500', wordBreak: 'break-word' }}>
                          {result.filename}
                        </div>
                        <div style={{ opacity: 0.8, fontSize: '0.7rem' }}>
                          {result.status === 'success'
                            ? `${result.chunks} chunks`
                            : result.message}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {documents.length > 0 ? (
            <button
              onClick={() => setShowDocumentsModal(true)}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-dark)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '500',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                transition: 'all 0.2s'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'var(--bg-dark)'}
            >
              <span>View all</span>
              <span style={{fontSize: '0.75rem', color: 'var(--text-secondary)'}}>{documents.length} files</span>
            </button>
          ) : (
            <div style={{
              textAlign: 'center',
              padding: '2rem 1rem',
              color: 'var(--text-secondary)',
              fontSize: '0.875rem'
            }}>
              No documents yet. Upload some to get started!
            </div>
          )}
        </aside>

        {/* Main Chat Area */}
        <main className="chat-container">
          <div className={`messages ${messages.length === 0 ? 'empty' : ''}`}>
            {messages.length === 0 ? (
              <div className="empty-state">
                <h2 style={{marginBottom: '2rem', fontSize: '2rem', color: 'var(--text-primary)'}}>Agentic RAG</h2>
                <form className="centered-input-form" onSubmit={handleSubmit}>
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question..."
                    disabled={loading}
                    className="input-field"
                    style={{
                      flex: 1,
                      padding: '0.875rem 1.25rem',
                      background: 'var(--bg-dark)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '0.75rem',
                      color: 'var(--text-primary)',
                      fontSize: '1rem',
                      minWidth: '400px'
                    }}
                  />
                  <button
                    type="submit"
                    disabled={loading || !question.trim()}
                    className="send-button"
                    style={{
                      padding: '0.875rem 1.5rem',
                      background: 'var(--accent-blue)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '0.75rem',
                      cursor: loading || !question.trim() ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: loading || !question.trim() ? 0.5 : 1
                    }}
                  >
                    <Send size={20} />
                  </button>
                </form>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message ${msg.role}`}
                >
                  <div className="message-content">
                    {msg.role === 'assistant' ? (
                      <>
                        <ReactMarkdown
                          components={{
                            code({node, inline, className, children, ...props}) {
                              const match = /language-(\w+)/.exec(className || '')
                              return !inline && match ? (
                                <SyntaxHighlighter
                                  style={vscDarkPlus}
                                  language={match[1]}
                                  PreTag="div"
                                  {...props}
                                >
                                  {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                              ) : (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              )
                            }
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                        {msg.isStreaming && (
                          <span className="cursor-blink" style={{
                            display: 'inline-block',
                            width: '2px',
                            height: '1.2em',
                            backgroundColor: 'var(--accent-blue)',
                            marginLeft: '2px',
                            animation: 'blink 1s infinite'
                          }}>▊</span>
                        )}
                      </>
                    ) : (
                      msg.content
                    )}
                  </div>
                  {msg.metadata && (
                    <div className="message-metadata">
                      <span
                        className="method-badge"
                        style={{backgroundColor: getMethodColor(msg.metadata.method)}}
                      >
                        {getMethodLabel(msg.metadata.method)}
                      </span>
                      <span className="metadata-item">
                        {msg.metadata.latency.toFixed(2)}s
                      </span>
                      <span className="metadata-item">
                        €{msg.metadata.cost.toFixed(3)}
                      </span>
                      {msg.metadata.fromCache && (
                        <span className="cache-badge">
                          <Database size={12} />
                          Cached
                        </span>
                      )}
                      {msg.metadata.fallbackUsed && (
                        <span className="cache-badge" style={{background: 'var(--accent-orange)'}}>
                          <Globe size={12} />
                          Web Search
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
            {loading && (
              <div className="message assistant loading-message">
                <div className="loading-dots">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            )}
          </div>

          {messages.length > 0 && (
            <form className="input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading}
              className="input-field"
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="send-button"
            >
              <Send size={20} />
            </button>
          </form>
          )}
        </main>
      </div>

      {/* Documents Modal */}
      {showDocumentsModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1rem'
          }}
          onClick={() => setShowDocumentsModal(false)}
        >
          <div
            style={{
              background: 'var(--bg-card)',
              borderRadius: '1rem',
              maxWidth: '600px',
              width: '100%',
              maxHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
              border: '1px solid var(--border-color)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{
              padding: '1.5rem',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Documents ({documents.length})</h2>
              <button
                onClick={() => setShowDocumentsModal(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-secondary)',
                  padding: '0.5rem',
                  display: 'flex',
                  borderRadius: '0.25rem'
                }}
                onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
              >
                <X size={20} />
              </button>
            </div>

            {/* Search Bar */}
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-color)' }}>
              <input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  background: 'var(--bg-dark)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  color: 'var(--text-primary)',
                  fontSize: '0.875rem'
                }}
              />
            </div>

            {/* Documents List */}
            <div style={{ padding: '1rem 1.5rem', overflowY: 'auto', flex: 1 }}>
              {documents
                .filter(doc => doc.source.toLowerCase().includes(searchQuery.toLowerCase()))
                .map((doc, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '0.75rem',
                      background: 'var(--bg-dark)',
                      borderRadius: '0.5rem',
                      border: '1px solid var(--border-color)',
                      marginBottom: '0.5rem'
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: '0.875rem',
                        fontWeight: '600',
                        color: 'var(--text-primary)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {doc.source}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {doc.chunks} chunks
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        handleDelete(doc.source)
                        setShowDocumentsModal(false)
                      }}
                      style={{
                        padding: '0.5rem',
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--accent-red)',
                        cursor: 'pointer',
                        borderRadius: '0.25rem',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                      onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              {documents.filter(doc => doc.source.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 && (
                <div style={{
                  textAlign: 'center',
                  padding: '2rem',
                  color: 'var(--text-secondary)',
                  fontSize: '0.875rem'
                }}>
                  No documents found
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
