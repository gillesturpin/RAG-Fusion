import { useState, useEffect } from 'react'
import {
  Award, TrendingUp, CheckCircle, XCircle,
  BarChart3, FileText, Target, ChevronDown, ChevronRight
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = {
  pass: '#10b981',
  fail: '#ef4444',
  primary: '#3b82f6',
  secondary: '#8b5cf6'
}

const GRADE_COLORS = {
  'A+': '#10b981',
  'A': '#34d399',
  'B': '#fbbf24',
  'C': '#fb923c',
  'D': '#f87171',
  'F': '#ef4444'
}

export default function CertificationDashboard({ API_URL }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [expandedRow, setExpandedRow] = useState(null)

  useEffect(() => {
    fetchCertificationData()
  }, [])

  const fetchCertificationData = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${API_URL}/api/certification`)
      if (!response.ok) throw new Error('Failed to load certification data')
      const json = await response.json()
      setData(json)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="certification-container">
        <div className="certification-loading">
          <div className="spinner"></div>
          <p>Loading results...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="certification-container">
        <div className="certification-error">
          <XCircle size={48} />
          <h3>Error</h3>
          <p>{error}</p>
          <button onClick={fetchCertificationData} className="btn-retry">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) return null

  const { summary, detailed_results } = data
  const isPassed = data.certification_passed

  // Filter questions by category
  const filteredQuestions = selectedCategory === 'all'
    ? detailed_results
    : detailed_results.filter(q => q.metadata?.category === selectedCategory)

  // Get unique categories
  const categories = ['all', ...new Set(detailed_results.map(q => q.metadata?.category).filter(Boolean))]

  // Prepare data for charts
  const categoryData = categories.slice(1).map(cat => {
    const questions = detailed_results.filter(q => q.metadata?.category === cat)
    const passed = questions.filter(q => q.verdict?.passed).length
    const failed = questions.length - passed
    const avgScore = questions.reduce((sum, q) => sum + q.overall_score, 0) / questions.length

    return {
      category: cat,
      passed,
      failed,
      avgScore: (avgScore * 100).toFixed(1)
    }
  })

  const metricsData = [
    {
      name: 'Context Precision',
      value: (summary.average_scores.context_precision * 100).toFixed(1),
      threshold: 70
    },
    {
      name: 'Answer Similarity',
      value: (summary.average_scores.answer_similarity * 100).toFixed(1),
      threshold: 75
    }
  ]

  return (
    <div className="certification-container">
      {/* Header */}
      <div className="cert-header">
        <div className="cert-title-section">
          <div>
            <h1>Evaluation Results</h1>
            <p className="cert-subtitle">
              RAG Evaluation - {new Date(data.timestamp).toLocaleDateString('en-US')}
            </p>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="cert-summary-grid">
        <div className="cert-card">
          <div className="cert-card-header">
            <Target size={20} />
            <span>Overall Score</span>
          </div>
          <div className="cert-card-value" style={{ color: isPassed ? COLORS.pass : COLORS.fail }}>
            {(summary.overall_score * 100).toFixed(1)}%
          </div>
        </div>

        <div className="cert-card">
          <div className="cert-card-header">
            <BarChart3 size={20} />
            <span>Success Rate</span>
          </div>
          <div className="cert-card-value">
            {summary.passed}/{summary.total_questions}
          </div>
        </div>

        <div className="cert-card">
          <div className="cert-card-header">
            <TrendingUp size={20} />
            <span>Context Precision</span>
          </div>
          <div className="cert-card-value" style={{
            color: summary.average_scores.context_precision >= 0.7 ? COLORS.pass : COLORS.fail
          }}>
            {(summary.average_scores.context_precision * 100).toFixed(1)}%
          </div>
        </div>

        <div className="cert-card">
          <div className="cert-card-header">
            <FileText size={20} />
            <span>Answer Similarity</span>
          </div>
          <div className="cert-card-value" style={{
            color: summary.average_scores.answer_similarity >= 0.75 ? COLORS.pass : COLORS.fail
          }}>
            {(summary.average_scores.answer_similarity * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Questions Table */}
      <div className="cert-questions-section">
        <div className="cert-questions-header">
          <h3>Detailed Questions</h3>
        </div>

        <div className="cert-table-container">
          <table className="cert-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}></th>
                <th>ID</th>
                <th>Question</th>
                <th>Category</th>
                <th>Difficulty</th>
                <th>CP</th>
                <th>AS</th>
                <th>Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredQuestions.map((q) => {
                const isExpanded = expandedRow === q.metadata?.id
                return (
                  <>
                    <tr
                      key={q.metadata?.id || Math.random()}
                      onClick={() => setExpandedRow(isExpanded ? null : q.metadata?.id)}
                      style={{ cursor: 'pointer' }}
                      className={isExpanded ? 'expanded-row' : ''}
                    >
                      <td>
                        {isExpanded ? (
                          <ChevronDown size={16} />
                        ) : (
                          <ChevronRight size={16} />
                        )}
                      </td>
                      <td>{q.metadata?.id}</td>
                      <td className="question-cell" title={q.question}>
                        {q.question.substring(0, 60)}...
                      </td>
                      <td>
                        <span className="category-badge">
                          {q.metadata?.category}
                        </span>
                      </td>
                      <td>
                        <span className={`difficulty-badge ${q.metadata?.difficulty}`}>
                          {q.metadata?.difficulty}
                        </span>
                      </td>
                      <td className="metric-cell">
                        {(q.scores?.context_precision * 100).toFixed(0)}%
                      </td>
                      <td className="metric-cell">
                        {(q.scores?.answer_similarity * 100).toFixed(0)}%
                      </td>
                      <td className="score-cell">
                        <span style={{ color: GRADE_COLORS[q.verdict?.grade?.split(' ')[0]] || '#666' }}>
                          {(q.overall_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td>
                        {q.verdict?.passed ? (
                          <CheckCircle size={20} color={COLORS.pass} />
                        ) : (
                          <XCircle size={20} color={COLORS.fail} />
                        )}
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr className="expanded-details">
                        <td colSpan="9">
                          <div className="question-details">
                            <div className="detail-section">
                              <h4>Question</h4>
                              <p>{q.question}</p>
                            </div>

                            <div className="detail-section">
                              <h4>Ground Truth</h4>
                              <p>{q.ground_truth || 'N/A'}</p>
                            </div>

                            <div className="detail-section">
                              <h4>Generated Answer</h4>
                              <p>{q.answer || 'N/A'}</p>
                            </div>

                            {q.contexts && q.contexts.length > 0 && (
                              <div className="detail-section">
                                <h4>Retrieved Contexts ({q.contexts.length})</h4>
                                {q.contexts.map((ctx, idx) => (
                                  <div key={idx} className="context-item">
                                    <strong>Context {idx + 1}:</strong>
                                    <p>{ctx}</p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
