import { useEffect, useMemo, useRef, useState } from 'react'

import { chartPoints, formatNumber, formatPercent, rotatePoint } from './utils.js'

const ACTIVE_STATES = new Set(['starting', 'running', 'paused', 'saving', 'stopping'])

const STATUS_LABELS = {
  starting: '正在启动',
  running: '训练中',
  paused: '已暂停',
  saving: '正在保存',
  stopping: '正在停止',
  stopped: '已停止',
  completed: '训练完成',
  failed: '训练失败',
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`)
  return payload
}

function rollingAverage(values, window = 20) {
  return values.map((_, index) => {
    const start = Math.max(0, index - window + 1)
    const slice = values.slice(start, index + 1)
    return slice.reduce((sum, value) => sum + Number(value || 0), 0) / slice.length
  })
}

function interpolateAngle(from, to, alpha) {
  let delta = (to - from + Math.PI) % (Math.PI * 2) - Math.PI
  if (delta < -Math.PI) delta += Math.PI * 2
  return from + delta * alpha
}

function interpolateSnapshot(from, to, alpha) {
  if (!from?.frame || !to?.frame) return to
  if (from.training?.episode !== to.training?.episode) return to
  const fromPositions = from.frame.body_positions || []
  const toPositions = to.frame.body_positions || []
  const fromAngles = from.frame.body_angles || []
  const toAngles = to.frame.body_angles || []
  if (fromPositions.length !== toPositions.length || fromAngles.length !== toAngles.length) return to
  return {
    ...to,
    frame: {
      ...to.frame,
      body_positions: toPositions.map((position, index) => [
        fromPositions[index][0] + (position[0] - fromPositions[index][0]) * alpha,
        fromPositions[index][1] + (position[1] - fromPositions[index][1]) * alpha,
      ]),
      body_angles: toAngles.map((angle, index) => interpolateAngle(fromAngles[index], angle, alpha)),
    },
  }
}

function drawLiveScene(canvas, snapshot, trail) {
  if (!canvas || !snapshot?.metadata || !snapshot?.frame) return
  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const targetWidth = Math.max(1, Math.floor(rect.width * dpr))
  const targetHeight = Math.max(1, Math.floor(rect.height * dpr))
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth
    canvas.height = targetHeight
  }
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  const width = rect.width
  const height = rect.height
  const metadata = snapshot.metadata
  const frame = snapshot.frame
  const worldWidth = metadata.room.width
  const worldHeight = metadata.room.height
  const padding = 22
  const scale = Math.min((width - padding * 2) / worldWidth, (height - padding * 2) / worldHeight)
  const offsetX = (width - worldWidth * scale) / 2
  const offsetY = (height - worldHeight * scale) / 2
  const point = ([x, y]) => [offsetX + x * scale, height - offsetY - y * scale]

  ctx.clearRect(0, 0, width, height)
  const background = ctx.createLinearGradient(0, 0, 0, height)
  background.addColorStop(0, '#0d1c2e')
  background.addColorStop(1, '#06101c')
  ctx.fillStyle = background
  ctx.fillRect(0, 0, width, height)

  ctx.strokeStyle = 'rgba(148, 163, 184, .075)'
  ctx.lineWidth = 1
  for (let x = 0; x <= worldWidth; x += 1) {
    const [px] = point([x, 0])
    ctx.beginPath()
    ctx.moveTo(px, offsetY)
    ctx.lineTo(px, height - offsetY)
    ctx.stroke()
  }
  for (let y = 0; y <= worldHeight; y += 1) {
    const [, py] = point([0, y])
    ctx.beginPath()
    ctx.moveTo(offsetX, py)
    ctx.lineTo(width - offsetX, py)
    ctx.stroke()
  }

  ctx.strokeStyle = '#52657b'
  ctx.lineWidth = Math.max(3, scale * 0.055)
  ctx.strokeRect(offsetX, offsetY, worldWidth * scale, worldHeight * scale)

  for (const obstacle of metadata.obstacles || []) {
    const kind = String(obstacle.type || 'box').toLowerCase()
    if (['box', 'platform', 'wall'].includes(kind)) {
      const [cx, cy] = obstacle.position
      const [ow, oh] = obstacle.size
      const [left, top] = point([cx - ow / 2, cy + oh / 2])
      ctx.fillStyle = '#34465a'
      ctx.strokeStyle = '#74879c'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.roundRect(left, top, ow * scale, oh * scale, Math.min(7, ow * scale * 0.08))
      ctx.fill()
      ctx.stroke()
    }
  }

  const targetPosition = frame.target_position || metadata.target.position
  const target = { ...metadata.target, position: targetPosition }
  const [targetLeft, targetTop] = point([
    target.position[0] - target.size[0] / 2,
    target.position[1] + target.size[1] / 2,
  ])
  ctx.save()
  ctx.shadowColor = 'rgba(248, 69, 82, .8)'
  ctx.shadowBlur = 18
  ctx.fillStyle = '#ef4444'
  ctx.fillRect(targetLeft, targetTop, target.size[0] * scale, target.size[1] * scale)
  ctx.restore()
  ctx.strokeStyle = '#fecaca'
  ctx.strokeRect(targetLeft, targetTop, target.size[0] * scale, target.size[1] * scale)

  const activeWaypoint = frame.active_waypoint_index || 0
  const waypoints = frame.waypoints || metadata.waypoints || []
  for (let index = 0; index < waypoints.length; index += 1) {
    const waypoint = point(waypoints[index])
    ctx.beginPath()
    ctx.arc(...waypoint, Math.max(7, scale * 0.18), 0, Math.PI * 2)
    ctx.setLineDash([5, 5])
    ctx.strokeStyle = index < activeWaypoint ? '#34d399' : index === activeWaypoint ? '#fbbf24' : '#64748b'
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = '#dce7f5'
    ctx.font = '600 11px Inter, system-ui, sans-serif'
    ctx.fillText(`W${index + 1}`, waypoint[0] + 9, waypoint[1] - 8)
  }

  if (trail.length > 1) {
    ctx.strokeStyle = 'rgba(56, 189, 248, .38)'
    ctx.lineWidth = 2
    ctx.beginPath()
    trail.forEach((position, index) => {
      const trailPoint = point(position)
      if (index === 0) ctx.moveTo(...trailPoint)
      else ctx.lineTo(...trailPoint)
    })
    ctx.stroke()
  }

  const positions = frame.body_positions || []
  const angles = frame.body_angles || []
  metadata.body_names.forEach((name, index) => {
    const geometry = metadata.body_geometry[name]
    const position = positions[index]
    const angle = angles[index]
    if (!geometry || !position || angle === undefined) return
    const main = name === 'torso'
    ctx.fillStyle = main ? '#3b82f6' : '#d8e3ef'
    ctx.strokeStyle = main ? '#93c5fd' : '#172033'
    ctx.lineWidth = main ? 2 : 1.5
    if (geometry.kind === 'circle') {
      const center = point(rotatePoint(geometry.offset || [0, 0], angle, position))
      ctx.beginPath()
      ctx.arc(...center, Math.max(2, geometry.radius * scale), 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    } else if (geometry.kind === 'segment') {
      const a = point(rotatePoint(geometry.a, angle, position))
      const b = point(rotatePoint(geometry.b, angle, position))
      ctx.lineCap = 'round'
      ctx.lineWidth = Math.max(3, geometry.radius * scale * 2)
      ctx.strokeStyle = '#d8e3ef'
      ctx.beginPath()
      ctx.moveTo(...a)
      ctx.lineTo(...b)
      ctx.stroke()
      ctx.lineWidth = 1.2
      ctx.strokeStyle = '#172033'
      ctx.beginPath()
      ctx.moveTo(...a)
      ctx.lineTo(...b)
      ctx.stroke()
    } else if (geometry.kind === 'polygon') {
      const vertices = geometry.vertices.map((vertex) => point(rotatePoint(vertex, angle, position)))
      ctx.beginPath()
      vertices.forEach((vertex, vertexIndex) => {
        if (vertexIndex === 0) ctx.moveTo(...vertex)
        else ctx.lineTo(...vertex)
      })
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
    }
  })

  if (frame.info?.is_success) {
    ctx.fillStyle = 'rgba(16, 185, 129, .15)'
    ctx.fillRect(offsetX, offsetY, worldWidth * scale, worldHeight * scale)
    ctx.fillStyle = '#a7f3d0'
    ctx.font = '700 22px Inter, system-ui, sans-serif'
    ctx.fillText('GOAL REACHED', offsetX + 22, offsetY + 36)
  }
}

function LiveCanvas({ snapshot }) {
  const canvasRef = useRef(null)
  const trailRef = useRef([])
  const targetRef = useRef(null)
  const fromRef = useRef(null)
  const renderedRef = useRef(null)
  const transitionStartedRef = useRef(0)
  const lastFrameKeyRef = useRef('')

  useEffect(() => {
    if (!snapshot?.metadata || !snapshot?.frame) return
    fromRef.current = renderedRef.current || targetRef.current || snapshot
    targetRef.current = snapshot
    transitionStartedRef.current = performance.now()

    const frameKey = `${snapshot.training?.episode}-${snapshot.training?.num_timesteps}`
    if (frameKey !== lastFrameKeyRef.current) {
      const previousEpisode = lastFrameKeyRef.current.split('-')[0]
      const currentEpisode = String(snapshot.training?.episode || '')
      if (previousEpisode && previousEpisode !== currentEpisode) trailRef.current = []
      const torsoIndex = snapshot.metadata.body_names.indexOf('torso')
      const torso = snapshot.frame.body_positions?.[torsoIndex]
      if (torso) trailRef.current = [...trailRef.current.slice(-179), torso]
      lastFrameKeyRef.current = frameKey
    }
  }, [snapshot])

  useEffect(() => {
    let animationFrame = 0
    const draw = (now) => {
      const target = targetRef.current
      if (target?.metadata && target?.frame) {
        const alpha = Math.min(1, Math.max(0, (now - transitionStartedRef.current) / 42))
        const displayed = interpolateSnapshot(fromRef.current, target, alpha)
        renderedRef.current = displayed
        drawLiveScene(canvasRef.current, displayed, trailRef.current)
      }
      animationFrame = window.requestAnimationFrame(draw)
    }
    animationFrame = window.requestAnimationFrame(draw)
    return () => window.cancelAnimationFrame(animationFrame)
  }, [])

  if (!snapshot) {
    return <div className="canvas-empty">启动训练后，这里显示当前训练 episode 的真实物理状态。</div>
  }
  return <canvas className="physics-canvas" ref={canvasRef} aria-label="实时强化学习训练环境" />
}

function StatusBadge({ state = 'idle' }) {
  return <span className={`status-badge status-${state}`}><i />{STATUS_LABELS[state] || '等待训练'}</span>
}

function MetricCard({ label, value, detail, tone = 'blue' }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  )
}

function LineChart({ title, values, unit = '', empty = '等待数据' }) {
  const numeric = (values || []).filter((value) => value !== null && Number.isFinite(Number(value))).map(Number)
  const width = 420
  const height = 115
  const points = chartPoints(numeric, width, height, 10)
  const current = numeric.at(-1)
  return (
    <div className="chart-card">
      <div className="chart-heading">
        <span>{title}</span>
        <strong>{current === undefined ? empty : `${formatNumber(current, 3)}${unit}`}</strong>
      </div>
      {numeric.length > 1 ? (
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
        </svg>
      ) : <div className="chart-empty">{empty}</div>}
    </div>
  )
}

function ActionPanel({ snapshot }) {
  const names = snapshot?.metadata?.action_names || []
  const actions = snapshot?.training?.action || []
  return (
    <div className="panel action-panel">
      <div className="panel-heading"><strong>策略输出</strong><span>关节动作 −1 到 +1</span></div>
      <div className="action-grid">
        {names.length ? names.map((name, index) => {
          const value = Number(actions[index] || 0)
          return (
            <div className="action-row" key={name}>
              <span>{name.replaceAll('_', ' ')}</span>
              <div className="action-track"><i style={{ left: `${50 + value * 50}%` }} /></div>
              <b>{value.toFixed(2)}</b>
            </div>
          )
        }) : <div className="panel-empty">等待训练帧</div>}
      </div>
    </div>
  )
}

function TrainingForm({ options, disabled, onStart }) {
  const defaults = options?.defaults || {}
  const [stage, setStage] = useState(defaults.stage ?? 1)
  const [timesteps, setTimesteps] = useState(defaults.timesteps ?? 100000)
  const [seed, setSeed] = useState(defaults.seed ?? 0)
  const [trainConfig, setTrainConfig] = useState(defaults.train_config || 'configs/train_live.yaml')
  const [envConfig, setEnvConfig] = useState('')

  useEffect(() => {
    if (!options?.defaults) return
    setStage(options.defaults.stage)
    setTimesteps(options.defaults.timesteps)
    setSeed(options.defaults.seed)
    setTrainConfig(options.defaults.train_config)
  }, [options])

  return (
    <form className="training-form" onSubmit={(event) => {
      event.preventDefault()
      onStart({ stage: Number(stage), timesteps: Number(timesteps), seed: Number(seed), train_config: trainConfig, env_config: envConfig || null })
    }}>
      <div className="form-heading">
        <div><span>NEW PPO SESSION</span><h2>从随机权重开始训练</h2></div>
        <b>FROM SCRATCH</b>
      </div>
      <label>
        <span>课程阶段</span>
        <select value={stage} onChange={(event) => setStage(event.target.value)}>
          {(options?.stages || []).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </select>
      </label>
      <div className="form-pair">
        <label><span>总训练步数</span><input min="64" max="10000000" step="1" type="number" value={timesteps} onChange={(event) => setTimesteps(event.target.value)} /></label>
        <label><span>随机种子</span><input min="0" type="number" value={seed} onChange={(event) => setSeed(event.target.value)} /></label>
      </div>
      <label>
        <span>PPO 配置</span>
        <select value={trainConfig} onChange={(event) => setTrainConfig(event.target.value)}>
          {(options?.train_configs || []).map((path) => <option key={path} value={path}>{path.replace('configs/', '')}</option>)}
        </select>
      </label>
      <label>
        <span>环境覆盖配置（可选）</span>
        <select value={envConfig} onChange={(event) => setEnvConfig(event.target.value)}>
          <option value="">使用所选 Stage 默认配置</option>
          {(options?.env_configs || []).map((path) => <option key={path} value={path}>{path.replace('configs/', '')}</option>)}
        </select>
      </label>
      <button className="start-button" disabled={disabled} type="submit">
        {disabled ? '已有任务正在训练' : '▶ 创建并启动真实训练'}
      </button>
      <p>Stage 1 最适合验证从零学习；Stage 3 以上从随机权重直接训练会明显更难。</p>
    </form>
  )
}

function RunList({ runs, selectedId, onSelect }) {
  return (
    <div className="run-list">
      <div className="sidebar-section-title">训练记录</div>
      {runs.length ? runs.map((run) => (
        <button className={`run-item ${selectedId === run.run_id ? 'selected' : ''}`} key={run.run_id} onClick={() => onSelect(run.run_id)} type="button">
          <span className={`run-dot state-${run.status?.state || 'idle'}`} />
          <span><strong>{run.run_id}</strong><small>Stage {run.request?.stage ?? '?'} · {run.status?.num_timesteps || 0} steps</small></span>
          <em>{STATUS_LABELS[run.status?.state] || run.status?.state}</em>
        </button>
      )) : <div className="run-list-empty">还没有 UI 训练任务</div>}
    </div>
  )
}

function App() {
  const [options, setOptions] = useState(null)
  const [runs, setRuns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [run, setRun] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [streamFrame, setStreamFrame] = useState(null)
  const [streamConnected, setStreamConnected] = useState(false)
  const [streamFps, setStreamFps] = useState(0)

  const loadRuns = async () => {
    const payload = await requestJson('/api/training/runs')
    setRuns(payload.runs || [])
    if (!selectedId && payload.runs?.length) setSelectedId(payload.runs[0].run_id)
  }

  useEffect(() => {
    Promise.all([requestJson('/api/training/options'), requestJson('/api/training/runs')])
      .then(([optionPayload, runPayload]) => {
        setOptions(optionPayload)
        setRuns(runPayload.runs || [])
        if (runPayload.runs?.length) setSelectedId(runPayload.runs[0].run_id)
      })
      .catch((reason) => setError(reason.message))
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => loadRuns().catch(() => {}), 1800)
    return () => window.clearInterval(timer)
  }, [selectedId])

  useEffect(() => {
    if (!selectedId) {
      setRun(null)
      setStreamFrame(null)
      setStreamConnected(false)
      return undefined
    }
    let cancelled = false
    let socket = null
    let reconnectTimer = 0
    let frameCount = 0
    let fpsWindowStart = performance.now()

    const refresh = async () => {
      try {
        const payload = await requestJson(`/api/training/runs/${selectedId}`)
        if (!cancelled) {
          setRun(payload)
          if (!streamFrame && payload.frame) setStreamFrame(payload.frame)
        }
      } catch (reason) {
        if (!cancelled) setError(reason.message)
      }
    }

    const connect = () => {
      if (cancelled) return
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/api/training/runs/${selectedId}/stream`)
      socket.onopen = () => setStreamConnected(true)
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === 'snapshot') {
          setRun(message.payload)
          if (message.payload?.frame) setStreamFrame(message.payload.frame)
        } else if (message.type === 'frame') {
          setStreamFrame((previous) => ({ metadata: previous?.metadata, ...message.payload }))
          frameCount += 1
          const now = performance.now()
          const elapsed = now - fpsWindowStart
          if (elapsed >= 750) {
            setStreamFps((frameCount * 1000) / elapsed)
            frameCount = 0
            fpsWindowStart = now
          }
        } else if (message.type === 'status') {
          setRun((previous) => previous ? { ...previous, status: message.payload } : previous)
        } else if (message.type === 'metrics') {
          setRun((previous) => previous ? { ...previous, metrics: message.payload } : previous)
        } else if (message.type === 'last_save') {
          setRun((previous) => previous ? { ...previous, last_save: message.payload } : previous)
        }
      }
      socket.onerror = () => socket?.close()
      socket.onclose = () => {
        setStreamConnected(false)
        setStreamFps(0)
        if (!cancelled) reconnectTimer = window.setTimeout(connect, 700)
      }
    }

    setStreamFrame(null)
    refresh()
    connect()
    const fallbackTimer = window.setInterval(refresh, 1200)
    return () => {
      cancelled = true
      window.clearInterval(fallbackTimer)
      window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [selectedId])

  const activeRun = runs.find((item) => ACTIVE_STATES.has(item.status?.state))
  const status = run?.status || {}
  const request = run?.request || {}
  const episodes = run?.metrics?.episodes || []
  const updates = run?.metrics?.updates || []
  const rewards = episodes.map((episode) => episode.reward)
  const successRolling = rollingAverage(episodes.map((episode) => Number(episode.success)), 20)
  const distanceRolling = rollingAverage(episodes.map((episode) => episode.final_distance), 20)
  const valueLosses = updates.map((update) => update.value_loss).filter((value) => value !== null)
  const progressPercent = Math.max(0, Math.min(100, Number(status.progress || 0) * 100))
  const state = status.state || 'idle'
  const liveSnapshot = streamFrame || run?.frame

  const handleStart = async (payload) => {
    setBusy(true)
    setError('')
    try {
      const created = await requestJson('/api/training/runs', { method: 'POST', body: JSON.stringify(payload) })
      setSelectedId(created.run_id)
      setRun(created)
      await loadRuns()
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  const control = async (action) => {
    if (!selectedId) return
    setBusy(true)
    setError('')
    try {
      await requestJson(`/api/training/runs/${selectedId}/control`, { method: 'POST', body: JSON.stringify({ action }) })
      const payload = await requestJson(`/api/training/runs/${selectedId}`)
      setRun(payload)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">RL</div><div><strong>Stickman RL Lab</strong><span>LIVE TRAINING CONSOLE</span></div></div>
        <p className="sidebar-copy">直接创建 PPO 训练进程，观察当前 episode 的真实物理状态和训练指标。</p>
        <TrainingForm options={options} disabled={Boolean(activeRun) || busy} onStart={handleStart} />
        <RunList runs={runs} selectedId={selectedId} onSelect={setSelectedId} />
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">LIVE REINFORCEMENT LEARNING</span>
            <h1>实时训练控制台</h1>
            <p>模型权重从随机初始化开始更新；画面、奖励、成功率和 PPO 损失均来自当前训练进程。</p>
          </div>
          <div className="topbar-status"><StatusBadge state={state} /><span>{streamConnected ? `${streamFps.toFixed(0)} FPS source` : 'stream reconnecting'}</span><span>seed {request.seed ?? '—'}</span><span>stage {request.stage ?? '—'}</span></div>
        </header>

        {error && <div className="error-banner"><strong>操作失败</strong><span>{error}</span></div>}

        <section className="simulation-card">
          <div className="simulation-toolbar">
            <div><span className="live-pill"><i /> LIVE PHYSICS</span><b>{request.from_scratch === false ? 'resumed model' : 'randomly initialized PPO'}</b></div>
            <div className="training-controls">
              {state === 'paused' ? <button disabled={busy} onClick={() => control('resume')}>▶ 继续</button> : <button disabled={busy || state !== 'running'} onClick={() => control('pause')}>Ⅱ 暂停</button>}
              <button disabled={busy || !ACTIVE_STATES.has(state)} onClick={() => control('save')}>保存 checkpoint</button>
              <button className="danger" disabled={busy || !ACTIVE_STATES.has(state)} onClick={() => control('stop')}>停止训练</button>
            </div>
          </div>
          <div className="canvas-wrap">
            <LiveCanvas snapshot={liveSnapshot} />
            <div className="canvas-hud left"><span>TIMESTEPS</span><strong>{status.num_timesteps || 0} / {status.total_timesteps || request.timesteps || '—'}</strong></div>
            <div className="canvas-hud right"><span>EPISODE</span><strong>{status.episode || '—'} · step {status.episode_step || 0}</strong></div>
          </div>
          <div className="progress-footer">
            <div className="progress-track"><i style={{ width: `${progressPercent}%` }} /></div>
            <span>{progressPercent.toFixed(2)}%</span>
            <b>{formatNumber(status.fps, 1)} steps/s · {streamFps.toFixed(0)} source FPS · interpolated render</b>
          </div>
        </section>

        <section className="metric-grid">
          <MetricCard label="当前训练步数" value={(status.num_timesteps || 0).toLocaleString()} detail={`目标 ${(status.total_timesteps || request.timesteps || 0).toLocaleString()}`} tone="blue" />
          <MetricCard label="当前 Episode 奖励" value={formatNumber(status.current_episode_reward, 2)} detail={`episode ${status.episode || 0}`} tone="orange" />
          <MetricCard label="最近成功率" value={formatPercent(status.rolling_success_rate)} detail={`最近 ${Math.min(50, status.completed_episodes || 0)} 个 episode`} tone="green" />
          <MetricCard label="最近终点距离" value={formatNumber(status.rolling_final_distance, 3)} detail={`${status.completed_episodes || 0} episodes completed`} tone="purple" />
        </section>

        <section className="analysis-layout">
          <div className="charts-grid">
            <LineChart title="Episode reward" values={rewards} />
            <LineChart title="Rolling success rate" values={successRolling} />
            <LineChart title="Rolling final distance" values={distanceRolling} />
            <LineChart title="PPO value loss" values={valueLosses} />
          </div>
          <ActionPanel snapshot={liveSnapshot} />
        </section>

        <section className="detail-grid">
          <div className="panel session-panel">
            <div className="panel-heading"><strong>训练会话</strong><span>真实进程状态</span></div>
            <div className="session-grid">
              <div><span>RUN ID</span><code>{run?.run_id || '尚未创建'}</code></div>
              <div><span>INITIALIZATION</span><code>random policy weights</code></div>
              <div><span>TRAIN CONFIG</span><code>{request.train_config || '—'}</code></div>
              <div><span>ENV CONFIG</span><code>{request.env_config || `default stage ${request.stage ?? '—'}`}</code></div>
              <div><span>PROCESS</span><code>{status.process_alive ? `PID ${status.pid} alive` : status.pid ? `PID ${status.pid} ended` : '—'}</code></div>
              <div><span>CHECKPOINT</span><code>{run?.last_save?.path || status.final_checkpoint || '尚未保存'}</code></div>
            </div>
          </div>
          <div className="panel loss-panel">
            <div className="panel-heading"><strong>最近 PPO 更新</strong><span>{updates.length} updates</span></div>
            <dl>
              <div><dt>policy loss</dt><dd>{formatNumber(status.losses?.policy_loss, 5)}</dd></div>
              <div><dt>value loss</dt><dd>{formatNumber(status.losses?.value_loss, 5)}</dd></div>
              <div><dt>entropy loss</dt><dd>{formatNumber(status.losses?.entropy_loss, 5)}</dd></div>
              <div><dt>approx KL</dt><dd>{formatNumber(status.losses?.approx_kl, 6)}</dd></div>
              <div><dt>explained variance</dt><dd>{formatNumber(status.losses?.explained_variance, 4)}</dd></div>
              <div><dt>learning rate</dt><dd>{formatNumber(status.losses?.learning_rate, 7)}</dd></div>
            </dl>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
