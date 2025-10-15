//! Web server for animation UI

use crate::script::AnimationScript;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,

    /// Host to bind to
    pub host: String,

    /// Path to animation script
    pub script_path: PathBuf,

    /// Path to checkpoint directory
    pub checkpoint_dir: PathBuf,

    /// Path to web assets directory (Trunk dist)
    pub web_dist_path: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            script_path: PathBuf::from("animation.json"),
            checkpoint_dir: PathBuf::from("checkpoints"),
            web_dist_path: Some(PathBuf::from("crates/neural-net-animator/web/dist")),
        }
    }
}

/// Shared server state
#[derive(Clone)]
struct AppState {
    script: Arc<AnimationScript>,
    #[allow(dead_code)]  // TODO: Use for resolving relative checkpoint paths
    checkpoint_dir: PathBuf,
}

/// Start the animation server
pub async fn start_server(config: ServerConfig) -> anyhow::Result<()> {
    // Load animation script
    let script_content = std::fs::read_to_string(&config.script_path)?;
    let script: AnimationScript = serde_json::from_str(&script_content)?;

    tracing::info!(
        "Loaded animation script: {} ({} scenes, {:.1}s total)",
        script.metadata.title,
        script.scenes.len(),
        script.total_duration()
    );

    let state = AppState {
        script: Arc::new(script),
        checkpoint_dir: config.checkpoint_dir.clone(),
    };

    // Build router
    let mut app = Router::new()
        .route("/api/script", get(get_script))
        .route("/api/checkpoint/*path", get(get_checkpoint))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Serve static files if web_dist_path is provided
    if let Some(dist_path) = &config.web_dist_path {
        if dist_path.exists() {
            tracing::info!("Serving web assets from: {}", dist_path.display());
            app = app.fallback_service(ServeDir::new(dist_path));
        } else {
            tracing::warn!(
                "Web dist path does not exist: {}. Run 'trunk build' in web/ directory first.",
                dist_path.display()
            );
            app = app.route("/", get(placeholder_handler));
        }
    } else {
        app = app.route("/", get(placeholder_handler));
    }

    let addr = SocketAddr::from((
        config.host.parse::<std::net::IpAddr>()?,
        config.port,
    ));

    tracing::info!("Starting server on http://{}", addr);
    tracing::info!("Press Ctrl+C to stop");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve a placeholder page when Trunk hasn't built the frontend yet
async fn placeholder_handler() -> Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Animator</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        h1 {
            margin: 0 0 10px 0;
            color: #333;
        }

        .description {
            color: #666;
            margin: 0;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }

        .visualization {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        #network-svg {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .panel h3 {
            margin: 0 0 15px 0;
            font-size: 16px;
            color: #333;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #666;
            font-size: 14px;
        }

        .metric-value {
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }

        .controls {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            border-top: 1px solid #ddd;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }

        .controls-inner {
            max-width: 1400px;
            margin: 0 auto;
        }

        .timeline {
            margin-bottom: 15px;
        }

        .timeline-bar {
            width: 100%;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            position: relative;
            cursor: pointer;
        }

        .timeline-progress {
            height: 100%;
            background: #2196F3;
            border-radius: 3px;
            transition: width 0.1s ease;
        }

        .timeline-info {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }

        .button-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        button {
            padding: 10px 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }

        button:hover {
            background: #f5f5f5;
            border-color: #999;
        }

        button:active {
            transform: translateY(1px);
        }

        .play-button {
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }

        .play-button:hover {
            background: #1976D2;
            border-color: #1976D2;
        }

        .speed-indicator {
            padding: 10px;
            color: #666;
            font-size: 14px;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 id="animation-title">Loading...</h1>
            <p class="description" id="animation-description"></p>
        </div>

        <div class="main-content">
            <div class="visualization">
                <div id="network-svg" class="loading">
                    Loading animation...
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <h3>Metrics</h3>
                    <div id="metrics">
                        <div class="metric">
                            <span class="metric-label">Iteration:</span>
                            <span class="metric-value" id="iteration">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Accuracy:</span>
                            <span class="metric-value" id="accuracy">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Error:</span>
                            <span class="metric-value" id="error">0.0000</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Info</h3>
                    <div id="info">
                        <div class="metric">
                            <span class="metric-label">Architecture:</span>
                            <span class="metric-value" id="architecture">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Activation:</span>
                            <span class="metric-value" id="activation">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Scene:</span>
                            <span class="metric-value" id="scene">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="controls">
        <div class="controls-inner">
            <div class="timeline">
                <div class="timeline-bar" id="timeline-bar">
                    <div class="timeline-progress" id="timeline-progress"></div>
                </div>
                <div class="timeline-info">
                    <span id="current-time">00:00</span>
                    <span id="total-time">00:00</span>
                </div>
            </div>

            <div class="button-row">
                <button id="btn-restart" title="Restart">⏮</button>
                <button id="btn-step-back" title="Step Backward">◄</button>
                <button id="btn-play" class="play-button" title="Play/Pause">▶</button>
                <button id="btn-step-forward" title="Step Forward">►</button>
                <button id="btn-end" title="Jump to End">⏭</button>

                <span style="flex: 1"></span>

                <button id="btn-speed-down" title="Slower">0.5×</button>
                <span class="speed-indicator" id="speed">1×</span>
                <button id="btn-speed-up" title="Faster">2×</button>
            </div>
        </div>
    </div>

    <script>
        // Animation state
        let script = null;
        let isPlaying = false;
        let currentTime = 0;
        let speed = 1.0;
        let lastUpdate = null;

        // Load script on startup
        async function init() {
            try {
                const response = await fetch('/api/script');
                script = await response.json();

                document.getElementById('animation-title').textContent = script.metadata.title;
                document.getElementById('animation-description').textContent = script.metadata.description;
                document.getElementById('architecture').textContent = script.network_info.architecture;
                document.getElementById('activation').textContent = script.network_info.activation;
                document.getElementById('total-time').textContent = formatTime(script.scenes.reduce((sum, s) => sum + s.duration, 0));

                updateScene();
                console.log('Animation loaded:', script);
            } catch (error) {
                console.error('Failed to load animation:', error);
                document.getElementById('network-svg').innerHTML = '<div class="loading">Error loading animation</div>';
            }
        }

        // Format seconds as MM:SS
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }

        // Update current scene
        async function updateScene() {
            if (!script) return;

            // Find current scene
            let cumulative = 0;
            let currentScene = null;
            let sceneIndex = 0;

            for (let i = 0; i < script.scenes.length; i++) {
                const scene = script.scenes[i];
                if (currentTime >= cumulative && currentTime < cumulative + scene.duration) {
                    currentScene = scene;
                    sceneIndex = i;
                    break;
                }
                cumulative += scene.duration;
            }

            if (!currentScene) {
                currentScene = script.scenes[script.scenes.length - 1];
                sceneIndex = script.scenes.length - 1;
            }

            // Update UI
            document.getElementById('scene').textContent = `${sceneIndex + 1}/${script.scenes.length}`;
            document.getElementById('iteration').textContent = currentScene.network_state.iteration;

            if (currentScene.network_state.test_results) {
                const results = currentScene.network_state.test_results;
                document.getElementById('accuracy').textContent = `${(results.accuracy * 100).toFixed(1)}%`;
                document.getElementById('error').textContent = results.mean_error.toFixed(4);
            }

            // Load and display SVG
            // TODO: Load actual checkpoint and render SVG
            // For now, show placeholder
            document.getElementById('network-svg').innerHTML =
                `<div class="loading">Scene: ${currentScene.id}<br>Checkpoint: ${currentScene.network_state.checkpoint_path}</div>`;
        }

        // Animation loop
        function animate() {
            if (isPlaying && script) {
                const now = Date.now();
                if (lastUpdate) {
                    const delta = (now - lastUpdate) / 1000 * speed;
                    currentTime += delta;

                    const totalDuration = script.scenes.reduce((sum, s) => sum + s.duration, 0);
                    if (currentTime >= totalDuration) {
                        currentTime = totalDuration;
                        pause();
                    }
                }
                lastUpdate = now;

                updateTimeline();
                updateScene();
            }

            requestAnimationFrame(animate);
        }

        // Update timeline UI
        function updateTimeline() {
            if (!script) return;

            const totalDuration = script.scenes.reduce((sum, s) => sum + s.duration, 0);
            const progress = (currentTime / totalDuration) * 100;

            document.getElementById('timeline-progress').style.width = `${progress}%`;
            document.getElementById('current-time').textContent = formatTime(currentTime);
        }

        // Playback controls
        function play() {
            isPlaying = true;
            lastUpdate = Date.now();
            document.getElementById('btn-play').textContent = '❚❚';
        }

        function pause() {
            isPlaying = false;
            lastUpdate = null;
            document.getElementById('btn-play').textContent = '▶';
        }

        function togglePlayPause() {
            if (isPlaying) {
                pause();
            } else {
                play();
            }
        }

        function restart() {
            currentTime = 0;
            updateTimeline();
            updateScene();
        }

        function jumpToEnd() {
            if (!script) return;
            currentTime = script.scenes.reduce((sum, s) => sum + s.duration, 0);
            updateTimeline();
            updateScene();
        }

        function stepBackward() {
            currentTime = Math.max(0, currentTime - 1);
            updateTimeline();
            updateScene();
        }

        function stepForward() {
            if (!script) return;
            const totalDuration = script.scenes.reduce((sum, s) => sum + s.duration, 0);
            currentTime = Math.min(totalDuration, currentTime + 1);
            updateTimeline();
            updateScene();
        }

        function changeSpeed(delta) {
            const speeds = [0.25, 0.5, 1.0, 2.0, 4.0];
            let index = speeds.indexOf(speed);
            index = Math.max(0, Math.min(speeds.length - 1, index + delta));
            speed = speeds[index];
            document.getElementById('speed').textContent = `${speed}×`;
        }

        // Event listeners
        document.getElementById('btn-play').addEventListener('click', togglePlayPause);
        document.getElementById('btn-restart').addEventListener('click', restart);
        document.getElementById('btn-end').addEventListener('click', jumpToEnd);
        document.getElementById('btn-step-back').addEventListener('click', stepBackward);
        document.getElementById('btn-step-forward').addEventListener('click', stepForward);
        document.getElementById('btn-speed-down').addEventListener('click', () => changeSpeed(-1));
        document.getElementById('btn-speed-up').addEventListener('click', () => changeSpeed(1));

        // Timeline scrubbing
        document.getElementById('timeline-bar').addEventListener('click', (e) => {
            if (!script) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const progress = x / rect.width;
            const totalDuration = script.scenes.reduce((sum, s) => sum + s.duration, 0);
            currentTime = progress * totalDuration;
            updateTimeline();
            updateScene();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stepBackward();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    stepForward();
                    break;
                case 'Home':
                    e.preventDefault();
                    restart();
                    break;
                case 'End':
                    e.preventDefault();
                    jumpToEnd();
                    break;
            }
        });

        // Start
        init();
        requestAnimationFrame(animate);
    </script>
</body>
</html>
    "#;

    Html(html.to_string())
}

/// Get animation script
async fn get_script(State(state): State<AppState>) -> Json<AnimationScript> {
    Json((*state.script).clone())
}

/// Get checkpoint file
async fn get_checkpoint(
    State(_state): State<AppState>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> Result<Response, StatusCode> {
    // The path parameter contains the full path after /api/checkpoint/
    // which may include directories (e.g., "examples/example-2/.../file.json")
    let checkpoint_path = PathBuf::from(&path);

    // Resolve relative to project root or use as absolute path
    let full_path = if checkpoint_path.is_absolute() {
        checkpoint_path
    } else {
        std::env::current_dir()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .join(&checkpoint_path)
    };

    if !full_path.exists() {
        tracing::warn!("Checkpoint not found: {}", full_path.display());
        return Err(StatusCode::NOT_FOUND);
    }

    match std::fs::read(&full_path) {
        Ok(content) => Ok((
            [(header::CONTENT_TYPE, "application/json")],
            content,
        )
            .into_response()),
        Err(e) => {
            tracing::error!("Failed to read checkpoint: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
