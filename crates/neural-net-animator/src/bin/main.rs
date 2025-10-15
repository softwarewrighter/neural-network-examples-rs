//! Neural Network Animator CLI

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use neural_net_animator::{GeneratorConfig, ScriptGenerator, server::ServerConfig};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "neural-net-animator")]
#[command(about = "Animate neural network training", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start animation server
    Serve {
        /// Path to animation script (JSON)
        script: PathBuf,

        /// Port to serve on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Path to checkpoint directory
        #[arg(short, long, default_value = "checkpoints")]
        checkpoints: PathBuf,

        /// Open browser automatically
        #[arg(short, long)]
        open: bool,
    },

    /// Validate animation script
    Validate {
        /// Path to animation script
        script: PathBuf,
    },

    /// Generate animation script from checkpoints
    Generate {
        /// Output path for generated script
        #[arg(short, long)]
        output: PathBuf,

        /// Checkpoint files (in order)
        #[arg(required = true)]
        checkpoints: Vec<PathBuf>,

        /// Animation title
        #[arg(short, long)]
        title: Option<String>,

        /// Animation description
        #[arg(short, long)]
        description: Option<String>,

        /// Scene duration (seconds)
        #[arg(long, default_value = "2.0")]
        scene_duration: f64,

        /// Intro/outro duration (seconds)
        #[arg(long, default_value = "3.0")]
        intro_duration: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neural_net_animator=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            script,
            port,
            host,
            checkpoints,
            open,
        } => {
            let config = ServerConfig {
                port,
                host: host.clone(),
                script_path: script,
                checkpoint_dir: checkpoints,
                web_dist_path: Some(PathBuf::from("crates/neural-net-animator/web/dist")),
            };

            let url = format!("http://{}:{}", host, port);

            if open {
                if let Err(e) = open_browser(&url) {
                    tracing::warn!("Failed to open browser: {}", e);
                }
            }

            neural_net_animator::server::start_server(config).await?;
        }

        Commands::Validate { script } => {
            validate_script(&script)?;
        }

        Commands::Generate {
            output,
            checkpoints,
            title,
            description,
            scene_duration,
            intro_duration,
        } => {
            generate_script(GenerateConfig {
                output,
                checkpoints,
                title,
                description,
                scene_duration,
                intro_duration,
            })?;
        }
    }

    Ok(())
}

/// Validate animation script
fn validate_script(path: &PathBuf) -> Result<()> {
    tracing::info!("Validating script: {}", path.display());

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read script: {}", path.display()))?;

    let script: neural_net_animator::AnimationScript = serde_json::from_str(&content)
        .with_context(|| "Failed to parse script as JSON")?;

    tracing::info!("✓ Script is valid JSON");
    tracing::info!("  Title: {}", script.metadata.title);
    tracing::info!("  Scenes: {}", script.scenes.len());
    tracing::info!("  Duration: {:.1}s", script.total_duration());

    // Validate checkpoint files exist
    for (idx, scene) in script.scenes.iter().enumerate() {
        let checkpoint_path = PathBuf::from(&scene.network_state.checkpoint_path);
        if !checkpoint_path.exists() {
            tracing::warn!(
                "  Scene {}: Checkpoint not found: {}",
                idx + 1,
                checkpoint_path.display()
            );
        } else {
            tracing::info!(
                "  Scene {}: ✓ Checkpoint exists: {}",
                idx + 1,
                checkpoint_path.display()
            );
        }
    }

    println!("\n✓ Script validation passed");
    Ok(())
}

struct GenerateConfig {
    output: PathBuf,
    checkpoints: Vec<PathBuf>,
    title: Option<String>,
    description: Option<String>,
    scene_duration: f64,
    intro_duration: f64,
}

/// Generate animation script from checkpoints
fn generate_script(config: GenerateConfig) -> Result<()> {
    tracing::info!("Generating animation script from {} checkpoints", config.checkpoints.len());

    let gen_config = GeneratorConfig {
        title: config.title.unwrap_or_else(|| "Neural Network Training".to_string()),
        description: config.description.unwrap_or_else(|| "Watch the network learn".to_string()),
        scene_duration: config.scene_duration,
        intro_duration: config.intro_duration,
        include_tests: false, // TODO: Add CLI option
        test_inputs: None,
        test_targets: None,
        truth_table: None,
    };

    let generator = ScriptGenerator::new(gen_config);
    let script = generator.generate_from_checkpoints(&config.checkpoints)?;

    let json = serde_json::to_string_pretty(&script)?;
    std::fs::write(&config.output, json)
        .with_context(|| format!("Failed to write script to: {}", config.output.display()))?;

    tracing::info!("✓ Generated script: {}", config.output.display());
    tracing::info!("  Scenes: {}", script.scenes.len());
    tracing::info!("  Duration: {:.1}s", script.total_duration());

    println!("\n✓ Script generated successfully");
    println!("  File: {}", config.output.display());
    println!("\nRun with:");
    println!("  neural-net-animator serve {}", config.output.display());

    Ok(())
}

/// Open URL in default browser
fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(&["/C", "start", url])
            .spawn()?;
    }

    Ok(())
}
