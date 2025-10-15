//! Main animation player component

use gloo_net::http::Request;
use gloo_timers::callback::Interval;
use neural_net_animator::{AnimationScript, Timeline};
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

use super::dvr_controls::DvrControls;
use super::info_panel::InfoPanel;
use super::metrics_panel::MetricsPanel;
use super::network_canvas::NetworkCanvas;
use super::timeline::TimelineComponent;

#[function_component(AnimationPlayer)]
pub fn animation_player() -> Html {
    let script = use_state(|| None::<AnimationScript>);
    let loading = use_state(|| true);
    let error = use_state(|| None::<String>);

    // Load animation script on mount
    {
        let script = script.clone();
        let loading = loading.clone();
        let error = error.clone();

        use_effect_with((), move |_| {
            spawn_local(async move {
                match load_animation_script().await {
                    Ok(data) => {
                        script.set(Some(data));
                        loading.set(false);
                    }
                    Err(e) => {
                        error.set(Some(e));
                        loading.set(false);
                    }
                }
            });
            || ()
        });
    }

    html! {
        <div class="animation-player">
            {
                if *loading {
                    html! { <div class="loading">{"Loading animation..."}</div> }
                } else if let Some(err) = (*error).as_ref() {
                    html! {
                        <div class="error">
                            <h3>{"Error"}</h3>
                            <p>{err}</p>
                        </div>
                    }
                } else if let Some(script_data) = (*script).as_ref() {
                    html! { <AnimationPlayerInner script={script_data.clone()} /> }
                } else {
                    html! { <div>{"No animation loaded"}</div> }
                }
            }
        </div>
    }
}

#[derive(Properties, PartialEq, Clone)]
struct AnimationPlayerInnerProps {
    script: AnimationScript,
}

#[function_component(AnimationPlayerInner)]
fn animation_player_inner(props: &AnimationPlayerInnerProps) -> Html {
    let script = props.script.clone();
    let total_duration = script.total_duration();

    let timeline = use_state(|| Timeline::new(total_duration));
    let current_svg = use_state(|| None::<String>);

    // Load initial checkpoint
    {
        let current_svg = current_svg.clone();
        let initial_checkpoint = script.scenes.get(0).map(|s| s.network_state.checkpoint_path.clone());

        use_effect_with((), move |_| {
            if let Some(checkpoint_path) = initial_checkpoint {
                spawn_local(async move {
                    if let Ok(svg) = load_and_render_checkpoint(&checkpoint_path).await {
                        current_svg.set(Some(svg));
                    }
                });
            }
            || ()
        });
    }

    // Animation loop
    {
        let timeline = timeline.clone();
        let current_svg = current_svg.clone();
        let script = script.clone();

        use_effect_with((), move |_| {
            let interval = Interval::new(16, move || {
                let mut t = (*timeline).clone();
                if t.update() {
                    let current_time = t.current_time();
                    timeline.set(t);

                    // Update scene if needed
                    if let Some((_idx, scene, _)) = script.scene_at_time(current_time) {
                        let checkpoint_path = scene.network_state.checkpoint_path.clone();
                        let current_svg = current_svg.clone();
                        spawn_local(async move {
                            if let Ok(svg) = load_and_render_checkpoint(&checkpoint_path).await {
                                current_svg.set(Some(svg));
                            }
                        });
                    }
                }
            });

            move || drop(interval)
        });
    }

    // Get current scene index
    let current_scene_idx = {
        let t = (*timeline).clone();
        script.scene_at_time(t.current_time()).map(|(idx, _, _)| idx)
    };

    html! {
        <div class="player-inner">
            <div class="player-header">
                <h2>{&script.metadata.title}</h2>
                <p class="description">{&script.metadata.description}</p>
            </div>

            <div class="player-content">
                <div class="main-view">
                    <NetworkCanvas svg={(*current_svg).clone()} />
                    <InfoPanel scene_idx={current_scene_idx} script={script.clone()} />
                </div>

                <div class="side-panel">
                    <MetricsPanel scene_idx={current_scene_idx} script={script.clone()} />
                </div>
            </div>

            <div class="player-controls">
                <TimelineComponent timeline={timeline.clone()} />
                <DvrControls timeline={timeline.clone()} />
            </div>
        </div>
    }
}

/// Load animation script from server
async fn load_animation_script() -> Result<AnimationScript, String> {
    let response = Request::get("/api/script")
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.ok() {
        return Err(format!("Server error: {}", response.status()));
    }

    let script: AnimationScript = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(script)
}

/// Load checkpoint and render as SVG
async fn load_and_render_checkpoint(checkpoint_path: &str) -> Result<String, String> {
    let url = format!("/api/checkpoint/{}", checkpoint_path);
    let response = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.ok() {
        return Err(format!("Server error: {}", response.status()));
    }

    // For now, return placeholder SVG
    // TODO: Parse checkpoint and use neural-net-viz to generate real SVG
    let svg = format!("<svg viewBox=\"0 0 800 600\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"800\" height=\"600\" fill=\"#f8f9fa\"/><text x=\"400\" y=\"300\" text-anchor=\"middle\" font-size=\"20\" fill=\"#333\">Checkpoint: {}</text></svg>", checkpoint_path);

    Ok(svg)
}
