//! Metrics display panel

use neural_net_animator::AnimationScript;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct MetricsPanelProps {
    pub scene_idx: Option<usize>,
    pub script: AnimationScript,
}

#[function_component(MetricsPanel)]
pub fn metrics_panel(props: &MetricsPanelProps) -> Html {
    html! {
        <div class="metrics-panel">
            <h3>{"Metrics"}</h3>

            {
                if let Some(idx) = props.scene_idx {
                    if let Some(scene) = props.script.scenes.get(idx) {
                        if let Some(ref test_results) = scene.network_state.test_results {
                            html! {
                                <div class="metrics-content">
                                    <div class="metric">
                                        <span class="metric-label">{"Accuracy:"}</span>
                                        <span class="metric-value">
                                            {format!("{:.1}%", test_results.accuracy * 100.0)}
                                        </span>
                                    </div>

                                    <div class="metric">
                                        <span class="metric-label">{"Mean Error:"}</span>
                                        <span class="metric-value">
                                            {format!("{:.4}", test_results.mean_error)}
                                        </span>
                                    </div>

                                    <div class="examples">
                                        <h4>{"Test Examples"}</h4>
                                        <ul class="example-list">
                                            {
                                                test_results.examples.iter().enumerate().map(|(idx, ex)| {
                                                    let status_class = if ex.correct { "correct" } else { "incorrect" };
                                                    html! {
                                                        <li class={classes!("example", status_class)}>
                                                            <span class="example-idx">{idx + 1}</span>
                                                            <span class="example-inputs">
                                                                {format!("{:?}", ex.inputs)}
                                                            </span>
                                                            <span class="example-arrow">{"→"}</span>
                                                            <span class="example-output">
                                                                {format!("{:.2}", ex.actual[0])}
                                                            </span>
                                                            <span class="example-status">
                                                                {if ex.correct { "✓" } else { "✗" }}
                                                            </span>
                                                        </li>
                                                    }
                                                }).collect::<Html>()
                                            }
                                        </ul>
                                    </div>
                                </div>
                            }
                        } else {
                            html! { <div class="no-metrics">{"No test results available"}</div> }
                        }
                    } else {
                        html! { <div class="no-scene">{"Scene not found"}</div> }
                    }
                } else {
                    html! { <div class="no-scene">{"No scene loaded"}</div> }
                }
            }
        </div>
    }
}
