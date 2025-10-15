//! Information display panel

use neural_net_animator::AnimationScript;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct InfoPanelProps {
    pub scene_idx: Option<usize>,
    pub script: AnimationScript,
}

#[function_component(InfoPanel)]
pub fn info_panel(props: &InfoPanelProps) -> Html {
    html! {
        <div class="info-panel">
            <div class="network-info">
                <h3>{"Network Architecture"}</h3>
                <p class="architecture">{&props.script.network_info.architecture}</p>
                <p class="activation">
                    {"Activation: "}
                    <span class="value">{&props.script.network_info.activation}</span>
                </p>
            </div>

            {
                if let Some(idx) = props.scene_idx {
                    if let Some(scene) = props.script.scenes.get(idx) {
                        html! {
                            <div class="scene-info">
                                <h3>{"Current Scene"}</h3>
                                <p class="scene-id">
                                    {"Scene: "}
                                    <span class="value">{&scene.id}</span>
                                </p>
                                <p class="iteration">
                                    {"Iteration: "}
                                    <span class="value">
                                        {format!("{}", scene.network_state.iteration)}
                                    </span>
                                </p>

                                {
                                    if !scene.annotations.is_empty() {
                                        html! {
                                            <div class="annotations">
                                                <h4>{"Annotations"}</h4>
                                                <ul class="annotation-list">
                                                    {
                                                        scene.annotations.iter().map(|ann| {
                                                            html! {
                                                                <li class="annotation">
                                                                    <strong>{format!("{:?}: ", ann.annotation_type)}</strong>
                                                                    {&ann.text}
                                                                </li>
                                                            }
                                                        }).collect::<Html>()
                                                    }
                                                </ul>
                                            </div>
                                        }
                                    } else {
                                        html! {}
                                    }
                                }
                            </div>
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
