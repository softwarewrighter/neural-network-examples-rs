//! Timeline scrubber component

use neural_net_animator::Timeline;
use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, InputEvent};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct TimelineComponentProps {
    pub timeline: UseStateHandle<Timeline>,
}

#[function_component(TimelineComponent)]
pub fn timeline_component(props: &TimelineComponentProps) -> Html {
    let timeline = props.timeline.clone();

    let on_seek = {
        let timeline = timeline.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(input) = e
                .target()
                .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
            {
                if let Ok(progress) = input.value().parse::<f64>() {
                    let mut t = (*timeline).clone();
                    t.seek_to_progress(progress / 100.0);
                    timeline.set(t);
                }
            }
        })
    };

    let current_time = (*timeline).format_time();
    let total_duration = (*timeline).format_duration();
    let progress_percent = (*timeline).progress() * 100.0;

    html! {
        <div class="timeline-component">
            <div class="timeline-info">
                <span class="current-time">{current_time}</span>
                <span class="separator">{" / "}</span>
                <span class="total-duration">{total_duration}</span>
            </div>

            <div class="timeline-bar">
                <input
                    type="range"
                    min="0"
                    max="100"
                    step="0.1"
                    value={progress_percent.to_string()}
                    oninput={on_seek}
                    class="timeline-slider"
                />
                <div
                    class="timeline-progress"
                    style={format!("width: {}%", progress_percent)}
                />
            </div>
        </div>
    }
}
