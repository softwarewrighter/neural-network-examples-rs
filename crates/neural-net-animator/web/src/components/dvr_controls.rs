//! DVR-style playback controls

use neural_net_animator::{PlaybackState, Timeline};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct DvrControlsProps {
    pub timeline: UseStateHandle<Timeline>,
}

#[function_component(DvrControls)]
pub fn dvr_controls(props: &DvrControlsProps) -> Html {
    let timeline = props.timeline.clone();

    let on_restart = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            t.skip_to_start();
            timeline.set(t);
        })
    };

    let on_step_back = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            t.step_backward(1.0);
            timeline.set(t);
        })
    };

    let on_play_pause = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            t.toggle_play_pause();
            timeline.set(t);
        })
    };

    let on_step_forward = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            t.step_forward(1.0);
            timeline.set(t);
        })
    };

    let on_jump_end = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            t.skip_to_end();
            timeline.set(t);
        })
    };

    let on_slower = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            let current = t.speed();
            t.set_speed(current.cycle_backward());
            timeline.set(t);
        })
    };

    let on_faster = {
        let timeline = timeline.clone();
        Callback::from(move |_| {
            let mut t = (*timeline).clone();
            let current = t.speed();
            t.set_speed(current.cycle_forward());
            timeline.set(t);
        })
    };

    let is_playing = (*timeline).state() == PlaybackState::Playing;
    let speed_text = format!("{}×", (*timeline).speed().multiplier());

    html! {
        <div class="dvr-controls">
            <div class="control-group">
                <button class="control-btn" onclick={on_restart} title="Restart">
                    {"⏮"}
                </button>

                <button class="control-btn" onclick={on_step_back} title="Step Back">
                    {"◄"}
                </button>

                <button
                    class="control-btn play-pause"
                    onclick={on_play_pause}
                    title={if is_playing { "Pause" } else { "Play" }}
                >
                    {if is_playing { "❚❚" } else { "▶" }}
                </button>

                <button class="control-btn" onclick={on_step_forward} title="Step Forward">
                    {"►"}
                </button>

                <button class="control-btn" onclick={on_jump_end} title="Jump to End">
                    {"⏭"}
                </button>
            </div>

            <div class="speed-controls">
                <button class="control-btn speed-btn" onclick={on_slower} title="Slower">
                    {"0.5×"}
                </button>

                <span class="speed-display">{speed_text}</span>

                <button class="control-btn speed-btn" onclick={on_faster} title="Faster">
                    {"2×"}
                </button>
            </div>
        </div>
    }
}
