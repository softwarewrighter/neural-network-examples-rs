//! Main application component

use super::animation_player::AnimationPlayer;
use yew::prelude::*;

#[function_component(App)]
pub fn app() -> Html {
    html! {
        <main class="app-container">
            <h1>{"Neural Network Animator"}</h1>
            <AnimationPlayer />
        </main>
    }
}
