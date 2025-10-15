//! Main application component

use yew::prelude::*;
use super::animation_player::AnimationPlayer;

#[function_component(App)]
pub fn app() -> Html {
    html! {
        <main class="app-container">
            <h1>{"Neural Network Animator"}</h1>
            <AnimationPlayer />
        </main>
    }
}
