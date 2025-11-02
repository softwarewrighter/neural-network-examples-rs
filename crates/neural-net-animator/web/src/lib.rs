//! Neural Network Animator - Yew WASM Frontend

use wasm_bindgen::prelude::*;

mod components;
use components::app::App;

/// Entry point for WASM
#[wasm_bindgen(start)]
pub fn run_app() {
    console_error_panic_hook::set_once();
    yew::Renderer::<App>::new().render();
}
