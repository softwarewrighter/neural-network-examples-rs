//! Network visualization canvas component

use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct NetworkCanvasProps {
    pub svg: Option<String>,
}

#[function_component(NetworkCanvas)]
pub fn network_canvas(props: &NetworkCanvasProps) -> Html {
    html! {
        <div class="network-canvas">
            {
                if let Some(svg_content) = &props.svg {
                    Html::from_html_unchecked(AttrValue::from(svg_content.clone()))
                } else {
                    html! { <div class="loading-network">{"Loading network..."}</div> }
                }
            }
        </div>
    }
}
