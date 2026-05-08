pub(crate) const METAL_SOURCE: &str = concat!(
    include_str!("shaders/generic.metal"),
    include_str!("shaders/qwen.metal"),
    include_str!("shaders/matvec.metal"),
    include_str!("shaders/reductions.metal"),
);
