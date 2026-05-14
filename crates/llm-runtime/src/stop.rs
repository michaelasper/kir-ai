pub(crate) fn apply_stop_sequences(content: &mut String, stop: &[String]) -> bool {
    let Some(stop_at) = stop
        .iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
    else {
        return false;
    };
    content.truncate(stop_at);
    true
}

pub(crate) fn earliest_stop_index(content: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
}

pub(crate) fn max_stop_sequence_len(stop: &[String]) -> usize {
    stop.iter().map(String::len).max().unwrap_or(0)
}

pub(crate) fn safe_stream_emit_len(content: &str, max_stop_len: usize) -> usize {
    if max_stop_len <= 1 {
        return content.len();
    }
    floor_char_boundary(content, content.len().saturating_sub(max_stop_len - 1))
}

fn floor_char_boundary(content: &str, mut index: usize) -> usize {
    while !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}
