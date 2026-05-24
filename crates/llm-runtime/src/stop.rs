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

#[cfg(test)]
pub(crate) fn earliest_stop_index(content: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
}

pub(crate) fn max_stop_sequence_len(stop: &[String]) -> usize {
    stop.iter().map(String::len).max().unwrap_or(0)
}

#[derive(Debug, Clone)]
pub(crate) struct IncrementalStopDetector {
    max_stop_len: usize,
    previous_len: usize,
}

impl IncrementalStopDetector {
    pub(crate) fn new(stop: &[String]) -> Self {
        Self {
            max_stop_len: max_stop_sequence_len(stop),
            previous_len: 0,
        }
    }

    pub(crate) fn observe(&mut self, content: &str, stop: &[String]) -> Option<usize> {
        if stop.is_empty() {
            self.previous_len = content.len();
            return None;
        }
        if stop.iter().any(String::is_empty) {
            self.previous_len = content.len();
            return Some(0);
        }

        let overlap = self.max_stop_len.saturating_sub(1);
        let search_start = floor_char_boundary(content, self.previous_len.saturating_sub(overlap));
        self.previous_len = content.len();

        stop.iter()
            .filter_map(|sequence| {
                content[search_start..]
                    .find(sequence)
                    .map(|index| search_start + index)
            })
            .min()
    }
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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn incremental_stop_detector_finds_stop_split_across_chunks() {
        let stop = vec![" STOP".to_owned()];
        let mut detector = IncrementalStopDetector::new(&stop);
        let mut content = String::new();

        content.push_str("hello ST");
        assert_eq!(detector.observe(&content, &stop), None);

        content.push_str("OP ignored");
        assert_eq!(detector.observe(&content, &stop), Some("hello".len()));
    }

    #[test]
    fn incremental_stop_detector_matches_full_scan_with_utf8_overlap() {
        let stop = vec!["🙂END".to_owned(), "終わり".to_owned()];
        let chunks = ["alpha 🙂", "EN", "D beta 終", "わり"];
        let mut detector = IncrementalStopDetector::new(&stop);
        let mut content = String::new();

        for chunk in chunks {
            content.push_str(chunk);
            let incremental = detector.observe(&content, &stop);
            assert_eq!(incremental, earliest_stop_index(&content, &stop));
            if incremental.is_some() {
                break;
            }
        }
    }
}
