//! Inline-tag parser for Granite Speech 4.1 plus.
//!
//! Splits a raw decoded transcript that contains `[Speaker N]:` and
//! `[T:N]` markers into clean text plus structured speaker / timestamp
//! lists. Lives in its own file so the engine module stays under the
//! 700-line ceiling and so the parser is easy to unit-test in isolation
//! from the live ort sessions.

use crate::granite_plus::{SpeakerSegment, TimedWord};

/// Splits a raw decoded transcript into
/// `(clean_text, speaker_segments, timed_words)` and updates the
/// rollover counter as `[T:N]` tags are consumed. Public-in-crate so
/// unit tests can exercise it without constructing a real
/// [`crate::granite_plus::GranitePlus`] with live ort sessions.
#[derive(Debug, Clone, Default)]
pub(crate) struct TagParserState {
    pub rollover: u32,
    pub last_centiseconds: Option<u32>,
}

impl TagParserState {
    pub(crate) fn parse(&mut self, raw: &str) -> (String, Vec<SpeakerSegment>, Vec<TimedWord>) {
        let mut clean = String::with_capacity(raw.len());
        let mut segments: Vec<SpeakerSegment> = Vec::new();
        let mut words: Vec<TimedWord> = Vec::new();
        let mut current_speaker: Option<u32> = None;
        let mut current_segment_text = String::new();
        let mut pending_word = String::new();

        let mut i = 0;
        while i < raw.len() {
            if let Some((spk, end)) = parse_speaker_tag(raw, i) {
                if let Some(prev) = current_speaker {
                    let st = current_segment_text.trim().to_string();
                    if !st.is_empty() {
                        segments.push(SpeakerSegment {
                            speaker: prev,
                            text: st,
                        });
                    }
                }
                current_speaker = Some(spk);
                current_segment_text.clear();
                i = end;
                continue;
            }
            if let Some((cs, end)) = parse_timestamp_tag(raw, i) {
                let absolute_cs = self.advance_rollover(cs);
                let word_str = pending_word.trim().to_string();
                let is_silence = word_str == "_";
                if !word_str.is_empty() {
                    words.push(TimedWord {
                        word: word_str,
                        end_time: absolute_cs as f32 / 100.0,
                        is_silence,
                    });
                }
                pending_word.clear();
                i = end;
                continue;
            }
            let c = raw[i..]
                .chars()
                .next()
                .expect("byte index on char boundary");
            let cl = c.len_utf8();
            clean.push(c);
            current_segment_text.push(c);
            // Build up the pending word until the next [T:N] tag
            // consumes it. We deliberately don't reset on whitespace
            // here - in timestamp mode, multiple words are separated
            // by `[T:N]` tags, not by spaces, so the tag is the
            // consume point. Trimming and silence-detection happen at
            // tag time.
            pending_word.push(c);
            i += cl;
        }
        if let Some(prev) = current_speaker {
            let st = current_segment_text.trim().to_string();
            if !st.is_empty() {
                segments.push(SpeakerSegment {
                    speaker: prev,
                    text: st,
                });
            }
        }
        (collapse_whitespace(&clean), segments, words)
    }

    /// Centiseconds-mod-1000 rollover heuristic. The model emits
    /// `N = round(t*100) mod 1000`, which wraps every 10 s. We
    /// increment the rollover counter when a value drops by more than
    /// 250 cs (2.5 s), which is far larger than any plausible
    /// inter-word gap inside a single 10 s window but well below the
    /// 10 s wrap boundary. The 250 cs threshold is heuristic; pick a
    /// smaller one if you observe missed rollovers in long mid-word
    /// silences.
    fn advance_rollover(&mut self, centiseconds_mod_1000: u32) -> u32 {
        if let Some(prev) = self.last_centiseconds {
            if centiseconds_mod_1000 + 250 < prev {
                self.rollover = self.rollover.saturating_add(1);
            }
        }
        self.last_centiseconds = Some(centiseconds_mod_1000);
        centiseconds_mod_1000 + 1000 * self.rollover
    }
}

fn parse_speaker_tag(raw: &str, start: usize) -> Option<(u32, usize)> {
    let rest = raw.get(start..)?;
    let prefix = "[Speaker ";
    if !rest.starts_with(prefix) {
        return None;
    }
    let after_prefix = &rest[prefix.len()..];
    let digits_end = after_prefix.find(|c: char| !c.is_ascii_digit())?;
    if digits_end == 0 {
        return None;
    }
    let n: u32 = after_prefix[..digits_end].parse().ok()?;
    let after_digits = &after_prefix[digits_end..];
    if !after_digits.starts_with("]:") {
        return None;
    }
    let consumed = prefix.len() + digits_end + "]:".len();
    Some((n, start + consumed))
}

fn parse_timestamp_tag(raw: &str, start: usize) -> Option<(u32, usize)> {
    let rest = raw.get(start..)?;
    let prefix = "[T:";
    if !rest.starts_with(prefix) {
        return None;
    }
    let after_prefix = &rest[prefix.len()..];
    let digits_end = after_prefix.find(|c: char| !c.is_ascii_digit())?;
    if digits_end == 0 {
        return None;
    }
    let n: u32 = after_prefix[..digits_end].parse().ok()?;
    let after_digits = &after_prefix[digits_end..];
    if !after_digits.starts_with(']') {
        return None;
    }
    let consumed = prefix.len() + digits_end + 1;
    Some((n, start + consumed))
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_speaker_tag_works() {
        assert_eq!(parse_speaker_tag("[Speaker 1]: hi", 0), Some((1, 12)));
        assert_eq!(parse_speaker_tag("[Speaker 23]: hi", 0), Some((23, 13)));
        assert_eq!(parse_speaker_tag("not a tag", 0), None);
        assert_eq!(parse_speaker_tag("[Speaker ]: bad", 0), None);
        assert_eq!(parse_speaker_tag("[Speaker 1] without colon", 0), None);
    }

    #[test]
    fn parse_timestamp_tag_works() {
        assert_eq!(parse_timestamp_tag("[T:42]", 0), Some((42, 6)));
        assert_eq!(parse_timestamp_tag("[T:999]", 0), Some((999, 7)));
        assert_eq!(parse_timestamp_tag("not a tag", 0), None);
        assert_eq!(parse_timestamp_tag("[T:]", 0), None);
    }

    #[test]
    fn rollover_advances_only_on_large_drops() {
        let mut s = TagParserState::default();
        assert_eq!(s.advance_rollover(5), 5);
        assert_eq!(s.advance_rollover(92), 92);
        // 92 -> 12: drops by 80, under the 250 threshold -> still
        // monotonic-ish, no rollover.
        assert_eq!(s.advance_rollover(12), 12);
        assert_eq!(s.rollover, 0);
        assert_eq!(s.advance_rollover(850), 850);
        // 850 -> 5 large drop -> rollover
        assert_eq!(s.advance_rollover(5), 5 + 1000);
        assert_eq!(s.rollover, 1);
    }

    #[test]
    fn parse_splits_speakers_and_strips_markers() {
        let mut s = TagParserState::default();
        let raw = "[Speaker 1]: hello world [Speaker 2]: hi there";
        let (clean, segs, words) = s.parse(raw);
        assert_eq!(clean, "hello world hi there");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker, 1);
        assert_eq!(segs[0].text, "hello world");
        assert_eq!(segs[1].speaker, 2);
        assert_eq!(segs[1].text, "hi there");
        assert!(words.is_empty());
    }

    #[test]
    fn parse_extracts_timestamps_with_rollover() {
        let mut s = TagParserState::default();
        let raw = "hello [T:42] world [T:850] _ [T:5] end [T:120]";
        let (clean, _, words) = s.parse(raw);
        assert_eq!(clean, "hello world _ end");
        assert_eq!(words.len(), 4);
        assert!((words[0].end_time - 0.42).abs() < 1e-6);
        assert!((words[1].end_time - 8.50).abs() < 1e-6);
        // 850 -> 5 large drop -> rollover; absolute = 5 + 1000 cs = 10.05 s
        assert!((words[2].end_time - 10.05).abs() < 1e-6);
        assert!(words[2].is_silence);
        // 5 -> 120 monotonic, same rollover bucket: 120 + 1000 = 1120 cs = 11.20 s
        assert!((words[3].end_time - 11.20).abs() < 1e-6);
    }

    #[test]
    fn parse_handles_mixed_speaker_and_timestamp() {
        let mut s = TagParserState::default();
        let raw = "[Speaker 1]: hello [T:42] world [T:89] [Speaker 2]: hi [T:150]";
        let (clean, segs, words) = s.parse(raw);
        assert_eq!(clean, "hello world hi");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker, 1);
        assert_eq!(segs[1].speaker, 2);
        assert_eq!(words.len(), 3);
    }
}
