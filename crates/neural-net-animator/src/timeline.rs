//! Timeline management for animation playback

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackState {
    Playing,
    Paused,
    Stopped,
}

/// Playback speed multiplier
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PlaybackSpeed {
    /// 0.25x speed
    SlowMotion,
    /// 1.0x speed (normal)
    Normal,
    /// 2.0x speed
    Fast,
    /// 4.0x speed
    VeryFast,
}

impl PlaybackSpeed {
    pub fn multiplier(self) -> f64 {
        match self {
            Self::SlowMotion => 0.25,
            Self::Normal => 1.0,
            Self::Fast => 2.0,
            Self::VeryFast => 4.0,
        }
    }

    pub fn from_multiplier(mult: f64) -> Self {
        if (mult - 0.25).abs() < 0.01 {
            Self::SlowMotion
        } else if (mult - 1.0).abs() < 0.01 {
            Self::Normal
        } else if (mult - 2.0).abs() < 0.01 {
            Self::Fast
        } else {
            Self::VeryFast
        }
    }

    pub fn cycle_forward(self) -> Self {
        match self {
            Self::SlowMotion => Self::Normal,
            Self::Normal => Self::Fast,
            Self::Fast => Self::VeryFast,
            Self::VeryFast => Self::VeryFast,
        }
    }

    pub fn cycle_backward(self) -> Self {
        match self {
            Self::VeryFast => Self::Fast,
            Self::Fast => Self::Normal,
            Self::Normal => Self::SlowMotion,
            Self::SlowMotion => Self::SlowMotion,
        }
    }
}

/// Timeline controller
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Current playback state
    state: PlaybackState,

    /// Playback speed
    speed: PlaybackSpeed,

    /// Current time in seconds
    current_time: f64,

    /// Total duration in seconds
    total_duration: f64,

    /// Last update timestamp
    last_update: Option<Instant>,

    /// Whether to loop
    looping: bool,
}

impl PartialEq for Timeline {
    fn eq(&self, other: &Self) -> bool {
        // Compare all fields except last_update (Instant doesn't implement PartialEq)
        self.state == other.state
            && self.speed == other.speed
            && (self.current_time - other.current_time).abs() < 0.001
            && (self.total_duration - other.total_duration).abs() < 0.001
            && self.looping == other.looping
    }
}

impl Timeline {
    /// Create a new timeline with given duration
    pub fn new(total_duration: f64) -> Self {
        Self {
            state: PlaybackState::Stopped,
            speed: PlaybackSpeed::Normal,
            current_time: 0.0,
            total_duration,
            last_update: None,
            looping: false,
        }
    }

    /// Get current time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get total duration
    pub fn total_duration(&self) -> f64 {
        self.total_duration
    }

    /// Get playback state
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Get playback speed
    pub fn speed(&self) -> PlaybackSpeed {
        self.speed
    }

    /// Set playback speed
    pub fn set_speed(&mut self, speed: PlaybackSpeed) {
        self.speed = speed;
    }

    /// Get progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_duration > 0.0 {
            (self.current_time / self.total_duration).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Play the animation
    pub fn play(&mut self) {
        if self.state != PlaybackState::Playing {
            self.state = PlaybackState::Playing;
            self.last_update = Some(Instant::now());
        }
    }

    /// Pause the animation
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
        self.last_update = None;
    }

    /// Stop the animation (reset to beginning)
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time = 0.0;
        self.last_update = None;
    }

    /// Toggle play/pause
    pub fn toggle_play_pause(&mut self) {
        match self.state {
            PlaybackState::Playing => self.pause(),
            PlaybackState::Paused | PlaybackState::Stopped => self.play(),
        }
    }

    /// Seek to specific time
    pub fn seek(&mut self, time: f64) {
        self.current_time = time.clamp(0.0, self.total_duration);
        if self.state == PlaybackState::Playing {
            self.last_update = Some(Instant::now());
        }
    }

    /// Seek to specific progress (0.0 to 1.0)
    pub fn seek_to_progress(&mut self, progress: f64) {
        let time = progress.clamp(0.0, 1.0) * self.total_duration;
        self.seek(time);
    }

    /// Step forward by duration
    pub fn step_forward(&mut self, duration: f64) {
        self.seek(self.current_time + duration);
    }

    /// Step backward by duration
    pub fn step_backward(&mut self, duration: f64) {
        self.seek(self.current_time - duration);
    }

    /// Skip to beginning
    pub fn skip_to_start(&mut self) {
        self.seek(0.0);
    }

    /// Skip to end
    pub fn skip_to_end(&mut self) {
        self.seek(self.total_duration);
    }

    /// Update timeline (call on animation frame)
    /// Returns true if time changed
    pub fn update(&mut self) -> bool {
        if self.state != PlaybackState::Playing {
            return false;
        }

        let now = Instant::now();
        if let Some(last) = self.last_update {
            let delta = now.duration_since(last);
            let delta_secs = delta.as_secs_f64() * self.speed.multiplier();

            self.current_time += delta_secs;

            // Handle end of animation
            if self.current_time >= self.total_duration {
                if self.looping {
                    self.current_time %= self.total_duration;
                } else {
                    self.current_time = self.total_duration;
                    self.pause();
                }
            }
        }

        self.last_update = Some(now);
        true
    }

    /// Set looping
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Get looping state
    pub fn is_looping(&self) -> bool {
        self.looping
    }

    /// Format current time as MM:SS
    pub fn format_time(&self) -> String {
        let total_secs = self.current_time as u64;
        let minutes = total_secs / 60;
        let seconds = total_secs % 60;
        format!("{:02}:{:02}", minutes, seconds)
    }

    /// Format total duration as MM:SS
    pub fn format_duration(&self) -> String {
        let total_secs = self.total_duration as u64;
        let minutes = total_secs / 60;
        let seconds = total_secs % 60;
        format!("{:02}:{:02}", minutes, seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};

    #[test]
    fn test_timeline_creation() {
        let timeline = Timeline::new(10.0);
        assert_eq!(timeline.current_time(), 0.0);
        assert_eq!(timeline.total_duration(), 10.0);
        assert_eq!(timeline.state(), PlaybackState::Stopped);
        assert_eq!(timeline.progress(), 0.0);
    }

    #[test]
    fn test_playback_speed() {
        assert_eq!(PlaybackSpeed::SlowMotion.multiplier(), 0.25);
        assert_eq!(PlaybackSpeed::Normal.multiplier(), 1.0);
        assert_eq!(PlaybackSpeed::Fast.multiplier(), 2.0);
        assert_eq!(PlaybackSpeed::VeryFast.multiplier(), 4.0);

        let speed = PlaybackSpeed::Normal;
        let speed = speed.cycle_forward();
        assert_eq!(speed, PlaybackSpeed::Fast);
    }

    #[test]
    fn test_seek() {
        let mut timeline = Timeline::new(10.0);

        timeline.seek(5.0);
        assert_eq!(timeline.current_time(), 5.0);
        assert_eq!(timeline.progress(), 0.5);

        timeline.seek_to_progress(0.75);
        assert_eq!(timeline.current_time(), 7.5);

        timeline.seek(20.0); // Beyond duration
        assert_eq!(timeline.current_time(), 10.0); // Clamped
    }

    #[test]
    fn test_step() {
        let mut timeline = Timeline::new(10.0);

        timeline.step_forward(2.0);
        assert_eq!(timeline.current_time(), 2.0);

        timeline.step_forward(3.0);
        assert_eq!(timeline.current_time(), 5.0);

        timeline.step_backward(1.0);
        assert_eq!(timeline.current_time(), 4.0);

        timeline.step_backward(10.0); // Beyond start
        assert_eq!(timeline.current_time(), 0.0); // Clamped
    }

    #[test]
    fn test_play_pause() {
        let mut timeline = Timeline::new(10.0);

        assert_eq!(timeline.state(), PlaybackState::Stopped);

        timeline.play();
        assert_eq!(timeline.state(), PlaybackState::Playing);

        timeline.pause();
        assert_eq!(timeline.state(), PlaybackState::Paused);

        timeline.toggle_play_pause();
        assert_eq!(timeline.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_update() {
        let mut timeline = Timeline::new(1.0);

        // Not playing, update should return false
        assert!(!timeline.update());

        timeline.play();

        // Small delay
        thread::sleep(Duration::from_millis(50));

        // Update should advance time
        let updated = timeline.update();
        assert!(updated);
        assert!(timeline.current_time() > 0.0);
    }

    #[test]
    fn test_format_time() {
        let mut timeline = Timeline::new(125.0);

        timeline.seek(0.0);
        assert_eq!(timeline.format_time(), "00:00");

        timeline.seek(65.0);
        assert_eq!(timeline.format_time(), "01:05");

        assert_eq!(timeline.format_duration(), "02:05");
    }
}
