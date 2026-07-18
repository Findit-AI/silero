pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
mod graph_optimization_level {
  use super::GraphOptimizationLevel;
  use serde::*;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    Disable,
    Level1,
    Level2,
    #[default]
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Disable
  }
}

/// Options for constructing an ONNX session.
///
/// This type intentionally stays small. Deployment-specific runtime
/// policy such as `intra_threads` / `inter_threads` should normally be
/// configured one layer up, then passed down via
/// [`crate::Session::from_ort_session`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "graph_optimization_level::default",
      with = "graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
}

impl Default for SessionOptions {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl SessionOptions {
  /// Create a new `SessionOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      optimization_level: GraphOptimizationLevel::Level3,
    }
  }

  /// Returns the graph optimization level to use when constructing the ONNX session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// Set the graph optimization level to use when constructing the ONNX session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }
}

#[cfg(test)]
mod tests {
  use super::{GraphOptimizationLevel, SessionOptions};

  #[test]
  fn session_options_default_to_unopinionated_core_settings() {
    let options = SessionOptions::default();
    assert_eq!(options.optimization_level(), GraphOptimizationLevel::Level3,);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn test_serde() {
    let opts = SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level2);
    let serialized = serde_json::to_string(&opts).expect("serialize options");
    let deserialized: SessionOptions =
      serde_json::from_str(&serialized).expect("deserialize options");
    assert_eq!(opts.optimization_level, deserialized.optimization_level);

    let default_deserialized: SessionOptions =
      serde_json::from_str("{}").expect("deserialize default options");
    assert!(matches!(
      default_deserialized.optimization_level,
      GraphOptimizationLevel::Disable
    ));

    // level1
    let level1_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level1);
    let level1_serialized = serde_json::to_string(&level1_opts).expect("serialize level1 options");
    let level1_deserialized: SessionOptions =
      serde_json::from_str(&level1_serialized).expect("deserialize level1 options");
    assert!(matches!(
      level1_deserialized.optimization_level,
      GraphOptimizationLevel::Level1
    ));

    // level2
    let level2_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level2);
    let level2_serialized = serde_json::to_string(&level2_opts).expect("serialize level2 options");
    let level2_deserialized: SessionOptions =
      serde_json::from_str(&level2_serialized).expect("deserialize level2 options");
    assert!(matches!(
      level2_deserialized.optimization_level,
      GraphOptimizationLevel::Level2
    ));

    // level3
    let level3_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level3);
    let level3_serialized = serde_json::to_string(&level3_opts).expect("serialize level3 options");
    let level3_deserialized: SessionOptions =
      serde_json::from_str(&level3_serialized).expect("deserialize level3 options");
    assert!(matches!(
      level3_deserialized.optimization_level,
      GraphOptimizationLevel::Level3
    ));

    // all
    let all_opts = SessionOptions::default().with_optimization_level(GraphOptimizationLevel::All);
    let all_serialized = serde_json::to_string(&all_opts).expect("serialize all options");
    let all_deserialized: SessionOptions =
      serde_json::from_str(&all_serialized).expect("deserialize all options");
    assert!(matches!(
      all_deserialized.optimization_level,
      GraphOptimizationLevel::All
    ));

    // disable
    let disable_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Disable);
    let disable_serialized =
      serde_json::to_string(&disable_opts).expect("serialize disable options");
    let disable_deserialized: SessionOptions =
      serde_json::from_str(&disable_serialized).expect("deserialize disable options");
    assert!(matches!(
      disable_deserialized.optimization_level,
      GraphOptimizationLevel::Disable
    ));
  }
}
