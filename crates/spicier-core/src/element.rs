//! Element trait and element storage for circuits.

use crate::NodeId;

/// A circuit element that connects nodes.
pub trait Element: std::fmt::Debug {
    /// Get the element's name.
    fn name(&self) -> &str;

    /// Get the nodes this element connects to.
    fn nodes(&self) -> Vec<NodeId>;

    /// Get the number of additional current variables this element requires.
    /// Voltage sources and inductors need one current variable each.
    fn num_current_vars(&self) -> usize {
        0
    }
}
