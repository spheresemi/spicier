//! Node representation for circuit graphs.

use std::fmt;

/// Unique identifier for a node in the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// The ground node (node 0).
    pub const GROUND: NodeId = NodeId(0);

    /// Create a new NodeId from a raw value.
    pub fn new(id: u32) -> Self {
        NodeId(id)
    }

    /// Get the raw node ID value.
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Check if this is the ground node.
    pub fn is_ground(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ground() {
            write!(f, "GND")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

/// A node in the circuit graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier for this node.
    id: NodeId,
    /// Optional name for the node (from netlist).
    name: Option<String>,
}

impl Node {
    /// Create a new node with the given ID.
    pub fn new(id: NodeId) -> Self {
        Self { id, name: None }
    }

    /// Create a new node with the given ID and name.
    pub fn with_name(id: NodeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: Some(name.into()),
        }
    }

    /// Get the node's ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get the node's name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Check if this is the ground node.
    pub fn is_ground(&self) -> bool {
        self.id.is_ground()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_node() {
        assert!(NodeId::GROUND.is_ground());
        assert_eq!(NodeId::GROUND.as_u32(), 0);
        assert_eq!(NodeId::GROUND.to_string(), "GND");
    }

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert!(!id.is_ground());
        assert_eq!(id.as_u32(), 42);
        assert_eq!(id.to_string(), "42");
    }

    #[test]
    fn test_node_with_name() {
        let node = Node::with_name(NodeId::new(1), "vdd");
        assert_eq!(node.id().as_u32(), 1);
        assert_eq!(node.name(), Some("vdd"));
    }
}
