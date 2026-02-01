//! Circuit graph representation.

use indexmap::IndexMap;

use crate::node::{Node, NodeId};

/// A circuit containing nodes and elements.
#[derive(Debug, Default)]
pub struct Circuit {
    /// Map from node ID to node data.
    nodes: IndexMap<NodeId, Node>,
    /// Next available node ID.
    next_node_id: u32,
    /// Circuit title/name.
    title: Option<String>,
}

impl Circuit {
    /// Create a new empty circuit.
    pub fn new() -> Self {
        let mut circuit = Self {
            nodes: IndexMap::new(),
            next_node_id: 1, // 0 is reserved for ground
            title: None,
        };
        // Always include ground node
        circuit
            .nodes
            .insert(NodeId::GROUND, Node::new(NodeId::GROUND));
        circuit
    }

    /// Create a new circuit with a title.
    pub fn with_title(title: impl Into<String>) -> Self {
        let mut circuit = Self::new();
        circuit.title = Some(title.into());
        circuit
    }

    /// Get the circuit title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Set the circuit title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
    }

    /// Add a node to the circuit, returning its ID.
    pub fn add_node(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.insert(id, Node::new(id));
        id
    }

    /// Add a named node to the circuit, returning its ID.
    pub fn add_named_node(&mut self, name: impl Into<String>) -> NodeId {
        let id = NodeId::new(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.insert(id, Node::with_name(id, name));
        id
    }

    /// Get or create a node with the given ID.
    /// Used when parsing netlists where node numbers are specified.
    pub fn get_or_create_node(&mut self, id: u32) -> NodeId {
        let node_id = NodeId::new(id);
        if !self.nodes.contains_key(&node_id) {
            self.nodes.insert(node_id, Node::new(node_id));
            if id >= self.next_node_id {
                self.next_node_id = id + 1;
            }
        }
        node_id
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get the ground node.
    pub fn ground(&self) -> &Node {
        self.nodes.get(&NodeId::GROUND).expect("ground node exists")
    }

    /// Iterate over all nodes (excluding ground).
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values().filter(|n| !n.is_ground())
    }

    /// Iterate over all nodes including ground.
    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Get the number of nodes (excluding ground).
    pub fn node_count(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Check if a node exists.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_circuit_has_ground() {
        let circuit = Circuit::new();
        assert!(circuit.has_node(NodeId::GROUND));
        assert_eq!(circuit.node_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut circuit = Circuit::new();
        let n1 = circuit.add_node();
        let n2 = circuit.add_node();

        assert_eq!(n1.as_u32(), 1);
        assert_eq!(n2.as_u32(), 2);
        assert_eq!(circuit.node_count(), 2);
    }

    #[test]
    fn test_named_node() {
        let mut circuit = Circuit::new();
        let vdd = circuit.add_named_node("vdd");

        assert_eq!(circuit.node(vdd).unwrap().name(), Some("vdd"));
    }

    #[test]
    fn test_get_or_create_node() {
        let mut circuit = Circuit::new();

        // Create node 5 directly
        let n5 = circuit.get_or_create_node(5);
        assert_eq!(n5.as_u32(), 5);
        assert!(circuit.has_node(n5));

        // Getting same node again returns same ID
        let n5_again = circuit.get_or_create_node(5);
        assert_eq!(n5, n5_again);
    }

    #[test]
    fn test_circuit_title() {
        let circuit = Circuit::with_title("Test Circuit");
        assert_eq!(circuit.title(), Some("Test Circuit"));
    }
}
