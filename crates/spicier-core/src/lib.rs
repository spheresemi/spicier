//! Core circuit representation and MNA matrix structures for Spicier.
//!
//! This crate provides the fundamental data structures for representing
//! circuits, including nodes, branches, and the Modified Nodal Analysis (MNA)
//! matrix system.
//!
//! # Overview
//!
//! The core abstractions are:
//!
//! - [`NodeId`] - Identifies a node in the circuit (ground is node 0)
//! - [`MnaSystem`](mna::MnaSystem) - The MNA matrix equation Ax = b
//! - [`Netlist`] - A complete circuit with devices and analysis commands
//! - [`Stamper`] - Trait for devices that can stamp into the MNA matrix
//!
//! # Modified Nodal Analysis (MNA)
//!
//! MNA is a systematic method for formulating circuit equations. The system
//! `Ax = b` contains:
//!
//! - Node voltages (V₁, V₂, ..., Vₙ)
//! - Branch currents through voltage sources and inductors
//!
//! # Example: Building an MNA System
//!
//! ```rust
//! use spicier_core::mna::MnaSystem;
//!
//! // Create a simple voltage divider: V1=10V, R1=1k, R2=1k
//! // Circuit: V1 -- R1 -- node1 -- R2 -- GND
//! let mut mna = MnaSystem::new(1, 1); // 1 node, 1 voltage source
//!
//! // Stamp R1 (1kΩ) between voltage source node (via branch current) and node 0
//! // For simplicity, we'll model this as: V1 at node 0, R1 from 0 to 1, R2 from 1 to GND
//! // But node 0 is connected to V1, so we need 2 nodes total
//! let mut mna = MnaSystem::new(2, 1); // 2 nodes, 1 voltage source
//!
//! // Stamp voltage source V1=10V at node 0 (positive) to ground (negative)
//! mna.stamp_voltage_source(Some(0), None, 0, 10.0);
//!
//! // Stamp R1=1kΩ between node 0 and node 1
//! mna.stamp_conductance(Some(0), Some(1), 1.0 / 1000.0);
//!
//! // Stamp R2=1kΩ between node 1 and ground
//! mna.stamp_conductance(Some(1), None, 1.0 / 1000.0);
//!
//! // The system is now ready to solve
//! let matrix = mna.to_dense_matrix();
//! assert_eq!(matrix.nrows(), 3); // 2 nodes + 1 current variable
//! ```
//!
//! # Example: Using NodeId
//!
//! ```rust
//! use spicier_core::NodeId;
//!
//! // Ground is always node 0
//! let ground = NodeId::GROUND;
//! assert!(ground.is_ground());
//! assert_eq!(ground.as_u32(), 0);
//!
//! // Other nodes are 1-indexed in SPICE
//! let node1 = NodeId::new(1);
//! assert!(!node1.is_ground());
//! assert_eq!(node1.as_u32(), 1);
//!
//! let node2 = NodeId::new(2);
//! assert_eq!(node2.as_u32(), 2);
//!
//! // For matrix indexing, subtract 1 from non-ground nodes
//! // (ground is not in the matrix)
//! let matrix_idx = node1.as_u32() as usize - 1; // = 0
//! ```

pub mod circuit;
pub mod element;
pub mod error;
pub mod mna;
pub mod netlist;
pub mod node;
pub mod units;

pub use circuit::Circuit;
pub use element::Element;
pub use error::{Error, Result};
pub use netlist::{AcDeviceInfo, Netlist, Stamper, TransientDeviceInfo};
pub use node::{Node, NodeId};
