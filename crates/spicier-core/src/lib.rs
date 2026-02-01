//! Core circuit representation and MNA matrix structures for Spicier.
//!
//! This crate provides the fundamental data structures for representing
//! circuits, including nodes, branches, and the Modified Nodal Analysis (MNA)
//! matrix system.

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
pub use netlist::{AcDeviceInfo, Netlist, Stamper};
pub use node::{Node, NodeId};
