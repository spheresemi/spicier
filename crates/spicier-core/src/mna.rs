//! Modified Nodal Analysis (MNA) matrix structures.

use nalgebra::{DMatrix, DVector};

/// MNA system: Ax = b
/// Where A is the conductance/coefficient matrix,
/// x is the solution vector (node voltages + branch currents),
/// and b is the RHS vector (current sources + voltage sources).
///
/// Uses sparse triplet storage - duplicate entries are summed when constructing
/// the sparse matrix. Dense matrix can be built on demand for small circuits.
#[derive(Debug, Clone)]
pub struct MnaSystem {
    /// The right-hand side vector.
    pub rhs: DVector<f64>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
    /// Number of voltage sources (and other current variables).
    pub num_vsources: usize,
    /// Triplet accumulator for sparse matrix construction: (row, col, value).
    /// Duplicates are allowed and will be summed during matrix construction.
    pub triplets: Vec<(usize, usize, f64)>,
}

impl MnaSystem {
    /// Create a new MNA system with the given dimensions.
    ///
    /// # Arguments
    /// * `num_nodes` - Number of nodes excluding ground
    /// * `num_vsources` - Number of voltage sources (adds current variables)
    pub fn new(num_nodes: usize, num_vsources: usize) -> Self {
        let size = num_nodes + num_vsources;
        Self {
            rhs: DVector::zeros(size),
            num_nodes,
            num_vsources,
            triplets: Vec::new(),
        }
    }

    /// Get the total size of the system (nodes + current variables).
    pub fn size(&self) -> usize {
        self.num_nodes + self.num_vsources
    }

    /// Clear the triplets and RHS to prepare for re-stamping.
    pub fn clear(&mut self) {
        self.rhs.fill(0.0);
        self.triplets.clear();
    }

    /// Add a value to the coefficient matrix at (row, col).
    ///
    /// Values are stored as triplets. Duplicate entries at the same position
    /// are summed when the matrix is constructed.
    pub fn add_element(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push((row, col, value));
    }

    /// Build a dense matrix from the triplets.
    ///
    /// Use this for small circuits or testing. For large circuits, use the
    /// triplets directly with a sparse solver.
    pub fn to_dense_matrix(&self) -> DMatrix<f64> {
        let size = self.size();
        let mut matrix = DMatrix::zeros(size, size);
        for &(row, col, value) in &self.triplets {
            matrix[(row, col)] += value;
        }
        matrix
    }

    /// Add a value to the RHS vector at the given row.
    pub fn add_rhs(&mut self, row: usize, value: f64) {
        self.rhs[row] += value;
    }

    /// Stamp a conductance between two nodes.
    ///
    /// For a conductance G between nodes i and j, the matrix stamps are:
    /// - A\[i,i\] += G
    /// - A\[j,j\] += G
    /// - A\[i,j\] -= G
    /// - A\[j,i\] -= G
    ///
    /// Node indices are 0-based (ground is not included in matrix).
    pub fn stamp_conductance(&mut self, node_i: Option<usize>, node_j: Option<usize>, g: f64) {
        if let Some(i) = node_i {
            self.add_element(i, i, g);
        }
        if let Some(j) = node_j {
            self.add_element(j, j, g);
        }
        if let (Some(i), Some(j)) = (node_i, node_j) {
            self.add_element(i, j, -g);
            self.add_element(j, i, -g);
        }
    }

    /// Stamp a current source from node i to node j.
    ///
    /// Current flows from node i to node j (positive current enters node j).
    pub fn stamp_current_source(
        &mut self,
        node_i: Option<usize>,
        node_j: Option<usize>,
        current: f64,
    ) {
        if let Some(i) = node_i {
            self.add_rhs(i, -current);
        }
        if let Some(j) = node_j {
            self.add_rhs(j, current);
        }
    }

    /// Stamp a voltage source between two nodes.
    ///
    /// # Arguments
    /// * `node_pos` - Positive node (None for ground)
    /// * `node_neg` - Negative node (None for ground)
    /// * `vsource_idx` - Index of this voltage source (0-based)
    /// * `voltage` - Voltage value
    pub fn stamp_voltage_source(
        &mut self,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        vsource_idx: usize,
        voltage: f64,
    ) {
        let row = self.num_nodes + vsource_idx;

        // B and C matrices (coupling between nodes and voltage source current)
        if let Some(i) = node_pos {
            self.add_element(i, row, 1.0);
            self.add_element(row, i, 1.0);
        }
        if let Some(j) = node_neg {
            self.add_element(j, row, -1.0);
            self.add_element(row, j, -1.0);
        }

        // RHS for voltage source
        self.rhs[row] = voltage;
    }

    /// Get a reference to the RHS vector.
    pub fn rhs(&self) -> &DVector<f64> {
        &self.rhs
    }

    /// Get a mutable reference to the RHS vector.
    pub fn rhs_mut(&mut self) -> &mut DVector<f64> {
        &mut self.rhs
    }

    /// Stamp Gmin conductance from each node to ground.
    ///
    /// This adds a small conductance from every node to ground, which helps
    /// with convergence by preventing floating nodes and numerical instability.
    /// Used by the Gmin stepping convergence aid.
    ///
    /// # Arguments
    /// * `gmin` - Small conductance value (typically 1e-12 to 1e-9)
    pub fn stamp_gmin(&mut self, gmin: f64) {
        for i in 0..self.num_nodes {
            self.add_element(i, i, gmin);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_system() {
        let sys = MnaSystem::new(3, 1);
        assert_eq!(sys.size(), 4);
        assert_eq!(sys.num_nodes, 3);
        assert_eq!(sys.num_vsources, 1);
    }

    #[test]
    fn test_stamp_conductance() {
        let mut sys = MnaSystem::new(2, 0);

        // 1 ohm resistor between nodes 0 and 1
        sys.stamp_conductance(Some(0), Some(1), 1.0);
        let matrix = sys.to_dense_matrix();

        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 1)], 1.0);
        assert_eq!(matrix[(0, 1)], -1.0);
        assert_eq!(matrix[(1, 0)], -1.0);
    }

    #[test]
    fn test_stamp_conductance_to_ground() {
        let mut sys = MnaSystem::new(2, 0);

        // 1 ohm resistor between node 0 and ground
        sys.stamp_conductance(Some(0), None, 1.0);
        let matrix = sys.to_dense_matrix();

        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 1)], 0.0);
    }

    #[test]
    fn test_stamp_current_source() {
        let mut sys = MnaSystem::new(2, 0);

        // 1A current source from ground to node 0
        sys.stamp_current_source(None, Some(0), 1.0);

        assert_eq!(sys.rhs[0], 1.0);
        assert_eq!(sys.rhs[1], 0.0);
    }

    #[test]
    fn test_stamp_voltage_source() {
        let mut sys = MnaSystem::new(2, 1);

        // 5V source between node 0 (+) and ground (-)
        sys.stamp_voltage_source(Some(0), None, 0, 5.0);
        let matrix = sys.to_dense_matrix();

        // Check B matrix (node 0 row, vsource column)
        assert_eq!(matrix[(0, 2)], 1.0);
        // Check C matrix (vsource row, node 0 column)
        assert_eq!(matrix[(2, 0)], 1.0);
        // Check RHS
        assert_eq!(sys.rhs[2], 5.0);
    }
}
