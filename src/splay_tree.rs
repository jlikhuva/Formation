//! A splay tree is a binary search tree that achieves these four key properties:
//!
//! ## Properties
//! 1. The balance property which guarantees that the amortized cost of any lookup in the tree is
//!    at least (lg n)
//! 2. The entropy property which guarantees that the cost of a lookup is
//!    less than logarithmic if some elements are more likely to be queried than others.
//! 3. The Dynamic Finger Property (aka the spatial locality property)
//! 4. The working set property (aka the temporal locality property)
//!
//! A splay tree achieves all the 4 properties discussed above by moving
//! nodes around the tree after each operation. in particular, whenever some
//! element x is accessed — through any of the tree's dictionary operations,
//! we move it to the root of the tree in a process called splaying.
//! In a splay tree, any sequence of k operations takes a total
//! runtime of O(k \lg n). This means that on average, each operation
//! takes O(\lg n) — that is, the amortized runtime of each operation is logarithmic.

use std::cmp::Ordering::{Equal, Greater, Less};
/// Indicates whether a given node is a left or right child of its
/// parent
#[derive(Debug)]
pub enum ChildType {
    Left,
    Right,
}

/// The three configurations that dictate the number, order,
/// and nature of the rotations we perform during the splay operation
#[derive(Debug)]
pub enum NodeConfig {
    /// A node is in a Zig configuration if it is a
    /// child of the root.
    Zig(ChildType),

    /// A node is in a ZIgZig configuration if it is
    /// a left child of a left child or a right child of
    /// a right child
    ZigZig(ChildType),

    /// A node is in a ZigZag configuration if
    /// it is left child of a right child or
    /// a right child of a left child
    ZigZag(ChildType),
}

/// We use the new type index pattern to make our code more
/// understandable. Using indexes to simulate pointers
/// can lead to opaqueness. Using a concrete type instead of
/// raw indexes ameliorates this. The 'insane' derive ensures
/// that the new type has all the crucial properties of the
/// underlying index
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct SplayNodeIdx(usize);

impl From<usize> for SplayNodeIdx {
    /// Allows us to quickly construct tree indexes from
    /// a raw index
    /// # Example
    ///
    /// ```
    /// let idx: SplayNodeIdx = 5.into();
    /// ```
    fn from(idx: usize) -> Self {
        SplayNodeIdx(idx)
    }
}

impl<K: Ord, V> std::ops::Index<SplayNodeIdx> for Vec<SplayNode<K, V>> {
    type Output = SplayNode<K, V>;

    /// This allows us to use SplayNodeIdx directly as an index
    ///
    /// # Example
    /// ```
    /// let v: Vec<SplayNode<K, V>> = ... // snip
    /// let idx =  SplayNodeIdx(0);
    /// let first_node = v[idx]
    /// ```
    fn index(&self, index: SplayNodeIdx) -> &Self::Output {
        &self[index.0]
    }
}

impl<K: Ord, V> std::ops::IndexMut<SplayNodeIdx> for Vec<SplayNode<K, V>> {
    fn index_mut(&mut self, index: SplayNodeIdx) -> &mut Self::Output {
        &mut self[index.0]
    }
}

/// A single entry in the tree
#[derive(Debug)]
pub struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K: Ord, V> From<(K, V)> for Entry<K, V> {
    /// Allows us to quickly construct an entry from
    /// a key value tuple
    ///
    /// # Example
    ///
    /// ```
    /// let entry: Entry<&str, usize> = ("usa", 245).into();
    /// ```
    fn from(e: (K, V)) -> Self {
        Entry { key: e.0, value: e.1 }
    }
}

/// A single node in the tree. This is the main unit of
/// computation in the tree. That is, all operations
/// operate on nodes. It is parameterized by a key which
/// should be orderable and an arbitrary value
#[derive(Debug)]
struct SplayNode<K: Ord, V> {
    entry: Entry<K, V>,
    left: Option<SplayNodeIdx>,
    right: Option<SplayNodeIdx>,
    parent: Option<SplayNodeIdx>,
}

impl<K: Ord, V> SplayNode<K, V> {
    /// Create a new splay tree node with the given entry.
    ///
    /// # Example
    /// ```
    /// let entry: Entry<&str, usize> = ("usa", 245).into();
    /// let node = SplayNode::new(entry);
    /// ```
    pub fn new(entry: Entry<K, V>) -> Self {
        SplayNode {
            entry,
            left: None,
            right: None,
            parent: None,
        }
    }

    /// Retrieve the key in this node
    pub fn key(&self) -> &K {
        &self.entry.key
    }
}
/// A type alias for ergonomic reasons
type Nodes<K, V> = Vec<SplayNode<K, V>>;

/// A splay tree implemented using indexes
#[derive(Debug, Default)]
pub struct SplayTree<K: Ord, V> {
    /// A growable container of all the nodes in the tree
    elements: Option<Nodes<K, V>>,

    // The location of the root. Optional because we first create
    // an empty tree. We keep track of it because its location
    // can change as we make structural changes to the tree
    root: Option<SplayNodeIdx>,
}

/// Implementation of Read operations. These procedures
/// do not lead to structural changes in the tree
impl<K: Ord + Default, V: Default> SplayTree<K, V> {
    /// Create a new, empty red black tree
    ///
    /// # Example
    /// ```
    /// let tree: SplayTree<&str, usize> = SplayTree::new();
    /// ```
    pub fn new() -> Self {
        SplayTree::default()
    }

    /// Retrieves the Key-Value pair associated with
    /// the provided key id it exists in the tree.
    ///
    /// # Examples
    /// ```
    /// use crate::splay_tree::SplayTree;
    /// let mut tree = SplayTree::new();
    /// assert!(tree.get("cat").is_none())
    ///
    /// tree.insert(("cat", "tabby").into());
    /// assert!(tree.get("cat").is_some());
    /// assert_eq!(tree.get("cat").unwrap(), ("cat", "tabby").into());
    /// ```
    pub fn get(&mut self, k: K) -> Option<&Entry<K, V>> {
        match &mut self.elements {
            None => None,
            Some(nodes) => {
                Self::get_helper(nodes, self.root, k).and_then(move |key_index| Some(&nodes[key_index].entry))
            }
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the largest key smaller than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the smallest element in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn pred(&mut self, k: K) -> Option<&Entry<K, V>> {
        match &mut self.elements {
            None => None,
            Some(nodes) => Self::get_helper(nodes, self.root, k).and_then(move |key_idx| {
                let right = nodes[key_idx].right;
                let pred = match right {
                    None => Self::lra(nodes, Some(key_idx)),
                    Some(right_idx) => Self::max_helper(nodes, Some(right_idx)),
                };
                pred.and_then(move |pred_idx| Some(&nodes[pred_idx].entry))
            }),
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key larger than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the largest element in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn successor(&mut self, k: K) -> Option<&Entry<K, V>> {
        match &mut self.elements {
            None => None,
            Some(nodes) => Self::get_helper(nodes, self.root, k).and_then(move |key_idx| {
                let right = nodes[key_idx].right;
                let successor = match right {
                    None => Self::lla(nodes, Some(key_idx)),
                    Some(right_idx) => Self::min_helper(nodes, Some(right_idx)),
                };
                successor.and_then(move |succ_idx| Some(&nodes[succ_idx].entry))
            }),
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the largest key in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn max(&mut self) -> Option<&Entry<K, V>> {
        match &mut self.elements {
            None => None,
            Some(nodes) => Self::max_helper(nodes, self.root).and_then(move |max_idx| Some(&nodes[max_idx].entry)),
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn min(&mut self) -> Option<&Entry<K, V>> {
        match &mut self.elements {
            None => None,
            Some(nodes) => Self::min_helper(nodes, self.root).and_then(move |min_idx| Some(&nodes[min_idx].entry)),
        }
    }
}

/// Implementation of Write operations. These procedures
/// do lead to structural changes in the tree
impl<K: Ord, V> SplayTree<K, V> {
    /// Adds a new entry into the red black tree in amortized O(lg n) time.
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn insert(&mut self, e: Entry<K, V>) -> Option<&Entry<K, V>> {
        todo!()
    }

    /// Removes the entry associated with the provided key
    /// from the splay tree in amortized O(lg n) time.
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn delete(&mut self, k: K) -> Option<Entry<K, V>> {
        todo!()
    }
}

/// Implementation of internal helper functions to read operations. We call the
/// splay procedure here to make the public API as clean as possible
impl<K: Ord, V> SplayTree<K, V> {
    /// Searches for the location of the entry with the provided key starting
    /// at the specified location index. If such an entry exists, we return its index.
    /// If not, we return [`None`]. The catch is, we always splay even when the
    /// `key` is not present in the tree
    fn get_helper(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>, key: K) -> Option<SplayNodeIdx> {
        todo!()
    }

    /// Searches for the location of the entry with the largest key value
    fn max_helper(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| match nodes[cur_idx].right {
            None => start,
            Some(right_child_idx) => Self::max_helper(nodes, Some(right_child_idx)),
        })
    }

    /// Searches for the location of the entry with the smallest key value
    fn min_helper(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| match nodes[cur_idx].left {
            None => start,
            Some(left_child_idx) => Self::min_helper(nodes, Some(left_child_idx)),
        })
    }

    /// Searches for the location of the lowest ancestor node that is a right child of
    /// its parent. This is a sub-procedure used when computing the predecessor of a node
    /// `lra` stands for lowest right ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    fn lra(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, start).and_then(|child_type| match child_type {
                ChildType::Left => Self::lra(nodes, nodes[cur_idx].parent),
                ChildType::Right => nodes[cur_idx].parent,
            })
        })
    }

    /// Searches for the location of the lowest ancestor node that is a left child of
    /// its parent. This is a sub-procedure used when computing the successor of a node
    /// `lla` stands for lowest left ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    fn lla(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, start).and_then(|child_type| match child_type {
                ChildType::Right => Self::lla(nodes, nodes[cur_idx].parent),
                ChildType::Left => nodes[cur_idx].parent,
            })
        })
    }

    /// Is this node a left child or right child of its parent
    fn child_type(nodes: &Nodes<K, V>, cur: Option<SplayNodeIdx>) -> Option<ChildType> {
        cur.and_then(|cur_idx| {
            let parent = nodes[cur_idx].parent;
            parent.and_then(|parent_idx| match nodes[parent_idx].left.cmp(&cur) {
                Equal => Some(ChildType::Left),
                _ => Some(ChildType::Right),
            })
        })
    }
}

/// Implementation of internal helper functions to write operations. We call
/// the splay procedure here to make the public API as clean as possible
impl<K: Ord, V> SplayTree<K, V> {
    fn insert_helper(nodes: &mut Nodes<K, V>, e: Entry<K, V>, cur: SplayNodeIdx) -> SplayNodeIdx {
        todo!()
    }
    fn delete_helper(nodes: &mut Nodes<K, V>, k: K) -> SplayNodeIdx {
        todo!()
    }

    fn transplant(nodes: &mut Nodes<K, V>, left: SplayNodeIdx) {
        todo!()
    }

    fn swap_remove(nodes: &mut Nodes<K, V>, delete_idx: SplayNodeIdx) -> Option<V> {
        todo!()
    }
}

/// Implementation of the bottom up splay operation
impl<K: Ord, V> SplayTree<K, V> {
    /// Moves the target node to the root of the tree using a series of rotations.
    fn splay(&mut self, target: SplayNodeIdx) {
        match &mut self.elements {
            None => (),
            Some(nodes) => {
                while nodes[target].parent.is_some() {
                    match Self::get_cur_config(nodes, target) {
                        NodeConfig::Zig(typ) => Self::zig(nodes, target, typ),
                        NodeConfig::ZigZig(typ) => Self::zig_zig(nodes, target, typ),
                        NodeConfig::ZigZag(typ) => Self::zig_zag(nodes, target, typ),
                    }
                }
                self.root = Some(target);
            }
        }
    }

    /// The final splay operation to move the target node to the root of the tree
    fn zig(nodes: &mut Nodes<K, V>, target: SplayNodeIdx, child_type: ChildType) {
        match child_type {
            ChildType::Left => Self::rotate_right(nodes, target),
            ChildType::Right => Self::rotate_left(nodes, target),
        }
    }

    /// Applied when the target node is in a cis-configuration with its parent
    fn zig_zig(nodes: &mut Nodes<K, V>, target: SplayNodeIdx, child_type: ChildType) {
        let grand_parent = Self::get_grand_parent(nodes, target);
        match child_type {
            ChildType::Left => {
                Self::rotate_right(nodes, grand_parent);
                Self::rotate_right(nodes, grand_parent);
            }
            ChildType::Right => {
                Self::rotate_left(nodes, grand_parent);
                Self::rotate_left(nodes, grand_parent);
            }
        }
    }

    /// Applied when the target node is in a trans-configuration with  its parent
    fn zig_zag(nodes: &mut Nodes<K, V>, target: SplayNodeIdx, child_type: ChildType) {
        let parent = nodes[target].parent.unwrap();
        let grand_parent = Self::get_grand_parent(nodes, target);
        match child_type {
            ChildType::Right => {
                Self::rotate_left(nodes, parent);
                Self::rotate_right(nodes, grand_parent);
            }
            ChildType::Left => {
                Self::rotate_right(nodes, parent);
                Self::rotate_left(nodes, grand_parent);
            }
        }
    }

    /// Swaps the entries of the nodes stored at the given indexes.
    /// Again, this does not swap the nodes. It swaps the payloads
    /// while leaving the child and parent pointers in place.
    fn swap_entries(nodes: &mut Nodes<K, V>, a: SplayNodeIdx, b: SplayNodeIdx) {
        unsafe {
            // We cannot take two mutable loans from a vector.
            // Furthermore, we cannot use nodes.swap(..) because that
            // swaps the actual nodes. std::mem::swap(..) does not
            // work either because it would require us to
            // take two mutable refs into nodes. Therefore,
            // we implement a swapping mechanism that mirrors
            // how the vector's swap(..) method is implemented.
            let entry_a: *mut Entry<K, V> = &mut nodes[a].entry;
            let entry_b: *mut Entry<K, V> = &mut nodes[b].entry;
            std::ptr::swap(entry_a, entry_b);
        }
    }

    /// We are give  a node `x` which is the root of some sub-tree and we would
    /// like to exchange it with its left child `y` which must exist. `x` can be
    /// the root of the tree in which case it has no parent.
    fn rotate_right(nodes: &mut Nodes<K, V>, x_idx: SplayNodeIdx) {
        // Step 1: Swap the entry at `x_idx`, not the node, the entry, with the entry
        //         of the left child. This operation does not modify any of the child
        //         or parent pointers.
        let x_left_idx = nodes[x_idx].left.unwrap();
        Self::swap_entries(nodes, x_idx, x_left_idx);

        // Keep in mind that swapping nodes takes the subtrees with it
        // Whereas swapping entries does not modify the tree structure.

        // Step 2: We now operate on the children of `x_left_idx`
        //         Either one, or both, of these kids could be
        //         `None`. We want to swap the nodes, not the entries,
        //         the nodes themselves.
        // TODO: x_left.swap(right_child, left_child)

        // Step 3: We now want to move the right child of x_left
        //         (which was previously the left child of x_left)
        //         to become the right child of x_idx. Since these
        //         two nodes do not share a parent, we have to also
        //         update their parent pointers. Note that we do not
        //         need to tell their parents anything
        // TODO: x_right.swap_with(x_left.right)

        // Step 4: This is the final step. We want to switch the left child
        // of x with the right child of x. To do so, we simply swap nodes
    }

    /// We are give  a node `y` which is the root of some sub-tree and we would
    /// like to exchange it with its right child `x` which must exist. `y` can be
    /// the root of the tree in which case it has no parent.
    fn rotate_left(nodes: &mut Nodes<K, V>, y_idx: SplayNodeIdx) {
        todo!()
    }

    fn get_grand_parent(nodes: &Nodes<K, V>, target: SplayNodeIdx) -> SplayNodeIdx {
        let parent = &nodes[target];
        debug_assert!(parent.parent.is_some());
        parent.parent.unwrap()
    }

    fn get_cur_config(nodes: &Nodes<K, V>, target: SplayNodeIdx) -> NodeConfig {
        match Self::child_type(nodes, Some(target)) {
            Some(ChildType::Left) => match Self::child_type(nodes, nodes[target].parent) {
                None => NodeConfig::Zig(ChildType::Left),
                Some(ChildType::Left) => NodeConfig::ZigZig(ChildType::Left),
                Some(ChildType::Right) => NodeConfig::ZigZag(ChildType::Left),
            },
            Some(ChildType::Right) => match Self::child_type(nodes, nodes[target].parent) {
                None => NodeConfig::Zig(ChildType::Right),
                Some(ChildType::Left) => NodeConfig::ZigZag(ChildType::Right),
                Some(ChildType::Right) => NodeConfig::ZigZig(ChildType::Right),
            },
            None => panic!("This procedure should not be called on the root node"),
        }
    }
}

#[cfg(test)]
mod test_splay {}

#[cfg(test)]
mod test_read {}

#[cfg(test)]
mod test_write {}
