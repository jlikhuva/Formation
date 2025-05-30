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
    /// let v: Vec<SplayNode<i32, String>> = Vec::new(); // Example instantiation
    /// // let idx =  SplayNodeIdx(0);
    /// // let first_node = v[idx]; // This would panic if v is empty
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
#[derive(Debug, Clone)]
pub struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K: Ord, V> Entry<K, V> {
    /// Retrieve the key in this entry
    pub fn key(&self) -> &K {
        &self.key
    }
    /// Retrieve the value in this entry
    pub fn value(&self) -> &V {
        &self.value
    }
}


impl<K: Ord, V> From<(K, V)> for Entry<K, V> {
    /// Allows us to quickly construct an entry from
    /// a key value tuple
    ///
    /// # Example
    ///
    /// ```
    /// // Use concrete types for example to be runnable
    /// // use crate::splay_tree::Entry; // If used outside the module
    /// // let entry: Entry<&str, usize> = ("usa", 245).into();
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
    /// // use crate::splay_tree::{Entry, SplayNode}; // If used outside the module
    /// // let entry: Entry<i32, String> = (245, "usa".to_string()).into();
    /// // let node = SplayNode::new(entry);
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
impl<K: Ord + Default + Clone + std::fmt::Debug, V: Default> SplayTree<K, V> { // Added K: std::fmt::Debug
    /// Create a new, empty SplayTree tree
    ///
    /// # Example
    /// ```
    /// // use crate::splay_tree::SplayTree; // If used outside the module
    /// // let tree: SplayTree<i32, &str> = SplayTree::new();
    /// ```
    pub fn new() -> Self {
        SplayTree::default()
    }

    /// Retrieves the Key-Value pair associated with
    /// the provided key id it exists in the tree.
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, &str> = SplayTree::new();
    /// // assert!(tree.get(0).is_none());
    /// // tree.insert(Entry::from((10, "ten")));
    /// // assert!(tree.get(10).is_some());
    /// // assert_eq!(tree.get(10).unwrap().key(), &10);
    /// ```
    pub fn get(&mut self, k: K) -> Option<&Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }

        let key_clone_for_get_helper = k.clone(); // K needs Clone due to this
        
        let idx_to_splay_opt;
        { // Scope for mutable borrow of self.elements for get_helper
            let nodes_for_get_helper = self.elements.as_mut().unwrap();
            let root_for_get_helper = self.root; 
            idx_to_splay_opt = SplayTree::get_helper(nodes_for_get_helper, root_for_get_helper, key_clone_for_get_helper);
        }
        
        if let Some(idx_to_splay) = idx_to_splay_opt {
            self.splay(idx_to_splay);
            
            let current_nodes = self.elements.as_ref().unwrap();
            let current_root_idx = self.root.unwrap(); // splay ensures root is Some if elements is Some.
            
            if current_nodes[current_root_idx].key() == &k {
                Some(&current_nodes[current_root_idx].entry)
            } else {
                None // Key not found after splaying closest match
            }
        } else {
             // This implies the tree was empty (handled by initial check),
             // or get_helper returned None unexpectedly (e.g. if start was None and tree wasn't empty).
            None
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the largest key smaller than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the smallest element in the tree
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, &str> = SplayTree::new();
    /// // tree.insert((10, "ten").into());
    /// // tree.insert((5, "five").into());
    /// // tree.insert((15, "fifteen").into());
    /// // assert_eq!(tree.pred(10).unwrap().key(), &5);
    /// // assert_eq!(tree.pred(15).unwrap().key(), &10);
    /// // assert!(tree.pred(5).is_none());
    /// ```
    pub fn pred(&mut self, k: K) -> Option<&Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }

        let key_clone_for_get_helper = k.clone();
        let idx_from_get_helper_opt;
        {
            let nodes_for_get_helper = self.elements.as_mut().unwrap();
            let root_for_get_helper = self.root;
            idx_from_get_helper_opt = SplayTree::get_helper(nodes_for_get_helper, root_for_get_helper, key_clone_for_get_helper);
        }
        
        if let Some(idx_from_get_helper) = idx_from_get_helper_opt {
            self.splay(idx_from_get_helper); 

            let nodes_after_splay = self.elements.as_ref().unwrap();
            let current_root_idx = self.root.unwrap(); 
            let root_key_ref = nodes_after_splay[current_root_idx].key();

            let pred_idx_opt = if root_key_ref < &k { 
                Some(current_root_idx)
            } else { 
                SplayTree::max_helper(nodes_after_splay, nodes_after_splay[current_root_idx].left)
            };
            
            match pred_idx_opt {
                Some(pred_idx) => {
                    if pred_idx == current_root_idx && nodes_after_splay[pred_idx].key() >= &k {
                        return None;
                    }
                    self.splay(pred_idx); 
                    Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry)
                }
                None => None,
            }
        } else {
            None
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key larger than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the largest element in the tree
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, &str> = SplayTree::new();
    /// // tree.insert((10, "ten").into());
    /// // tree.insert((5, "five").into());
    /// // tree.insert((15, "fifteen").into());
    /// // assert_eq!(tree.successor(10).unwrap().key(), &15);
    /// // assert_eq!(tree.successor(5).unwrap().key(), &10);
    /// // assert!(tree.successor(15).is_none());
    /// ```
    pub fn successor(&mut self, k: K) -> Option<&Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }

        let key_clone_for_get_helper = k.clone();
        let idx_from_get_helper_opt;
        {
            let nodes_for_get_helper = self.elements.as_mut().unwrap();
            let root_for_get_helper = self.root;
            idx_from_get_helper_opt = SplayTree::get_helper(nodes_for_get_helper, root_for_get_helper, key_clone_for_get_helper);
        }

        if let Some(idx_from_get_helper) = idx_from_get_helper_opt {
            self.splay(idx_from_get_helper);

            let nodes_after_splay = self.elements.as_ref().unwrap();
            let current_root_idx = self.root.unwrap();
            let root_key_ref = nodes_after_splay[current_root_idx].key();

            let succ_idx_opt = if root_key_ref > &k { 
                Some(current_root_idx)
            } else { 
                SplayTree::min_helper(nodes_after_splay, nodes_after_splay[current_root_idx].right)
            };

            match succ_idx_opt {
                Some(succ_idx) => {
                     if succ_idx == current_root_idx && nodes_after_splay[succ_idx].key() <= &k {
                        return None;
                    }
                    self.splay(succ_idx); 
                    Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry)
                }
                None => None,
            }
        } else {
            None
        }
    }


    /// Retrieves the Key-Value pair associated with
    /// the largest key in the tree
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, &str> = SplayTree::new();
    /// // tree.insert((10, "ten").into());
    /// // tree.insert((5, "five").into());
    /// // tree.insert((15, "fifteen").into());
    /// // assert_eq!(tree.max().unwrap().key(), &15);
    /// ```
    pub fn max(&mut self) -> Option<&Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }
        match SplayTree::max_helper(self.elements.as_ref().unwrap(), self.root) {
            Some(max_idx) => {
                self.splay(max_idx);
                Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry)
            }
            None => None,
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key in the tree
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, &str> = SplayTree::new();
    /// // tree.insert((10, "ten").into());
    /// // tree.insert((5, "five").into());
    /// // tree.insert((15, "fifteen").into());
    /// // assert_eq!(tree.min().unwrap().key(), &5);
    /// ```
    pub fn min(&mut self) -> Option<&Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }
        match SplayTree::min_helper(self.elements.as_ref().unwrap(), self.root) {
            Some(min_idx) => {
                self.splay(min_idx);
                Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry)
            }
            None => None,
        }
    }
}

/// Implementation of Write operations. These procedures
/// do lead to structural changes in the tree
impl<K: Ord + Clone + std::fmt::Debug, V: Clone> SplayTree<K, V> { // Added K: std::fmt::Debug
    /// Adds a new entry into the splay tree in amortized O(lg n) time.
    /// If the key already exists, its value is updated.
    /// The newly inserted or updated node is splayed to the root.
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, String> = SplayTree::new();
    /// // tree.insert(Entry::from((10, "ten".to_string())));
    /// // tree.insert(Entry::from((5, "five".to_string())));
    /// // if let Some(entry) = tree.get(10) { // get will splay 10
    /// //     assert_eq!(entry.key(), &10);
    /// // }
    /// ```
    pub fn insert(&mut self, e: Entry<K, V>) -> Option<&Entry<K, V>> {
        if self.elements.is_none() {
            self.elements = Some(Vec::new());
        }
        
        if self.root.is_none() { // Case 1: Tree is empty
            let nodes = self.elements.as_mut().unwrap();
            let new_node = SplayNode::new(e);
            nodes.push(new_node);
            let new_node_idx = SplayNodeIdx(nodes.len() - 1); 
            self.root = Some(new_node_idx);
            // Splay is trivial for a single node, as it has no parent.
            // The splay method's loop condition `parent.is_some()` handles this.
            // self.splay(new_node_idx); 
            return Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry);
        }

        // Case 2: Tree is not empty
        let key_to_insert = e.key.clone();
        
        let mut current_idx = self.root.unwrap();
        // parent_idx_for_search_phase will hold the parent of current_idx during search,
        // or the node itself if the key is found, or the parent for a new node.
        let mut parent_idx_for_search_phase: SplayNodeIdx;

        // Phase 1: Find insertion point or existing node.
        // This loop only reads node structure for navigation.
        loop {
            let nodes_ro = self.elements.as_ref().unwrap(); 
            let current_node_key_ref = nodes_ro[current_idx].key();
             // Update parent_idx_for_search_phase before current_idx potentially changes
            parent_idx_for_search_phase = current_idx;

            match key_to_insert.cmp(current_node_key_ref) {
                Less => {
                    if let Some(left_child_idx) = nodes_ro[current_idx].left {
                        current_idx = left_child_idx;
                    } else {
                        break; // Found insertion spot as left child of parent_idx_for_search_phase
                    }
                }
                Greater => {
                    if let Some(right_child_idx) = nodes_ro[current_idx].right {
                        current_idx = right_child_idx;
                    } else {
                        break; // Found insertion spot as right child of parent_idx_for_search_phase
                    }
                }
                Equal => { // Key exists at current_idx (which is parent_idx_for_search_phase here)
                    break;
                }
            }
        }
        
        // Phase 2: Modify tree and determine node to splay
        let idx_to_splay;
        { 
            let nodes = self.elements.as_mut().unwrap(); 
            // parent_idx_for_search_phase is the node at which search terminated.
            // It's the node to update if key found, or parent if new node inserted.
            if key_to_insert.cmp(nodes[parent_idx_for_search_phase].key()) == Equal { 
                nodes[parent_idx_for_search_phase].entry.value = e.value;
                idx_to_splay = parent_idx_for_search_phase;
            } else { 
                let new_splay_node = SplayNode::new(e);
                nodes.push(new_splay_node);
                let new_node_idx = SplayNodeIdx(nodes.len() - 1);
                nodes[new_node_idx].parent = Some(parent_idx_for_search_phase);

                if key_to_insert.cmp(nodes[parent_idx_for_search_phase].key()) == Less {
                    nodes[parent_idx_for_search_phase].left = Some(new_node_idx);
                } else {
                    nodes[parent_idx_for_search_phase].right = Some(new_node_idx);
                }
                idx_to_splay = new_node_idx;
            }
        } 
        
        self.splay(idx_to_splay);
        Some(&self.elements.as_ref().unwrap()[self.root.unwrap()].entry)
    }


    /// Removes the entry associated with the provided key
    /// from the splay tree in amortized O(lg n) time.
    ///
    /// # Examples
    /// ```
    /// // use crate::splay_tree::{SplayTree, Entry};
    /// // let mut tree: SplayTree<i32, String> = SplayTree::new();
    /// // tree.insert(Entry::from((10, "ten".to_string())));
    /// // tree.insert(Entry::from((5, "five".to_string())));
    /// // let deleted_entry = tree.delete(5);
    /// // assert!(deleted_entry.is_some());
    /// // assert!(tree.get(5).is_none());
    /// ```
    pub fn delete(&mut self, k: K) -> Option<Entry<K, V>> {
        if self.elements.is_none() || self.root.is_none() {
            return None;
        }

        let target_idx_from_get_helper_opt;
        { 
            let nodes_for_get_helper = self.elements.as_mut().unwrap();
            target_idx_from_get_helper_opt = SplayTree::get_helper(nodes_for_get_helper, self.root, k.clone());
        }
        
        let target_idx_from_get_helper = match target_idx_from_get_helper_opt {
            Some(idx) => idx,
            None => return None, 
        };
        
        self.splay(target_idx_from_get_helper);

        let root_idx_after_splay = self.root.unwrap(); 
        
        let key_matches;
        { 
            let current_nodes_ro = self.elements.as_ref().unwrap();
            key_matches = current_nodes_ro[root_idx_after_splay].key() == &k;
        }

        if !key_matches {
            return None; 
        }

        let deleted_entry = self.elements.as_ref().unwrap()[root_idx_after_splay].entry.clone();

        let left_subtree_root_idx;
        let right_subtree_root_idx;
        { 
            let nodes = self.elements.as_mut().unwrap();
            left_subtree_root_idx = nodes[root_idx_after_splay].left.take();
            right_subtree_root_idx = nodes[root_idx_after_splay].right.take();
        }
        
        if let Some(l_idx) = left_subtree_root_idx {
            self.elements.as_mut().unwrap()[l_idx].parent = None;
            
            let _old_main_root_idx = self.root; 
            self.root = Some(l_idx); 

            let max_in_left_idx;
            { 
                let nodes_ro_for_max_helper = self.elements.as_ref().unwrap();
                max_in_left_idx = SplayTree::max_helper(nodes_ro_for_max_helper, Some(l_idx))
                    .expect("Left subtree was not empty, so max must exist.");
            }
            self.splay(max_in_left_idx); 
            
            let nodes_for_right_attach = self.elements.as_mut().unwrap();
            nodes_for_right_attach[self.root.unwrap()].right = right_subtree_root_idx;
            if let Some(r_idx) = right_subtree_root_idx {
                nodes_for_right_attach[r_idx].parent = self.root;
            }
        } else { 
            self.root = right_subtree_root_idx;
            if let Some(r_idx) = right_subtree_root_idx {
                self.elements.as_mut().unwrap()[r_idx].parent = None;
            }
        }
        
        Some(deleted_entry)
    }
}

/// Implementation of internal helper functions to read operations. We call the
/// splay procedure here to make the public API as clean as possible
impl<K: Ord, V> SplayTree<K, V> {
    /// Searches for the location of the entry with the provided key starting
    /// at the specified location index. If such an entry exists, we return its index.
    /// If not, we return the index of the last node visited on the search path. 
    /// Returns `None` only if the tree is empty (start is None).
    fn get_helper(nodes: &mut Nodes<K, V>, start: Option<SplayNodeIdx>, key: K) -> Option<SplayNodeIdx> {
        let mut current_idx_opt = start;
        let mut last_visited_idx_opt: Option<SplayNodeIdx> = None;

        while let Some(current_idx) = current_idx_opt {
            last_visited_idx_opt = Some(current_idx);
            
            let node_key = &nodes[current_idx].entry.key; 
            let left_child = nodes[current_idx].left;
            let right_child = nodes[current_idx].right;

            match key.cmp(node_key) {
                Less => current_idx_opt = left_child,
                Greater => current_idx_opt = right_child,
                Equal => return Some(current_idx), // Key found
            }
        }
        last_visited_idx_opt 
    }

    /// Searches for the location of the entry with the largest key value
    fn max_helper(nodes: &Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        let mut current_idx_opt = start;
        let mut max_idx_opt: Option<SplayNodeIdx> = None; 
        while let Some(current_idx) = current_idx_opt {
            max_idx_opt = Some(current_idx);
            current_idx_opt = nodes[current_idx].right;
        }
        max_idx_opt
    }

    /// Searches for the location of the entry with the smallest key value
    fn min_helper(nodes: &Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        let mut current_idx_opt = start;
        let mut min_idx_opt: Option<SplayNodeIdx> = None; 
        while let Some(current_idx) = current_idx_opt {
            min_idx_opt = Some(current_idx);
            current_idx_opt = nodes[current_idx].left;
        }
        min_idx_opt
    }


    /// Searches for the location of the lowest ancestor node that is a right child of
    /// its parent. This is a sub-procedure used when computing the predecessor of a node
    /// `lra` stands for lowest right ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    #[allow(dead_code)] // Currently not used by refined pred/succ
    fn lra(nodes: &Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, Some(cur_idx)).and_then(|child_type| match child_type {
                ChildType::Left => Self::lra(nodes, nodes[cur_idx].parent),
                ChildType::Right => nodes[cur_idx].parent,
            })
        })
    }

    /// Searches for the location of the lowest ancestor node that is a left child of
    /// its parent. This is a sub-procedure used when computing the successor of a node
    /// `lla` stands for lowest left ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    #[allow(dead_code)] // Currently not used by refined pred/succ
    fn lla(nodes: &Nodes<K, V>, start: Option<SplayNodeIdx>) -> Option<SplayNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, Some(cur_idx)).and_then(|child_type| match child_type {
                ChildType::Right => Self::lla(nodes, nodes[cur_idx].parent),
                ChildType::Left => nodes[cur_idx].parent,
            })
        })
    }

    /// Is this node a left child or right child of its parent
    fn child_type(nodes: &Nodes<K, V>, cur: Option<SplayNodeIdx>) -> Option<ChildType> {
        cur.and_then(|cur_idx| {
            nodes[cur_idx].parent.and_then(|parent_idx| {
                if nodes[parent_idx].left == Some(cur_idx) {
                    Some(ChildType::Left)
                } else if nodes[parent_idx].right == Some(cur_idx) {
                    Some(ChildType::Right)
                } else {
                    None 
                }
            })
        })
    }
}


/// Implementation of the bottom up splay operation
impl<K: Ord + std::fmt::Debug, V> SplayTree<K, V> { // Added K: std::fmt::Debug
    /// Moves the target node to the root of the tree using a series of rotations.
    fn splay(&mut self, target: SplayNodeIdx) {
        if self.elements.is_none() { return; }
        let elements_ref_check = self.elements.as_ref().unwrap();
        if elements_ref_check.is_empty() { return; } 

        // Check if target is valid
        if target.0 >= elements_ref_check.len() {
            eprintln!("[SPLAY] Error: Target index {:?} out of bounds.", target);
            return;
        }
        
        { // Logging scope for initial state
            let nodes_for_log = self.elements.as_ref().unwrap();
            eprintln!("[SPLAY] Starting splay for target: NodeIdx({}), Key({:?})", target.0, nodes_for_log[target].key());
        }

        // Check if target has a parent (i.e., not already root)
        if elements_ref_check[target].parent.is_none() {
            // Target is root or invalid, no splay needed or possible.
            // If target is valid and has no parent, it should be the root.
            self.root = Some(target);
            { // Logging scope
                let nodes_for_log = self.elements.as_ref().unwrap();
                eprintln!("[SPLAY] Target NodeIdx({}), Key({:?}) is already root or has no parent. Splay not needed.", target.0, nodes_for_log[target].key());
            }
            return;
        }

        while self.elements.as_ref().unwrap()[target].parent.is_some() {
            let parent_idx = self.elements.as_ref().unwrap()[target].parent.unwrap();
            let grand_parent_idx_opt = self.elements.as_ref().unwrap()[parent_idx].parent;
            let target_is_left_child_of_parent = self.elements.as_ref().unwrap()[parent_idx].left == Some(target);

            { // Logging scope for pre-rotation state
                let nodes_for_log = self.elements.as_ref().unwrap();
                let target_key = nodes_for_log[target].key();
                let parent_key = nodes_for_log[parent_idx].key();
                if let Some(gp_idx) = grand_parent_idx_opt {
                    let gp_key = nodes_for_log[gp_idx].key();
                    eprintln!("[SPLAY] Pre-rotation: Target: {:?}({:?}), Parent: {:?}({:?}), Grandparent: {:?}({:?})",
                              target.0, target_key, parent_idx.0, parent_key, gp_idx.0, gp_key);
                } else {
                    eprintln!("[SPLAY] Pre-rotation: Target: {:?}({:?}), Parent: {:?}({:?}), No Grandparent (Parent is root)",
                              target.0, target_key, parent_idx.0, parent_key);
                }
            }

            if grand_parent_idx_opt.is_none() { // Parent is root: Zig case
                let nodes = self.elements.as_mut().unwrap();
                let op_type = if target_is_left_child_of_parent { "Zig (L)" } else { "Zig (R)" };
                eprintln!("[SPLAY] Target: {:?}, Op: {}, Rotate around P_Key: {:?}", nodes[target].key(), op_type, nodes[parent_idx].key());
                if target_is_left_child_of_parent {
                    Self::rotate_right(nodes, parent_idx);
                } else {
                    Self::rotate_left(nodes, parent_idx);
                }
            } else { // Parent is not root: ZigZig or ZigZag
                let grand_parent_idx = grand_parent_idx_opt.unwrap();
                let parent_is_left_child_of_grandparent = self.elements.as_ref().unwrap()[grand_parent_idx].left == Some(parent_idx);
                let nodes = self.elements.as_mut().unwrap();

                if parent_is_left_child_of_grandparent { // Parent is Left Child
                    if target_is_left_child_of_parent { // Target is Left Child: ZigZig (Left-Left)
                        eprintln!("[SPLAY] Target: {:?}, Op: ZigZig L-L, RotateRight G_Key: {:?}, then RotateRight P_Key: {:?}", nodes[target].key(), nodes[grand_parent_idx].key(), nodes[parent_idx].key());
                        Self::rotate_right(nodes, grand_parent_idx);
                        Self::rotate_right(nodes, parent_idx);
                    } else { // Target is Right Child: ZigZag (Left-Right)
                        eprintln!("[SPLAY] Target: {:?}, Op: ZigZag L-R, RotateLeft P_Key: {:?}, then RotateRight G_Key: {:?}", nodes[target].key(), nodes[parent_idx].key(), nodes[grand_parent_idx].key());
                        Self::rotate_left(nodes, parent_idx);
                        Self::rotate_right(nodes, grand_parent_idx);
                    }
                } else { // Parent is Right Child
                    if target_is_left_child_of_parent { // Target is Left Child: ZigZag (Right-Left)
                        eprintln!("[SPLAY] Target: {:?}, Op: ZigZag R-L, RotateRight P_Key: {:?}, then RotateLeft G_Key: {:?}", nodes[target].key(), nodes[parent_idx].key(), nodes[grand_parent_idx].key());
                        Self::rotate_right(nodes, parent_idx);
                        Self::rotate_left(nodes, grand_parent_idx);
                    } else { // Target is Right Child: ZigZig (Right-Right)
                        eprintln!("[SPLAY] Target: {:?}, Op: ZigZig R-R, RotateLeft G_Key: {:?}, then RotateLeft P_Key: {:?}", nodes[target].key(), nodes[grand_parent_idx].key(), nodes[parent_idx].key());
                        Self::rotate_left(nodes, grand_parent_idx);
                        Self::rotate_left(nodes, parent_idx);
                    }
                }
            }
            { // Logging scope for post-rotation state
                let nodes_for_log = self.elements.as_ref().unwrap();
                let target_key = nodes_for_log[target].key();
                let t_p_key = nodes_for_log[target].parent.map(|pi| nodes_for_log[pi].key());
                let t_l_key = nodes_for_log[target].left.map(|li| nodes_for_log[li].key());
                let t_r_key = nodes_for_log[target].right.map(|ri| nodes_for_log[ri].key());
                eprintln!("[SPLAY] Target: {:?}({:?}), After rotation: Parent: {:?}, Left: {:?}, Right: {:?}",
                          target.0, target_key, t_p_key, t_l_key, t_r_key);
            }
        }
        self.root = Some(target);
        { // Logging scope
            let nodes_for_log = self.elements.as_ref().unwrap();
            eprintln!("[SPLAY] Finished splay. Target NodeIdx({}), Key({:?}) is now root.", target.0, nodes_for_log[target].key());
        }
    }
    
    /// We are give a node `x` which is the root of some sub-tree and we would
    /// like to exchange it with its left child `y` which must exist. `x` can be
    /// the root of the tree in which case it has no parent.
    fn rotate_right(nodes: &mut Nodes<K, V>, x_idx: SplayNodeIdx) { // K must be Debug for logging key()
        // eprintln!("[ROTATE_RIGHT] Rotating right around NodeIdx({}), Key({:?})", x_idx.0, nodes[x_idx].key()); // K must be Debug
        let y_idx = nodes[x_idx].left.expect("Left child must exist for rotate_right");
        // eprintln!("[ROTATE_RIGHT] Left child (y) is NodeIdx({}), Key({:?})", y_idx.0, nodes[y_idx].key());
        let beta_idx = nodes[y_idx].right;
        // if let Some(b_idx) = beta_idx {
        //     eprintln!("[ROTATE_RIGHT] y's right child (beta) is NodeIdx({}), Key({:?})", b_idx.0, nodes[b_idx].key());
        // } else {
        //     eprintln!("[ROTATE_RIGHT] y's right child (beta) is None");
        // }

        nodes[y_idx].parent = nodes[x_idx].parent;
        // if let Some(p_idx) = nodes[y_idx].parent {
        //     eprintln!("[ROTATE_RIGHT] y's new parent is NodeIdx({}), Key({:?})", p_idx.0, nodes[p_idx].key());
        // } else {
        //     eprintln!("[ROTATE_RIGHT] y is now root");
        // }
        if let Some(p_idx) = nodes[x_idx].parent {
            if nodes[p_idx].left == Some(x_idx) {
                nodes[p_idx].left = Some(y_idx);
                // eprintln!("[ROTATE_RIGHT] x's parent NodeIdx({}), Key({:?}) now points left to y", p_idx.0, nodes[p_idx].key());
            } else {
                nodes[p_idx].right = Some(y_idx);
                // eprintln!("[ROTATE_RIGHT] x's parent NodeIdx({}), Key({:?}) now points right to y", p_idx.0, nodes[p_idx].key());
            }
        }
        
        nodes[x_idx].left = beta_idx;
        // eprintln!("[ROTATE_RIGHT] x's left child is now beta (NodeIdx({:?}))", beta_idx.map(|i|i.0));
        if let Some(b_idx) = beta_idx {
            nodes[b_idx].parent = Some(x_idx);
            // eprintln!("[ROTATE_RIGHT] beta's parent is now x (NodeIdx({}), Key({:?}))", x_idx.0, nodes[x_idx].key());
        }

        nodes[y_idx].right = Some(x_idx);
        // eprintln!("[ROTATE_RIGHT] y's right child is now x (NodeIdx({}), Key({:?}))", x_idx.0, nodes[x_idx].key());
        nodes[x_idx].parent = Some(y_idx);
        // eprintln!("[ROTATE_RIGHT] x's parent is now y (NodeIdx({}), Key({:?}))", y_idx.0, nodes[y_idx].key());
    }
    
    /// We are give  a node `y` which is the root of some sub-tree and we would
    /// like to exchange it with its right child `x` which must exist. `y` can be
    /// the root of the tree in which case it has no parent.
    fn rotate_left(nodes: &mut Nodes<K, V>, y_idx: SplayNodeIdx) { // K must be Debug for logging key()
        // eprintln!("[ROTATE_LEFT] Rotating left around NodeIdx({}), Key({:?})", y_idx.0, nodes[y_idx].key()); // K must be Debug
        let x_idx = nodes[y_idx].right.expect("Right child must exist for rotate_left");
        // eprintln!("[ROTATE_LEFT] Right child (x) is NodeIdx({}), Key({:?})", x_idx.0, nodes[x_idx].key());
        let beta_idx = nodes[x_idx].left;
        // if let Some(b_idx) = beta_idx {
        //     eprintln!("[ROTATE_LEFT] x's left child (beta) is NodeIdx({}), Key({:?})", b_idx.0, nodes[b_idx].key());
        // } else {
        //     eprintln!("[ROTATE_LEFT] x's left child (beta) is None");
        // }

        nodes[x_idx].parent = nodes[y_idx].parent;
        // if let Some(p_idx) = nodes[x_idx].parent {
        //     eprintln!("[ROTATE_LEFT] x's new parent is NodeIdx({}), Key({:?})", p_idx.0, nodes[p_idx].key());
        // } else {
        //     eprintln!("[ROTATE_LEFT] x is now root");
        // }
        if let Some(p_idx) = nodes[y_idx].parent {
            if nodes[p_idx].left == Some(y_idx) {
                nodes[p_idx].left = Some(x_idx);
                // eprintln!("[ROTATE_LEFT] y's parent NodeIdx({}), Key({:?}) now points left to x", p_idx.0, nodes[p_idx].key());
            } else {
                nodes[p_idx].right = Some(x_idx);
                // eprintln!("[ROTATE_LEFT] y's parent NodeIdx({}), Key({:?}) now points right to x", p_idx.0, nodes[p_idx].key());
            }
        }
        
        nodes[y_idx].right = beta_idx;
        // eprintln!("[ROTATE_LEFT] y's right child is now beta (NodeIdx({:?}))", beta_idx.map(|i|i.0));
        if let Some(b_idx) = beta_idx {
            nodes[b_idx].parent = Some(y_idx);
            // eprintln!("[ROTATE_LEFT] beta's parent is now y (NodeIdx({}), Key({:?}))", y_idx.0, nodes[y_idx].key());
        }

        nodes[x_idx].left = Some(y_idx);
        // eprintln!("[ROTATE_LEFT] x's left child is now y (NodeIdx({}), Key({:?}))", y_idx.0, nodes[y_idx].key());
        nodes[y_idx].parent = Some(x_idx);
        // eprintln!("[ROTATE_LEFT] y's parent is now x (NodeIdx({}), Key({:?}))", x_idx.0, nodes[x_idx].key());
    }
}

// The conceptual empty impl block for K: Debug is removed as the constraint is added directly.

#[cfg(test)]
mod test_splay_tree { 
    use super::*;

    #[test]
    fn test_new_empty() {
        let tree: SplayTree<i32, String> = SplayTree::new();
        assert!(tree.root.is_none());
        assert!(tree.elements.is_none()); 
    }

    #[test]
    fn test_insert_into_empty_tree() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        assert!(tree.root.is_some());
        let root_idx = tree.root.unwrap();
        let elements = tree.elements.as_ref().unwrap();
        assert_eq!(elements[root_idx].key(), &10);
        assert_eq!(elements[root_idx].entry.value(), "ten");
        assert!(elements[root_idx].parent.is_none());
    }

    #[test]
    fn test_insert_multiple_and_get() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        tree.insert(Entry::from((5, "five".to_string())));
        tree.insert(Entry::from((15, "fifteen".to_string())));
        tree.insert(Entry::from((3, "three".to_string())));
        tree.insert(Entry::from((7, "seven".to_string())));

        assert_eq!(tree.get(10).unwrap().value(), "ten");
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &10);


        assert_eq!(tree.get(3).unwrap().value(), "three");
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &3); 

        assert!(tree.get(100).is_none());
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &15);
    }
    
    #[test]
    fn test_insert_duplicate_updates_value() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten_v1".to_string())));
        assert_eq!(tree.get(10).unwrap().value(), "ten_v1");
        
        tree.insert(Entry::from((10, "ten_v2".to_string())));
        assert_eq!(tree.get(10).unwrap().value(), "ten_v2");
        // Length should be 1 if duplicates update
        let elements = tree.elements.as_ref().unwrap();
        assert_eq!(elements.len(), 1); 
        assert_eq!(elements[tree.root.unwrap()].key(), &10); 
    }

    #[test]
    fn test_delete_leaf() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string()))); 
        tree.insert(Entry::from((5, "five".to_string()))); 
        
        let deleted = tree.delete(10); 
        assert!(deleted.is_some());
        assert_eq!(deleted.unwrap().key(), &10);
        assert!(tree.get(10).is_none()); 
        assert!(tree.root.is_some()); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &5);
    }

    #[test]
    fn test_delete_node_with_one_child() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        tree.insert(Entry::from((5, "five".to_string()))); 
        tree.insert(Entry::from((3, "three".to_string()))); 
        
        let deleted = tree.delete(5);
        assert_eq!(deleted.as_ref().unwrap().key(), &5);

        // Perform structural checks first
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &3);
        assert_eq!(tree.elements.as_ref().unwrap()[tree.elements.as_ref().unwrap()[tree.root.unwrap()].right.unwrap()].key(), &10);

        // Then check get after structural assertions are done
        assert!(tree.get(5).is_none());
    }

    #[test]
    fn test_delete_node_with_two_children() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        tree.insert(Entry::from((5, "five".to_string())));
        tree.insert(Entry::from((15, "fifteen".to_string())));
        tree.insert(Entry::from((3, "three".to_string())));
        tree.insert(Entry::from((7, "seven".to_string())));
        tree.insert(Entry::from((12, "twelve".to_string())));
        tree.insert(Entry::from((17, "seventeen".to_string())));

        assert_eq!(tree.get(10).unwrap().key(), &10); // Initial get to splay 10 to root
        
        let deleted = tree.delete(10);
        assert_eq!(deleted.as_ref().expect("Deleted entry should be Some").key(), &10);

        // Structural assertions first
        let elements_ref = tree.elements.as_ref().expect("Elements should exist");
        let root_idx = tree.root.expect("Root should exist");

        let root_key = elements_ref[root_idx].key();
        assert_eq!(root_key, &7, "Root key should be 7");

        let node_7_right_child_idx = elements_ref[root_idx].right.expect("Root (7) should have a right child (node 15)");
        let node_7_right_child_key = elements_ref[node_7_right_child_idx].key();
        assert_eq!(node_7_right_child_key, &15, "Root's (7) right child key should be 15");

        // Check children of node 15 (which is node_7_right_child_idx)
        let node_15_left_child_idx = elements_ref[node_7_right_child_idx].left.expect("Node 15 should have a left child (node 12)");
        let node_15_left_child_key = elements_ref[node_15_left_child_idx].key();
        assert_eq!(node_15_left_child_key, &12, "Node 15's left child key should be 12");

        let node_15_right_child_idx = elements_ref[node_7_right_child_idx].right.expect("Node 15 should have a right child (node 17)");
        let node_15_right_child_key = elements_ref[node_15_right_child_idx].key();
        assert_eq!(node_15_right_child_key, &17, "Node 15's right child key should be 17");

        // Then check get for the deleted key
        assert!(tree.get(10).is_none(), "Getting deleted key 10 should return None");
    }
    
    #[test]
    fn test_delete_root() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        tree.insert(Entry::from((5, "five".to_string())));
        tree.insert(Entry::from((15, "fifteen".to_string()))); 
        
        let deleted = tree.delete(15); 
        assert_eq!(deleted.as_ref().unwrap().key(), &15);
        assert!(tree.get(15).is_none());
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &10); 
    }

    #[test]
    fn test_delete_non_existent() {
        let mut tree: SplayTree<i32, String> = SplayTree::new();
        tree.insert(Entry::from((10, "ten".to_string())));
        tree.insert(Entry::from((5, "five".to_string()))); 
        
        assert!(tree.delete(100).is_none());
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &10);
    }

    #[test]
    fn test_min_max_pred_succ_after_various_ops() {
        let mut tree: SplayTree<i32, &str> = SplayTree::new();
        let entries = [(10,"a"), (20,"b"), (30,"c"), (5,"d"), (15,"e"), (25,"f"), (35,"g"), (1,"h"), (8,"i")];
        for (k,v) in entries {
            tree.insert(Entry::from((k,v)));
        }

        assert_eq!(tree.min().unwrap().key(), &1);
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &1); 
        
        assert_eq!(tree.max().unwrap().key(), &35);
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &35); 

        assert_eq!(tree.pred(20).unwrap().key(), &15);
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &15); 

        assert_eq!(tree.successor(20).unwrap().key(), &25);
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &25); 
        
        assert!(tree.pred(1).is_none()); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &1); 

        assert!(tree.successor(35).is_none()); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &35); 
        
        assert_eq!(tree.pred(22).unwrap().key(), &20); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &20); 

        assert_eq!(tree.successor(22).unwrap().key(), &25); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &25);

        assert_eq!(tree.delete(20).unwrap().key(), &20); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &15);

        assert_eq!(tree.delete(15).unwrap().key(), &15); 
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &8);

        let succ_key_val = tree.successor(8).unwrap().key();
        eprintln!("[DEBUG TEST] In test_min_max_pred_succ_after_various_ops: successor(8) returned key: {:?}", succ_key_val);
        eprintln!("[DEBUG TEST] In test_min_max_pred_succ_after_various_ops: Expected key for successor(8) is: {:?}", &10);
        assert_eq!(succ_key_val, &10, "Successor of 8 should be 10");
        assert_eq!(tree.elements.as_ref().unwrap()[tree.root.unwrap()].key(), &10);
    }
}

#[cfg(test)]
mod test_splay_tree_properties {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};
    use std::collections::{HashMap, BTreeSet};

    // Implement Arbitrary for Entry<i32, i32> for property tests
    // Note: K and V must be Clone for Entry to be Clone.
    // K must be Ord for SplayTree.
    // K and V must be Default for SplayTree's new() and some read ops.
    // For these tests, we'll use i32 which satisfies these.
    impl Arbitrary for Entry<i32, i32> {
        fn arbitrary(g: &mut Gen) -> Self {
            Entry {
                key: i32::arbitrary(g),
                value: i32::arbitrary(g),
            }
        }
    }

    quickcheck! {
        fn prop_insert_get_one(key: i32, value: i32) -> bool {
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let entry = Entry::from((key, value));
            tree.insert(entry.clone());
            
            let retrieved = tree.get(key);
            retrieved.is_some() && retrieved.unwrap().key() == &key && retrieved.unwrap().value() == &value
        }

        fn prop_insert_get_multiple(entries_tuples: Vec<(i32, i32)>) -> bool {
            if entries_tuples.is_empty() {
                return true; 
            }
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let mut map = HashMap::new();

            for (k, v) in entries_tuples {
                tree.insert(Entry::from((k, v)));
                map.insert(k, v); 
            }

            for (key, value) in map.iter() {
                if let Some(retrieved_entry) = tree.get(*key) {
                    if retrieved_entry.key() != key || retrieved_entry.value() != value {
                        return false;
                    }
                } else {
                    return false; 
                }
            }
            true
        }

        fn prop_insert_delete_get(key: i32, value: i32) -> bool {
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let entry = Entry::from((key, value));
            tree.insert(entry.clone());

            let deleted_entry_opt = tree.delete(key);
            if deleted_entry_opt.is_none() { return false; }
            let deleted_entry = deleted_entry_opt.unwrap();
            if deleted_entry.key() != &key || deleted_entry.value() != &value { return false; }

            tree.get(key).is_none()
        }

        fn prop_delete_non_existent(entries_tuples: Vec<(i32, i32)>, non_existent_key: i32) -> TestResult {
            let mut non_existent_key = non_existent_key; // Added line
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let mut keys_present = BTreeSet::new();
            for (k, v) in entries_tuples {
                tree.insert(Entry::from((k, v)));
                keys_present.insert(k);
            }

            // Ensure the key is truly non-existent
            while keys_present.contains(&non_existent_key) {
                 // Attempt to find a non-existent key by incrementing.
                 // This is a simple strategy; more robust would be to draw from a different range.
                if non_existent_key == i32::MAX { 
                     return TestResult::discard(); // Hard to find a non-existent key if all are present.
                }
                non_existent_key = non_existent_key.wrapping_add(1);
            }
            
            TestResult::from_bool(tree.delete(non_existent_key).is_none())
        }
        
        fn prop_min_max_after_inserts(entries_tuples: Vec<(i32, i32)>) -> TestResult {
            if entries_tuples.is_empty() {
                return TestResult::discard();
            }
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let mut min_key_opt: Option<i32> = None;
            let mut max_key_opt: Option<i32> = None;
            
            for (k, v) in entries_tuples {
                tree.insert(Entry::from((k, v)));
                min_key_opt = Some(min_key_opt.map_or(k, |mk| std::cmp::min(mk, k)));
                max_key_opt = Some(max_key_opt.map_or(k, |mk| std::cmp::max(mk, k)));
            }
            
            let min_res = tree.min().map(|e| *e.key()) == min_key_opt;
            let max_res = tree.max().map(|e| *e.key()) == max_key_opt;
            
            TestResult::from_bool(min_res && max_res)
        }

        fn prop_successor_predecessor_consistency(entries_tuples: Vec<(i32, i32)>) -> TestResult {
            if entries_tuples.len() < 2 {
                return TestResult::discard();
            }
            let mut tree: SplayTree<i32, i32> = SplayTree::new();
            let mut unique_sorted_keys = BTreeSet::new();

            for (k, v) in entries_tuples {
                tree.insert(Entry::from((k,v)));
                unique_sorted_keys.insert(k);
            }
            
            if unique_sorted_keys.len() < 2 { 
                return TestResult::discard();
            }

            let sorted_keys: Vec<i32> = unique_sorted_keys.into_iter().collect();

            for i in 0..sorted_keys.len() {
                let k = sorted_keys[i];
                
                if i > 0 {
                    let expected_pred = sorted_keys[i-1];
                    if let Some(pred_entry) = tree.pred(k) {
                        if *pred_entry.key() != expected_pred { return TestResult::error(format!("Pred: For key {}, expected {}, got {}", k, expected_pred, pred_entry.key())); }
                    } else {
                        return TestResult::error(format!("Pred: For key {}, expected Some({}), got None", k, expected_pred));
                    }
                } else { 
                    if tree.pred(k).is_some() { return TestResult::error(format!("Pred: For smallest key {}, expected None, got Some", k)); }
                }

                if i < sorted_keys.len() - 1 {
                    let expected_succ = sorted_keys[i+1];
                    if let Some(succ_entry) = tree.successor(k) {
                        if *succ_entry.key() != expected_succ { return TestResult::error(format!("Succ: For key {}, expected {}, got {}", k, expected_succ, succ_entry.key())); }
                    } else {
                        return TestResult::error(format!("Succ: For key {}, expected Some({}), got None", k, expected_succ));
                    }
                } else { 
                    if tree.successor(k).is_some() { return TestResult::error(format!("Succ: For largest key {}, expected None, got Some", k)); }
                }
            }
            TestResult::passed()
        }
    }
}
