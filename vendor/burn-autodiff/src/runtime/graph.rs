use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    NodeId,
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::{Parent, StepBoxed},
    runtime::server::NodeCleaner,
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::sync::Arc;
use burn_common::stub::Mutex;
use burn_tensor::backend::Backend;
use hashbrown::{HashMap, HashSet};

/// A client for managing multiple graphs using mutex-based synchronization.
///
/// The biggest benefit of using this client implementation is that each graph can modify its own
/// data without blocking other graphs, which is essential for multi-device training.
///
/// # Notes
///
/// The [AutodiffServer] fully supports multiple graphs with sharing nodes, however those type of
/// graphs will be stored under a single mutex-protected graph by the client, limiting
/// parallelisation.
#[derive(Clone, new, Debug)]
pub struct GraphMutexClient;

/// Manages a collection of graphs, mapping [node ids](NodeId) to their respective graph.
///
/// The `GraphLocator` is responsible for selecting and merging graphs based on their IDs and parent
/// dependencies, ensuring proper synchronization and server allocation.
///
/// # Notes
///
/// Multiple node ids can point to the same graph, where the autodiff graph is stored.
#[derive(Default)]
pub struct GraphLocator {
    graphs: HashMap<NodeId, Arc<Graph>>,
    /// We keep a mapping of each original node id (graph id) => all nodes that point to that graph.
    /// This is to ensure that when merging graphs, we correctly move all previous graphs to
    /// the new merged one.
    keys: HashMap<NodeId, HashSet<NodeId>>,
    untracked_nodes: HashSet<NodeId>,
}

/// Represents a single computation graph with a mutex-protected server.
///
/// Each `Graph` contains an [AutodiffServer] and the original [NodeId] where the server was
/// first created.
pub(crate) struct Graph {
    origin: NodeId,
    state: Mutex<GraphState>,
}

#[derive(Default)]
struct GraphState {
    server: AutodiffServer,
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("origin", &self.origin)
            .finish()
    }
}

static STATE: spin::Mutex<Option<GraphLocator>> = spin::Mutex::new(None);

#[cfg(feature = "std")]
std::thread_local! {
    static LOCAL_SERVERS: std::cell::RefCell<Vec<AutodiffServer>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) struct LocalGraphGuard;

impl LocalGraphGuard {
    pub(crate) fn enter() -> Self {
        #[cfg(feature = "std")]
        LOCAL_SERVERS.with(|servers| servers.borrow_mut().push(AutodiffServer::default()));
        Self
    }
}

impl Drop for LocalGraphGuard {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        LOCAL_SERVERS.with(|servers| {
            servers.borrow_mut().pop();
        });
    }
}

pub(crate) fn cleanup_dead_nodes() {
    let graphs: Vec<Arc<Graph>> = {
        let state = STATE.lock();
        let Some(locator) = state.as_ref() else {
            return;
        };
        locator.graphs.values().cloned().collect()
    };
    for graph in graphs {
        let mut graph_state = graph.state.lock().unwrap();
        graph_state.server.sweep_dead_nodes();
    }
    prune_empty_graphs_global();
}

impl GraphMutexClient {
    /// Retrieves or creates a graph for the given [NodeId] and parent dependencies.
    ///
    /// # Parameters
    /// - `node`: The unique identifier for the stream.
    /// - `parents`: A slice of parent nodes that the stream depends on.
    ///
    /// # Returns
    /// An `Arc<Graph>` representing the selected or newly created stream.
    fn graph(node: NodeId, parents: &[Parent]) -> Arc<Graph> {
        let mut state = STATE.lock();

        match state.as_mut() {
            Some(locator) => locator.select(node, parents),
            None => {
                let mut locator = GraphLocator::default();
                let stream = locator.select(node, parents);
                *state = Some(locator);
                stream
            }
        }
    }

}

impl AutodiffClient for GraphMutexClient {
    fn register(&self, node_id_ref: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let mut step = Some(step);
        let mut actions = Some(actions);
        #[cfg(feature = "std")]
        {
            let handled = LOCAL_SERVERS.with(|servers| {
                let mut servers = servers.borrow_mut();
                if let Some(server) = servers.last_mut() {
                    server.register(
                        node_id_ref.clone(),
                        step.take().expect("local step already consumed"),
                        actions.take().expect("local actions already consumed"),
                    );
                    true
                } else {
                    false
                }
            });
            if handled {
                return;
            }
        }
        let node_id = *node_id_ref;
        let step = step.expect("global step already consumed");
        let actions = actions.expect("global actions already consumed");
        let graph = GraphMutexClient::graph(node_id, step.parents());
        let mut state = graph.state.lock().unwrap();

        state.server.register(node_id_ref, step, actions);
    }

    fn register_untracked(
        &self,
        node_id_ref: NodeRefCount,
        step: StepBoxed,
        actions: CheckpointerBuilder,
    ) {
        #[cfg(feature = "std")]
        {
            let local = LOCAL_SERVERS.with(|servers| !servers.borrow().is_empty());
            if local {
                self.register(node_id_ref, step, actions);
                return;
            }
        }
        let node_id = *node_id_ref;
        // Register normally (might be needed by tracked children, required for checkpointing)
        self.register(node_id_ref, step, actions);

        // But mark this node for cleanup after backward
        let mut state = STATE.lock();
        if let Some(locator) = state.as_mut() {
            locator.mark_untracked(node_id);
        }
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let debug_cleanup = std::env::var_os("AFTERBURNER_DEBUG_GRAPH_CLEANUP").is_some();
        let backward_started = std::time::Instant::now();
        let node_id = root.node.id;
        let graph = GraphMutexClient::graph(root.node.id, &[]);

        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut state = graph.state.lock().unwrap();

        let grads = state.server.backward::<GraphCleaner>(grads, node_id);
        if debug_cleanup {
            eprintln!(
                "[graph-cleanup] server_backward_complete elapsed={:.3}s",
                backward_started.elapsed().as_secs_f64()
            );
        }

        // The backward server owns the temporary Steps and checkpoint builders
        // for this graph. Release the mutex before compacting the global
        // locator; pruning only graphs whose server and memory-management
        // state are both empty is safe for the next forward pass.
        core::mem::drop(state);

        let mut cleaner = GraphCleaner::init();
        let cleanup_started = std::time::Instant::now();
        cleaner.cleanup_orphaned_entries();
        if debug_cleanup {
            eprintln!(
                "[graph-cleanup] orphan_cleanup_complete elapsed={:.3}s total={:.3}s",
                cleanup_started.elapsed().as_secs_f64(),
                backward_started.elapsed().as_secs_f64()
            );
        }
        // Do not hold GraphCleaner::guard while taking graph state locks.
        // The previous lock order (global STATE -> graph.state) could wait
        // forever when a local checkpoint graph was being finalized.
        core::mem::drop(cleaner);
        prune_empty_graphs_global();
        if debug_cleanup {
            eprintln!(
                "[graph-cleanup] prune_complete elapsed={:.3}s total={:.3}s",
                cleanup_started.elapsed().as_secs_f64(),
                backward_started.elapsed().as_secs_f64()
            );
        }

        grads
    }

}

struct NoopCleaner;

impl NodeCleaner for NoopCleaner {
    fn init() -> Self { Self }
    fn clean(&mut self, _node: &NodeId) {}
}

impl GraphMutexClient {
    pub(crate) fn backward_local_with_grad<B: Backend>(
        root: AutodiffTensor<B>,
        root_grad: B::FloatTensorPrimitive,
    ) -> Gradients {
        let node_id = root.node.id;
        let root_node = root.node.clone();
        let root_primitive = root.primitive.clone();
        let mut grads = Gradients::new::<B>(root_node, root_primitive);
        grads.register::<B>(node_id, root_grad);
        #[cfg(feature = "std")]
        {
            let result = LOCAL_SERVERS.with(|servers| {
                let mut servers = servers.borrow_mut();
                servers
                    .last_mut()
                    .map(|server| server.backward::<NoopCleaner>(grads, node_id))
            });
            if let Some(grads) = result {
                return grads;
            }
        }
        panic!("backward_local_with_grad called outside LocalGraphGuard");
    }
}

struct GraphCleaner<'a> {
    guard: spin::MutexGuard<'a, Option<GraphLocator>>,
}

impl<'a> GraphCleaner<'a> {
    fn cleanup_orphaned_entries(&mut self) {
        if let Some(state) = self.guard.as_mut() {
            state.cleanup_untracked();
        }
    }

}

impl<'a> NodeCleaner for GraphCleaner<'a> {
    fn init() -> Self {
        let guard = STATE.lock();
        Self { guard }
    }

    fn clean(&mut self, node: &NodeId) {
        if let Some(state) = self.guard.as_mut() {
            state.remove_entry(node);
        }
    }
}

impl GraphLocator {
    fn prune_empty_graphs(&mut self) {
        let mut empty_origins = HashSet::new();
        for graph in self.graphs.values() {
            let origin = graph.origin;
            if empty_origins.contains(&origin) {
                continue;
            }
            let state = graph.state.lock().unwrap();
            if state.server.is_empty() {
                empty_origins.insert(origin);
            }
        }

        if empty_origins.is_empty() {
            return;
        }

        self.graphs
            .retain(|_, graph| !empty_origins.contains(&graph.origin));
        self.keys
            .retain(|origin, _| !empty_origins.contains(origin));
        self.untracked_nodes
            .retain(|node| !empty_origins.contains(node));
    }

    /// Selects a single graph for the given [NodeId], considering parent dependencies.
    ///
    /// If multiple graphs are found, they are merged into a single one.
    ///
    /// # Parameters
    /// - `node`: The node ID of the graph to select.
    /// - `parents`: A slice of parent nodes that the graph depends on.
    ///
    /// # Returns
    ///
    /// An `Arc<Graph>` representing the selected or merged graph.
    pub(crate) fn select(&mut self, node: NodeId, parents: &[Parent]) -> Arc<Graph> {
        match self.analyse(node, parents) {
            GraphAnalysis::NoCollision(graph) => {
                if graph.origin != node {
                    self.graphs.insert(node, graph.clone());
                    self.register_key(graph.origin, node);
                }

                graph
            }
            GraphAnalysis::Collisions(graphs) => self.merge(node, graphs),
        }
    }

    /// Analyses the graph for a given node and its parents, returning the associated `GraphAnalysis`.
    fn analyse(&mut self, node: NodeId, parents: &[Parent]) -> GraphAnalysis {
        // If no parents, there is no collision, therefore a single graph is ok.
        if parents.is_empty() {
            let graph = match self.graphs.get(&node) {
                Some(val) => val.clone(),
                None => self.new_graph(node),
            };
            return GraphAnalysis::NoCollision(graph);
        };

        // We collect all graphs of parents and of the current node based on their origin node id.
        let mut graphs = HashMap::<NodeId, Arc<Graph>>::new();

        if let Some(val) = self.graphs.get(&node) {
            graphs.insert(val.origin, val.clone());
        }

        for parent in parents {
            match self.graphs.get(&parent.id) {
                Some(graph) => graphs.insert(graph.origin, graph.clone()),
                None => continue,
            };
        }

        if graphs.is_empty() {
            return match self.graphs.get(&node) {
                Some(old) => GraphAnalysis::NoCollision(old.clone()),
                None => GraphAnalysis::NoCollision(self.new_graph(node)),
            };
        }

        if graphs.len() == 1 {
            return GraphAnalysis::NoCollision(graphs.drain().next().unwrap().1);
        }

        GraphAnalysis::Collisions(graphs)
    }

    /// Merges multiple graphs associated with a node into a single graph.
    fn merge(&mut self, node: NodeId, mut graphs: HashMap<NodeId, Arc<Graph>>) -> Arc<Graph> {
        let mut graphs = graphs.drain().map(|g| g.1);

        let main = graphs.next().expect("At least one graph");
        self.register_key(main.origin, node);

        let mut state = main.state.lock().unwrap();

        for graph in graphs {
            self.merge_two(&mut state, &main, graph);
        }

        self.graphs.insert(main.origin, main.clone());
        self.graphs.insert(node, main.clone());

        core::mem::drop(state);

        main
    }

    /// Registers a key for a given origin node.
    fn register_key(&mut self, origin: NodeId, key: NodeId) {
        if !self.keys.contains_key(&origin) {
            // Ensure an entry exists for this origin
            self.keys.insert(origin, HashSet::new());
        }

        if origin != key {
            // Register this node to point to the origin graph
            self.keys.get_mut(&origin).unwrap().insert(key);
        }
    }

    /// Merges two graphs by combining their states and updating graph mappings.
    fn merge_two(&mut self, main_state: &mut GraphState, main: &Arc<Graph>, merged: Arc<Graph>) {
        let mut locked = merged.state.lock().unwrap();
        let mut state_old = GraphState::default();
        core::mem::swap(&mut state_old, &mut locked);
        main_state.server.extend(state_old.server);

        // Re-map merged origin to the main graph
        self.graphs.insert(merged.origin, main.clone());

        // Move all keys (node IDs) from the merged graph to the main graph
        if let Some(locator_keys) = self.keys.remove(&merged.origin) {
            for k in locator_keys.iter() {
                self.graphs.insert(*k, main.clone());
            }

            let locator_keys_main = self
                .keys
                .get_mut(&main.origin)
                .expect("Should be init before the merge.");
            locator_keys_main.extend(locator_keys);
        }
    }

    /// Creates a new graph for a given node.
    fn new_graph(&mut self, origin: NodeId) -> Arc<Graph> {
        let graph = Arc::new(Graph {
            origin,
            state: Mutex::new(GraphState::default()),
        });
        self.graphs.insert(origin, graph.clone());
        self.keys.insert(origin, HashSet::new());
        graph
    }

    fn mark_untracked(&mut self, node_id: NodeId) {
        self.untracked_nodes.insert(node_id);
    }

    /// Clean up untracked nodes.
    fn cleanup_untracked(&mut self) {
        let mut nodes = core::mem::take(&mut self.untracked_nodes);
        for node in nodes.drain() {
            self.remove_entry(&node);
        }
    }

    fn remove_entry(&mut self, node: &NodeId) {
        if let Some(graph) = self.graphs.remove(node) {
            let mut remove = false;

            if let Some(entry) = self.keys.get_mut(&graph.origin) {
                entry.remove(node);
                if entry.is_empty() {
                    remove = true;
                }
            }

            if remove {
                self.keys.remove(&graph.origin);
            }
        }
    }
}

/// Remove empty graph servers without holding the global locator lock while
/// acquiring an individual graph lock. Local checkpoint backward creates and
/// destroys many short-lived graphs, so lock ordering here must remain
/// strictly independent of the graph server locks.
fn prune_empty_graphs_global() {
    // Do not inspect graph.state here. Burn's mutex stub has no non-blocking
    // lock operation, and nested checkpoint backward can still own one of
    // those states. The locator's key map is the ownership index: once an
    // origin has no key, no live tensor can reach that graph. Remove those
    // graph entries directly without waiting on the graph server mutex.
    let mut state = STATE.lock();
    let Some(locator) = state.as_mut() else {
        return;
    };
    let live_origins = locator.keys.keys().copied().collect::<HashSet<_>>();
    locator
        .graphs
        .retain(|_, graph| live_origins.contains(&graph.origin));
    locator
        .untracked_nodes
        .retain(|node| locator.graphs.contains_key(node));
}

/// Represents the analysis result of graph operations for a given node and its parents.
#[derive(Debug)]
enum GraphAnalysis {
    /// No collision detected, contains the graph associated with the node.
    NoCollision(Arc<Graph>),
    /// Collision detected, contains a map of node IDs to their associated graphs.
    Collisions(HashMap<NodeId, Arc<Graph>>),
}
