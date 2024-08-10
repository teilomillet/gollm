// File: internal/opto/graph.go

package opto

import (
	"sync"
)

// Node represents a node in the computational graph
type Node interface {
	ID() string
	Type() string
	Parents() []Node
	Children() []Node
	Value() interface{}
	Parameter() Parameter          // New method
	UpdateValue(interface{}) error // New method
}

// Graph represents the computational graph
type Graph interface {
	AddNode(node Node)
	AddWorkflowNode(node WorkflowNode)
	Nodes() []Node
	WorkflowNodes() []WorkflowNode
	ExtractMinimalSubgraph(start, end Node) Graph
	GetParameterNodes() []Node
}

type node struct {
	id        string
	nodeType  string
	value     interface{}
	parameter Parameter // New field
	parents   []Node
	children  []Node
	mu        sync.RWMutex
}

type graph struct {
	nodes         map[string]Node
	workflowNodes map[string]WorkflowNode
	mu            sync.RWMutex
}

func NewNode(id, nodeType string, value interface{}, parameter Parameter) Node {
	return &node{id: id, nodeType: nodeType, value: value, parameter: parameter}
}

func (n *node) ID() string           { return n.id }
func (n *node) Type() string         { return n.nodeType }
func (n *node) Value() interface{}   { return n.value }
func (n *node) Parameter() Parameter { return n.parameter } // New method

func (n *node) UpdateValue(v interface{}) error { // New method
	n.mu.Lock()
	defer n.mu.Unlock()
	n.value = v
	return nil
}

func (n *node) Parents() []Node {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.parents
}

func (n *node) Children() []Node {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.children
}

func (n *node) AddParent(parent Node) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.parents = append(n.parents, parent)
}

func (n *node) AddChild(child Node) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.children = append(n.children, child)
}

func NewGraph() Graph {
	return &graph{
		nodes:         make(map[string]Node),
		workflowNodes: make(map[string]WorkflowNode),
	}
}

func (g *graph) AddNode(node Node) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nodes[node.ID()] = node
}

func (g *graph) Nodes() []Node {
	g.mu.RLock()
	defer g.mu.RUnlock()
	nodes := make([]Node, 0, len(g.nodes))
	for _, node := range g.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

func (g *graph) ExtractMinimalSubgraph(start, end Node) Graph {
	subgraph := NewGraph()
	visited := make(map[string]bool)

	var dfs func(Node)
	dfs = func(n Node) {
		if visited[n.ID()] {
			return
		}
		visited[n.ID()] = true
		subgraph.AddNode(n)

		if n == end {
			return
		}

		for _, child := range n.Children() {
			dfs(child)
		}
	}

	dfs(start)

	return subgraph
}

func (g *graph) GetParameterNodes() []Node { // New method
	g.mu.RLock()
	defer g.mu.RUnlock()
	var paramNodes []Node
	for _, node := range g.nodes {
		if node.Parameter() != nil {
			paramNodes = append(paramNodes, node)
		}
	}
	return paramNodes
}

// WorkflowNodeType represents different types of nodes in a computational workflow
type WorkflowNodeType int

const (
	InputNode WorkflowNodeType = iota
	OperationNode
	OutputNode
	ParameterNode
)

// WorkflowNode represents a node in a computational workflow
type WorkflowNode interface {
	Node
	NodeType() WorkflowNodeType
	Dependencies() []WorkflowNode
	AddDependency(dep WorkflowNode)
}

type workflowNode struct {
	node
	nodeType     WorkflowNodeType
	dependencies []WorkflowNode
}

func NewWorkflowNode(id string, nodeType WorkflowNodeType, value interface{}, parameter Parameter) WorkflowNode {
	return &workflowNode{
		node:     node{id: id, nodeType: nodeType.String(), value: value, parameter: parameter},
		nodeType: nodeType,
	}
}

// Add String method to WorkflowNodeType
func (wnt WorkflowNodeType) String() string {
	switch wnt {
	case InputNode:
		return "InputNode"
	case OperationNode:
		return "OperationNode"
	case OutputNode:
		return "OutputNode"
	case ParameterNode:
		return "ParameterNode"
	default:
		return "UnknownNode"
	}
}

func (wn *workflowNode) NodeType() WorkflowNodeType {
	return wn.nodeType
}

func (wn *workflowNode) Dependencies() []WorkflowNode {
	return wn.dependencies
}

func (wn *workflowNode) AddDependency(dep WorkflowNode) {
	wn.dependencies = append(wn.dependencies, dep)
	wn.AddParent(dep)
}

func (g *graph) AddWorkflowNode(node WorkflowNode) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.workflowNodes[node.ID()] = node
	g.nodes[node.ID()] = node
}

func (g *graph) WorkflowNodes() []WorkflowNode {
	g.mu.RLock()
	defer g.mu.RUnlock()
	nodes := make([]WorkflowNode, 0, len(g.workflowNodes))
	for _, node := range g.workflowNodes {
		nodes = append(nodes, node)
	}
	return nodes
}
