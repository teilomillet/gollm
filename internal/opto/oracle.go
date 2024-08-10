// File: internal/opto/oracle.go

package opto

import (
	"context"
	"fmt"
	"reflect"
)

// TraceOracle encapsulates the core OPTO functionality
type TraceOracle interface {
	Execute(ctx context.Context, params map[string]Parameter) (Graph, *Feedback, error)
}

type traceOracle struct {
	executor          func(context.Context, string) (interface{}, error)
	graphBuilder      func(string, interface{}) Graph
	feedbackGenerator func(interface{}) *Feedback
}

func NewTraceOracle(
	executor func(context.Context, string) (interface{}, error),
	graphBuilder func(string, interface{}) Graph,
	feedbackGenerator func(interface{}) *Feedback,
) TraceOracle {
	return &traceOracle{
		executor:          executor,
		graphBuilder:      graphBuilder,
		feedbackGenerator: feedbackGenerator,
	}
}

func (to *traceOracle) Execute(ctx context.Context, params map[string]Parameter) (Graph, *Feedback, error) {
	// Assume we're dealing with a single prompt parameter for simplicity
	var promptText string
	for _, param := range params {
		if promptParam, ok := param.(*PromptParameter); ok {
			promptText = promptParam.GetPromptText()
			break
		}
	}

	if promptText == "" {
		return nil, nil, fmt.Errorf("no valid prompt parameter found")
	}

	result, err := to.executor(ctx, promptText)
	if err != nil {
		return nil, nil, fmt.Errorf("execution failed: %w", err)
	}

	graph := to.graphBuilder(promptText, result)
	feedback := to.feedbackGenerator(result)

	return graph, feedback, nil
}

// NewDefaultGraphBuilder creates a default graph builder function for prompt-based workflows
func NewDefaultGraphBuilder() func(string, interface{}) Graph {
	return func(prompt string, result interface{}) Graph {
		g := NewGraph()

		// Create nodes
		promptNode := NewWorkflowNode("prompt", InputNode, prompt, nil)
		resultNode := NewWorkflowNode("result", OutputNode, result, nil)

		// Add nodes to the graph
		g.AddWorkflowNode(promptNode)
		g.AddWorkflowNode(resultNode)

		// Connect nodes
		resultNode.AddDependency(promptNode)

		return g
	}
}

// NewDefaultFeedbackGenerator creates a default feedback generator function
func NewDefaultFeedbackGenerator() func(interface{}) *Feedback {
	return func(result interface{}) *Feedback {
		// This is a simple placeholder. In a real implementation,
		// you would analyze the result and generate appropriate feedback.
		return NewFeedback(0.5, "Default feedback based on execution result")
	}
}

func buildGraph(g Graph, v reflect.Value, name string, parent WorkflowNode) WorkflowNode {
	if !v.IsValid() {
		return nil
	}

	var newNode WorkflowNode
	nodeType := OperationNode // Default node type

	switch v.Kind() {
	case reflect.Ptr, reflect.Interface:
		if v.IsNil() {
			return nil
		}
		return buildGraph(g, v.Elem(), name, parent)
	case reflect.Struct:
		newNode = NewWorkflowNode(name, nodeType, v.Interface(), nil)
		g.AddWorkflowNode(newNode)
		if parent != nil {
			newNode.AddDependency(parent)
		}
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			fieldName := v.Type().Field(i).Name
			childNode := buildGraph(g, field, fieldName, newNode)
			if childNode != nil {
				newNode.AddDependency(childNode)
			}
		}
	case reflect.Slice, reflect.Array:
		newNode = NewWorkflowNode(name, nodeType, v.Interface(), nil)
		g.AddWorkflowNode(newNode)
		if parent != nil {
			newNode.AddDependency(parent)
		}
		for i := 0; i < v.Len(); i++ {
			childNode := buildGraph(g, v.Index(i), fmt.Sprintf("%s[%d]", name, i), newNode)
			if childNode != nil {
				newNode.AddDependency(childNode)
			}
		}
	case reflect.Map:
		newNode = NewWorkflowNode(name, nodeType, v.Interface(), nil)
		g.AddWorkflowNode(newNode)
		if parent != nil {
			newNode.AddDependency(parent)
		}
		for _, key := range v.MapKeys() {
			childNode := buildGraph(g, v.MapIndex(key), fmt.Sprintf("%s[%v]", name, key.Interface()), newNode)
			if childNode != nil {
				newNode.AddDependency(childNode)
			}
		}
	default:
		newNode = NewWorkflowNode(name, nodeType, v.Interface(), nil)
		g.AddWorkflowNode(newNode)
		if parent != nil {
			newNode.AddDependency(parent)
		}
	}

	// Check if the newNode represents a Parameter
	if v.Type().Implements(reflect.TypeOf((*Parameter)(nil)).Elem()) {
		newNode = NewWorkflowNode(name, ParameterNode, v.Interface(), v.Interface().(Parameter))
		g.AddWorkflowNode(newNode)
	}

	return newNode
}
