// Package gollm provides a high-level interface for interacting with Language Learning Models (LLMs).
package gollm

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// MOAConfig represents the configuration for the Mixture of Agents (MOA) system.
// MOA is an ensemble learning approach that combines multiple language models
// to produce higher quality outputs through iterative refinement and aggregation.
type MOAConfig struct {
	// Iterations specifies the number of refinement cycles to perform
	Iterations int

	// Models defines the set of language models to use in the mixture
	// Each model is configured using ConfigOption functions
	Models []ConfigOption

	// MaxParallel limits the number of concurrent model executions
	// If set to 0, all models run concurrently without limitation
	MaxParallel int

	// AgentTimeout specifies the maximum duration for each agent's execution
	// If set to 0, no timeout is applied
	AgentTimeout time.Duration
}

// MOALayer represents a single layer in the Mixture of Agents architecture.
// Each layer contains one or more language models that process inputs in parallel.
type MOALayer struct {
	// Models contains the LLM instances that operate within this layer
	Models []llm.LLM
}

// MOA (Mixture of Agents) implements an ensemble learning system that combines
// multiple language models to produce higher quality outputs. It processes inputs
// through multiple layers and iterations, then aggregates the results using a
// designated aggregator model.
type MOA struct {
	// Config holds the MOA system configuration
	Config MOAConfig

	// Layers contains the processing layers, each with one or more models
	Layers []MOALayer

	// Aggregator is the LLM instance used to synthesize outputs from multiple models
	Aggregator llm.LLM
}

// NewMOA creates a new Mixture of Agents instance with the specified configuration
// and aggregator options.
//
// Parameters:
//   - moaConfig: Configuration for the MOA system
//   - aggregatorOpts: Configuration options for the aggregator model
//
// Returns:
//   - A configured MOA instance
//   - Any error encountered during initialization
//
// Example:
//
//	moa, err := NewMOA(MOAConfig{
//	    Iterations:   3,
//	    Models:      []ConfigOption{SetModel("gpt-4"), SetModel("claude-3")},
//	    MaxParallel: 2,
//	})
func NewMOA(moaConfig MOAConfig, aggregatorOpts ...ConfigOption) (*MOA, error) {
	if len(moaConfig.Models) == 0 {
		return nil, fmt.Errorf("invalid model configuration: at least one model must be specified")
	}

	registry := providers.NewProviderRegistry()
	logger := utils.NewLogger(utils.LogLevelInfo)

	moa := &MOA{
		Config: moaConfig,
		Layers: make([]MOALayer, len(moaConfig.Models)),
	}

	// Initialize each layer with its corresponding model
	for i, modelOpt := range moaConfig.Models {
		cfg := &config.Config{}
		modelOpt(cfg)
		llmInstance, err := llm.NewLLM(cfg, logger, registry)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM for model %d: %w", i, err)
		}
		moa.Layers[i] = MOALayer{
			Models: []llm.LLM{llmInstance},
		}
	}

	// Create the aggregator LLM
	aggregatorConfig := &config.Config{}
	for _, opt := range aggregatorOpts {
		opt(aggregatorConfig)
	}
	aggregator, err := llm.NewLLM(aggregatorConfig, logger, registry)
	if err != nil {
		return nil, fmt.Errorf("failed to create aggregator LLM: %w", err)
	}
	moa.Aggregator = aggregator

	return moa, nil
}

// Generate processes the input through the Mixture of Agents and returns the final output.
// It runs the input through multiple iterations of processing layers, then aggregates
// the results using the designated aggregator model.
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - input: The text input to process
//
// Returns:
//   - The final processed output
//   - Any error encountered during processing
//
// The processing flow is:
// 1. Input is processed through each layer in sequence
// 2. Each layer's output becomes the input for the next layer
// 3. This process repeats for the specified number of iterations
// 4. Results from all iterations are aggregated into the final output
func (moa *MOA) Generate(ctx context.Context, input string) (string, error) {
	var layerOutputs []string

	// Process through each iteration
	for i := 0; i < moa.Config.Iterations; i++ {
		layerInput := input
		// Process each layer
		for _, layer := range moa.Layers {
			layerOutput, err := moa.processLayer(ctx, layer, layerInput)
			if err != nil {
				return "", fmt.Errorf("error processing layer: %w", err)
			}
			layerInput = layerOutput
		}
		layerOutputs = append(layerOutputs, layerInput)
	}

	// Aggregate the results from all iterations
	return moa.aggregate(ctx, layerOutputs)
}

// processLayer handles the processing of a single layer, potentially in parallel.
// It distributes the input to all models in the layer and collects their outputs.
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - layer: The MOALayer containing models to process the input
//   - input: The text input to process
//
// Returns:
//   - Combined output from all models in the layer
//   - Any error encountered during processing
//
// Features:
//   - Supports parallel processing with configurable concurrency limits
//   - Implements per-agent timeouts when configured
//   - Combines outputs from all models in the layer
func (moa *MOA) processLayer(ctx context.Context, layer MOALayer, input string) (string, error) {
	results := make([]string, len(layer.Models))
	errors := make([]error, len(layer.Models))

	// Create a worker pool if parallel processing is enabled
	var wg sync.WaitGroup
	var workerPool chan struct{}
	if moa.Config.MaxParallel > 0 {
		workerPool = make(chan struct{}, moa.Config.MaxParallel)
	}

	for i, model := range layer.Models {
		wg.Add(1)
		go func(index int, llmInstance llm.LLM) {
			defer wg.Done()
			if workerPool != nil {
				workerPool <- struct{}{}        // Acquire a worker
				defer func() { <-workerPool }() // Release the worker
			}

			// Create a context with timeout if AgentTimeout is set
			var cancel context.CancelFunc
			if moa.Config.AgentTimeout > 0 {
				ctx, cancel = context.WithTimeout(ctx, moa.Config.AgentTimeout)
				defer cancel()
			}

			output, err := llmInstance.Generate(ctx, llm.NewPrompt(input))
			if err != nil {
				errors[index] = err
				return
			}
			results[index] = output
		}(i, model)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return "", fmt.Errorf("error in layer processing: %w", err)
		}
	}

	return moa.combineResults(results), nil
}

// combineResults merges the results from multiple models in a layer into a single string.
// Each model's output is separated by a delimiter for clear distinction.
//
// Parameters:
//   - results: Slice of output strings from individual models
//
// Returns:
//   - A combined string containing all model outputs
func (moa *MOA) combineResults(results []string) string {
	combined := ""
	for _, result := range results {
		combined += result + "\n---\n"
	}
	return combined
}

// aggregate uses the aggregator LLM to synthesize the final output from multiple iterations.
// It combines all iteration outputs and prompts the aggregator to create a cohesive response.
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - outputs: Slice of outputs from different iterations
//
// Returns:
//   - The final synthesized output
//   - Any error encountered during aggregation
func (moa *MOA) aggregate(ctx context.Context, outputs []string) (string, error) {
	aggregationPrompt := fmt.Sprintf("Synthesise these responses into a single, high-quality response:\n\n%s", moa.combineResults(outputs))
	response, err := moa.Aggregator.Generate(ctx, llm.NewPrompt(aggregationPrompt))

	return response, err
}
