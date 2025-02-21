// Package presets provides a collection of high-level tools and utilities for
// enhancing Language Learning Model interactions with specific reasoning patterns
// and problem-solving strategies.
package presets

import (
	"context"
	"fmt"
	"unicode/utf8"

	"github.com/mauza/gollm"
)

// chainOfThoughtTemplate defines a structured prompt template for guiding
// the LLM through explicit step-by-step reasoning. It encourages the model
// to break down complex problems and show its reasoning process.
//
// The template includes:
// - Problem breakdown directive
// - Explicit reasoning requirement
// - Structured output format
//
// Example generated prompt:
//
//	Perform a chain of thought reasoning for the following question:
//
//	What is the result of (17 * 6) + (23 * 4)?
//
//	Directives:
//	- Break down the problem into steps
//	- Show your reasoning for each step
//
//	Chain of Thought:
var chainOfThoughtTemplate = gollm.NewPromptTemplate(
	"ChainOfThought",
	"Perform a chain of thought reasoning",
	"Perform a chain of thought reasoning for the following question:\n\n{{.Question}}\n\nPlease number each step (1., 2., etc.) in your response.",
	gollm.WithPromptOptions(
		gollm.WithDirectives(
			"Break down the problem into steps",
			"Show your reasoning for each step",
			"Number each step (1., 2., etc.)",
		),
		gollm.WithOutput("Chain of Thought:"),
	),
)

// ChainOfThought performs chain of thought reasoning on a given question.
// This approach encourages the LLM to break down complex problems into
// smaller steps and explicitly show its reasoning process, leading to
// more reliable and explainable responses.
//
// The function supports various prompt enhancement options:
//   - WithMaxLength: Control response length
//   - WithContext: Add relevant background information
//   - WithExamples: Provide guiding examples
//   - WithDirectives: Add specific reasoning instructions
//   - WithTemperature: Adjust response creativity
//   - WithTopP: Control response diversity
//
// The function:
// 1. Creates a structured prompt using the chain of thought template
// 2. Applies any additional prompt options
// 3. Generates a response showing step-by-step reasoning
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - l: LLM instance to use for generation
//   - question: The question or problem to reason about
//   - opts: Optional prompt configuration options
//
// Returns:
//   - string: The generated chain of thought response
//   - error: Any error encountered during generation
//
// Example usage for mathematical reasoning:
//
//	llm := gollm.NewLLM(...)
//	response, err := ChainOfThought(ctx, llm,
//	    "What is the result of (17 * 6) + (23 * 4)?",
//	    gollm.WithMaxLength(200),
//	)
//
// Example usage for complex analysis:
//
//	llm := gollm.NewLLM(...)
//	response, err := ChainOfThought(ctx, llm,
//	    "How might climate change affect global agriculture?",
//	    gollm.WithMaxLength(300),
//	    gollm.WithContext("Climate change is causing global temperature increases and changing precipitation patterns."),
//	    gollm.WithExamples("Effect: Shifting growing seasons, Adaptation: Developing heat-resistant crops"),
//	    gollm.WithDirectives(
//	        "Break down the problem into steps",
//	        "Show your reasoning for each step",
//	    ),
//	)
//
// The response will provide structured reasoning like:
//
//	Let me break this down:
//	1. Temperature Effects:
//	   - Rising global temperatures affect crop growth cycles
//	   - Some regions become too hot for traditional crops
//	   - New growing zones emerge in previously cold areas
//
//	2. Precipitation Changes:
//	   - Altered rainfall patterns impact irrigation needs
//	   - More frequent droughts in some regions
//	   - Increased flooding in other areas
//
//	3. Adaptation Requirements:
//	   - Development of heat-resistant crop varieties
//	   - Implementation of water-efficient farming methods
//	   - Shifts in planting and harvesting schedules
func ChainOfThought(ctx context.Context, l gollm.LLM, question string, opts ...gollm.PromptOption) (string, error) {
	if ctx == nil {
		return "", fmt.Errorf("context cannot be nil")
	}

	if l == nil {
		return "", fmt.Errorf("LLM instance cannot be nil")
	}

	if question == "" {
		return "", fmt.Errorf("question cannot be empty")
	}

	// Validate UTF-8 encoding
	if !utf8.ValidString(question) {
		return "", fmt.Errorf("question contains invalid UTF-8 characters")
	}

	prompt, err := chainOfThoughtTemplate.Execute(map[string]interface{}{
		"Question": question,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute chain of thought template: %w", err)
	}
	prompt.Apply(opts...)
	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}
