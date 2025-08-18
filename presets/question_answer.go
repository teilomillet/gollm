// Package presets provides utilities for enhancing Language Learning Model interactions
// with specific reasoning patterns and question-answering capabilities.
package presets

import (
	"context"
	"fmt"

	"github.com/weave-labs/gollm"
)

// QuestionAnswer performs question answering with support for context, examples,
// and custom directives. It enhances answer quality by allowing additional
// context and guidance through prompt options.
//
// The function supports various prompt enhancement options:
//   - WithContext: Add relevant background information
//   - WithExamples: Provide example Q&A pairs
//   - WithMaxLength: Control answer length
//   - WithDirectives: Add specific answering instructions
//   - WithTemperature: Adjust answer creativity
//   - WithTopP: Control response diversity
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - l: LLM instance to use for generation
//   - question: The question to be answered
//   - opts: Optional prompt configuration options
//
// Returns:
//   - string: The generated answer
//   - error: Any error encountered during generation
//
// Example usage with basic question:
//
//	llm := gollm.NewLLM(...)
//	answer, err := QuestionAnswer(ctx, llm,
//	    "What is the capital of France?",
//	    gollm.WithMaxLength(100),
//	)
//
// Example usage with context and examples:
//
//	contextInfo := `Quantum computing is an emerging field that uses
//	quantum-mechanical phenomena such as superposition and entanglement
//	to perform computation. It has the potential to solve certain problems
//	much faster than classical computers.`
//
//	answer, err := QuestionAnswer(ctx, llm,
//	    "What are the main challenges in quantum computing?",
//	    gollm.WithContext(contextInfo),
//	    gollm.WithExamples("Challenge: Decoherence, Solution: Error correction techniques"),
//	    gollm.WithMaxLength(200),
//	    gollm.WithDirectives(
//	        "Provide a concise answer",
//	        "Address the main challenges mentioned in the question",
//	    ),
//	)
//
// Example response:
//
//	The main challenges in quantum computing include:
//
//	1. Decoherence: Quantum states are extremely fragile and can collapse
//	   due to environmental interactions. This requires sophisticated error
//	   correction techniques.
//
//	2. Scalability: Building large-scale quantum computers while maintaining
//	   coherence is technically challenging.
//
//	3. Error Rates: Current quantum gates have relatively high error rates,
//	   making reliable computations difficult.
//
//	4. Cost and Complexity: Quantum computers require extremely precise
//	   control systems and specialized operating conditions.
func QuestionAnswer(ctx context.Context, l gollm.LLM, question string, opts ...gollm.PromptOption) (string, error) {
	questionAnswerTemplate := gollm.NewPromptTemplate(
		"QuestionAnswer",
		"Answer the given question",
		"Answer the following question:\n\n{{.Question}}",
		gollm.WithPromptOptions(
			gollm.WithDirectives("Provide a clear and concise answer"),
			gollm.WithOutput("Answer:"),
		),
	)

	if ctx == nil {
		ctx = context.Background()
	}
	prompt, err := questionAnswerTemplate.Execute(map[string]any{
		"Question": question,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute question answer template: %w", err)
	}
	prompt.Apply(opts...)
	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response.AsText(), nil
}
