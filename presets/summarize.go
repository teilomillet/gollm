// Package presets provides utilities for enhancing Language Learning Model interactions
// with specific reasoning patterns and text processing capabilities.
package presets

import (
	"context"
	"errors"
	"fmt"

	"github.com/weave-labs/gollm"
)

// Summarize generates a concise summary of the provided text while preserving
// key information and main points. It uses a templated prompt to guide the LLM
// in producing consistent, high-quality summaries.
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - l: LLM instance to use for summarization
//   - text: The text to be summarized
//   - opts: Optional prompt configuration options
//
// Returns:
//   - string: The generated summary
//   - error: Any error encountered during summarization
//
// Example usage with basic options:
//
//	text := `Artificial intelligence (AI) is transforming various sectors of society,
//	         including healthcare, finance, and transportation. While AI offers numerous
//	         benefits such as improved efficiency and decision-making, it also raises
//	         concerns about privacy, job displacement, and ethical considerations.`
//
//	summary, err := Summarize(ctx, llm, text,
//	    gollm.WithMaxLength(50),
//	    gollm.WithDirectives(
//	        "Provide a concise summary",
//	        "Focus on main impacts",
//	    ),
//	)
//
// Example usage with advanced options:
//
//	text := `[Long technical paper or article...]`
//
//	summary, err := Summarize(ctx, llm, text,
//	    gollm.WithMaxLength(200),
//	    gollm.WithTemperature(0.3),
//	    gollm.WithDirectives(
//	        "Maintain technical accuracy",
//	        "Include key findings and methodology",
//	        "Preserve important statistics and data",
//	        "Structure the summary with clear sections",
//	    ),
//	    gollm.WithOutput("Technical Summary:"),
//	)
//
// Common use cases:
//   - Article summarization
//   - Document condensation
//   - Meeting minutes generation
//   - Research paper abstracts
//   - News briefing creation
//
// Customization options:
//   - WithMaxLength: Control summary length
//   - WithTemperature: Adjust creativity vs. precision
//   - WithDirectives: Guide summarization focus
//   - WithOutput: Customize output format
//
// Best practices:
//  1. Set appropriate length limits based on source text
//  2. Use lower temperature for factual summaries
//  3. Add specific directives for focus areas
//  4. Consider audience when setting style
//
// The function handles:
//   - Context management
//   - Template execution
//   - Error propagation
//   - Response generation
func Summarize(ctx context.Context, l gollm.LLM, text string, opts ...gollm.PromptOption) (string, error) {
	summarizeTemplate := gollm.NewPromptTemplate(
		"Summarize",
		"Summarize the given text",
		"Summarize the following text:\n\n{{.Text}}",
		gollm.WithPromptOptions(
			gollm.WithDirectives(
				"Provide a concise summary",
				"Capture the main points and key details",
			),
			gollm.WithOutput("Summary:"),
		),
	)

	if ctx == nil {
		ctx = context.Background()
	}
	if l == nil {
		return "", errors.New("LLM instance cannot be nil")
	}

	prompt, err := summarizeTemplate.Execute(map[string]any{
		"Text": text,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute summarize template: %w", err)
	}
	prompt.Apply(opts...)
	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response.AsText(), nil
}
