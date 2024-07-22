// File: prompt.go

package goal

import (
	"fmt"
	"strings"
)

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	Input      string
	Output     string
	Directives []string
	Context    string
	MaxLength  int
	Examples   []string
}

// PromptOption is a function type that modifies a Prompt
type PromptOption func(*Prompt)

// NewPrompt creates a new Prompt with the given input and applies any provided options
func NewPrompt(input string, opts ...PromptOption) *Prompt {
	p := &Prompt{
		Input: input,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithDirectives adds directives to the Prompt
func WithDirectives(directives ...string) PromptOption {
	return func(p *Prompt) {
		p.Directives = append(p.Directives, directives...)
	}
}

// WithOutput adds an output specification to the Prompt
func WithOutput(output string) PromptOption {
	return func(p *Prompt) {
		p.Output = output
	}
}

// WithContext adds context to the Prompt
func WithContext(context string) PromptOption {
	return func(p *Prompt) {
		p.Context = context
	}
}

// WithMaxLength sets the maximum length for the output
func WithMaxLength(length int) PromptOption {
	return func(p *Prompt) {
		p.MaxLength = length
	}
}

// WithExamples adds examples to the Prompt
func WithExamples(examples ...string) PromptOption {
	return func(p *Prompt) {
		p.Examples = append(p.Examples, examples...)
	}
}

// Apply applies the given options to the Prompt
func (p *Prompt) Apply(opts ...PromptOption) {
	for _, opt := range opts {
		opt(p)
	}
}

// String returns the formatted prompt as a string
func (p *Prompt) String() string {
	var builder strings.Builder

	if p.Context != "" {
		builder.WriteString("Context: ")
		builder.WriteString(p.Context)
		builder.WriteString("\n\n")
	}

	if len(p.Directives) > 0 {
		builder.WriteString("Directives:\n")
		for _, d := range p.Directives {
			builder.WriteString("- ")
			builder.WriteString(d)
			builder.WriteString("\n")
		}
		builder.WriteString("\n")
	}

	builder.WriteString(p.Input)

	if p.Output != "" {
		builder.WriteString("\n\nExpected Output Format:\n")
		builder.WriteString(p.Output)
	}

	if len(p.Examples) > 0 {
		builder.WriteString("\n\nExamples:\n")
		for _, example := range p.Examples {
			builder.WriteString("- ")
			builder.WriteString(example)
			builder.WriteString("\n")
		}
	}

	if p.MaxLength > 0 {
		builder.WriteString(fmt.Sprintf("\n\nPlease limit your response to approximately %d words.", p.MaxLength))
	}

	return builder.String()
}

