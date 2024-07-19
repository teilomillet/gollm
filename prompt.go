package goal

import (
	"github.com/teilomillet/goal/internal/llm"
)

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	*llm.Prompt
}

// NewPrompt creates a new Prompt
func NewPrompt(input string) *Prompt {
	return &Prompt{llm.NewPrompt(input)}
}

// WithDirective adds a directive to the Prompt
func (p *Prompt) WithDirective(directive string) *Prompt {
	p.Prompt.WithDirective(directive)
	return p
}

// WithOutput adds an output specification to the Prompt
func (p *Prompt) WithOutput(output string) *Prompt {
	p.Prompt.WithOutput(output)
	return p
}

// WithInput adds or updates the input of the Prompt
func (p *Prompt) WithInput(input string) *Prompt {
	p.Prompt.Input = input
	return p
}
