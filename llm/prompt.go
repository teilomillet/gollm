package llm

import (
	"go.uber.org/zap"
	"strings"
)

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	Input      string
	Output     string
	Directives []string
}

// NewPrompt creates a new Prompt
func NewPrompt(input string) *Prompt {
	Logger.Debug("Creating new prompt", zap.String("input", input))
	return &Prompt{
		Input: input,
	}
}

// WithOutput adds an output specification to the Prompt
func (p *Prompt) WithOutput(output string) *Prompt {
	Logger.Debug("Adding output to prompt", zap.String("output", output))
	p.Output = output
	return p
}

// WithDirective adds a directive to the Prompt
func (p *Prompt) WithDirective(directive string) *Prompt {
	Logger.Debug("Adding directive to prompt", zap.String("directive", directive))
	p.Directives = append(p.Directives, directive)
	return p
}

// String returns the formatted prompt as a string
func (p *Prompt) String() string {
	var sb strings.Builder

	if len(p.Directives) > 0 {
		sb.WriteString("Directives:\n")
		for _, d := range p.Directives {
			sb.WriteString("- ")
			sb.WriteString(d)
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}

	sb.WriteString(p.Input)

	if p.Output != "" {
		sb.WriteString("\n\n")
		sb.WriteString(p.Output)
	}

	fullPrompt := sb.String()
	Logger.Debug("Generated full prompt", zap.String("prompt", fullPrompt))
	return fullPrompt
}

