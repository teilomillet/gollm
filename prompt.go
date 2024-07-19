package goal

import (
	"fmt"
	"strings"

	"github.com/teilomillet/goal/internal/llm"
)

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	*llm.Prompt
	examples  []string
	maxLength *int
	context   string
}

// NewPrompt creates a new Prompt
func NewPrompt(input string) *Prompt {
	return &Prompt{Prompt: llm.NewPrompt(input)}
}

// Directive adds a directive to the Prompt
func (p *Prompt) Directive(directive string) *Prompt {
	p.Prompt.WithDirective(directive)
	return p
}

// Output adds an output specification to the Prompt
func (p *Prompt) Output(output string) *Prompt {
	p.Prompt.WithOutput(output)
	return p
}

// Input adds or updates the input of the Prompt
func (p *Prompt) Input(input string) *Prompt {
	p.Prompt.Input = input
	return p
}

// Examples adds examples to the Prompt
func (p *Prompt) Examples(filePath string, n int, order string) *Prompt {
	examples, err := readExamplesFromFile(filePath)
	if err != nil {
		// Log the error and return the prompt without examples
		// You might want to use your logging system here
		return p
	}

	p.examples = selectExamples(examples, n, order)
	return p
}

// MaxLength sets the maximum length for the output
func (p *Prompt) MaxLength(length int) *Prompt {
	p.maxLength = &length
	return p
}

// Context adds context to the Prompt
func (p *Prompt) Context(context string) *Prompt {
	p.context = context
	return p
}

// String returns the formatted prompt as a string
func (p *Prompt) String() string {
	var builder strings.Builder

	if p.context != "" {
		builder.WriteString("Context: ")
		builder.WriteString(p.context)
		builder.WriteString("\n\n")
	}

	builder.WriteString(p.Prompt.String())

	if p.maxLength != nil {
		builder.WriteString(fmt.Sprintf("\nPlease limit your response to approximately %d words.", *p.maxLength))
	}

	if len(p.examples) > 0 {
		builder.WriteString("\nExamples:\n")
		for _, example := range p.examples {
			builder.WriteString("- ")
			builder.WriteString(example)
			builder.WriteString("\n")
		}
	}

	return builder.String()
}
