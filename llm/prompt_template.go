// Package llm provides a unified interface for interacting with various Language Learning Model providers.
package llm

import (
	"bytes"
	"fmt"
	"text/template"
)

// PromptTemplate represents a reusable template for generating prompts dynamically.
// It provides a structured way to create consistent prompt patterns that can be
// filled with different values at runtime.
//
// Example:
//
//	template := NewPromptTemplate(
//	    "translator",
//	    "Translates text to a target language",
//	    "Translate the following text to {{.language}}:\n{{.text}}",
//	    WithPromptOptions(WithMaxLength(100)),
//	)
//
//	prompt, err := template.Execute(map[string]any{
//	    "language": "French",
//	    "text": "Hello, world!",
//	})
type PromptTemplate struct {
	Name        string         // Unique identifier for the template
	Description string         // Human-readable description of the template's purpose
	Template    string         // Go template string for generating prompts
	Options     []PromptOption // Configuration options for generated prompts
}

// PromptTemplateOption is a function type that modifies a PromptTemplate.
// It follows the functional options pattern for flexible configuration.
//
// Example:
//
//	func WithCustomOption(value string) PromptTemplateOption {
//	    return func(pt *PromptTemplate) {
//	        pt.Description = value
//	    }
//	}
type PromptTemplateOption func(*PromptTemplate)

// NewPromptTemplate creates a new PromptTemplate with the given parameters.
//
// Parameters:
//   - name: Unique identifier for the template
//   - description: Human-readable description of the template's purpose
//   - template: Go template string for generating prompts
//   - opts: Optional configuration functions
//
// Returns:
//   - Configured PromptTemplate instance
//
// Example:
//
//	template := NewPromptTemplate(
//	    "qa",
//	    "Answers questions about a given topic",
//	    "Answer this question: {{.question}}\nContext: {{.context}}",
//	    WithPromptOptions(WithMaxLength(200)),
//	)
func NewPromptTemplate(name, description, tmpl string, opts ...PromptTemplateOption) *PromptTemplate {
	pt := &PromptTemplate{
		Name:        name,
		Description: description,
		Template:    tmpl,
	}
	for _, opt := range opts {
		opt(pt)
	}
	return pt
}

// WithPromptOptions adds PromptOptions to the PromptTemplate.
// These options will be applied to every prompt generated from this template.
//
// Parameters:
//   - options: Variadic list of PromptOption to apply
//
// Returns:
//   - PromptTemplateOption function that can be passed to NewPromptTemplate
//
// Example:
//
//	template := NewPromptTemplate(
//	    "summarizer",
//	    "Summarizes text",
//	    "Summarize: {{.text}}",
//	    WithPromptOptions(
//	        WithMaxLength(100),
//	        WithSystemPrompt("You are a concise summarizer", CacheTypeEphemeral),
//	    ),
//	)
func WithPromptOptions(options ...PromptOption) PromptTemplateOption {
	return func(pt *PromptTemplate) {
		pt.Options = append(pt.Options, options...)
	}
}

// Execute generates a Prompt from the PromptTemplate with the given data.
// It applies the template's options to the generated prompt and validates
// the result.
//
// Parameters:
//   - data: Map of key-value pairs to substitute in the template
//
// Returns:
//   - Generated and configured Prompt instance
//   - Error if template parsing, execution, or validation fails
//
// Example:
//
//	prompt, err := template.Execute(map[string]any{
//	    "text": "Long article to summarize...",
//	    "maxWords": 50,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
func (pt *PromptTemplate) Execute(data map[string]any) (*Prompt, error) {
	tmpl, err := template.New(pt.Name).Parse(pt.Template)
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, fmt.Errorf("failed to execute template: %w", err)
	}

	prompt := NewPrompt(buf.String())
	prompt.Apply(pt.Options...)

	return prompt, nil
}
