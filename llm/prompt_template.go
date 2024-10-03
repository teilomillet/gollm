package llm

import (
	"bytes"
	"text/template"
)

// PromptTemplate represents a template for generating prompts
type PromptTemplate struct {
	Name        string
	Description string
	Template    string
	Options     []PromptOption
}

// PromptTemplateOption is a function type that modifies a PromptTemplate
type PromptTemplateOption func(*PromptTemplate)

// NewPromptTemplate creates a new PromptTemplate with the given name, description, and template
func NewPromptTemplate(name, description, template string, opts ...PromptTemplateOption) *PromptTemplate {
	pt := &PromptTemplate{
		Name:        name,
		Description: description,
		Template:    template,
	}
	for _, opt := range opts {
		opt(pt)
	}
	return pt
}

// WithPromptOptions adds PromptOptions to the PromptTemplate
func WithPromptOptions(options ...PromptOption) PromptTemplateOption {
	return func(pt *PromptTemplate) {
		pt.Options = append(pt.Options, options...)
	}
}

// Execute generates a Prompt from the PromptTemplate with the given data
func (pt *PromptTemplate) Execute(data map[string]interface{}) (*Prompt, error) {
	tmpl, err := template.New(pt.Name).Parse(pt.Template)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, err
	}

	prompt := NewPrompt(buf.String())
	prompt.Apply(pt.Options...)

	return prompt, nil
}
