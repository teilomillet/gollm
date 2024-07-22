package goal

import (
	"bytes"
	"text/template"
)

// PromptTemplate represents a template for generating prompts
type PromptTemplate struct {
	Name        string
	Description string
	Template    string
	Directives  []string
	Output      string
	Options     []PromptOption
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

	prompt := NewPrompt(buf.String(), pt.Options...)

	return prompt, nil
}
