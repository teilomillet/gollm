// File: prompt.go

package goal

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"
)

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	Input      string   `json:"input" jsonschema:"required,description=The main input text for the LLM" validate:"required"`
	Output     string   `json:"output,omitempty" jsonschema:"description=Specification for the expected output format"`
	Directives []string `json:"directives,omitempty" jsonschema:"description=List of directives to guide the LLM"`
	Context    string   `json:"context,omitempty" jsonschema:"description=Additional context for the LLM"`
	MaxLength  int      `json:"maxLength,omitempty" jsonschema:"minimum=1,description=Maximum length of the response in words" validate:"omitempty,min=1"`
	Examples   []string `json:"examples,omitempty" jsonschema:"description=List of examples to guide the LLM"`
}

// SchemaOption is a function type for schema generation options
type SchemaOption func(*jsonschema.Reflector)

// WithExpandedStruct sets the ExpandedStruct option for schema generation
func WithExpandedStruct(expanded bool) SchemaOption {
	return func(r *jsonschema.Reflector) {
		r.ExpandedStruct = expanded
	}
}

// GenerateJSONSchema returns the JSON Schema for the given struct
func GenerateJSONSchema(v interface{}, opts ...SchemaOption) ([]byte, error) {
	reflector := &jsonschema.Reflector{}
	for _, opt := range opts {
		opt(reflector)
	}
	schema := reflector.Reflect(v)
	return json.MarshalIndent(schema, "", "  ")
}

// GenerateJSONSchema returns the JSON Schema for the Prompt struct
func (p *Prompt) GenerateJSONSchema(opts ...SchemaOption) ([]byte, error) {
	return GenerateJSONSchema(p, append(opts, WithExpandedStruct(true))...)
}

// Validate checks if the Prompt is valid according to its validation rules
func (p *Prompt) Validate() error {
	return validate.Struct(p)
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
