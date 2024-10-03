// File: prompt.go

package gollm

import (
	"github.com/teilomillet/gollm/llm"
)

// Reuse types from llm package
type (
	CacheType = llm.CacheType
	Message   = llm.PromptMessage
	Function  = llm.Function
	Tool      = llm.Tool
)

// Reuse constants
const (
	CacheTypeEphemeral = llm.CacheTypeEphemeral
)

// Reuse PromptOption type
type PromptOption = llm.PromptOption

// Prompt represents a structured prompt for an LLM
type Prompt struct {
	*llm.Prompt
}

// NewPrompt creates a new Prompt with the given input and applies any provided options
func NewPrompt(input string, opts ...llm.PromptOption) *Prompt {
	p := &Prompt{
		Prompt: llm.NewPrompt(input, opts...),
	}
	return p
}

// ToLLMPrompt converts the gollm.Prompt to an llm.Prompt
func (p *Prompt) ToLLMPrompt() *llm.Prompt {
	return p.Prompt
}

// Wrap and export llm package functions
var (
	CacheOption        = llm.CacheOption
	WithSystemPrompt   = llm.WithSystemPrompt
	WithMessage        = llm.WithMessage
	WithTools          = llm.WithTools
	WithToolChoice     = llm.WithToolChoice
	WithMessages       = llm.WithMessages
	WithDirectives     = llm.WithDirectives
	WithOutput         = llm.WithOutput
	WithContext        = llm.WithContext
	WithMaxLength      = llm.WithMaxLength
	WithExamples       = llm.WithExamples
	WithExpandedStruct = llm.WithExpandedStruct
)

// Prompt methods
func (p *Prompt) Validate() error {
	return p.Prompt.Validate()
}

func (p *Prompt) GenerateJSONSchema(opts ...llm.SchemaOption) ([]byte, error) {
	return p.Prompt.GenerateJSONSchema(opts...)
}

func (p *Prompt) Apply(opts ...PromptOption) {
	p.Prompt.Apply(opts...)
}

func (p *Prompt) String() string {
	return p.Prompt.String()
}

