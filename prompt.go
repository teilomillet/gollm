// File: gollm/prompt.go
package gollm

import (
	"github.com/teilomillet/gollm/llm"
	"strings"
)

// Re-export types from llm package
type (
	Prompt         = llm.Prompt
	CacheType      = llm.CacheType
	PromptMessage  = llm.PromptMessage
	Function       = llm.Function
	Tool           = llm.Tool
	PromptOption   = llm.PromptOption
	SchemaOption   = llm.SchemaOption
	ToolCall       = llm.ToolCall
	MemoryMessage  = llm.MemoryMessage
	PromptTemplate = llm.PromptTemplate
)

// Re-export constants
const (
	CacheTypeEphemeral = llm.CacheTypeEphemeral
)

// Re-export functions
var (
	NewPrompt          = llm.NewPrompt
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
	NewPromptTemplate  = llm.NewPromptTemplate
	WithPromptOptions  = llm.WithPromptOptions
)

// GenerateOption is a function type for configuring generate options
type GenerateOption func(*GenerateConfig)

// GenerateConfig holds configuration options for the Generate method
type GenerateConfig struct {
	UseJSONSchema bool
}

// WithJSONSchemaValidation returns a GenerateOption that enables JSON schema validation
func WithJSONSchemaValidation() GenerateOption {
	return func(c *GenerateConfig) {
		c.UseJSONSchema = true
	}
}

// CleanResponse removes markdown code block syntax and trims the JSON response
func CleanResponse(response string) string {
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}
	return strings.TrimSpace(response)
}
