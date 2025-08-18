// Package gollm provides prompt handling and manipulation functionality for Language Learning Models.
// This file contains type definitions, re-exports, and utility functions for working with prompts
// and their associated components like caching, templates, and message handling.
//
//nolint:gochecknoglobals // These are re-exports to simplify the API surface.
package gollm

import (
	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/providers"
)

// The following types are re-exported from the llm package to provide a cleaner API surface.
// These types form the core components of the prompt system and are essential for
// interacting with Language Learning Models.
type (
	// Prompt represents a complete prompt structure that can be sent to an LLM.
	// It includes system messages, user messages, and optional components like tools and cache settings.
	Prompt = llm.Prompt

	// CacheType defines how prompts should be cached in the system.
	// Different cache types affect the persistence and availability of cached responses.
	CacheType = llm.CacheType

	// PromptMessage represents a single message in a conversation with the LLM.
	// It can be a system message, user message, or assistant message.
	PromptMessage = llm.PromptMessage

	// PromptOption defines a function that can modify a prompt's configuration.
	// These are used to customize prompt behavior in a flexible, chainable way.
	PromptOption = llm.PromptOption

	// SchemaOption defines options for JSON schema generation.
	// These control how prompts are validated against schemas.
	SchemaOption = llm.SchemaOption

	// ToolCall represents a request from the LLM to use a specific tool.
	// It includes the tool name and any arguments needed for execution.
	ToolCall = llm.ToolCall

	// PromptTemplate defines a reusable template for generating prompts.
	// Templates can include variables that are filled in at runtime.
	PromptTemplate = llm.PromptTemplate

	// Response represents the output from an LLM after processing a prompt.
	Response = providers.Response
)

// Cache type constants define the available caching strategies.
const (
	// CacheTypeEphemeral indicates that cached responses should only persist
	// for the duration of the program's execution.
	CacheTypeEphemeral = llm.CacheTypeEphemeral
)

// The following variables are re-exported functions from the llm package.
// They provide the primary means of constructing and customizing prompts.
var (
	// NewPrompt creates a new prompt instance with the given options.
	NewPrompt = llm.NewPrompt

	// CacheOption configures caching behavior for a prompt.
	CacheOption = llm.CacheOption

	// WithSystemPrompt adds a system-level prompt message.
	WithSystemPrompt = llm.WithSystemPrompt

	// WithMessage adds a single message to the prompt.
	WithMessage = llm.WithMessage

	// WithTools configures available tools for the prompt.
	WithTools = llm.WithTools

	// WithToolChoice specifies how tools should be selected.
	WithToolChoice = llm.WithToolChoice

	// WithMessages adds multiple messages to the prompt.
	WithMessages = llm.WithMessages

	// WithDirectives adds special instructions or constraints.
	WithDirectives = llm.WithDirectives

	// WithOutput configures the expected output format.
	WithOutput = llm.WithOutput

	// WithContext adds contextual information to the prompt.
	WithContext = llm.WithContext

	// WithMaxLength sets the maximum length for generated responses.
	WithMaxLength = llm.WithMaxLength

	// WithExamples adds example conversations or outputs.
	WithExamples = llm.WithExamples

	// WithExpandedStruct enables detailed structure expansion.
	WithExpandedStruct = llm.WithExpandedStruct

	// NewPromptTemplate creates a new template for generating prompts.
	NewPromptTemplate = llm.NewPromptTemplate

	// WithPromptOptions adds multiple prompt options at once.
	WithPromptOptions = llm.WithPromptOptions
)

// WithStructuredResponseSchema re-exports the llm generic option while preserving the type parameter.
func WithStructuredResponseSchema[T any]() llm.GenerateOption {
	return llm.WithStructuredResponse[T]()
}

// IsKnownProvider returns true if the given provider name is recognized by the library.
// A provider is considered known if it has a ProviderConfig registered in the default registry.
// Note: This does not guarantee the provider can be instantiated (constructor may be missing).
func IsKnownProvider(name string) bool {
	return providers.IsKnownProvider(name)
}
