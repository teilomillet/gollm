package llm

import (
	"encoding/json"
	"fmt"

	"strings"

	"github.com/invopop/jsonschema"

	"github.com/weave-labs/gollm/internal/models"
	"github.com/weave-labs/gollm/providers"
)

// CacheType defines how prompts and responses should be cached in the system.
type CacheType string

const (
	// CacheTypeEphemeral indicates that cached responses should only persist
	// for the duration of the program's execution.
	CacheTypeEphemeral CacheType = "ephemeral"
)

// PromptMessage represents a single message in a conversation with an LLM.
// It can be a system message, user message, or assistant message, and may include
// tool calls and caching configuration.
type PromptMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	CacheType  CacheType  `json:"cache_type,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

func (pm *PromptMessage) ToMessage() providers.Message {
	msg := providers.Message{
		Role:    pm.Role,
		Content: pm.Content,
		Name:    pm.Name,
	}
	if len(pm.ToolCalls) > 0 {
		var toolCalls []providers.ToolCall
		for _, tc := range pm.ToolCalls {
			toolCalls = append(toolCalls, providers.ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: providers.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		msg.ToolCalls = toolCalls
	}
	return msg
}

func ToMessages(pm []PromptMessage) []providers.Message {
	messages := make([]providers.Message, 0, len(pm))
	for i := range pm {
		messages = append(messages, pm[i].ToMessage())
	}
	return messages
}

// ToolCall represents a request from the LLM to use a specific tool.
// It includes the tool's identifier, type, and any arguments needed for execution.
type ToolCall struct {
	ID       string `json:"id"`   // Unique identifier for the tool call
	Type     string `json:"type"` // Type of tool being called
	Function struct {
		Name      string          `json:"name"`      // Name of the function to call
		Arguments json.RawMessage `json:"arguments"` // Arguments for the function call
	} `json:"function"`
}

// Prompt represents a complete prompt structure that can be sent to an LLM.
// It includes various components like system messages, user input, context,
// and optional elements like tools and examples.
type Prompt struct {
	ToolChoice      map[string]any  `json:"tool_choice,omitempty" jsonschema:"description=Configuration for tool selection behavior"`
	Input           string          `json:"input" jsonschema:"required,description=The main input text for the LLM" validate:"required"`
	Output          string          `json:"output,omitempty" jsonschema:"description=Specification for the expected output format"`
	Context         string          `json:"context,omitempty" jsonschema:"description=Additional context for the LLM"`
	SystemPrompt    string          `json:"system_prompt,omitempty" jsonschema:"description=System prompt for the LLM"`
	SystemCacheType CacheType       `json:"system_cache_type,omitempty" jsonschema:"description=Cache type for the system prompt"`
	Directives      []string        `json:"directives,omitempty" jsonschema:"description=List of directives to guide the LLM"`
	Examples        []string        `json:"examples,omitempty" jsonschema:"description=List of examples to guide the LLM"`
	Messages        []PromptMessage `json:"messages,omitempty" jsonschema:"description=List of messages for the conversation"`
	Tools           []models.Tool   `json:"tools,omitempty" jsonschema:"description=Available tools for the LLM to use"`
	MaxLength       int             `json:"max_length,omitempty" jsonschema:"minimum=1,description=Maximum length of the response in words" validate:"omitempty,min=1"`
}

// PromptOption is a function type that modifies a Prompt.
// It follows the functional options pattern for flexible configuration.
type PromptOption func(*Prompt)

// NewPrompt creates a new Prompt with the given input and applies any provided options.
//
// Parameters:
//   - input: The main input text for the LLM
//   - opts: Optional configuration functions
//
// Returns:
//   - Configured Prompt instance
//
// Example:
//
//	prompt := NewPrompt("Translate this to French",
//	    WithSystemPrompt("You are a helpful translator.", CacheTypeEphemeral),
//	    WithMaxLength(100),
//	)
func NewPrompt(input string, opts ...PromptOption) *Prompt {
	p := &Prompt{
		Input:    input,
		Messages: []PromptMessage{{Role: "user", Content: input}},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// CacheOption sets the cache type for the last message in the prompt.
//
// Parameters:
//   - cacheType: The type of caching to apply
func CacheOption(cacheType CacheType) PromptOption {
	return func(p *Prompt) {
		if len(p.Messages) > 0 {
			p.Messages[len(p.Messages)-1].CacheType = cacheType
		}
	}
}

// WithSystemPrompt adds a system-level prompt message with optional caching.
//
// Parameters:
//   - prompt: The system prompt text
//   - cacheType: Optional caching strategy for the system prompt
func WithSystemPrompt(prompt string, cacheType CacheType) PromptOption {
	return func(p *Prompt) {
		p.SystemPrompt = prompt
		p.SystemCacheType = cacheType
	}
}

// WithMessage adds a single message to the prompt.
//
// Parameters:
//   - role: Role of the message sender
//   - content: Content of the message
//   - cacheType: Optional caching strategy
func WithMessage(role, content string, cacheType CacheType) PromptOption {
	return func(p *Prompt) {
		p.Messages = append(p.Messages, PromptMessage{Role: role, Content: content, CacheType: cacheType})
	}
}

// WithTools configures the available tools for the LLM to use.
//
// Parameters:
//   - tools: List of available tools
func WithTools(tools []models.Tool) PromptOption {
	return func(p *Prompt) {
		p.Tools = tools
	}
}

// WithToolChoice specifies how tools should be selected by the LLM.
//
// Parameters:
//   - choice: Tool selection strategy
func WithToolChoice(choice string) PromptOption {
	return func(p *Prompt) {
		p.ToolChoice = map[string]any{
			"type": choice,
		}
	}
}

// WithMessages sets the complete list of conversation messages.
//
// Parameters:
//   - messages: List of messages to set
func WithMessages(messages []PromptMessage) PromptOption {
	return func(p *Prompt) {
		p.Messages = messages
	}
}

// WithDirectives adds special instructions or constraints to guide the LLM.
//
// Parameters:
//   - directives: List of directive strings
func WithDirectives(directives ...string) PromptOption {
	return func(p *Prompt) {
		p.Directives = append(p.Directives, directives...)
	}
}

// WithOutput specifies the expected format or structure of the LLM's response.
//
// Parameters:
//   - output: Output specification string
func WithOutput(output string) PromptOption {
	return func(p *Prompt) {
		p.Output = output
	}
}

// WithContext adds background information or context for the LLM.
//
// Parameters:
//   - context: Contextual information string
func WithContext(context string) PromptOption {
	return func(p *Prompt) {
		p.Context = context
	}
}

// WithMaxLength sets the maximum length for the LLM's response.
//
// Parameters:
//   - length: Maximum response length in words
func WithMaxLength(length int) PromptOption {
	return func(p *Prompt) {
		p.MaxLength = length
	}
}

// WithExamples adds example conversations or outputs to guide the LLM.
// If a single example ends with .txt or .jsonl, it's treated as a file path.
//
// Parameters:
//   - examples: List of example strings or a file path
func WithExamples(examples ...string) PromptOption {
	return func(p *Prompt) {
		if len(examples) == 1 && strings.HasSuffix(examples[0], ".txt") || strings.HasSuffix(examples[0], ".jsonl") {
			fileExamples, err := ReadExamplesFromFile(examples[0])
			if err != nil {
				panic(fmt.Sprintf("Failed to read examples from file: %v", err))
			}
			p.Examples = append(p.Examples, fileExamples...)
		} else {
			p.Examples = append(p.Examples, examples...)
		}
	}
}

// Apply applies the given options to modify the prompt's configuration.
//
// Parameters:
//   - opts: List of configuration functions to apply
func (p *Prompt) Apply(opts ...PromptOption) {
	for _, opt := range opts {
		opt(p)
	}
}

// String returns a formatted string representation of the prompt.
// It includes all components (system prompt, context, directives, etc.)
// in a human-readable format.
//
// Returns:
//   - Formatted prompt string
func (p *Prompt) String() string {
	var builder strings.Builder

	if p.SystemPrompt != "" {
		builder.WriteString("System: ")
		builder.WriteString(p.SystemPrompt)
		if p.SystemCacheType != "" {
			builder.WriteString(fmt.Sprintf(" (Cache: %s)", p.SystemCacheType))
		}
		builder.WriteString("\n\n")
	}

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

	if len(p.Messages) > 0 {
		builder.WriteString("\nMessages:\n")
		for _, msg := range p.Messages {
			builder.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
			if msg.CacheType != "" {
				builder.WriteString(fmt.Sprintf("(Cache: %s)\n", msg.CacheType))
			}
		}
	}

	return builder.String()
}

// Validate checks if the prompt configuration is valid according to
// its validation rules and constraints.
//
// Returns:
//   - Error if validation fails, nil otherwise
func (p *Prompt) Validate() error {
	return Validate(p)
}

// GenerateJSONSchema returns a JSON Schema representation of the prompt structure.
// This schema can be used for validation and documentation purposes.
//
// Parameters:
//   - opts: Optional schema generation configuration
//
// Returns:
//   - JSON schema as bytes
//   - Error if schema generation fails
func (p *Prompt) GenerateJSONSchema(opts ...SchemaOption) ([]byte, error) {
	reflector := &jsonschema.Reflector{}
	for _, opt := range opts {
		opt(reflector)
	}
	schema := reflector.Reflect(p)
	jsonData, err := schema.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal JSON schema: %w", err)
	}
	return jsonData, nil
}

// SchemaOption is a function type for configuring JSON schema generation.
type SchemaOption func(*jsonschema.Reflector)

// WithExpandedStruct enables or disables detailed structure expansion
// in the generated JSON schema.
//
// Parameters:
//   - expanded: Whether to expand nested structures
func WithExpandedStruct(expanded bool) SchemaOption {
	return func(r *jsonschema.Reflector) {
		r.ExpandedStruct = expanded
	}
}
