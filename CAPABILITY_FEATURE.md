# LLM Provider Capability System Implementation for GoLLM

## Problem Statement

The gollm library needs a capability system that:

- Tracks which capabilities (structured response, vision, function calling, etc.) are supported by different
  provider/model combinations
- Handles provider-specific quirks (e.g., Cohere only supports structured responses through tool calling)
- Avoids hardcoded model lists in each provider
- Provides compile-time type safety where possible
- Uses composition over inheritance
- Can be embedded into the existing Provider interface

## Solution: Functional Validator Pattern with Capabilities Interface

### Architecture Overview

```
Provider (implements Provider interface including Capabilities)
    ├── Composition: capabilities field (CapabilityResolver instance)
    │   ├── Uses: CapabilityRegistry (package-level singleton)
    │   └── Handles capability logic and registration
    ├── Delegates: Capabilities interface methods to resolver
    └── Core provider functionality (Name, Endpoint, PrepareRequest, etc.)
```

### Component Naming

The component that implements the Capabilities interface is called `CapabilityResolver` because it resolves what
capabilities are available for a given provider/model combination.

**Alternative names considered:**

- `CapabilityManager` - Manages capabilities for a provider/model
- `ModelCapabilities` - Direct description of what it represents
- `CapabilitySet` - Represents the set of capabilities
- `CapabilityChecker` - Original name, more procedural

## Complete Implementation

### 1. Core Types and Capabilities Interface

```go
// capability_types.go
package providers

// ProviderName ensures we use correct provider identifiers
type ProviderName string

const (
	ProviderOpenAI     ProviderName = "openai"
	ProviderGemini     ProviderName = "gemini"
	ProviderAnthropic  ProviderName = "anthropic"
	ProviderCohere     ProviderName = "cohere"
	ProviderOllama     ProviderName = "ollama"
	ProviderGroq       ProviderName = "groq"
	ProviderMistral    ProviderName = "mistral"
	ProviderDeepSeek   ProviderName = "deepseek"
	ProviderOpenRouter ProviderName = "openrouter"
)

// Capability represents a feature that a provider/model may support
type Capability string

const (
	CapStructuredResponse Capability = "structured_response"
	CapStreaming          Capability = "streaming"
	CapFunctionCalling    Capability = "function_calling"
	CapVision             Capability = "vision"
	CapToolUse            Capability = "tool_use"
	CapSystemPrompt       Capability = "system_prompt"
	CapCaching            Capability = "caching"
)

// Capabilities interface that will be embedded into Provider interface
type Capabilities interface {
	// Self-registration of capabilities
	Register()

	// Generic capability check
	HasCapability(cap Capability) bool

	// Typed configuration getters
	GetStructuredResponseConfig() (StructuredResponseConfig, bool)
	GetFunctionCallingConfig() (FunctionCallingConfig, bool)
	GetStreamingConfig() (StreamingConfig, bool)
	GetVisionConfig() (VisionConfig, bool)
	GetCachingConfig() (CachingConfig, bool)
}

// CapabilityConfig holds both support status and configuration
type CapabilityConfig struct {
	Supported bool
	Config    interface{}
}

// CapabilityValidator checks if a model supports a capability
type CapabilityValidator func(model string) CapabilityConfig
```

### 2. Configuration Structures

```go
// capability_configs.go
package providers

// StructuredResponseConfig defines how structured responses work
type StructuredResponseConfig struct {
	RequiresToolUse  bool // Cohere's quirk - only works via tool calling
	MaxSchemaDepth   int
	SupportedFormats []string // "json", "json_schema", "xml"
	SystemPromptHint string   // Some providers need specific prompts
	RequiresJSONMode bool     // OpenAI's JSON mode requirement
}

// FunctionCallingConfig defines function calling capabilities
type FunctionCallingConfig struct {
	MaxFunctions      int
	SupportsParallel  bool
	MaxParallelCalls  int
	RequiresToolRole  bool // Special formatting requirements
	SupportsStreaming bool // Can stream function calls
}

// StreamingConfig defines streaming behavior
type StreamingConfig struct {
	SupportsSSE    bool
	BufferSize     int
	ChunkDelimiter string
	SupportsUsage  bool // Can stream token usage
}

// VisionConfig defines image handling capabilities
type VisionConfig struct {
	MaxImageSize        int64
	SupportedFormats    []string
	MaxImagesPerRequest int
	SupportsImageGen    bool
	SupportsVideoFrames bool
}

// CachingConfig defines caching capabilities
type CachingConfig struct {
	SupportsCaching  bool
	MaxCacheSize     int64
	CacheTTLSeconds  int
	CacheKeyStrategy string
}
```

### 3. Updated Provider Interface

```go
// provider.go
package providers

import (
	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Provider defines the complete interface that all LLM providers must implement
type Provider interface {
	// Embed the Capabilities interface
	Capabilities

	// Core identification and configuration
	Name() string
	Endpoint() string
	Headers() map[string]string
	SetExtraHeaders(extraHeaders map[string]string)
	SetDefaultOptions(cfg *config.Config)
	SetOption(key string, value any)
	SetLogger(logger logging.Logger)

	// Request preparation
	PrepareRequest(req *Request, options map[string]any) ([]byte, error)
	PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error)

	// Response handling
	ParseResponse(body []byte) (*Response, error)
	ParseStreamResponse(chunk []byte) (*Response, error)
}
```

### 4. Capability Registry (Package-Level Singleton)

```go
// capability_registry.go
package providers

import (
	"sync"
)

var (
	capabilityRegistry *CapabilityRegistry
	registryOnce       sync.Once
)

// GetCapabilityRegistry returns the singleton capability registry
func GetCapabilityRegistry() *CapabilityRegistry {
	registryOnce.Do(func() {
		capabilityRegistry = NewCapabilityRegistry()
	})
	return capabilityRegistry
}

// CapabilityRegistry manages all capability validators
type CapabilityRegistry struct {
	mu         sync.RWMutex
	validators map[Capability]map[ProviderName]CapabilityValidator
}

// NewCapabilityRegistry creates a new registry instance
func NewCapabilityRegistry() *CapabilityRegistry {
	return &CapabilityRegistry{
		validators: make(map[Capability]map[ProviderName]CapabilityValidator),
	}
}

// Register adds a validator for a specific capability and provider
func (r *CapabilityRegistry) Register(cap Capability, provider ProviderName, validator CapabilityValidator) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.validators[cap] == nil {
		r.validators[cap] = make(map[ProviderName]CapabilityValidator)
	}
	r.validators[cap][provider] = validator
}

// RegisterAll registers multiple capabilities at once
func (r *CapabilityRegistry) RegisterAll(provider ProviderName, capabilities map[Capability]CapabilityValidator) {
	for cap, validator := range capabilities {
		r.Register(cap, provider, validator)
	}
}

// Check evaluates if a provider/model supports a capability
func (r *CapabilityRegistry) Check(provider ProviderName, model string, cap Capability) CapabilityConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if capValidators, ok := r.validators[cap]; ok {
		if validator, ok := capValidators[provider]; ok {
			return validator(model)
		}
	}
	return CapabilityConfig{Supported: false}
}

// HasCapability is a simple boolean check
func (r *CapabilityRegistry) HasCapability(provider ProviderName, model string, cap Capability) bool {
	return r.Check(provider, model, cap).Supported
}

// GetTypedConfig returns a typed configuration (generic helper)
func GetTypedConfig[T any](
	registry *CapabilityRegistry,
	provider ProviderName,
	model string,
	cap Capability
) (T, bool) {
	config := registry.Check(provider, model, cap)
	var zero T

	if !config.Supported {
		return zero, false
	}

	if typed, ok := config.Config.(T); ok {
		return typed, true
	}

	return zero, false
}
```

### 5. CapabilityResolver Implementation

```go
// capability_resolver.go
package providers

// CapabilityResolver is a helper that handles capability logic for providers.
// Providers compose this as a field and delegate Capabilities interface methods to it.
type CapabilityResolver struct {
	provider ProviderName
	model    string
	registry *CapabilityRegistry
}

// NewCapabilityResolver creates a new capability resolver
func NewCapabilityResolver(provider ProviderName, model string) *CapabilityResolver {
	return &CapabilityResolver{
		provider: provider,
		model:    model,
		registry: GetCapabilityRegistry(), // Use singleton
	}
}

// Register self-registers this provider/model's capabilities
func (cr *CapabilityResolver) Register() {
	validators := cr.getValidatorsForProvider()
	if validators != nil {
		cr.registry.RegisterAll(cr.provider, validators)
	}
}

// HasCapability checks if a capability is supported
func (cr *CapabilityResolver) HasCapability(cap Capability) bool {
	return cr.registry.HasCapability(cr.provider, cr.model, cap)
}

// GetStructuredResponseConfig returns typed structured response config
func (cr *CapabilityResolver) GetStructuredResponseConfig() (StructuredResponseConfig, bool) {
	return GetTypedConfig[StructuredResponseConfig](cr.registry, cr.provider, cr.model, CapStructuredResponse)
}

// GetFunctionCallingConfig returns typed function calling config
func (cr *CapabilityResolver) GetFunctionCallingConfig() (FunctionCallingConfig, bool) {
	return GetTypedConfig[FunctionCallingConfig](cr.registry, cr.provider, cr.model, CapFunctionCalling)
}

// GetStreamingConfig returns typed streaming config
func (cr *CapabilityResolver) GetStreamingConfig() (StreamingConfig, bool) {
	return GetTypedConfig[StreamingConfig](cr.registry, cr.provider, cr.model, CapStreaming)
}

// GetVisionConfig returns typed vision config
func (cr *CapabilityResolver) GetVisionConfig() (VisionConfig, bool) {
	return GetTypedConfig[VisionConfig](cr.registry, cr.provider, cr.model, CapVision)
}

// GetCachingConfig returns typed caching config
func (cr *CapabilityResolver) GetCachingConfig() (CachingConfig, bool) {
	return GetTypedConfig[CachingConfig](cr.registry, cr.provider, cr.model, CapCaching)
}

// getValidatorsForProvider returns validators for each provider
func (cr *CapabilityResolver) getValidatorsForProvider() map[Capability]CapabilityValidator {
	switch cr.provider {
	case ProviderOpenAI:
		return getOpenAIValidators()
	case ProviderCohere:
		return getCohereValidators()
	case ProviderAnthropic:
		return getAnthropicValidators()
	case ProviderGemini:
		return getGeminiValidators()
	case ProviderGroq:
		return getGroqValidators()
	default:
		return nil
	}
}
```

### 6. Capability Validators

```go
// validators_openai.go
package providers

import "strings"

// getOpenAIValidators returns all OpenAI model capability validators
func getOpenAIValidators() map[Capability]CapabilityValidator {
	return map[Capability]CapabilityValidator{
		CapStructuredResponse: func(model string) CapabilityConfig {
			// O1 models don't support structured output
			if strings.HasPrefix(model, "o1") {
				return CapabilityConfig{Supported: false}
			}

			if strings.HasPrefix(model, "gpt-4o") || strings.HasPrefix(model, "gpt-4-turbo") {
				return CapabilityConfig{
					Supported: true,
					Config: StructuredResponseConfig{
						RequiresToolUse:  false,
						MaxSchemaDepth:   15,
						SupportedFormats: []string{"json", "json_schema"},
						RequiresJSONMode: true,
					},
				}
			}

			if model == "gpt-3.5-turbo-0125" || model == "gpt-3.5-turbo-1106" {
				return CapabilityConfig{
					Supported: true,
					Config: StructuredResponseConfig{
						RequiresToolUse:  false,
						MaxSchemaDepth:   10,
						SupportedFormats: []string{"json"},
						RequiresJSONMode: true,
					},
				}
			}

			return CapabilityConfig{Supported: false}
		},

		CapVision: func(model string) CapabilityConfig {
			visionModels := []string{"gpt-4o", "gpt-4-turbo", "gpt-4-vision"}
			for _, vm := range visionModels {
				if strings.HasPrefix(model, vm) {
					return CapabilityConfig{
						Supported: true,
						Config: VisionConfig{
							MaxImageSize:        20 * 1024 * 1024,
							SupportedFormats:    []string{"jpeg", "png", "gif", "webp"},
							MaxImagesPerRequest: 10,
							SupportsVideoFrames: strings.Contains(model, "4o"),
						},
					}
				}
			}
			return CapabilityConfig{Supported: false}
		},

		CapFunctionCalling: func(model string) CapabilityConfig {
			// O1 models don't support function calling
			if strings.HasPrefix(model, "o1") {
				return CapabilityConfig{Supported: false}
			}

			if strings.HasPrefix(model, "gpt-4") {
				return CapabilityConfig{
					Supported: true,
					Config: FunctionCallingConfig{
						MaxFunctions:      128,
						SupportsParallel:  true,
						MaxParallelCalls:  10,
						SupportsStreaming: true,
					},
				}
			}

			if strings.HasPrefix(model, "gpt-3.5-turbo") {
				return CapabilityConfig{
					Supported: true,
					Config: FunctionCallingConfig{
						MaxFunctions:      64,
						SupportsParallel:  true,
						MaxParallelCalls:  5,
						SupportsStreaming: false,
					},
				}
			}

			return CapabilityConfig{Supported: false}
		},

		CapStreaming: func(model string) CapabilityConfig {
			// All OpenAI models support streaming
			return CapabilityConfig{
				Supported: true,
				Config: StreamingConfig{
					SupportsSSE:    true,
					BufferSize:     4096,
					ChunkDelimiter: "data: ",
					SupportsUsage:  strings.HasPrefix(model, "gpt-4"),
				},
			}
		},
	}
}
```

```go
// validators_cohere.go
package providers

import (
"slices"
"strings"


)

// getCohereValidators returns all Cohere model capability validators
func getCohereValidators() map[Capability]CapabilityValidator {
	return map[Capability]CapabilityValidator{
		CapStructuredResponse: func(model string) CapabilityConfig {
			// IMPORTANT: Cohere only supports structured response through tool calling
			supportedModels := []string{
				"command-a-03-2025",
				"command-r-plus-08-2024",
				"command-r-plus",
				"command-r-08-2024",
				"command-r",
			}

			if slices.Contains(supportedModels, model) {
				return CapabilityConfig{
					Supported: true,
					Config: StructuredResponseConfig{
						RequiresToolUse:  true, // THE COHERE QUIRK!
						MaxSchemaDepth:   5,
						SupportedFormats: []string{"json"},
						SystemPromptHint: "You must use the provided tool to structure your response",
					},
				}
			}
			return CapabilityConfig{Supported: false}
		},

		CapFunctionCalling: func(model string) CapabilityConfig {
			if strings.Contains(model, "command-r") {
				return CapabilityConfig{
					Supported: true,
					Config: FunctionCallingConfig{
						MaxFunctions:      50,
						SupportsParallel:  false,
						RequiresToolRole:  true,
						SupportsStreaming: true,
					},
				}
			}

			if strings.Contains(model, "command") {
				return CapabilityConfig{
					Supported: true,
					Config: FunctionCallingConfig{
						MaxFunctions:      20,
						SupportsParallel:  false,
						RequiresToolRole:  true,
						SupportsStreaming: false,
					},
				}
			}

			return CapabilityConfig{Supported: false}
		},

		CapStreaming: func(model string) CapabilityConfig {
			return CapabilityConfig{
				Supported: true,
				Config: StreamingConfig{
					SupportsSSE:    true,
					BufferSize:     8192,
					ChunkDelimiter: "\n",
					SupportsUsage:  false,
				},
			}
		},
	}
}
```

### 7. Provider Implementation Examples

```go
// openai.go
package providers

import (
	"encoding/json"
	"fmt"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

type OpenAIProvider struct {
	// Composition: use a resolver field instead of embedding
	capabilities CapabilityResolver

	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewOpenAIProvider creates a new OpenAI provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) *OpenAIProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	p := &OpenAIProvider{
		capabilities: NewCapabilityResolver(ProviderOpenAI, model),
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}

	// Self-register capabilities
	p.Register()

	return p
}

// Implement Capabilities interface methods by delegating to the resolver

// Register self-registers this provider's capabilities
func (p *OpenAIProvider) Register() {
	p.capabilities.Register()
}

// HasCapability checks if a capability is supported
func (p *OpenAIProvider) HasCapability(cap Capability) bool {
	return p.capabilities.HasCapability(cap)
}

// GetStructuredResponseConfig returns typed structured response config
func (p *OpenAIProvider) GetStructuredResponseConfig() (StructuredResponseConfig, bool) {
	return p.capabilities.GetStructuredResponseConfig()
}

// GetFunctionCallingConfig returns typed function calling config
func (p *OpenAIProvider) GetFunctionCallingConfig() (FunctionCallingConfig, bool) {
	return p.capabilities.GetFunctionCallingConfig()
}

// GetStreamingConfig returns typed streaming config
func (p *OpenAIProvider) GetStreamingConfig() (StreamingConfig, bool) {
	return p.capabilities.GetStreamingConfig()
}

// GetVisionConfig returns typed vision config
func (p *OpenAIProvider) GetVisionConfig() (VisionConfig, bool) {
	return p.capabilities.GetVisionConfig()
}

// GetCachingConfig returns typed caching config
func (p *OpenAIProvider) GetCachingConfig() (CachingConfig, bool) {
	return p.capabilities.GetCachingConfig()
}

// Core provider methods
func (p *OpenAIProvider) Name() string {
	return string(ProviderOpenAI)
}

func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

func (p *OpenAIProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	// Check capabilities before preparing request
	if req.ResponseSchema != nil {
		if structConfig, ok := p.GetStructuredResponseConfig(); ok {
			if structConfig.RequiresJSONMode {
				// Add JSON mode to request
				if options == nil {
					options = make(map[string]any)
				}
				options["response_format"] = map[string]string{"type": "json_object"}
			}
		} else {
			return nil, fmt.Errorf("model %s does not support structured responses", p.model)
		}
	}

	// Continue with request preparation...
	requestBody := map[string]any{
		"model":    p.model,
		"messages": req.Messages,
	}

	// Add options
	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// ... other Provider interface methods
```

```go
// cohere.go
package providers

import (
	"encoding/json"
	"fmt"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

type CohereProvider struct {
	// Composition: use a resolver field instead of embedding
	capabilities CapabilityResolver

	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewCohereProvider creates a new Cohere provider instance
func NewCohereProvider(apiKey, model string, extraHeaders map[string]string) *CohereProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	p := &CohereProvider{
		capabilities: *NewCapabilityResolver(ProviderCohere, model),
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}

	// Self-register capabilities
	p.Register()

	return p
}

// Implement Capabilities interface methods by delegating to the resolver

// Register self-registers this provider's capabilities
func (p *CohereProvider) Register() {
	p.capabilities.Register()
}

// HasCapability checks if a capability is supported
func (p *CohereProvider) HasCapability(cap Capability) bool {
	return p.capabilities.HasCapability(cap)
}

// GetStructuredResponseConfig returns typed structured response config
func (p *CohereProvider) GetStructuredResponseConfig() (StructuredResponseConfig, bool) {
	return p.capabilities.GetStructuredResponseConfig()
}

// GetFunctionCallingConfig returns typed function calling config  
func (p *CohereProvider) GetFunctionCallingConfig() (FunctionCallingConfig, bool) {
	return p.capabilities.GetFunctionCallingConfig()
}

// GetStreamingConfig returns typed streaming config
func (p *CohereProvider) GetStreamingConfig() (StreamingConfig, bool) {
	return p.capabilities.GetStreamingConfig()
}

// GetVisionConfig returns typed vision config
func (p *CohereProvider) GetVisionConfig() (VisionConfig, bool) {
	return p.capabilities.GetVisionConfig()
}

// GetCachingConfig returns typed caching config
func (p *CohereProvider) GetCachingConfig() (CachingConfig, bool) {
	return p.capabilities.GetCachingConfig()
}

func (p *CohereProvider) Name() string {
	return string(ProviderCohere)
}

func (p *CohereProvider) Endpoint() string {
	return "https://api.cohere.com/v2/chat"
}

func (p *CohereProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	// Handle structured response with Cohere's quirk
	if req.ResponseSchema != nil {
		if structConfig, ok := p.GetStructuredResponseConfig(); ok {
			if structConfig.RequiresToolUse {
				// Convert to tool calling format for Cohere
				return p.prepareToolCallRequest(req, options)
			}
		} else {
			return nil, fmt.Errorf("model %s does not support structured responses", p.model)
		}
	}

	// Handle function calling
	if len(req.Messages) > 0 {
		if funcConfig, ok := p.GetFunctionCallingConfig(); ok {
			if funcConfig.RequiresToolRole {
				return p.prepareFormattedToolRequest(req, options)
			}
		}
	}

	return p.prepareStandardRequest(req, options)
}

func (p *CohereProvider) prepareToolCallRequest(req *Request, options map[string]any) ([]byte, error) {
	// Convert structured response to Cohere's tool format
	toolDef := map[string]interface{}{
		"name":        "respond_with_json",
		"description": "Respond with structured JSON matching the provided schema",
		"parameters":  req.ResponseSchema,
	}

	cohereReq := map[string]interface{}{
		"model":   p.model,
		"message": req.Messages[len(req.Messages)-1].Content,
		"tools":   []interface{}{toolDef},
		"tool_choice": map[string]string{
			"type": "tool",
			"name": "respond_with_json",
		},
	}

	// Add system prompt hint if configured
	if structConfig, ok := p.GetStructuredResponseConfig(); ok {
		if structConfig.SystemPromptHint != "" {
			cohereReq["preamble"] = structConfig.SystemPromptHint
		}
	}

	return json.Marshal(cohereReq)
}

// ... other Provider interface methods
```

### 8. Usage Examples

```go
// main.go
package main

import (
	"fmt"

	"github.com/weave-labs/gollm/providers"
)

func main() {
	// Create providers - they self-register their capabilities
	openai := providers.NewOpenAIProvider("api-key", "gpt-4o", nil)
	cohere := providers.NewCohereProvider("api-key", "command-r-plus", nil)

	// Direct capability checking on provider
	if openai.HasCapability(providers.CapVision) {
		fmt.Println("✓ OpenAI supports vision")
		if visionCfg, ok := openai.GetVisionConfig(); ok {
			fmt.Printf("  Max images: %d\n", visionCfg.MaxImagesPerRequest)
			fmt.Printf("  Max size: %d MB\n", visionCfg.MaxImageSize/(1024*1024))
		}
	}

	// Check Cohere's structured response quirk
	if structCfg, ok := cohere.GetStructuredResponseConfig(); ok {
		if structCfg.RequiresToolUse {
			fmt.Println("⚠️ Cohere requires tool use for structured responses")
		}
	}

	// Work with any provider through the interface
	validateProvider(openai)
	validateProvider(cohere)
}

func validateProvider(p providers.Provider) error {
	// Works with any provider that implements the Provider interface
	if !p.HasCapability(providers.CapStreaming) {
		return fmt.Errorf("provider %s does not support streaming", p.Name())
	}

	if funcConfig, ok := p.GetFunctionCallingConfig(); ok {
		fmt.Printf("Provider supports up to %d functions\n", funcConfig.MaxFunctions)
		if funcConfig.SupportsParallel {
			fmt.Println("Provider can execute functions in parallel")
		}
	}

	return nil
}

// Validate capabilities before making requests
func prepareRequestWithValidation(p providers.Provider, req *providers.Request) error {
	// Check structured response support
	if req.ResponseSchema != nil {
		if structCfg, ok := p.GetStructuredResponseConfig(); ok {
			if structCfg.RequiresToolUse {
				fmt.Println("Note: Converting to tool format for structured response")
			}
			if structCfg.SystemPromptHint != "" {
				// Add system prompt
				req.SystemPrompt = structCfg.SystemPromptHint
			}
		} else {
			return fmt.Errorf("structured response not supported")
		}
	}

	// Check function calling limits
	if len(req.Messages) > 0 {
		if funcCfg, ok := p.GetFunctionCallingConfig(); ok {
			// Validate against limits
			fmt.Printf("Provider supports %d functions max\n", funcCfg.MaxFunctions)
		} else {
			return fmt.Errorf("function calling not supported")
		}
	}

	return nil
}
```

### 9. Testing

```go
// capability_test.go
package providers

import (
	"testing"
)

func TestCapabilities(t *testing.T) {
	tests := []struct {
		name     string
		provider Provider
		cap      Capability
		expected bool
	}{
		{
			name:     "Cohere supports structured response",
			provider: NewCohereProvider("key", "command-r-plus", nil),
			cap:      CapStructuredResponse,
			expected: true,
		},
		{
			name:     "OpenAI GPT-4o supports vision",
			provider: NewOpenAIProvider("key", "gpt-4o", nil),
			cap:      CapVision,
			expected: true,
		},
		{
			name:     "OpenAI o1 does not support function calling",
			provider: NewOpenAIProvider("key", "o1-preview", nil),
			cap:      CapFunctionCalling,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.provider.HasCapability(tt.cap)
			if got != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, got)
			}
		})
	}
}

func TestCohereStructuredResponseQuirk(t *testing.T) {
	provider := NewCohereProvider("key", "command-r-plus", nil)

	// Should support structured response
	if !provider.HasCapability(CapStructuredResponse) {
		t.Fatal("expected Cohere to support structured response")
	}

	// Should require tool use
	cfg, ok := provider.GetStructuredResponseConfig()
	if !ok {
		t.Fatal("expected to get structured response config")
	}

	if !cfg.RequiresToolUse {
		t.Error("expected Cohere to require tool use for structured response")
	}
}
```

## Key Design Decisions

1. **Capabilities Interface with Register()**: Self-registration method included in the interface
2. **CapabilityResolver**: Clear name that describes its purpose (resolving capabilities)
3. **Functional Validators**: Capability logic encoded as functions returning configuration
4. **Package-Level Singleton**: Registry accessed via GetCapabilityRegistry()
5. **Composition Over Inheritance**: Providers embed CapabilityResolver
6. **Type Safety**: Typed configuration structs and constants for provider/capability names
7. **Self-Registration**: Providers register their own capabilities during construction

## Implementation Checklist

- [ ] Create Capabilities interface with Register() method
- [ ] Implement core types and capability constants
- [ ] Create capability configuration structures
- [ ] Build CapabilityRegistry as package-level singleton
- [ ] Implement CapabilityResolver that implements Capabilities interface
- [ ] Create validator functions for each provider (OpenAI, Cohere, Gemini, Anthropic, etc.)
- [ ] Update existing providers to embed CapabilityResolver
- [ ] Update Provider interface to embed Capabilities
- [ ] Add comprehensive tests
- [ ] Document edge cases and quirks

## Edge Cases to Handle

1. **Cohere Structured Response**: Only works through tool calling (`RequiresToolUse: true`)
2. **OpenAI O1 Models**: Don't support function calling, vision, or structured responses
3. **OpenAI JSON Mode**: Some models require `response_format: {type: "json_object"}`
4. **Model Versioning**: Use string prefix matching for model families
5. **Provider-Specific Formatting**: Some providers need special role formatting for tools

## Benefits

- **Self-Contained**: Providers manage their own capability registration
- **Clean Separation**: Capabilities logic separate from provider implementation
- **Type Safe**: Compile-time checking for capability names and configurations
- **Extensible**: Easy to add new capabilities or providers
- **Testable**: Each component can be tested in isolation
- **Maintainable**: Centralized capability definitions, no scattered conditionals
- **No Backward Compatibility Concerns**: Clean break from old system
