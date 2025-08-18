## Capability System Architecture

### 1. Core Capability Types

```go

// Capability represents a specific feature that a provider/model combination may support

type Capability string

const (

CapabilityStructuredResponse Capability = "structured_response"

CapabilityStreaming Capability = "streaming"

CapabilityFunctionCalling Capability = "function_calling"

CapabilityVision Capability = "vision"

CapabilityToolUse Capability = "tool_use"

CapabilitySystemPrompt Capability = "system_prompt"

CapabilityCaching Capability = "caching"

// Future capabilities can be added here easily

)

// CapabilitySet represents a collection of capabilities

type CapabilitySet map[Capability]bool

// Has checks if a capability is supported

func (cs CapabilitySet) Has(cap Capability) bool {

return cs[cap]

}

// Add adds a capability to the set

func (cs CapabilitySet) Add(cap Capability) {

cs[cap] = true

}

// Remove removes a capability from the set

func (cs CapabilitySet) Remove(cap Capability) {

delete(cs, cap)

}

```

### 2. Model Capability Registry

```go

// ModelCapabilities holds the capability information for a specific model

type ModelCapabilities struct {

ModelPattern string // e.g., "gpt-4*", "gemini-2.0-*", exact model names

Capabilities CapabilitySet

}

// ProviderCapabilityRegistry manages capabilities for all models of a provider

type ProviderCapabilityRegistry struct {

providerName string

models []ModelCapabilities

}

// NewProviderCapabilityRegistry creates a new registry for a provider

func NewProviderCapabilityRegistry(providerName string) *ProviderCapabilityRegistry {

return &ProviderCapabilityRegistry{

providerName: providerName,

models: make([]ModelCapabilities, 0),

}

}

// RegisterModel adds capability information for a model or model pattern

func (pcr *ProviderCapabilityRegistry) RegisterModel(pattern string, capabilities ...Capability) {

capSet := make(CapabilitySet)

for _, cap := range capabilities {

capSet.Add(cap)

}

pcr.models = append(pcr.models, ModelCapabilities{

ModelPattern: pattern,

Capabilities: capSet,

})

}

// GetCapabilities returns the capabilities for a specific model

func (pcr *ProviderCapabilityRegistry) GetCapabilities(model string) CapabilitySet {

// Check for exact matches first

for _, mc := range pcr.models {

if mc.ModelPattern == model {

return mc.Capabilities

}

}

// Check for pattern matches

for _, mc := range pcr.models {

if matched, _ := filepath.Match(mc.ModelPattern, model); matched {

return mc.Capabilities

}

}

// Return empty set if no match found

return make(CapabilitySet)

}

// HasCapability checks if a specific model supports a capability

func (pcr *ProviderCapabilityRegistry) HasCapability(model string, capability Capability) bool {

return pcr.GetCapabilities(model).Has(capability)

}

```

### 3. Global Capability Manager

```go

// CapabilityManager manages capabilities across all providers

type CapabilityManager struct {

registries map[string]*ProviderCapabilityRegistry

}

// NewCapabilityManager creates a new capability manager

func NewCapabilityManager() *CapabilityManager {

return &CapabilityManager{

registries: make(map[string]*ProviderCapabilityRegistry),

}

}

// RegisterProvider registers a provider's capability registry

func (cm *CapabilityManager) RegisterProvider(registry *ProviderCapabilityRegistry) {

cm.registries[registry.providerName] = registry

}

// HasCapability checks if a provider/model combination supports a capability

func (cm *CapabilityManager) HasCapability(providerName, model string, capability Capability) bool {

if registry, exists := cm.registries[providerName]; exists {

return registry.HasCapability(model, capability)

}

return false

}

// GetCapabilities returns all capabilities for a provider/model combination

func (cm *CapabilityManager) GetCapabilities(providerName, model string) CapabilitySet {

if registry, exists := cm.registries[providerName]; exists {

return registry.GetCapabilities(model)

}

return make(CapabilitySet)

}

```

### 4. Updated Provider Interface

```go

// Provider interface updated to use the capability system

type Provider interface {

// Core identification and configuration

Name() string

Model() string // Add this to get the current model

Endpoint() string

Headers() map[string]string

SetExtraHeaders(extraHeaders map[string]string)

SetDefaultOptions(cfg *config.Config)

SetOption(key string, value any)

SetLogger(logger logging.Logger)

// Request preparation - unified interface

PrepareRequest(req *Request, options map[string]any) ([]byte, error)

PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error)

// Response handling

ParseResponse(body []byte) (*Response, error)

ParseStreamResponse(chunk []byte) (*Response, error)

// Capability management - delegated to capability manager

HasCapability(capability Capability) bool

GetCapabilities() CapabilitySet

}

```

### 5. Example Provider Implementation

```go

// GeminiProvider with capability system

type GeminiProvider struct {

apiKey string

model string

endpoint string

headers map[string]string

capManager *CapabilityManager

// ... other fields

}

// Initialize Gemini capabilities (called during provider registration)

func initGeminiCapabilities() *ProviderCapabilityRegistry {

registry := NewProviderCapabilityRegistry("gemini")

// Register capabilities for different model patterns

registry.RegisterModel("gemini-2.5-*",

CapabilityStructuredResponse,

CapabilityStreaming,

CapabilityFunctionCalling,

CapabilityVision,

CapabilitySystemPrompt,

)

registry.RegisterModel("gemini-2.0-*",

CapabilityStructuredResponse,

CapabilityStreaming,

CapabilityFunctionCalling,

CapabilityVision,

CapabilitySystemPrompt,

)

registry.RegisterModel("gemini-1.5-*",

CapabilityStreaming,

CapabilityFunctionCalling,

CapabilityVision,

CapabilitySystemPrompt,

)

// Older models with limited capabilities

registry.RegisterModel("gemini-pro",

CapabilityStreaming,

)

return registry

}

// Provider methods using capability system

func (p *GeminiProvider) HasCapability(capability Capability) bool {

return p.capManager.HasCapability(p.Name(), p.model, capability)

}

func (p *GeminiProvider) GetCapabilities() CapabilitySet {

return p.capManager.GetCapabilities(p.Name(), p.model)

}

// Legacy method support (deprecated but maintained for compatibility)

func (p *GeminiProvider) SupportsStructuredResponse() bool {

return p.HasCapability(CapabilityStructuredResponse)

}

```

### 6. Initialization and Registration

```go

// Global capability manager instance

var GlobalCapabilityManager = NewCapabilityManager()

// Initialize all provider capabilities during startup

func init() {

// Register all providers

GlobalCapabilityManager.RegisterProvider(initGeminiCapabilities())

GlobalCapabilityManager.RegisterProvider(initOpenAICapabilities())

GlobalCapabilityManager.RegisterProvider(initAnthropicCapabilities())

// ... other providers

}

// Provider constructor updated

func NewGeminiProvider(apiKey, model string, extraHeaders map[string]string) Provider {

return &GeminiProvider{

apiKey: apiKey,

model: model,

endpoint: "https://generativelanguage.googleapis.com/v1beta/models",

headers: make(map[string]string),

capManager: GlobalCapabilityManager,

}

}

```

## Benefits of This Architecture

1. **Maintainable**: Capability definitions are centralized and declarative

2. **Scalable**: Easy to add new capabilities or model patterns

3. **Flexible**: Supports pattern matching for model families

4. **Testable**: Capability logic is separate from provider logic

5. **Discoverable**: Easy to query what capabilities are available

6. **Consistent**: All providers use the same capability system

## Usage Examples

```go

// Check if a provider/model supports a capability

if provider.HasCapability(CapabilityStructuredResponse) {

// Use structured response features

}

// Get all capabilities for debugging/logging

caps := provider.GetCapabilities()

for cap := range caps {

fmt.Printf("Supports: %s\n", cap)

}

// Query capabilities without instantiating provider

hasVision := GlobalCapabilityManager.HasCapability("gemini", "gemini-2.5-pro", CapabilityVision)

```

This architecture eliminates the need for complex switch statements in each provider and makes it trivial to add new capabilities or update model support.