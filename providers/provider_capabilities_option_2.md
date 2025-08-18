## Strongly Typed Capability System

### 1. Capability Interface Hierarchy

```go
// Capability is the base interface that all capabilities must implement
type Capability interface {
	Name() string
	Description() string
}

// CapabilityChecker provides type-safe capability checking
type CapabilityChecker[T Capability] interface {
	HasCapability(T) bool
	GetCapabilityDetails(T) (T, bool)
}

// Provider extends the basic provider interface with type-safe capability checking
type CapabilityAwareProvider interface {
	Provider
	CapabilityChecker[Capability]
	
	// Type-safe capability checkers for specific capability types
	CheckStructuredResponse() StructuredResponseChecker
	CheckStreaming() StreamingChecker  
	CheckFunctionCalling() FunctionCallingChecker
	CheckVision() VisionChecker
	CheckCaching() CachingChecker
}
```

### 2. Specific Capability Types

```go
// StreamingCapability defines streaming-specific features
type StreamingCapability interface {
	Capability
	SupportsServerSentEvents() bool
	SupportsChunkedTransfer() bool
	MaxConcurrentStreams() int
}

// StructuredResponseCapability defines structured output features  
type StructuredResponseCapability interface {
	Capability
	SupportedSchemas() []SchemaType
	MaxSchemaComplexity() int
	RequiresSystemPrompt() bool
}

// FunctionCallingCapability defines function calling features
type FunctionCallingCapability interface {
	Capability
	MaxFunctions() int
	SupportsParallelCalls() bool
	SupportedParameterTypes() []ParameterType
}

// VisionCapability defines vision/image processing features
type VisionCapability interface {
	Capability
	SupportedImageFormats() []ImageFormat
	MaxImageSize() int64
	SupportsImageGeneration() bool
}

// CachingCapability defines caching features
type CachingCapability interface {
	Capability
	SupportedCacheTypes() []CacheType
	MaxCacheSize() int64
	CacheTTL() time.Duration
}
```

### 3. Concrete Capability Implementations

```go
// Streaming capabilities
type BasicStreaming struct{}
func (BasicStreaming) Name() string { return "streaming" }
func (BasicStreaming) Description() string { return "Basic streaming support" }
func (BasicStreaming) SupportsServerSentEvents() bool { return true }
func (BasicStreaming) SupportsChunkedTransfer() bool { return true }
func (BasicStreaming) MaxConcurrentStreams() int { return 1 }

type AdvancedStreaming struct{}
func (AdvancedStreaming) Name() string { return "streaming" }
func (AdvancedStreaming) Description() string { return "Advanced streaming with parallel support" }
func (AdvancedStreaming) SupportsServerSentEvents() bool { return true }
func (AdvancedStreaming) SupportsChunkedTransfer() bool { return true }
func (AdvancedStreaming) MaxConcurrentStreams() int { return 10 }

// Structured response capabilities
type BasicStructuredResponse struct{}
func (BasicStructuredResponse) Name() string { return "structured_response" }
func (BasicStructuredResponse) Description() string { return "Basic JSON schema support" }
func (BasicStructuredResponse) SupportedSchemas() []SchemaType { 
	return []SchemaType{SchemaTypeJSONSchema} 
}
func (BasicStructuredResponse) MaxSchemaComplexity() int { return 100 }
func (BasicStructuredResponse) RequiresSystemPrompt() bool { return false }

type AdvancedStructuredResponse struct{}
func (AdvancedStructuredResponse) Name() string { return "structured_response" }
func (AdvancedStructuredResponse) Description() string { return "Advanced structured response with complex schemas" }
func (AdvancedStructuredResponse) SupportedSchemas() []SchemaType { 
	return []SchemaType{SchemaTypeJSONSchema, SchemaTypeOpenAPI, SchemaTypeCustom} 
}
func (AdvancedStructuredResponse) MaxSchemaComplexity() int { return 1000 }
func (AdvancedStructuredResponse) RequiresSystemPrompt() bool { return false }

// Function calling capabilities
type BasicFunctionCalling struct{}
func (BasicFunctionCalling) Name() string { return "function_calling" }
func (BasicFunctionCalling) Description() string { return "Basic function calling support" }
func (BasicFunctionCalling) MaxFunctions() int { return 10 }
func (BasicFunctionCalling) SupportsParallelCalls() bool { return false }
func (BasicFunctionCalling) SupportedParameterTypes() []ParameterType {
	return []ParameterType{ParamTypeString, ParamTypeNumber, ParamTypeBoolean}
}

type AdvancedFunctionCalling struct{}
func (AdvancedFunctionCalling) Name() string { return "function_calling" }
func (AdvancedFunctionCalling) Description() string { return "Advanced function calling with parallel execution" }
func (AdvancedFunctionCalling) MaxFunctions() int { return 100 }
func (AdvancedFunctionCalling) SupportsParallelCalls() bool { return true }
func (AdvancedFunctionCalling) SupportedParameterTypes() []ParameterType {
	return []ParameterType{ParamTypeString, ParamTypeNumber, ParamTypeBoolean, ParamTypeObject, ParamTypeArray}
}
```

### 4. Type-Safe Capability Checkers

```go
// StreamingChecker provides type-safe streaming capability checking
type StreamingChecker interface {
	IsSupported() bool
	GetCapability() (StreamingCapability, bool)
}

// StructuredResponseChecker provides type-safe structured response checking
type StructuredResponseChecker interface {
	IsSupported() bool
	GetCapability() (StructuredResponseCapability, bool)
	SupportsSchema(SchemaType) bool
}

// FunctionCallingChecker provides type-safe function calling checking
type FunctionCallingChecker interface {
	IsSupported() bool
	GetCapability() (FunctionCallingCapability, bool)
	CanHandleFunctionCount(int) bool
}

// VisionChecker provides type-safe vision capability checking
type VisionChecker interface {
	IsSupported() bool
	GetCapability() (VisionCapability, bool)
	SupportsImageFormat(ImageFormat) bool
}

// CachingChecker provides type-safe caching capability checking
type CachingChecker interface {
	IsSupported() bool
	GetCapability() (CachingCapability, bool)
	SupportsCacheType(CacheType) bool
}
```

### 5. Model Capability Registry (Strongly Typed)

```go
// ModelCapabilitySet holds strongly typed capabilities for a model
type ModelCapabilitySet struct {
	Streaming          StreamingCapability
	StructuredResponse StructuredResponseCapability
	FunctionCalling    FunctionCallingCapability
	Vision             VisionCapability
	Caching            CachingCapability
}

// HasCapability performs type-safe capability checking
func (mcs *ModelCapabilitySet) HasCapability(capType reflect.Type) bool {
	switch capType {
	case reflect.TypeOf((*StreamingCapability)(nil)).Elem():
		return mcs.Streaming != nil
	case reflect.TypeOf((*StructuredResponseCapability)(nil)).Elem():
		return mcs.StructuredResponse != nil
	case reflect.TypeOf((*FunctionCallingCapability)(nil)).Elem():
		return mcs.FunctionCalling != nil
	case reflect.TypeOf((*VisionCapability)(nil)).Elem():
		return mcs.Vision != nil
	case reflect.TypeOf((*CachingCapability)(nil)).Elem():
		return mcs.Caching != nil
	default:
		return false
	}
}

// GetCapability returns a specific capability with type safety
func (mcs *ModelCapabilitySet) GetCapability(capType reflect.Type) (Capability, bool) {
	switch capType {
	case reflect.TypeOf((*StreamingCapability)(nil)).Elem():
		if mcs.Streaming != nil {
			return mcs.Streaming, true
		}
	case reflect.TypeOf((*StructuredResponseCapability)(nil)).Elem():
		if mcs.StructuredResponse != nil {
			return mcs.StructuredResponse, true
		}
	case reflect.TypeOf((*FunctionCallingCapability)(nil)).Elem():
		if mcs.FunctionCalling != nil {
			return mcs.FunctionCalling, true
		}
	case reflect.TypeOf((*VisionCapability)(nil)).Elem():
		if mcs.Vision != nil {
			return mcs.Vision, true
		}
	case reflect.TypeOf((*CachingCapability)(nil)).Elem():
		if mcs.Caching != nil {
			return mcs.Caching, true
		}
	}
	return nil, false
}

// ProviderCapabilityRegistry with strong typing
type ProviderCapabilityRegistry struct {
	providerName string
	models       map[string]*ModelCapabilitySet
}

// RegisterModel with strongly typed capabilities
func (pcr *ProviderCapabilityRegistry) RegisterModel(modelPattern string, capabilities *ModelCapabilitySet) {
	if pcr.models == nil {
		pcr.models = make(map[string]*ModelCapabilitySet)
	}
	pcr.models[modelPattern] = capabilities
}

// GetCapabilities returns strongly typed capability set
func (pcr *ProviderCapabilityRegistry) GetCapabilities(model string) *ModelCapabilitySet {
	// Direct match first
	if caps, exists := pcr.models[model]; exists {
		return caps
	}
	
	// Pattern matching
	for pattern, caps := range pcr.models {
		if matched, _ := filepath.Match(pattern, model); matched {
			return caps
		}
	}
	
	return &ModelCapabilitySet{} // Empty set
}
```

### 6. Type-Safe Provider Implementation

```go
// GeminiProvider with strongly typed capabilities
type GeminiProvider struct {
	apiKey      string
	model       string
	endpoint    string
	headers     map[string]string
	registry    *ProviderCapabilityRegistry
	// ... other fields
}

// Type-safe capability checking methods
func (p *GeminiProvider) CheckStreaming() StreamingChecker {
	return &geminiStreamingChecker{
		capabilities: p.registry.GetCapabilities(p.model),
	}
}

func (p *GeminiProvider) CheckStructuredResponse() StructuredResponseChecker {
	return &geminiStructuredResponseChecker{
		capabilities: p.registry.GetCapabilities(p.model),
	}
}

func (p *GeminiProvider) CheckFunctionCalling() FunctionCallingChecker {
	return &geminiFunctionCallingChecker{
		capabilities: p.registry.GetCapabilities(p.model),
	}
}

func (p *GeminiProvider) CheckVision() VisionChecker {
	return &geminiVisionChecker{
		capabilities: p.registry.GetCapabilities(p.model),
	}
}

func (p *GeminiProvider) CheckCaching() CachingChecker {
	return &geminiCachingChecker{
		capabilities: p.registry.GetCapabilities(p.model),
	}
}

// Generic capability checking with type safety
func (p *GeminiProvider) HasCapability(cap Capability) bool {
	capabilities := p.registry.GetCapabilities(p.model)
	return capabilities.HasCapability(reflect.TypeOf(cap))
}

func (p *GeminiProvider) GetCapabilityDetails(cap Capability) (Capability, bool) {
	capabilities := p.registry.GetCapabilities(p.model)
	return capabilities.GetCapability(reflect.TypeOf(cap))
}
```

### 7. Concrete Checker Implementations

```go
// geminiStreamingChecker implements StreamingChecker
type geminiStreamingChecker struct {
	capabilities *ModelCapabilitySet
}

func (gsc *geminiStreamingChecker) IsSupported() bool {
	return gsc.capabilities.Streaming != nil
}

func (gsc *geminiStreamingChecker) GetCapability() (StreamingCapability, bool) {
	if gsc.capabilities.Streaming != nil {
		return gsc.capabilities.Streaming, true
	}
	return nil, false
}

// geminiStructuredResponseChecker implements StructuredResponseChecker
type geminiStructuredResponseChecker struct {
	capabilities *ModelCapabilitySet
}

func (gsrc *geminiStructuredResponseChecker) IsSupported() bool {
	return gsrc.capabilities.StructuredResponse != nil
}

func (gsrc *geminiStructuredResponseChecker) GetCapability() (StructuredResponseCapability, bool) {
	if gsrc.capabilities.StructuredResponse != nil {
		return gsrc.capabilities.StructuredResponse, true
	}
	return nil, false
}

func (gsrc *geminiStructuredResponseChecker) SupportsSchema(schemaType SchemaType) bool {
	if cap, ok := gsrc.GetCapability(); ok {
		for _, supported := range cap.SupportedSchemas() {
			if supported == schemaType {
				return true
			}
		}
	}
	return false
}
```

### 8. Registration and Usage

```go
// Initialize Gemini capabilities with strong typing
func initGeminiCapabilities() *ProviderCapabilityRegistry {
	registry := &ProviderCapabilityRegistry{
		providerName: "gemini",
		models:       make(map[string]*ModelCapabilitySet),
	}
	
	// Gemini 2.5 Pro - Full capabilities
	registry.RegisterModel("gemini-2.5-pro", &ModelCapabilitySet{
		Streaming:          &AdvancedStreaming{},
		StructuredResponse: &AdvancedStructuredResponse{},
		FunctionCalling:    &AdvancedFunctionCalling{},
		Vision:             &geminiVisionCapability{},
		Caching:            &geminiCachingCapability{},
	})
	
	// Gemini 1.5 Pro - Limited capabilities
	registry.RegisterModel("gemini-1.5-pro", &ModelCapabilitySet{
		Streaming:       &BasicStreaming{},
		FunctionCalling: &BasicFunctionCalling{},
		Vision:          &geminiVisionCapability{},
	})
	
	// Gemini Pro - Basic capabilities only
	registry.RegisterModel("gemini-pro", &ModelCapabilitySet{
		Streaming: &BasicStreaming{},
	})
	
	return registry
}
```

### 9. Type-Safe Usage Examples

```go
// Type-safe capability checking with detailed information
func useProvider(provider CapabilityAwareProvider) {
	// Check streaming with detailed capability info
	streamChecker := provider.CheckStreaming()
	if streamChecker.IsSupported() {
		if cap, ok := streamChecker.GetCapability(); ok {
			fmt.Printf("Max concurrent streams: %d\n", cap.MaxConcurrentStreams())
			fmt.Printf("Supports SSE: %v\n", cap.SupportsServerSentEvents())
		}
	}
	
	// Check structured response with schema validation
	structChecker := provider.CheckStructuredResponse()
	if structChecker.IsSupported() {
		if structChecker.SupportsSchema(SchemaTypeJSONSchema) {
			// Use JSON schema
			if cap, ok := structChecker.GetCapability(); ok {
				fmt.Printf("Max schema complexity: %d\n", cap.MaxSchemaComplexity())
			}
		}
	}
	
	// Check function calling with parameter validation
	funcChecker := provider.CheckFunctionCalling()
	if funcChecker.IsSupported() {
		if funcChecker.CanHandleFunctionCount(5) {
			if cap, ok := funcChecker.GetCapability(); ok {
				fmt.Printf("Supports parallel calls: %v\n", cap.SupportsParallelCalls())
			}
		}
	}
}

// Generic type-safe capability checking
func checkCapability[T Capability](provider CapabilityAwareProvider, capability T) bool {
	return provider.HasCapability(capability)
}

// Usage
provider := NewGeminiProvider("api-key", "gemini-2.5-pro", nil)
hasAdvancedStreaming := checkCapability(provider, &AdvancedStreaming{})
```

This strongly typed approach provides:

1. **Compile-time safety** - Wrong capability types are caught at compile time
2. **Rich capability information** - Each capability can have detailed configuration
3. **Type-specific checkers** - Specialized interfaces for each capability type
4. **Extensibility** - Easy to add new capability types and implementations
5. **Performance** - No runtime string comparisons or reflection in hot paths
6. **IntelliSense support** - IDEs can provide proper autocomplete and type hints