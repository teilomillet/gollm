# Structured Message Caching Example

This example demonstrates how to use structured messages with caching in gollm. The implementation shows:

1. How to create an LLM instance with memory
2. How to enable structured messages instead of flattened messages
3. How to use cache control to optimize message processing

## Key Concepts

### Structured vs Flattened Messages

Traditional memory in LLMs often flattens conversations into a single text string. 
With our new structured message implementation, each message is preserved as a separate object with its own metadata.

#### Benefits of Structured Messages:
- Better preservation of conversation structure
- More efficient caching (especially with Anthropic and similar APIs)
- Better control over individual message properties

### Default Behavior

By default, all newly created `LLMWithMemory` instances use structured messages. Users only need to explicitly call `SetUseStructuredMessages(false)` if they want to revert to the old flattened approach.

### Cache Control

The example shows how to use cache control to mark messages for caching:

```go
// Add message with cache control
memLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")
```

The cache control options are:
- `"ephemeral"`: The message can be cached but may be evicted if space is needed
- `"persistent"`: The message should be kept in cache indefinitely
- `""` (empty string): No special caching instruction

## API Usage Examples

### Creating a Memory-Enabled LLM with Structured Messages

```go
// Create base LLM instance
baseLLM, err := llm.NewLLM(cfg, logger, registry)
if err != nil {
    log.Fatalf("Failed to create LLM: %v", err)
}

// Create LLM with memory (uses structured messages by default)
memoryLLM, err := llm.NewLLMWithMemory(baseLLM, 4000, cfg.Model)
if err != nil {
    log.Fatalf("Failed to create LLM with memory: %v", err)
}

// Cast to access specific methods
memLLM := memoryLLM.(*llm.LLMWithMemory)
```

### Adding Messages with Cache Control

```go
// Regular message addition (no cache control)
memLLM.AddToMemory("user", "Hello, how are you?")

// Message with cache control
memLLM.AddStructuredMessage("user", "Tell me about machine learning", "ephemeral")
```

### Switching Between Structured and Flattened Messages

```go
// Use structured messages (default)
memLLM.SetUseStructuredMessages(true)

// Use flattened messages (legacy approach)
memLLM.SetUseStructuredMessages(false)
```

## Running the Example

To run the example:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run the example
go run examples/caching_structured/main.go
```

## Expected Output

The program runs two tests:

1. **Traditional vs Structured Messages**: Compares response times between flattened and structured approaches

2. **Caching Performance**: Demonstrates the speed improvement when using caching with structured messages

With caching, you should see the second run complete significantly faster than the first run, often 2-3x faster, depending on network conditions and prompt complexity. 