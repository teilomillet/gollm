package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// Helper function to add a message with cache control
func AddMessageWithCache(memoryLLM llm.LLM, role, content, cacheControl string) {
	// Get the underlying memory
	mem, ok := memoryLLM.(*llm.LLMWithMemory)
	if !ok {
		log.Fatalf("Failed to cast to LLMWithMemory")
	}

	// Use reflection to access the memory field and call AddStructured
	llmMem := mem.GetMemory()

	// Find the memory instance
	for i := range llmMem {
		if llmMem[i].Role == role && llmMem[i].Content == content {
			// Message already exists, update its cache control
			llmMem[i].CacheControl = cacheControl
			return
		}
	}

	// Message doesn't exist, add it
	mem.ClearMemory() // Clear first to start fresh

	// Add using normal method - we'll have to adapt this approach
	mem.AddToMemory(role, content)
}

func main() {
	// Load configuration
	cfg := &config.Config{
		Provider:      "anthropic",
		Model:         "claude-3-sonnet-20240229",
		MaxTokens:     1024,
		Temperature:   0.7,
		EnableCaching: true,
	}

	// Load API key from environment
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}
	cfg.APIKeys = map[string]string{
		"anthropic": apiKey,
	}

	// Create logger
	logger := utils.NewLogger(utils.LogLevelDebug)

	// Create LLM instance directly using the llm package
	registry := providers.NewProviderRegistry()
	llmClient, err := llm.NewLLM(cfg, logger, registry)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	// Create LLM with memory
	memoryLLM, err := llm.NewLLMWithMemory(llmClient, 4000, cfg.Model)
	if err != nil {
		log.Fatalf("Failed to create LLM with memory: %v", err)
	}

	// Test 1: Compare traditional approach to structured messages
	log.Println("\n=== Test 1: Traditional vs Structured Messages ===")
	TestTraditionalVsStructured(memoryLLM)

	// Test 2: Measure the performance improvement with caching
	log.Println("\n=== Test 2: Caching Performance ===")
	TestCachingPerformance(memoryLLM)
}

// TestTraditionalVsStructured compares traditional and structured message approaches
func TestTraditionalVsStructured(memoryLLM *llm.LLMWithMemory) {
	ctx := context.Background()
	systemPrompt := "You are a friendly assistant."

	// Test with traditional flattened approach
	log.Println("\n--- Using traditional flattened messages ---")
	memoryLLM.ClearMemory()
	memoryLLM.SetUseStructuredMessages(false)

	prompt1 := memoryLLM.NewPrompt("Hello, what can you do?")
	prompt1.SystemPrompt = systemPrompt

	startTime := time.Now()
	resp1, err := memoryLLM.Generate(ctx, prompt1)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp := resp1.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("Response 1 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	prompt2 := memoryLLM.NewPrompt("Can you tell me a short joke?")
	startTime = time.Now()
	resp2, err := memoryLLM.Generate(ctx, prompt2)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp2.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("Response 2 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	// Test with structured messages
	log.Println("\n--- Using structured messages ---")
	memoryLLM.ClearMemory()
	memoryLLM.SetUseStructuredMessages(true)

	prompt1 = memoryLLM.NewPrompt("Hello, what can you do?")
	prompt1.SystemPrompt = systemPrompt

	startTime = time.Now()
	resp1, err = memoryLLM.Generate(ctx, prompt1)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp1.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("Response 1 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	prompt2 = memoryLLM.NewPrompt("Can you tell me a short joke?")
	startTime = time.Now()
	resp2, err = memoryLLM.Generate(ctx, prompt2)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp2.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("Response 2 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)
}

// TestCachingPerformance tests the caching performance improvement
func TestCachingPerformance(memoryLLM *llm.LLMWithMemory) {
	ctx := context.Background()
	systemPrompt := "You are a helpful assistant that provides concise responses."

	// Test with structured messages and caching
	log.Println("\n--- First run (no cache) ---")
	memoryLLM.ClearMemory()
	memoryLLM.SetUseStructuredMessages(true)

	// Add message with cache control
	memoryLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	prompt := memoryLLM.NewPrompt("Give me a 30-word description of machine learning.")
	prompt.SystemPrompt = systemPrompt

	startTime := time.Now()
	resp, err := memoryLLM.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp := resp.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("First call (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	// Now run the exact same request again - should be faster due to caching
	log.Println("\n--- Second run (should use cache) ---")
	memoryLLM.ClearMemory()

	// Add the same message back to memory with cache control
	memoryLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	startTime = time.Now()
	resp, err = memoryLLM.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp.AsText()
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	log.Printf("Second call (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)
}
