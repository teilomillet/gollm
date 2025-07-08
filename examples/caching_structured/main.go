package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/guiperry/gollm_cerebras/config"
	"github.com/guiperry/gollm_cerebras/llm"
	"github.com/guiperry/gollm_cerebras/providers"
	"github.com/guiperry/gollm_cerebras/utils"
)

// Helper function to add a message with cache control
func AddMessageWithCache(memLLM llm.LLM, role, content, cacheControl string) {
	// Get the underlying memory
	mem := memLLM.(*llm.LLMWithMemory)

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

	// Get memory LLM as the correct type
	memLLM, ok := memoryLLM.(*llm.LLMWithMemory)
	if !ok {
		log.Fatalf("Failed to cast to LLMWithMemory")
	}

	// Test 1: Compare traditional approach to structured messages
	fmt.Println("\n=== Test 1: Traditional vs Structured Messages ===")
	TestTraditionalVsStructured(memLLM)

	// Test 2: Measure the performance improvement with caching
	fmt.Println("\n=== Test 2: Caching Performance ===")
	TestCachingPerformance(memLLM)
}

// TestTraditionalVsStructured compares traditional and structured message approaches
func TestTraditionalVsStructured(memLLM *llm.LLMWithMemory) {
	ctx := context.Background()
	systemPrompt := "You are a friendly assistant."

	// Test with traditional flattened approach
	fmt.Println("\n--- Using traditional flattened messages ---")
	memLLM.ClearMemory()
	memLLM.SetUseStructuredMessages(false)

	prompt1 := memLLM.NewPrompt("Hello, what can you do?")
	prompt1.SystemPrompt = systemPrompt

	startTime := time.Now()
	resp1, err := memLLM.Generate(ctx, prompt1)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp := resp1
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("Response 1 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	prompt2 := memLLM.NewPrompt("Can you tell me a short joke?")
	startTime = time.Now()
	resp2, err := memLLM.Generate(ctx, prompt2)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp2
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("Response 2 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	// Test with structured messages
	fmt.Println("\n--- Using structured messages ---")
	memLLM.ClearMemory()
	memLLM.SetUseStructuredMessages(true)

	prompt1 = memLLM.NewPrompt("Hello, what can you do?")
	prompt1.SystemPrompt = systemPrompt

	startTime = time.Now()
	resp1, err = memLLM.Generate(ctx, prompt1)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp1
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("Response 1 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	prompt2 = memLLM.NewPrompt("Can you tell me a short joke?")
	startTime = time.Now()
	resp2, err = memLLM.Generate(ctx, prompt2)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp2
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("Response 2 (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)
}

// TestCachingPerformance tests the caching performance improvement
func TestCachingPerformance(memLLM *llm.LLMWithMemory) {
	ctx := context.Background()
	systemPrompt := "You are a helpful assistant that provides concise responses."

	// Test with structured messages and caching
	fmt.Println("\n--- First run (no cache) ---")
	memLLM.ClearMemory()
	memLLM.SetUseStructuredMessages(true)

	// Add message with cache control
	memLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	prompt := memLLM.NewPrompt("Give me a 30-word description of machine learning.")
	prompt.SystemPrompt = systemPrompt

	startTime := time.Now()
	resp, err := memLLM.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp := resp
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("First call (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)

	// Now run the exact same request again - should be faster due to caching
	fmt.Println("\n--- Second run (should use cache) ---")
	memLLM.ClearMemory()

	// Add the same message back to memory with cache control
	memLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	startTime = time.Now()
	resp, err = memLLM.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Truncate response for display
	displayResp = resp
	if len(displayResp) > 100 {
		displayResp = displayResp[:100] + "..."
	}
	fmt.Printf("Second call (%.2fs): %s\n", time.Since(startTime).Seconds(), displayResp)
}
