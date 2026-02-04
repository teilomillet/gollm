package gollm_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
)

// createTestLLM attempts to create a test LLM with the given options.
// Returns the LLM instance and true if successful, or nil and false if the provider
// is not available (in which case the test should be skipped).
func createTestLLM(t *testing.T, opts ...gollm.ConfigOption) (gollm.LLM, bool) {
	t.Helper()
	llm, err := gollm.NewLLM(opts...)
	if err != nil {
		t.Skipf("Skipping test: could not create LLM (provider may not be available): %v", err)
		return nil, false
	}
	return llm, true
}

// TestLLMWithoutMemory tests memory methods when memory is not enabled
func TestLLMWithoutMemory(t *testing.T) {
	llm, ok := createTestLLM(t,
		gollm.SetProvider("lmstudio"),
		gollm.SetModel("local-model"),
		gollm.SetMaxTokens(100),
		// No SetMemory - memory is disabled
	)
	if !ok {
		return
	}

	t.Run("HasMemory returns false when memory not enabled", func(t *testing.T) {
		assert.False(t, llm.HasMemory())
	})

	t.Run("GetMemory returns nil when memory not enabled", func(t *testing.T) {
		messages := llm.GetMemory()
		assert.Nil(t, messages)
	})

	t.Run("ClearMemory is safe when memory not enabled", func(t *testing.T) {
		// Should not panic
		llm.ClearMemory()
	})

	t.Run("AddToMemory is safe when memory not enabled", func(t *testing.T) {
		// Should not panic
		llm.AddToMemory("user", "Hello")
		// Memory still nil
		assert.Nil(t, llm.GetMemory())
	})

	t.Run("AddStructuredMessage is safe when memory not enabled", func(t *testing.T) {
		// Should not panic
		llm.AddStructuredMessage("user", "Hello", "ephemeral")
		// Memory still nil
		assert.Nil(t, llm.GetMemory())
	})

	t.Run("SetUseStructuredMessages is safe when memory not enabled", func(t *testing.T) {
		// Should not panic
		llm.SetUseStructuredMessages(true)
		llm.SetUseStructuredMessages(false)
	})
}

// TestLLMWithMemory tests memory methods when memory is enabled
func TestLLMWithMemory(t *testing.T) {
	llm, ok := createTestLLM(t,
		gollm.SetProvider("lmstudio"),
		gollm.SetModel("local-model"),
		gollm.SetMaxTokens(100),
		gollm.SetMemory(4096), // Enable memory
	)
	if !ok {
		return
	}

	t.Run("HasMemory returns true when memory enabled", func(t *testing.T) {
		assert.True(t, llm.HasMemory())
	})

	t.Run("GetMemory returns empty slice initially", func(t *testing.T) {
		llm.ClearMemory()
		messages := llm.GetMemory()
		assert.NotNil(t, messages)
		assert.Equal(t, 0, len(messages))
	})

	t.Run("AddToMemory adds messages", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddToMemory("user", "Hello")
		llm.AddToMemory("assistant", "Hi there")

		messages := llm.GetMemory()
		assert.Equal(t, 2, len(messages))
		assert.Equal(t, "user", messages[0].Role)
		assert.Equal(t, "Hello", messages[0].Content)
		assert.Equal(t, "assistant", messages[1].Role)
	})

	t.Run("AddStructuredMessage adds message with cache control", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddStructuredMessage("user", "Cached message", "ephemeral")

		messages := llm.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "ephemeral", messages[0].CacheControl)
	})

	t.Run("ClearMemory clears all messages", func(t *testing.T) {
		llm.AddToMemory("user", "Test")
		assert.Greater(t, len(llm.GetMemory()), 0)

		llm.ClearMemory()
		assert.Equal(t, 0, len(llm.GetMemory()))
	})

	t.Run("SetUseStructuredMessages toggles mode", func(t *testing.T) {
		// Should not panic and should work
		llm.SetUseStructuredMessages(false)
		llm.SetUseStructuredMessages(true)
	})
}

// TestMemoryEdgeCases tests edge cases for memory methods
func TestMemoryEdgeCases(t *testing.T) {
	llm, ok := createTestLLM(t,
		gollm.SetProvider("lmstudio"),
		gollm.SetModel("local-model"),
		gollm.SetMemory(4096),
	)
	if !ok {
		return
	}

	t.Run("AddToMemory with empty content", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddToMemory("user", "")

		messages := llm.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].Content)
	})

	t.Run("AddToMemory with empty role", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddToMemory("", "Hello")

		messages := llm.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].Role)
	})

	t.Run("AddStructuredMessage with empty cache control", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddStructuredMessage("user", "Hello", "")

		messages := llm.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].CacheControl)
	})

	t.Run("multiple ClearMemory calls are safe", func(t *testing.T) {
		llm.ClearMemory()
		llm.ClearMemory()
		llm.ClearMemory()
		assert.Equal(t, 0, len(llm.GetMemory()))
	})

	t.Run("GetMemory returns independent copy", func(t *testing.T) {
		llm.ClearMemory()
		llm.AddToMemory("user", "Original")

		messages1 := llm.GetMemory()
		messages2 := llm.GetMemory()

		// Modifying one copy should not affect the other
		messages1[0].Content = "Modified"
		assert.Equal(t, "Original", messages2[0].Content)
	})
}
