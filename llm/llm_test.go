package llm

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/guiperry/gollm_cerebras/utils"
)

func TestConcurrentOptionsAccess(t *testing.T) {
	mockLogger := &utils.MockLogger{}
	// Configure mock logger to ignore any method calls
	mockLogger.On("Debug", mock.Anything, mock.Anything).Return()

	llm := &LLMImpl{
		Options: make(map[string]interface{}),
		logger:  mockLogger,
	}

	// Test concurrent SetOption calls
	var wg sync.WaitGroup
	concurrency := 100
	wg.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key%d", i)
			llm.SetOption(key, i)
		}(i)
	}

	wg.Wait()

	// Verify all options were set correctly
	for i := 0; i < concurrency; i++ {
		key := fmt.Sprintf("key%d", i)
		llm.optionsMutex.RLock()
		val, ok := llm.Options[key]
		llm.optionsMutex.RUnlock()
		assert.True(t, ok, "Option %s should exist", key)
		assert.Equal(t, i, val, "Option %s should have value %d", key, i)
	}
}
