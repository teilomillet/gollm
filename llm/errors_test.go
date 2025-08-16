package llm

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/weave-labs/gollm/utils"
)

func TestLLMError(t *testing.T) {
	testCases := []struct {
		name          string
		errType       ErrorType
		message       string
		underlyingErr error
		expectedStr   string
	}{
		{
			name:          "Provider error with underlying error",
			errType:       ErrorTypeProvider,
			message:       "Failed to connect",
			underlyingErr: errors.New("connection refused"),
			expectedStr:   "ProviderError (Failed to connect): connection refused",
		},
		{
			name:        "API error without underlying error",
			errType:     ErrorTypeAPI,
			message:     "Rate limit exceeded",
			expectedStr: "APIError: Rate limit exceeded",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			llmErr := NewLLMError(tc.errType, tc.message, tc.underlyingErr)

			assert.Equal(t, tc.errType, llmErr.Type)
			assert.Equal(t, tc.message, llmErr.Message)
			assert.Equal(t, tc.underlyingErr, llmErr.Err)
			assert.Equal(t, tc.expectedStr, llmErr.Error())

			if tc.underlyingErr != nil {
				assert.Equal(t, tc.underlyingErr, errors.Unwrap(llmErr))
			}

			fields := llmErr.LoggableFields()
			assert.Len(t, fields, 6)
			assert.Equal(t, "error_type", fields[0])
			assert.Equal(t, llmErr.TypeString(), fields[1])
		})
	}
}

func TestHandleError(t *testing.T) {
	mockLogger := new(utils.MockLogger)

	t.Run("Handle LLMError", func(t *testing.T) {
		mockLogger.On("Error", mock.Anything, mock.Anything).Return()
		llmErr := NewLLMError(ErrorTypeAPI, "API Error", nil)
		HandleError(llmErr, false, mockLogger)

		mockLogger.AssertCalled(t, "Error", "API Error", mock.Anything)
		assert.Equal(t, 1, mockLogger.ErrorCallCount)
		assert.Equal(t, "API Error", mockLogger.LastErrorMessage)
	})

	t.Run("Handle generic error", func(t *testing.T) {
		mockLogger.On("Error", mock.Anything, mock.Anything).Return()
		genericErr := errors.New("generic error")
		HandleError(genericErr, false, mockLogger)

		mockLogger.AssertCalled(t, "Error", "An error occurred", mock.Anything)
		assert.Equal(t, 2, mockLogger.ErrorCallCount)
		assert.Equal(t, "An error occurred", mockLogger.LastErrorMessage)
	})

	t.Run("Fatal error", func(t *testing.T) {
		mockLogger.On("Error", mock.Anything, mock.Anything).Return()
		defer func() {
			r := recover()
			require.NotNil(t, r, "The code did not panic")
			assert.Equal(t, "fatal error", r.(error).Error())
		}()

		fatalErr := errors.New("fatal error")
		HandleError(fatalErr, true, mockLogger)
	})
}
