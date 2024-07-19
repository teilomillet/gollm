package llm

import (
	"fmt"
	"go.uber.org/zap"
	"os"
)

// ErrorType represents the type of an error
type ErrorType int

const (
	ErrorTypeUnknown ErrorType = iota
	ErrorTypeProvider
	ErrorTypeRequest
	ErrorTypeResponse
	ErrorTypeAPI
	ErrorTypeRateLimit
	ErrorTypeAuthentication
	ErrorTypeInvalidInput
)

// LLMError represents an error in the LLM package
type LLMError struct {
	Type    ErrorType
	Message string
	Err     error
}

func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s (%s): %v", e.TypeString(), e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.TypeString(), e.Message)
}

func (e *LLMError) Unwrap() error {
	return e.Err
}

func (e *LLMError) TypeString() string {
	switch e.Type {
	case ErrorTypeProvider:
		return "ProviderError"
	case ErrorTypeRequest:
		return "RequestError"
	case ErrorTypeResponse:
		return "ResponseError"
	case ErrorTypeAPI:
		return "APIError"
	case ErrorTypeRateLimit:
		return "RateLimitError"
	case ErrorTypeAuthentication:
		return "AuthenticationError"
	case ErrorTypeInvalidInput:
		return "InvalidInputError"
	default:
		return "UnknownError"
	}
}

// NewLLMError creates a new LLMError
func NewLLMError(errType ErrorType, message string, err error) *LLMError {
	return &LLMError{
		Type:    errType,
		Message: message,
		Err:     err,
	}
}

// HandleError logs the error and exits the program if it's a fatal error
func HandleError(err error, fatal bool) {
	if err == nil {
		return
	}

	if llmErr, ok := err.(*LLMError); ok {
		Logger.Error(llmErr.Message,
			zap.String("error_type", llmErr.TypeString()),
			zap.Error(llmErr.Err))
	} else {
		Logger.Error("An error occurred", zap.Error(err))
	}

	if fatal {
		os.Exit(1)
	}
}
