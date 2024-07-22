package llm

import (
	"fmt"
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
func HandleError(err error, fatal bool, logger Logger) {
	if err == nil {
		return
	}

	if llmErr, ok := err.(*LLMError); ok {
		logger.Error(llmErr.Message, "error_type", llmErr.TypeString(), "error", llmErr.Err)
	} else {
		logger.Error("An error occurred", "error", err)
	}

	if fatal {
		// Consider using os.Exit(1) here or returning an error to let the caller decide
		panic(err)
	}
}
