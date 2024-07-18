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
)

// LLMError represents an error in the LLM package
type LLMError struct {
	Type    ErrorType
	Message string
	Err     error
}

func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Err)
	}
	return e.Message
}

func (e *LLMError) Unwrap() error {
	return e.Err
}

// NewLLMError creates a new LLMError
func NewLLMError(errType ErrorType, message string, err error) *LLMError {
	return &LLMError{
		Type:    errType,
		Message: message,
		Err:     err,
	}
}
