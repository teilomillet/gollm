package llm

import (
	"errors"
	"fmt"

	"github.com/weave-labs/gollm/internal/logging"
)

// ErrorType represents the category of an LLM error.
// It helps classify errors for appropriate handling and logging.
type ErrorType int

const (
	// ErrorTypeUnknown represents an unclassified error
	ErrorTypeUnknown ErrorType = iota

	// ErrorTypeProvider indicates an error from the LLM provider
	ErrorTypeProvider

	// ErrorTypeRequest indicates an error in preparing or sending the request
	ErrorTypeRequest

	// ErrorTypeResponse indicates an error in processing the response
	ErrorTypeResponse

	// ErrorTypeAPI indicates an error returned by the provider's API
	ErrorTypeAPI

	// ErrorTypeRateLimit indicates the provider's rate limit has been exceeded
	ErrorTypeRateLimit

	// ErrorTypeAuthentication indicates an authentication or authorization failure
	ErrorTypeAuthentication

	// ErrorTypeInvalidInput indicates invalid input parameters or prompt
	ErrorTypeInvalidInput

	// ErrorTypeUnsupported indicates a requested feature is not supported
	ErrorTypeUnsupported
)

// LLMError represents a structured error in the LLM package.
// It implements the error interface and provides additional context
// about the error type and underlying cause.
//
//nolint:revive // LLMError is intentionally named to be clear about its domain
type LLMError struct {
	Err     error
	Message string
	Type    ErrorType
}

// LoggableFields returns a slice of any containing error information
// in a format suitable for structured logging.
func (e *LLMError) LoggableFields() []any {
	return []any{
		"error_type", e.TypeString(),
		"message", e.Message,
		"error", e.Err,
	}
}

// Error implements the error interface.
// It returns a formatted string containing the error type, message,
// and underlying error (if present).
func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s (%s): %v", e.TypeString(), e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.TypeString(), e.Message)
}

// Unwrap returns the underlying error.
// This implements the Go 1.13+ error unwrapping interface.
func (e *LLMError) Unwrap() error {
	return e.Err
}

// TypeString returns a string representation of the error type.
// This is used for logging and error messages.
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
	case ErrorTypeUnsupported:
		return "UnsupportedError"
	default:
		return "UnknownError"
	}
}

// NewLLMError creates a new LLMError with the specified type, message,
// and underlying error.
//
// Parameters:
//   - errType: The category of the error
//   - message: A human-readable error message
//   - err: The underlying error, if any
//
// Returns:
//   - A new LLMError instance
func NewLLMError(errType ErrorType, message string, err error) *LLMError {
	return &LLMError{
		Type:    errType,
		Message: message,
		Err:     err,
	}
}

// HandleError processes an error based on its severity.
// It logs the error appropriately and can optionally terminate the program
// if the error is considered fatal.
//
// Parameters:
//   - err: The error to handle
//   - fatal: If true, the program will panic after logging
//   - logger: The logger to use for error reporting
func HandleError(err error, fatal bool, logger logging.Logger) {
	if err == nil {
		return
	}

	llmErr := &LLMError{}
	if errors.As(err, &llmErr) {
		logger.Error(llmErr.Message, "error_type", llmErr.TypeString(), "error", llmErr.Err)
	} else {
		logger.Error("An error occurred", "error", err)
	}

	if fatal {
		// Consider using os.Exit(1) here or returning an error to let the caller decide
		panic(err)
	}
}
