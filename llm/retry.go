package llm

import "time"

// RetryStrategy defines how to handle stream interruptions.
type RetryStrategy interface {
	// ShouldRetry determines if a retry should be attempted.
	ShouldRetry(err error) bool

	// NextDelay returns the delay before the next retry.
	NextDelay() time.Duration

	// Reset resets the retry state.
	Reset()
}

// DefaultRetryStrategy implements a simple exponential backoff strategy.
type DefaultRetryStrategy struct {
	MaxRetries  int
	InitialWait time.Duration
	MaxWait     time.Duration
	attempts    int
}

func (s *DefaultRetryStrategy) ShouldRetry(err error) bool {
	// Check if we've exceeded max retries
	if s.attempts >= s.MaxRetries {
		return false
	}
	// Could add logic here to check if err is retryable
	// For now, retry for any non-nil error
	return err != nil
}

const maxShiftAmount = 30 // Cap at 2^30 to prevent overflow

func (s *DefaultRetryStrategy) NextDelay() time.Duration {
	s.attempts++
	// Prevent integer overflow by capping shift amount
	shiftAmount := s.attempts - 1
	if shiftAmount > maxShiftAmount {
		shiftAmount = maxShiftAmount
	}
	delay := s.InitialWait * time.Duration(1<<min(shiftAmount, maxShiftAmount))
	if delay > s.MaxWait {
		delay = s.MaxWait
	}
	return delay
}

func (s *DefaultRetryStrategy) Reset() {
	s.attempts = 0
}
