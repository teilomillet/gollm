package llm

import "time"

// RetryStrategy defines how to handle stream interruptions.
type RetryStrategy interface {
	// ShouldRetry determines if a retry should be attempted.
	ShouldRetry(error) bool

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
	return s.attempts < s.MaxRetries
}

func (s *DefaultRetryStrategy) NextDelay() time.Duration {
	s.attempts++
	delay := s.InitialWait * time.Duration(1<<uint(s.attempts-1))
	if delay > s.MaxWait {
		delay = s.MaxWait
	}
	return delay
}

func (s *DefaultRetryStrategy) Reset() {
	s.attempts = 0
}
