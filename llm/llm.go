package llm

import (
	"context"
)

// LLM defines the common interface for all LLM providers
type LLM interface {
	Generate(ctx context.Context, prompt string) (string, error)
	SetOption(opt Option) error
	SetProviderOption(opt ProviderOption) error
}

// Option defines a function type for configuring common LLM options
type Option func(*BaseOptions) error

// ProviderOption defines a function type for configuring provider-specific options
type ProviderOption interface {
	apply(LLM) error
}

// BaseOptions contains common options for all LLM providers
type BaseOptions struct {
	Temperature      *float64
	MaxTokens        *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	TopP             *float64
	N                *int
	Stream           *bool
	Stop             []string
	User             *string
}

// LLMMessage defines the structure for individual messages in the request payload
type LLMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// BaseLLM provides a common implementation for all LLM providers
type BaseLLM struct {
	Options BaseOptions
}

func (b *BaseLLM) SetOption(opt Option) error {
	return opt(&b.Options)
}

// WithTemperature sets the temperature for the LLM
func WithTemperature(temperature float64) Option {
	return func(o *BaseOptions) error {
		o.Temperature = &temperature
		return nil
	}
}

// WithMaxTokens sets the max tokens for the LLM
func WithMaxTokens(maxTokens int) Option {
	return func(o *BaseOptions) error {
		o.MaxTokens = &maxTokens
		return nil
	}
}

func WithFrequencyPenalty(penalty float64) Option {
	return func(o *BaseOptions) error {
		o.FrequencyPenalty = &penalty
		return nil
	}
}

func WithPresencePenalty(penalty float64) Option {
	return func(o *BaseOptions) error {
		o.PresencePenalty = &penalty
		return nil
	}
}

func WithTopP(topP float64) Option {
	return func(o *BaseOptions) error {
		o.TopP = &topP
		return nil
	}
}

func WithN(n int) Option {
	return func(o *BaseOptions) error {
		o.N = &n
		return nil
	}
}

func WithStream(stream bool) Option {
	return func(o *BaseOptions) error {
		o.Stream = &stream
		return nil
	}
}

func WithStop(stop []string) Option {
	return func(o *BaseOptions) error {
		o.Stop = stop
		return nil
	}
}

func WithUser(user string) Option {
	return func(o *BaseOptions) error {
		o.User = &user
		return nil
	}
}

