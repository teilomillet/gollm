// file: internal/opto/parameter.go

package opto

import "fmt"

// Parameter represents a trainable parameter in the OPTO system
type Parameter interface {
	GetValue() interface{}
	SetValue(value interface{}) error
	GetConstraints() string
}

// PromptParameter is a concrete implementation of Parameter for prompt values
type PromptParameter struct {
	value       string
	constraints string
}

// NewPromptParameter creates a new PromptParameter
func NewPromptParameter(value, constraints string) *PromptParameter {
	return &PromptParameter{value: value, constraints: constraints}
}

func (p *PromptParameter) GetValue() interface{} {
	return p.value
}

func (p *PromptParameter) SetValue(value interface{}) error {
	if str, ok := value.(string); ok {
		p.value = str
		return nil
	}
	return fmt.Errorf("invalid type for PromptParameter: expected string")
}

func (p *PromptParameter) GetConstraints() string {
	return p.constraints
}

func (p *PromptParameter) GetPromptText() string {
	return p.value
}

// StringParameter is a concrete implementation of Parameter for string values
type StringParameter struct {
	value       string
	constraints string
}

func NewStringParameter(value, constraints string) *StringParameter {
	return &StringParameter{value: value, constraints: constraints}
}

func (p *StringParameter) GetValue() interface{} {
	return p.value
}

func (p *StringParameter) SetValue(value interface{}) error {
	if str, ok := value.(string); ok {
		p.value = str
		return nil
	}
	return fmt.Errorf("invalid type for StringParameter")
}

func (p *StringParameter) GetConstraints() string {
	return p.constraints
}

type NumberParameter struct {
	value       float64
	constraints string
}

func NewNumberParameter(value float64, constraints string) *NumberParameter {
	return &NumberParameter{value: value, constraints: constraints}
}

func (p *NumberParameter) GetValue() interface{} {
	return p.value
}

func (p *NumberParameter) SetValue(value interface{}) error {
	if num, ok := value.(float64); ok {
		p.value = num
		return nil
	}
	return fmt.Errorf("invalid type for NumberParameter")
}

func (p *NumberParameter) GetConstraints() string {
	return p.constraints
}

type CodeParameter struct {
	value       string
	constraints string
}

func NewCodeParameter(value, constraints string) *CodeParameter {
	return &CodeParameter{value: value, constraints: constraints}
}

func (p *CodeParameter) GetValue() interface{} {
	return p.value
}

func (p *CodeParameter) SetValue(value interface{}) error {
	if str, ok := value.(string); ok {
		p.value = str
		return nil
	}
	return fmt.Errorf("invalid type for CodeParameter")
}

func (p *CodeParameter) GetConstraints() string {
	return p.constraints
}
