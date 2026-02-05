package llm

import (
	"errors"
	"sync"
	"testing"
)

// testStruct is a simple struct for validation testing
type testStruct struct {
	Name  string `validate:"required"`
	Value int    `validate:"min=1,max=100"`
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		wantErr bool
	}{
		{
			name:    "valid struct",
			input:   &testStruct{Name: "test", Value: 50},
			wantErr: false,
		},
		{
			name:    "missing required field",
			input:   &testStruct{Name: "", Value: 50},
			wantErr: true,
		},
		{
			name:    "value below minimum",
			input:   &testStruct{Name: "test", Value: 0},
			wantErr: true,
		},
		{
			name:    "value above maximum",
			input:   &testStruct{Name: "test", Value: 101},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateWithCustomValidator(t *testing.T) {
	t.Run("nil custom validator uses default", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		invalid := &testStruct{Name: "", Value: 50}

		if err := ValidateWithCustomValidator(valid, nil); err != nil {
			t.Errorf("expected no error for valid struct, got %v", err)
		}

		if err := ValidateWithCustomValidator(invalid, nil); err == nil {
			t.Error("expected error for invalid struct, got nil")
		}
	})

	t.Run("custom validator that always passes", func(t *testing.T) {
		invalid := &testStruct{Name: "", Value: 0}
		alwaysPass := func(v interface{}) error { return nil }

		if err := ValidateWithCustomValidator(invalid, alwaysPass); err != nil {
			t.Errorf("expected custom validator to pass, got %v", err)
		}
	})

	t.Run("custom validator that always fails", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		customErr := errors.New("custom validation error")
		alwaysFail := func(v interface{}) error { return customErr }

		err := ValidateWithCustomValidator(valid, alwaysFail)
		if err != customErr {
			t.Errorf("expected custom error, got %v", err)
		}
	})

	t.Run("custom validator with fallback to default", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		invalid := &testStruct{Name: "", Value: 50}

		// Custom validator that only validates Name field, falls back for rest
		customValidator := func(v interface{}) error {
			if ts, ok := v.(*testStruct); ok {
				if ts.Name == "skip" {
					return nil // Skip validation for specific name
				}
			}
			return DefaultValidate(v)
		}

		// Valid struct should pass
		if err := ValidateWithCustomValidator(valid, customValidator); err != nil {
			t.Errorf("expected valid struct to pass, got %v", err)
		}

		// Invalid struct should fail (fallback to default)
		if err := ValidateWithCustomValidator(invalid, customValidator); err == nil {
			t.Error("expected invalid struct to fail")
		}

		// Struct with "skip" name should pass even if otherwise invalid
		skipStruct := &testStruct{Name: "skip", Value: 0}
		if err := ValidateWithCustomValidator(skipStruct, customValidator); err != nil {
			t.Errorf("expected 'skip' struct to pass, got %v", err)
		}
	})
}

func TestValidateWithCustomValidatorConcurrency(t *testing.T) {
	// This test verifies that custom validators are scoped and don't interfere
	// with each other when called concurrently from different goroutines.
	const numGoroutines = 100
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines*2)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(2)

		// Goroutine with custom validator that always passes
		go func() {
			defer wg.Done()
			invalid := &testStruct{Name: "", Value: 0}
			alwaysPass := func(v interface{}) error { return nil }

			err := ValidateWithCustomValidator(invalid, alwaysPass)
			if err != nil {
				errCh <- errors.New("alwaysPass validator should have passed")
			}
		}()

		// Goroutine with nil validator (uses default)
		go func() {
			defer wg.Done()
			invalid := &testStruct{Name: "", Value: 0}

			err := ValidateWithCustomValidator(invalid, nil)
			if err == nil {
				errCh <- errors.New("nil validator should have failed for invalid struct")
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}

func TestDefaultValidate(t *testing.T) {
	valid := &testStruct{Name: "test", Value: 50}
	invalid := &testStruct{Name: "", Value: 0}

	if err := DefaultValidate(valid); err != nil {
		t.Errorf("expected valid struct to pass, got %v", err)
	}

	if err := DefaultValidate(invalid); err == nil {
		t.Error("expected invalid struct to fail")
	}
}
