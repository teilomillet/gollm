package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPromptTemplate(t *testing.T) {
	t.Run("NewPromptTemplate", func(t *testing.T) {
		pt := NewPromptTemplate(
			"test",
			"A test template",
			"Hello, {{.Name}}!",
			WithPromptOptions(WithContext("Some context")),
		)

		assert.Equal(t, "test", pt.Name)
		assert.Equal(t, "A test template", pt.Description)
		assert.Equal(t, "Hello, {{.Name}}!", pt.Template)
		assert.Len(t, pt.Options, 1)
	})

	t.Run("Execute", func(t *testing.T) {
		pt := NewPromptTemplate(
			"greeting",
			"A greeting template",
			"Hello, {{.Name}}! Welcome to {{.Place}}.",
			WithPromptOptions(WithContext("Greeting context")),
		)

		data := map[string]interface{}{
			"Name":  "Alice",
			"Place": "Wonderland",
		}

		prompt, err := pt.Execute(data)
		require.NoError(t, err)
		assert.Equal(t, "Hello, Alice! Welcome to Wonderland.", prompt.Input)
		assert.Equal(t, "Greeting context", prompt.Context)
	})

	t.Run("Execute with invalid template", func(t *testing.T) {
		pt := NewPromptTemplate(
			"invalid",
			"An invalid template",
			"Hello, {{.Name}! Missing closing brace",
		)

		data := map[string]interface{}{
			"Name": "Bob",
		}

		_, err := pt.Execute(data)
		assert.Error(t, err)
	})

	t.Run("WithPromptOptions", func(t *testing.T) {
		pt := NewPromptTemplate(
			"test",
			"A test template",
			"Hello, {{.Name}}!",
			WithPromptOptions(
				WithContext("Test context"),
				WithMaxLength(100),
				WithDirectives("Be polite", "Use formal language"),
			),
		)

		data := map[string]interface{}{"Name": "Charlie"}
		prompt, err := pt.Execute(data)
		require.NoError(t, err)

		assert.Equal(t, "Hello, Charlie!", prompt.Input)
		assert.Equal(t, "Test context", prompt.Context)
		assert.Equal(t, 100, prompt.MaxLength)
		assert.Equal(t, []string{"Be polite", "Use formal language"}, prompt.Directives)
	})
}
