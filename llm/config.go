// File: llm/config.go

package llm

import (
	"os"
	"path/filepath"

	"go.uber.org/zap"
	"gopkg.in/yaml.v2"
)

type Config struct {
	Provider    string  `yaml:"provider"`
	Model       string  `yaml:"model"`
	Temperature float64 `yaml:"temperature"`
	MaxTokens   int     `yaml:"max_tokens"`
	LogLevel    string  `yaml:"log_level"`
}

var DefaultConfig = Config{
	Provider:    "anthropic",
	Model:       "claude-3-opus-20240229",
	Temperature: 0.7,
	MaxTokens:   100,
	LogLevel:    "info",
}

func LoadConfigs(paths ...string) (map[string]*Config, error) {
	configs := make(map[string]*Config)

	if len(paths) == 0 {
		configDir := filepath.Join(os.Getenv("HOME"), ".goal", "configs")
		paths, _ = filepath.Glob(filepath.Join(configDir, "*.yaml"))
	}

	for _, path := range paths {
		config, err := loadSingleConfig(path)
		if err != nil {
			Logger.Warn("Failed to load config", zap.String("path", path), zap.Error(err))
			continue
		}
		configs[filepath.Base(path)] = config
	}

	if len(configs) == 0 {
		Logger.Info("No valid configs found, using default")
		configs["default"] = &DefaultConfig
	}

	return configs, nil
}

func loadSingleConfig(path string) (*Config, error) {
	Logger.Debug("Loading config file", zap.String("path", path))

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	Logger.Info("Config loaded successfully", zap.String("path", path), zap.Any("config", config))
	return &config, nil
}

func (c *Config) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}

	Logger.Debug("Saving config file", zap.String("path", path))
	return os.WriteFile(path, data, 0644)
}
