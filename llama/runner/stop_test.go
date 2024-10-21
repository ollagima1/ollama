package main

import (
	"reflect"
	"testing"
)

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name          string
		pieces        []string
		stop          string
		expected      []string
		expectedTrunc bool
	}{
		{
			name:          "Single word",
			pieces:        []string{"hello", "world"},
			stop:          "world",
			expected:      []string{"hello"},
			expectedTrunc: false,
		},
		{
			name:          "Partial",
			pieces:        []string{"hello", "wor"},
			stop:          "or",
			expected:      []string{"hello", "w"},
			expectedTrunc: true,
		},
		{
			name:          "Suffix",
			pieces:        []string{"Hello", " there", "!"},
			stop:          "!",
			expected:      []string{"Hello", " there"},
			expectedTrunc: false,
		},
		{
			name:          "Suffix partial",
			pieces:        []string{"Hello", " the", "re!"},
			stop:          "there!",
			expected:      []string{"Hello", " "},
			expectedTrunc: true,
		},
		{
			name:          "Middle",
			pieces:        []string{"hello", " wor"},
			stop:          "llo w",
			expected:      []string{"he"},
			expectedTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultTrunc := truncateStop(tt.pieces, tt.stop)
			if !reflect.DeepEqual(result, tt.expected) || resultTrunc != tt.expectedTrunc {
				t.Errorf("truncateStop(%v, %s): have %v (%v); want %v (%v)", tt.pieces, tt.stop, result, resultTrunc, tt.expected, tt.expectedTrunc)
			}
		})
	}
}
