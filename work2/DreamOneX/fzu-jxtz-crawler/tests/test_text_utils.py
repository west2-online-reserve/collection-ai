"""
Tests for text cleaning utilities.
"""
from fzu_jxtz_crawler import normalize_line_endings, clean_whitespace, clean_text


class TestNormalizeLineEndings:
    """Tests for normalize_line_endings function."""
    
    def test_crlf_to_lf(self):
        """Test CRLF conversion to LF."""
        text = "Line 1\r\nLine 2\r\nLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        assert normalize_line_endings(text) == expected
    
    def test_cr_to_lf(self):
        """Test CR conversion to LF."""
        text = "Line 1\rLine 2\rLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        assert normalize_line_endings(text) == expected
    
    def test_mixed_line_endings(self):
        """Test mixed line ending conversion."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        expected = "Line 1\nLine 2\nLine 3\nLine 4"
        assert normalize_line_endings(text) == expected
    
    def test_already_normalized(self):
        """Test text that already has LF endings."""
        text = "Line 1\nLine 2\nLine 3"
        assert normalize_line_endings(text) == text
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_line_endings("") == ""
    
    def test_no_line_endings(self):
        """Test text without line endings."""
        text = "Single line text"
        assert normalize_line_endings(text) == text


class TestCleanWhitespace:
    """Tests for clean_whitespace function."""
    
    def test_strip_leading_trailing(self):
        """Test stripping leading and trailing whitespace."""
        text = "  Hello World  "
        expected = "Hello World"
        assert clean_whitespace(text) == expected
    
    def test_collapse_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        text = "Hello    World   Test"
        expected = "Hello World Test"
        assert clean_whitespace(text) == expected
    
    def test_collapse_tabs(self):
        """Test collapsing tabs."""
        text = "Hello\t\t\tWorld"
        expected = "Hello World"
        assert clean_whitespace(text) == expected
    
    def test_collapse_multiple_newlines(self):
        """Test collapsing multiple newlines to max 2."""
        text = "Paragraph 1\n\n\n\n\nParagraph 2"
        expected = "Paragraph 1\n\nParagraph 2"
        assert clean_whitespace(text) == expected
    
    def test_preserve_single_newlines(self):
        """Test that single newlines are preserved."""
        text = "Line 1\nLine 2\nLine 3"
        assert clean_whitespace(text) == text
    
    def test_preserve_double_newlines(self):
        """Test that double newlines (paragraph breaks) are preserved."""
        text = "Paragraph 1\n\nParagraph 2"
        assert clean_whitespace(text) == text
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert clean_whitespace("") == ""
    
    def test_complex_whitespace(self):
        """Test complex whitespace cleaning."""
        text = "  Hello   World  \n\n\n  Next  Paragraph  "
        expected = "Hello World\n\nNext Paragraph"
        assert clean_whitespace(text) == expected


class TestCleanText:
    """Tests for combined clean_text function."""
    
    def test_combined_cleaning(self):
        """Test combined line ending normalization and whitespace cleaning."""
        text = "  Line 1\r\nLine 2  \r\n\r\n\r\nLine 3  "
        expected = "Line 1\nLine 2\n\nLine 3"
        assert clean_text(text) == expected
    
    def test_crlf_with_extra_spaces(self):
        """Test CRLF conversion with extra spaces."""
        text = "Hello   World\r\n  Next   Line  "
        expected = "Hello World\nNext Line"
        assert clean_text(text) == expected
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert clean_text("") == ""
    
    def test_already_clean(self):
        """Test text that's already clean."""
        text = "Clean text\nWith proper formatting"
        assert clean_text(text) == text
