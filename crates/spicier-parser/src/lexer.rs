//! SPICE netlist lexer.

use crate::error::{Error, Result};

/// Token types for SPICE netlists.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Element name (R1, C1, V1, etc.)
    Name(String),
    /// Node identifier (0, 1, vdd, gnd, etc.)
    Node(String),
    /// Numeric value with optional suffix (1k, 4.7u, etc.)
    Value(String),
    /// Dot command (.op, .dc, .tran, etc.)
    Command(String),
    /// Equal sign for parameters
    Equals,
    /// Opening parenthesis
    LParen,
    /// Closing parenthesis
    RParen,
    /// Comma separator (for V(node1, node2) syntax)
    Comma,
    /// End of line
    Eol,
    /// End of file
    Eof,
    // Arithmetic operators (for behavioral source expressions)
    /// Multiply operator
    Star,
    /// Divide operator
    Slash,
    /// Power operator
    Caret,
}

/// A token with its source location.
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub line: usize,
    pub column: usize,
}

/// Lexer for SPICE netlists.
pub struct Lexer<'a> {
    #[allow(dead_code)]
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    line: usize,
    column: usize,
    at_line_start: bool,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            line: 1,
            column: 1,
            at_line_start: true,
        }
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> Result<SpannedToken> {
        self.skip_whitespace();

        let line = self.line;
        let column = self.column;

        match self.peek_char() {
            None => Ok(SpannedToken {
                token: Token::Eof,
                line,
                column,
            }),
            Some('\n') => {
                self.advance();
                self.line += 1;
                self.column = 1;
                self.at_line_start = true;
                Ok(SpannedToken {
                    token: Token::Eol,
                    line,
                    column,
                })
            }
            Some('*') if self.at_line_start => {
                // Comment line - skip to end
                self.skip_to_eol();
                self.next_token()
            }
            Some(';') => {
                // Inline comment - skip to end
                self.skip_to_eol();
                self.next_token()
            }
            Some('.') => {
                // Dot command
                self.advance();
                let cmd = self.read_identifier();
                Ok(SpannedToken {
                    token: Token::Command(cmd.to_uppercase()),
                    line,
                    column,
                })
            }
            Some('=') => {
                self.advance();
                Ok(SpannedToken {
                    token: Token::Equals,
                    line,
                    column,
                })
            }
            Some('(') => {
                self.advance();
                Ok(SpannedToken {
                    token: Token::LParen,
                    line,
                    column,
                })
            }
            Some(')') => {
                self.advance();
                Ok(SpannedToken {
                    token: Token::RParen,
                    line,
                    column,
                })
            }
            Some(',') => {
                self.advance();
                Ok(SpannedToken {
                    token: Token::Comma,
                    line,
                    column,
                })
            }
            Some('+') if self.at_line_start => {
                // Continuation line - treat as whitespace and continue
                self.advance();
                self.at_line_start = false;
                self.next_token()
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_identifier();
                self.at_line_start = false;
                Ok(SpannedToken {
                    token: Token::Name(ident),
                    line,
                    column,
                })
            }
            Some(c) if c.is_ascii_digit() || c == '-' || c == '+' => {
                let value = self.read_value();
                self.at_line_start = false;
                Ok(SpannedToken {
                    token: Token::Value(value),
                    line,
                    column,
                })
            }
            // Arithmetic operators for behavioral source expressions
            Some('*') => {
                self.advance();
                self.at_line_start = false;
                Ok(SpannedToken {
                    token: Token::Star,
                    line,
                    column,
                })
            }
            Some('/') => {
                self.advance();
                self.at_line_start = false;
                Ok(SpannedToken {
                    token: Token::Slash,
                    line,
                    column,
                })
            }
            Some('^') => {
                self.advance();
                self.at_line_start = false;
                Ok(SpannedToken {
                    token: Token::Caret,
                    line,
                    column,
                })
            }
            Some(c) => Err(Error::ParseError {
                line,
                message: format!("unexpected character: '{}'", c),
            }),
        }
    }

    /// Tokenize the entire input.
    pub fn tokenize(mut self) -> Result<Vec<SpannedToken>> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let is_eof = token.token == Token::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    fn advance(&mut self) -> Option<char> {
        if let Some((_, c)) = self.chars.next() {
            self.column += 1;
            Some(c)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c == ' ' || c == '\t' || c == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_to_eol(&mut self) {
        while let Some(c) = self.peek_char() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut ident = String::new();
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn read_value(&mut self) -> String {
        let mut value = String::new();

        // Optional sign
        if let Some(c) = self.peek_char()
            && (c == '-' || c == '+')
        {
            value.push(c);
            self.advance();
        }

        // Digits and decimal point
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Exponent
        if let Some(c) = self.peek_char()
            && (c == 'e' || c == 'E')
        {
            value.push(c);
            self.advance();

            // Optional exponent sign
            if let Some(c) = self.peek_char()
                && (c == '-' || c == '+')
            {
                value.push(c);
                self.advance();
            }

            // Exponent digits
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() {
                    value.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // SI suffix (k, M, u, n, p, etc.)
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphabetic() {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_resistor() {
        let input = "R1 1 0 1k";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 5); // R1, 1, 0, 1k, Eof
        assert_eq!(tokens[0].token, Token::Name("R1".into()));
        // Node numbers are lexed as Value tokens (start with digit)
        assert_eq!(tokens[1].token, Token::Value("1".into()));
        assert_eq!(tokens[2].token, Token::Value("0".into()));
        assert_eq!(tokens[3].token, Token::Value("1k".into()));
        assert_eq!(tokens[4].token, Token::Eof);
    }

    #[test]
    fn test_comment() {
        let input = "* This is a comment\nR1 1 0 1k";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        // Should skip comment, then get R1 line
        assert!(tokens.iter().any(|t| t.token == Token::Name("R1".into())));
    }

    #[test]
    fn test_dot_command() {
        let input = ".op";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token, Token::Command("OP".into()));
    }

    #[test]
    fn test_continuation() {
        let input = "R1 1\n+ 0 1k";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        // Should have: R1, 1, Eol, 0, 1k, Eof
        let names: Vec<_> = tokens
            .iter()
            .filter_map(|t| match &t.token {
                Token::Name(n) => Some(n.clone()),
                Token::Value(v) => Some(v.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(names, vec!["R1", "1", "0", "1k"]);
    }

    #[test]
    fn test_negative_value() {
        let input = "V1 1 0 -5";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        assert!(tokens.iter().any(|t| t.token == Token::Value("-5".into())));
    }

    #[test]
    fn test_scientific_notation() {
        let input = "C1 1 0 1e-12";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::Value("1e-12".into()))
        );
    }
}
